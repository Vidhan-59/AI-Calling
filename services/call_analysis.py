"""What a finished scenario call meant, stored on its scenario_calls row
(data/legacy_app.db) for planners and reports.

Two stages:
  store_details()  - right at hang-up, no AI: the reason in the caller's own
                     words, duration, turn count, latency, full transcript.
  analyze_call()   - one Qwen call after hang-up: sentiment, emotion, reason
                     category, the reason in English, a one-line summary and
                     whether a human should follow up.

The analysis deliberately runs after the call, never during it: a per-turn
sentiment call would compete with live classification for the GPU and add
latency to every reply. Calls are analysed one at a time so a burst of
hang-ups can't crowd out live calls.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from local_ai import llm_engine as local_llm_engine
from services import conversation_store, scenario_store
from services.call_logger import set_call_analysis

logger = logging.getLogger("call_analysis")

SENTIMENTS = ["positive", "neutral", "negative"]
EMOTIONS = ["calm", "happy", "apologetic", "frustrated", "angry", "anxious", "sad", "confused"]
REASON_CATEGORIES = [
    "none", "illness", "family", "emergency", "transport", "personal_work",
    "other_job", "planned_leave", "shift_timing", "pay_or_dispute", "other",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": SENTIMENTS},
        "sentiment_score": {"type": "number"},
        "emotion": {"type": "string", "enum": EMOTIONS},
        "reason_category": {"type": "string", "enum": REASON_CATEGORIES},
        "reason_english": {"type": "string"},
        "summary": {"type": "string"},
        "follow_up_required": {"type": "boolean"},
        "follow_up_note": {"type": "string"},
    },
    "required": ["sentiment", "sentiment_score", "emotion", "reason_category", "reason_english",
                 "summary", "follow_up_required", "follow_up_note"],
}

SYSTEM_PROMPT = """You review a finished phone call. An automated assistant from a staffing company \
called a worker (often in Hindi or Gujarati) about a shift. Judge the WORKER, not the assistant.
The recorded outcome was decided during the call and is correct - your summary must agree with it. (Callers often correct themselves: "nahi nahi, main aa jaunga" = "no no, I WILL come" = confirmed.)

Return JSON:
- sentiment: the worker's overall attitude - positive, neutral or negative. A polite "no, I'm sick" is \
neutral, not negative; negative means irritated, angry or upset.
- sentiment_score: -1.0 (very negative) to 1.0 (very positive), 0 for neutral.
- emotion: the single best fit.
- reason_category: why the worker cannot come. "none" if they are coming or gave no reason. \
illness = the worker's OWN health only; family = family matters, including a relative who is ill or \
in hospital, a function, "family issues"; emergency = accident, death, sudden crisis; transport = no bus/vehicle, distance; \
personal_work = their own errands; other_job = working elsewhere; planned_leave = leave/holiday/travel; \
shift_timing = the time or shift does not suit them; pay_or_dispute = pay, or says it is not their duty.
- reason_english: the worker's reason as a short English phrase, faithful to what they said. "" if none.
- summary: one plain English sentence - what the worker said and the result.
- follow_up_required: true only if a human should call back - angry or upset worker, a dispute, \
an emergency, asked for a supervisor, a wrong number, or the answer stayed unclear.
- follow_up_note: why, in a few English words. "" if no follow-up.
"""

_OUTCOME_MEANING = {
    "confirmed": "the worker WILL come",
    "confirmed_self_reported_verify": "the worker says they already confirmed - they will come",
    "absent_with_reason": "the worker will NOT come",
    "disputed_flagged_to_planner": "the worker disputes the duty",
    "escalated_supervisor_notified": "the worker asked for a supervisor",
    "data_mismatch_flagged": "wrong person or wrong details",
    "no_response": "no clear answer",
}

_analysis_lock = asyncio.Semaphore(1)
_tasks: set[asyncio.Task] = set()  # keep references so tasks aren't garbage-collected


def _transcript(messages: list[dict]) -> str:
    return "\n".join(
        f"{'Caller' if m['role'] == 'user' else 'Bot'}: {m['content']}"
        for m in messages if (m.get("content") or "").strip()
    )


def _reason(collected: dict) -> Optional[str]:
    if collected.get("absence_reason"):
        return collected["absence_reason"]
    return next((v for k, v in collected.items() if "reason" in k and v), None)


def _duration_secs(started_at: Optional[str], ended_at: Optional[str]) -> Optional[float]:
    try:
        return round((datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


def store_details(call_sid: str, avg_turn_latency_ms: Optional[int] = None) -> bool:
    """Write the no-AI fields. Call after scenario_store.record_outcome (it
    sets ended_at). Returns whether the caller said anything - i.e. whether
    there is something for analyze_call to analyse."""
    rec = scenario_store.get(call_sid)
    if not rec:
        return False
    messages = conversation_store.get_messages(call_sid)
    turns = sum(1 for m in messages if m["role"] == "user")
    scenario_store.record_call_details(
        call_sid,
        absence_reason=_reason(rec.get("collected") or {}),
        duration_secs=_duration_secs(rec.get("started_at"), rec.get("ended_at")),
        turn_count=turns,
        avg_turn_latency_ms=avg_turn_latency_ms,
        transcript_text=_transcript(messages),
        analysis_status="pending" if turns else "skipped",
    )
    return turns > 0


def _clean(raw: dict) -> Optional[dict]:
    """Validate the model's JSON; None if it isn't an analysis at all (the
    engine's invalid-JSON fallback returns a classification-shaped dict)."""
    if raw.get("sentiment") not in SENTIMENTS:
        return None
    try:
        score = max(-1.0, min(1.0, float(raw.get("sentiment_score", 0.0))))
    except (TypeError, ValueError):
        score = 0.0
    follow_up = bool(raw.get("follow_up_required"))
    return {
        "sentiment": raw["sentiment"],
        "sentiment_score": round(score, 2),
        "emotion": raw.get("emotion") if raw.get("emotion") in EMOTIONS else None,
        "reason_category": raw.get("reason_category") if raw.get("reason_category") in REASON_CATEGORIES else "other",
        "reason_english": (raw.get("reason_english") or "").strip() or None,
        "summary": (raw.get("summary") or "").strip() or None,
        "follow_up_required": follow_up,
        "follow_up_note": ((raw.get("follow_up_note") or "").strip() or None) if follow_up else None,
    }


async def analyze_call(call_sid: str) -> Optional[dict]:
    rec = scenario_store.get(call_sid)
    if not rec:
        return None
    messages = conversation_store.get_messages(call_sid)
    if not any(m["role"] == "user" for m in messages):
        scenario_store.record_analysis(call_sid, None, status="skipped")  # nobody spoke - nothing to judge
        return None
    reason = _reason(rec.get("collected") or {})
    user_text = (
        f"Call type: {rec['scenario_id']} | language: {rec['language']}\n"
        f"Recorded outcome: {rec.get('outcome')} ({_OUTCOME_MEANING.get(rec.get('outcome'), 'see transcript')})\n"
        f"Reason recorded (worker's words): {reason or '-'}\n\n"
        f"Transcript:\n{_transcript(messages)}"
    )
    try:
        async with _analysis_lock:
            raw = await local_llm_engine.classify_structured(
                system_prompt=SYSTEM_PROMPT, user_text=user_text, schema=SCHEMA, max_tokens=300,
            )
        analysis = _clean(raw)
    except Exception as e:  # noqa: BLE001 - Ollama down etc.; the call's own data is already saved
        logger.warning("[%s] post-call analysis failed: %r", call_sid, e)
        analysis = None
    if analysis is None:
        scenario_store.record_analysis(call_sid, None, status="failed")
        return None
    if analysis["reason_category"] == "none" and reason:
        analysis["reason_category"] = "other"  # a reason was collected - never report it as "none"
    scenario_store.record_analysis(call_sid, analysis, status="done")
    set_call_analysis(call_sid, analysis)
    logger.info(
        "[%s] analysis: %s (%s) reason=%s follow_up=%s | %s", call_sid, analysis["sentiment"],
        analysis["emotion"], analysis["reason_category"], analysis["follow_up_required"], analysis["summary"],
    )
    return analysis


async def _analyze_safely(call_sid: str) -> None:
    try:
        await analyze_call(call_sid)
    except Exception:  # noqa: BLE001
        logger.exception("[%s] post-call analysis crashed", call_sid)


def schedule_analysis(call_sid: str) -> None:
    """Fire-and-forget from the call's own teardown."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.warning("[%s] no event loop - analysis not scheduled", call_sid)
        return
    task = loop.create_task(_analyze_safely(call_sid))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
