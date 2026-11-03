"""Per-call transcript logger.

Every call (scenario or generic) gets a structured log file written to
data/call_logs/<call_sid>.json containing the full conversation transcript,
timing, outcome, and metadata. This supplements the SQLite storage and gives
an easy-to-grep, easy-to-ship file-per-call audit trail.

Files are written incrementally (appended per turn) so even a mid-call crash
preserves whatever was captured up to that point.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from config import DATA_DIR

CALL_LOGS_DIR = os.path.join(DATA_DIR, "call_logs")
os.makedirs(CALL_LOGS_DIR, exist_ok=True)

_lock = threading.Lock()


def _sanitize_sid(call_sid: str) -> str:
    """Make the call_sid safe for use as a filename."""
    return call_sid.replace(":", "_").replace("/", "_").replace("\\", "_")


def _log_path(call_sid: str) -> str:
    return os.path.join(CALL_LOGS_DIR, f"{_sanitize_sid(call_sid)}.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_log(call_sid: str) -> dict:
    path = _log_path(call_sid)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_log(call_sid: str, data: dict) -> None:
    path = _log_path(call_sid)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def init_call_log(
    call_sid: str,
    *,
    scenario_id: Optional[str] = None,
    language: Optional[str] = None,
    to_number: Optional[str] = None,
    from_number: Optional[str] = None,
    agent_id: Optional[str] = None,
    facts: Optional[dict] = None,
    attempt: int = 1,
    direction: str = "outbound",
) -> None:
    """Create the initial log file for a new call."""
    with _lock:
        data = {
            "call_sid": call_sid,
            "scenario_id": scenario_id,
            "agent_id": agent_id,
            "language": language,
            "to_number": to_number,
            "from_number": from_number,
            "direction": direction,
            "attempt": attempt,
            "facts": facts or {},
            "started_at": _now_iso(),
            "ended_at": None,
            "status": "in_progress",
            "outcome": None,
            "matched_branch": None,
            "collected": {},
            "detected_language": None,
            "turns": [],
            "escalation_flags": [],
        }
        _write_log(call_sid, data)


def log_turn(
    call_sid: str,
    *,
    role: str,
    content: str,
    branch: Optional[str] = None,
    outcome: Optional[str] = None,
    end_call: bool = False,
    sentiment: Optional[dict] = None,
    classification: Optional[dict] = None,
    timings_ms: Optional[dict] = None,
) -> int:
    """Append one turn (user utterance or assistant reply) to the log.
    `timings_ms` records where the turn's latency went (e.g. stt, classify,
    total) so slow turns can be diagnosed from the log alone.
    Returns the turn's seq number, so a caller that doesn't have the
    sentiment yet (see set_turn_sentiment) can log the turn immediately and
    backfill it once classification finishes, without disturbing transcript
    order."""
    with _lock:
        data = _read_log(call_sid)
        if not data:
            # Log wasn't initialized (shouldn't happen, but be defensive)
            data = {
                "call_sid": call_sid,
                "started_at": _now_iso(),
                "turns": [],
                "escalation_flags": [],
            }

        turn_entry: dict[str, Any] = {
            "seq": len(data.get("turns", [])) + 1,
            "timestamp": _now_iso(),
            "role": role,
            "content": content,
        }
        if branch:
            turn_entry["branch"] = branch
        if outcome:
            turn_entry["outcome"] = outcome
        if end_call:
            turn_entry["end_call"] = True
        if sentiment:
            turn_entry["sentiment"] = sentiment
        if classification:
            turn_entry["classification"] = classification
        if timings_ms:
            turn_entry["timings_ms"] = timings_ms

        data.setdefault("turns", []).append(turn_entry)

        if sentiment and sentiment.get("escalate"):
            data.setdefault("escalation_flags", []).append({
                "turn_seq": turn_entry["seq"],
                "emotion": sentiment.get("emotion"),
                "content": content,
            })

        _write_log(call_sid, data)
        return turn_entry["seq"]


def set_turn_sentiment(call_sid: str, seq: int, sentiment: dict) -> None:
    """Backfills sentiment onto a turn already written by log_turn - see
    that function's docstring for why this exists."""
    with _lock:
        data = _read_log(call_sid)
        if not data:
            return
        for turn_entry in data.get("turns", []):
            if turn_entry.get("seq") == seq:
                turn_entry["sentiment"] = sentiment
                if sentiment.get("escalate"):
                    data.setdefault("escalation_flags", []).append({
                        "turn_seq": seq,
                        "emotion": sentiment.get("emotion"),
                        "content": turn_entry.get("content"),
                    })
                break
        _write_log(call_sid, data)


def finalize_call_log(
    call_sid: str,
    *,
    status: str = "completed",
    outcome: Optional[str] = None,
    matched_branch: Optional[str] = None,
    collected: Optional[dict] = None,
    detected_language: Optional[str] = None,
) -> None:
    """Mark the call log as complete with final outcome."""
    with _lock:
        data = _read_log(call_sid)
        if not data:
            return

        data["ended_at"] = _now_iso()
        data["status"] = status
        if outcome:
            data["outcome"] = outcome
        if matched_branch:
            data["matched_branch"] = matched_branch
        if collected:
            data["collected"] = collected
        if detected_language:
            data["detected_language"] = detected_language

        # Compute summary stats
        turns = data.get("turns", [])
        data["summary"] = {
            "total_turns": len(turns),
            "user_turns": sum(1 for t in turns if t.get("role") == "user"),
            "assistant_turns": sum(1 for t in turns if t.get("role") == "assistant"),
            "escalation_count": len(data.get("escalation_flags", [])),
        }

        _write_log(call_sid, data)


def set_call_analysis(call_sid: str, analysis: dict) -> None:
    """Post-call analysis (services/call_analysis.py), mirrored into the log."""
    with _lock:
        data = _read_log(call_sid)
        if not data:
            return
        data["analysis"] = analysis
        _write_log(call_sid, data)


def get_call_log(call_sid: str) -> Optional[dict]:
    """Read a call log file, or None if not found."""
    with _lock:
        data = _read_log(call_sid)
        return data if data else None
