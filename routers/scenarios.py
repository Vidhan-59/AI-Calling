"""ShiftEaze scenario-calling API.

Endpoints, in the order you'd normally use them:

  GET  /scenarios                     - what's available (ids, languages, branches)
  GET  /scenarios/{id}                - one scenario's full knowledge base
  POST /scenarios/reload              - re-read scenarios/*.json after editing
  POST /scenarios/preview             - render exact spoken lines, no call placed
  POST /scenarios/test-turn           - conversation over HTTP, no phone needed
  POST /scenarios/test-batch          - run many utterances through one scenario
  POST /scenarios/call                - place the real phone call
  GET  /scenario-calls                - outcomes of placed calls
  GET  /scenario-calls/{call_sid}     - one call: outcome + transcript + sentiment
"""

import logging
import urllib.error
import urllib.request
import uuid
from xml.sax.saxutils import quoteattr

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from twilio.rest import Client

from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
from public_url import get_public_url
from scenario_schemas import (
    ScenarioCallRequest,
    ScenarioPreviewRequest,
    ScenarioTestBatchRequest,
    ScenarioTestTurnRequest,
)
from services import conversation_store, scenario_engine, scenario_store
from services.call_logger import get_call_log

logger = logging.getLogger("scenarios")
router = APIRouter(tags=["shifteaze-scenarios"])

# ---------------------------------------------------------------------------
# Gujarati transcription workaround
# ---------------------------------------------------------------------------
# Twilio ConversationRelay uses Google Cloud STT under the hood. Google STT
# does NOT support gu-IN (Gujarati) as a recognition language. Attempting to
# set transcriptionLanguage="gu-IN" causes Twilio to fall back to en-US,
# which produces garbage transcripts for Gujarati/Hindi speech.
#
# Workaround: use hi-IN for transcription. Gujarati operators at port
# terminals code-mix heavily with Hindi and many Gujarati words are close
# enough to Hindi that Google's Hindi model captures them well. The AI
# classifier prompt is trained to handle Gujarati words transliterated
# into Devanagari. TTS stays in Gujarati (gu-IN voices work fine for
# synthesis).
# ---------------------------------------------------------------------------
_TRANSCRIPTION_LANGUAGE_OVERRIDES = {
    "gu-IN": "hi-IN",
}


def _effective_transcription_language(lang_cfg: dict) -> str:
    """Return the transcription language to use, applying overrides for
    unsupported languages like Gujarati."""
    raw = lang_cfg["transcription_language"]
    return _TRANSCRIPTION_LANGUAGE_OVERRIDES.get(raw, raw)


def _check_tunnel_reachable(base_url: str) -> str | None:
    """Verify Twilio will actually be able to fetch our TwiML.

    Exists because of a real, expensive-to-diagnose failure: a dead tunnel makes
    Twilio dial the operator, fail to fetch the TwiML, and play "an application
    error occurred" before hanging up. From the caller's side that looks like a
    broken agent; the only trace is a Twilio 11200 alert. Checking here turns
    that into an immediate, obvious API error and costs one local HTTP request.

    Returns None if reachable, else a human-readable reason.
    """
    url = base_url.rstrip("/") + "/"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status >= 400:
                return f"returned HTTP {resp.status}"
        return None
    except urllib.error.HTTPError as e:
        return f"returned HTTP {e.code}"
    except Exception as e:
        return f"is unreachable ({type(e).__name__}: {e})"

# In-memory sessions for the HTTP test endpoints only. Live phone calls keep
# their session in the WebSocket handler's closure instead (see voice_scenario.py),
# so nothing here is on the real-call path.
_test_sessions: dict[str, scenario_engine.ScenarioSession] = {}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@router.get("/scenarios")
def list_scenarios():
    """All five ShiftEaze calling scenarios, their languages and branch keys."""
    return scenario_engine.list_scenarios()


@router.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: str):
    """Full knowledge base for one scenario, all languages - the scripts,
    keyword hints, outcomes and guardrail facts the AI is working from."""
    scenario = scenario_engine.get_scenario(scenario_id)
    if not scenario:
        raise HTTPException(status_code=404, detail=f"Unknown scenario '{scenario_id}'")
    return scenario


@router.post("/scenarios/reload")
def reload_scenarios():
    """Re-read scenarios/*.json from disk. Edit a script or keyword list, hit
    this, and the very next call uses it - no server restart."""
    try:
        scenarios = scenario_engine.load_scenarios(force=True)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Scenario knowledge base is invalid: {e}")
    return {"status": "reloaded", "count": len(scenarios), "ids": sorted(scenarios)}


# ---------------------------------------------------------------------------
# Preview + text testing (no telephony involved)
# ---------------------------------------------------------------------------


@router.post("/scenarios/preview")
def preview(payload: ScenarioPreviewRequest):
    """Render every line this call could speak, with facts filled in.

    Use this to proof-read the Hindi/Gujarati output before dialling a real
    operator - it shows the opening line, the follow-up wording, the silence
    close, and each branch's reply exactly as the caller would hear it.
    """
    lang_cfg = scenario_engine.get_language_config(payload.scenario_id, payload.language)
    if not lang_cfg:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{payload.scenario_id}' or language '{payload.language}'",
        )

    facts = payload.facts.model_dump(exclude_none=True)
    missing = scenario_engine.missing_facts(payload.scenario_id, facts)

    session = scenario_engine.ScenarioSession(
        scenario_id=payload.scenario_id,
        language=payload.language,
        facts=facts,
        attempt=payload.attempt,
    )

    return {
        "scenario_id": payload.scenario_id,
        "language": payload.language,
        "attempt": payload.attempt,
        "voice": lang_cfg["voice"],
        "tts_language": lang_cfg["tts_language"],
        "transcription_language": lang_cfg["transcription_language"],
        "missing_facts": missing,
        "opening_line": session.opening_line(),
        "silence_reply": scenario_engine.render(lang_cfg["silence_reply"], facts),
        "branches": {
            key: {
                "description": branch["description"],
                "reply": scenario_engine.render(branch["reply"], facts),
                "after_collect_reply": scenario_engine.render(
                    branch.get("after_collect_reply", ""), facts
                )
                or None,
                "outcome": branch["outcome"],
                "ends_call": bool(branch.get("end_call", True)),
                "example_phrasings": branch.get("keywords", []),
            }
            for key, branch in lang_cfg["branches"].items()
        },
    }


@router.post("/scenarios/test-turn")
async def test_turn(payload: ScenarioTestTurnRequest):
    """One conversational turn over plain HTTP - no phone, Twilio or tunnel.

    First call for a given session_id returns the opening line plus the reply
    to your message. Send an empty message to simulate silence/timeout.
    """
    if not scenario_engine.get_language_config(payload.scenario_id, payload.language):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{payload.scenario_id}' or language '{payload.language}'",
        )

    key = f"{payload.scenario_id}:{payload.language}:{payload.session_id}"
    session = _test_sessions.get(key)
    opening = None
    if session is None:
        facts = payload.facts.model_dump(exclude_none=True) if payload.facts else {}
        session = scenario_engine.ScenarioSession(
            scenario_id=payload.scenario_id,
            language=payload.language,
            facts=facts,
            attempt=payload.attempt,
        )
        _test_sessions[key] = session
        opening = session.opening_line()

    try:
        result = await session.handle_utterance(payload.message)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result.end_call:
        _test_sessions.pop(key, None)

    return {
        "session_id": payload.session_id,
        "opening_line": opening,
        "operator_said": payload.message,
        "agent_reply": result.reply,
        "matched_branch": result.branch,
        "outcome": result.outcome,
        "call_ended": result.end_call,
        "collected": result.collected,
        "reprompted": result.reprompted,
        "classification": (
            {
                "confidence": result.classification.confidence,
                "detected_language": result.classification.detected_language,
                "rationale": result.classification.rationale,
            }
            if result.classification
            else None
        ),
    }


@router.post("/scenarios/test-batch")
async def test_batch(payload: ScenarioTestBatchRequest):
    """Run a list of operator utterances through one scenario in a single request.

    Each utterance starts a fresh conversation, so this exercises one branch per
    line - the quickest way to verify all branches still classify correctly after
    editing a scenario JSON.
    """
    if not scenario_engine.get_language_config(payload.scenario_id, payload.language):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{payload.scenario_id}' or language '{payload.language}'",
        )

    facts = payload.facts.model_dump(exclude_none=True) if payload.facts else {}
    results = []
    for utterance in payload.utterances:
        session = scenario_engine.ScenarioSession(
            scenario_id=payload.scenario_id,
            language=payload.language,
            facts=facts,
            attempt=payload.attempt,
        )
        try:
            result = await session.handle_utterance(utterance)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        results.append(
            {
                "operator_said": utterance,
                "matched_branch": result.branch,
                "confidence": result.classification.confidence if result.classification else None,
                "detected_language": (
                    result.classification.detected_language if result.classification else None
                ),
                "agent_reply": result.reply,
                "outcome": result.outcome,
                "call_ended": result.end_call,
            }
        )
    return {
        "scenario_id": payload.scenario_id,
        "language": payload.language,
        "turns": results,
    }


# ---------------------------------------------------------------------------
# Placing a real call
# ---------------------------------------------------------------------------


@router.post("/scenarios/call")
def place_scenario_call(payload: ScenarioCallRequest):
    """Place a real outbound ShiftEaze scenario call.

    Rejects the request up front if a required fact is missing, rather than
    letting an operator hear an unfilled placeholder mid-sentence.
    """
    lang_cfg = scenario_engine.get_language_config(payload.scenario_id, payload.language)
    if not lang_cfg:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{payload.scenario_id}' or language '{payload.language}'",
        )

    facts = payload.facts.model_dump(exclude_none=True)
    missing = scenario_engine.missing_facts(payload.scenario_id, facts)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required facts for '{payload.scenario_id}': {', '.join(missing)}. "
                "The agent may only speak facts supplied here."
            ),
        )

    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
        raise HTTPException(status_code=500, detail="Twilio is not configured. Set TWILIO_* in .env.")
    # Read fresh from .env rather than at import time, so replacing a dead
    # tunnel needs only an .env edit - no server restart.
    public_url = get_public_url()
    if not public_url:
        raise HTTPException(
            status_code=500,
            detail="NGROK_URL in .env is not set to a real public https URL reachable by Twilio.",
        )

    problem = _check_tunnel_reachable(public_url)
    if problem:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Not placing the call: your public URL {public_url} {problem}. "
                "Twilio would dial the operator, fail to fetch the TwiML, and play "
                "'an application error occurred'. Start a fresh tunnel and put the new "
                "URL in NGROK_URL in .env - no server restart needed, it is re-read per "
                "call. Easiest fix: stop and re-run .\\run-all.ps1"
            ),
        )

    # Correlation id carried through the TwiML query string into the WebSocket,
    # so the media handler can look up which scenario/language/facts this call
    # is for without trusting anything the caller says.
    ref = uuid.uuid4().hex[:12]
    scenario_store.create(
        call_sid=f"pending:{ref}",
        scenario_id=payload.scenario_id,
        language=payload.language,
        to_number=payload.to_phone_number,
        facts=facts,
        attempt=payload.attempt,
        rgid=payload.rgid,
    )

    twiml_url = (
        f"{public_url}/scenarios/voice"
        f"?ref={ref}&scenario_id={payload.scenario_id}"
        f"&language={payload.language}&attempt={payload.attempt}"
    )

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            url=twiml_url,
            to=payload.to_phone_number,
            from_=TWILIO_PHONE_NUMBER,
        )
    except Exception as e:
        return {"status": "failed", "error": str(e)}

    return {
        "status": "success",
        "call_sid": call.sid,
        "ref": ref,
        "scenario_id": payload.scenario_id,
        "language": payload.language,
        "attempt": payload.attempt,
        "voice": lang_cfg["voice"],
        "will_say": scenario_engine.render(
            lang_cfg.get("attempt_2_opening" if payload.attempt > 1 else "opening")
            or lang_cfg["opening"],
            facts,
        ),
        "track_at": f"/scenario-calls/{call.sid}",
    }


@router.api_route("/scenarios/voice", methods=["GET", "POST"])
async def scenario_twiml(request: Request):
    """Twilio webhook: hands the call to ConversationRelay in the right language.

    ConversationRelay does speech-to-text and text-to-speech; our WebSocket only
    ever exchanges plain text. `transcriptionLanguage` is what makes Hindi and
    Gujarati recognition work - without it Twilio transcribes with an English
    model and the classifier receives garbage.

    NOTE: Gujarati (gu-IN) is NOT supported by Google Cloud STT, so we
    transparently use hi-IN for transcription while keeping gu-IN for TTS.
    The AI classifier handles Gujarati words in Devanagari transliteration.
    """
    params = dict(request.query_params)
    ref = params.get("ref", "")
    scenario_id = params.get("scenario_id", "shift_confirm")
    language = params.get("language", "hi")
    attempt = int(params.get("attempt", "1") or 1)

    lang_cfg = scenario_engine.get_language_config(scenario_id, language)
    if not lang_cfg:
        response = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Say>Configuration error. Goodbye.</Say><Hangup/></Response>"
        )
        return HTMLResponse(content=response, media_type="application/xml")

    record = scenario_store.get(f"pending:{ref}") if ref else None
    facts = (record or {}).get("facts", {})

    session = scenario_engine.ScenarioSession(
        scenario_id=scenario_id, language=language, facts=facts, attempt=attempt
    )
    greeting = session.opening_line()

    # Apply transcription language override for unsupported languages (Gujarati).
    effective_transcription_lang = _effective_transcription_language(lang_cfg)
    if effective_transcription_lang != lang_cfg["transcription_language"]:
        logger.info(
            "[%s/%s] transcription language overridden: %s -> %s (unsupported by Google STT)",
            scenario_id, language,
            lang_cfg["transcription_language"], effective_transcription_lang,
        )

    ws_url = f"wss://{request.url.hostname}/scenarios/media-stream"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f"<ConversationRelay url={quoteattr(ws_url)} "
        f"welcomeGreeting={quoteattr(greeting)} "
        f"ttsProvider=\"Google\" voice={quoteattr(lang_cfg['voice'])} "
        f"ttsLanguage={quoteattr(lang_cfg['tts_language'])} "
        f"transcriptionProvider=\"Google\" "
        f"transcriptionLanguage={quoteattr(effective_transcription_lang)} "
        'interruptible="true" dtmfDetection="false" debug="true">'
        f"<Parameter name=\"ref\" value={quoteattr(ref)}/>"
        f"<Parameter name=\"scenarioId\" value={quoteattr(scenario_id)}/>"
        f"<Parameter name=\"language\" value={quoteattr(language)}/>"
        f"<Parameter name=\"attempt\" value={quoteattr(str(attempt))}/>"
        "</ConversationRelay>"
        "</Connect></Response>"
    )
    return HTMLResponse(content=twiml, media_type="application/xml")


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


@router.get("/scenario-calls")
def list_scenario_calls(scenario_id: str | None = None, limit: int = 100):
    """Outcomes of placed scenario calls, newest first."""
    return scenario_store.list_calls(scenario_id=scenario_id, limit=limit)


@router.get("/scenario-calls/{call_sid}")
def get_scenario_call(call_sid: str):
    """One scenario call: the facts it used, the branch the AI matched, the
    outcome, plus the verbatim transcript with per-turn sentiment."""
    record = scenario_store.get(call_sid)
    if not record:
        raise HTTPException(status_code=404, detail="Scenario call not found")
    summary = conversation_store.get_call_summary(call_sid)
    record["transcript"] = (summary or {}).get("transcript", [])
    record["average_sentiment_score"] = (summary or {}).get("average_sentiment_score")
    record["escalation_recommended"] = (summary or {}).get("escalation_recommended", False)
    # Attach the file-based call log if it exists.
    call_log = get_call_log(call_sid)
    if call_log:
        record["call_log"] = call_log
    return record
