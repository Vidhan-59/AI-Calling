"""ConversationRelay WebSocket for ShiftEaze scenario calls.

Twilio ConversationRelay handles speech-to-text and text-to-speech, so this
socket only ever exchanges plain text:

  Twilio -> us:  {"type":"setup", "callSid":..., "customParameters":{...}}
                 {"type":"prompt", "voicePrompt":"<transcribed speech>", "last":true}
                 {"type":"interrupt"|"dtmf"|"error", ...}
  us -> Twilio:  {"type":"text", "token":"<words to speak>", "last":true}
                 {"type":"end", "handoffData":"<json>"}

The turn loop is deliberately thin: every decision lives in
services/scenario_engine.py so the identical flow can be driven over HTTP by
POST /scenarios/test-turn with no telephony at all. Sentiment classification
runs alongside for the escalation signal but never gates the reply.

ConversationRelay is a Twilio public-beta product. If a field name changes on
Twilio's side, this file is the only place that needs updating.
"""

import asyncio
import json
import logging
import os

from fastapi import APIRouter, WebSocket
from fastapi.websockets import WebSocketDisconnect

from services import claude_client, conversation_store, scenario_engine, scenario_store
from services.call_logger import init_call_log, log_turn, finalize_call_log

logger = logging.getLogger("voice_scenario")
router = APIRouter()


@router.websocket("/scenarios/media-stream")
async def scenario_media_stream(websocket: WebSocket):
    await websocket.accept()

    call_sid: str | None = None
    session: scenario_engine.ScenarioSession | None = None
    ended = False
    scenario_id_for_log: str | None = None
    language_for_log: str | None = None

    async def speak(text: str, last: bool = True) -> None:
        """Send text to ConversationRelay for TTS.

        last=True means 'I am done talking, start listening for the operator'.
        last=False means 'more text coming, keep accumulating before speaking'.

        CRITICAL: For multi-turn conversations (reason collection, re-asks),
        last MUST be True so ConversationRelay starts listening for the next
        operator utterance. Setting last=False would cause it to wait for more
        agent text, creating a dead-air hang.
        """
        if text:
            await websocket.send_json({"type": "text", "token": text, "last": last})

    async def hang_up(outcome: str | None) -> None:
        """Ask ConversationRelay to end the call, passing the outcome as handoff
        data so it also lands in Twilio's own call record, not just our DB."""
        await websocket.send_json(
            {"type": "end", "handoffData": json.dumps({"outcome": outcome or "unknown"})}
        )

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            # ---------------- setup ----------------
            if msg_type == "setup":
                call_sid = data.get("callSid") or "unknown"
                params = data.get("customParameters") or data.get("custom_parameters") or {}
                ref = params.get("ref", "")
                scenario_id = params.get("scenarioId") or "shift_confirm"
                language = params.get("language") or "hi"
                attempt = int(params.get("attempt") or 1)
                scenario_id_for_log = scenario_id
                language_for_log = language

                # Facts were stashed under pending:<ref> when the call was
                # placed; re-key that row onto the real Twilio call SID now
                # that we know it.
                pending = scenario_store.get(f"pending:{ref}") if ref else None
                facts = (pending or {}).get("facts", {})

                try:
                    session = scenario_engine.ScenarioSession(
                        scenario_id=scenario_id,
                        language=language,
                        facts=facts,
                        attempt=attempt,
                    )
                except ValueError as e:
                    logger.error("[%s] bad scenario config: %s", call_sid, e)
                    await hang_up("config_error")
                    return

                scenario_store.create(
                    call_sid=call_sid,
                    scenario_id=scenario_id,
                    language=language,
                    to_number=(pending or {}).get("to_number") or data.get("to", ""),
                    facts=facts,
                    attempt=attempt,
                    rgid=(pending or {}).get("rgid"),
                )
                conversation_store.create_call(
                    call_sid, scenario_id, data.get("from", ""), data.get("to", ""),
                    data.get("direction", "outbound"),
                )

                # Initialize the per-call log file
                init_call_log(
                    call_sid,
                    scenario_id=scenario_id,
                    language=language,
                    to_number=(pending or {}).get("to_number") or data.get("to", ""),
                    from_number=data.get("from", ""),
                    facts=facts,
                    attempt=attempt,
                )

                # The opening line was already spoken by Twilio via
                # welcomeGreeting - log it so the transcript is complete.
                opening = session.opening_line()
                conversation_store.add_message(call_sid, "assistant", opening)
                log_turn(call_sid, role="assistant", content=opening)

                logger.info(
                    "[%s] setup: %s/%s attempt=%d", call_sid, scenario_id, language, attempt
                )

            # ---------------- operator spoke ----------------
            elif msg_type == "prompt":
                if session is None or ended:
                    continue
                if not data.get("last", True):
                    continue  # interim transcript; wait for the final one

                utterance = (data.get("voicePrompt") or "").strip()

                # Sentiment runs concurrently with classification on Claude:
                # cloud capacity means the second call doesn't steal time from
                # the first, so it's free latency-wise. That assumption breaks
                # on the local backend - Ollama on a fixed-core CPU box can't
                # truly parallelize two inferences, "concurrent" just means
                # both compete for the same cores, close to doubling the real
                # wait. Sentiment isn't load-bearing for the scenario flow, so
                # skip it outright on local rather than pay that tax.
                is_local_backend = os.getenv("SCENARIO_LLM_BACKEND", "claude") == "local"
                sentiment_task = asyncio.create_task(
                    claude_client.classify_sentiment(utterance)
                ) if (utterance and not is_local_backend) else None

                try:
                    result = await session.handle_utterance(utterance)
                except Exception as e:
                    logger.error("[%s] classification failed: %s", call_sid, e)
                    # Fall back to the scripted silence close rather than
                    # improvising an unscripted line on a compliance call.
                    result = session.on_silence()

                sentiment = {}
                if sentiment_task is not None:
                    try:
                        sentiment = await sentiment_task
                    except Exception:
                        sentiment = {}

                # Log the operator's utterance
                conversation_store.add_message(
                    call_sid,
                    "user",
                    utterance,
                    sentiment_label=sentiment.get("sentiment"),
                    sentiment_score=sentiment.get("score"),
                    emotion=sentiment.get("emotion"),
                    escalate=bool(sentiment.get("escalate")),
                )
                log_turn(
                    call_sid,
                    role="user",
                    content=utterance,
                    sentiment=sentiment if sentiment else None,
                )

                if sentiment.get("escalate"):
                    logger.warning(
                        "[%s] ESCALATION FLAG: operator sounds %s",
                        call_sid, sentiment.get("emotion"),
                    )

                classification_dict = None
                if result.classification:
                    classification_dict = {
                        "branch": result.classification.branch,
                        "confidence": result.classification.confidence,
                        "detected_language": result.classification.detected_language,
                        "rationale": result.classification.rationale,
                    }

                logger.info(
                    '[%s] "%s" -> branch=%s outcome=%s end=%s collecting=%s',
                    call_sid, utterance, result.branch,
                    result.outcome, result.end_call,
                    session.collecting if session else None,
                )

                # Speak the reply. last=True so ConversationRelay starts
                # listening for the operator's next utterance. This is critical
                # for multi-turn flows (reason collection, re-asks, pay/hours
                # questions). Without last=True, the call would go silent.
                await speak(result.reply, last=True)

                # Log the assistant's reply
                conversation_store.add_message(call_sid, "assistant", result.reply)
                log_turn(
                    call_sid,
                    role="assistant",
                    content=result.reply,
                    branch=result.branch,
                    outcome=result.outcome,
                    end_call=result.end_call,
                    classification=classification_dict,
                )

                if result.end_call:
                    ended = True
                    scenario_store.record_outcome(
                        call_sid,
                        outcome=result.outcome or "unknown",
                        matched_branch=result.branch,
                        collected=result.collected,
                        detected_language=(
                            result.classification.detected_language
                            if result.classification
                            else None
                        ),
                    )
                    conversation_store.end_call(call_sid)
                    finalize_call_log(
                        call_sid,
                        status="completed",
                        outcome=result.outcome,
                        matched_branch=result.branch,
                        collected=result.collected,
                        detected_language=(
                            result.classification.detected_language
                            if result.classification
                            else None
                        ),
                    )
                    await hang_up(result.outcome)
                # else: call stays open. ConversationRelay will listen for
                # the next operator utterance and send another "prompt" message.
                # This is how multi-turn conversations work:
                #   - "no" branch -> asks for reason -> operator gives reason -> ends
                #   - "asks_pay" -> answers + re-asks -> operator answers yes/no -> ends
                #   - "repeat" -> re-states info -> operator answers -> ends

            elif msg_type == "interrupt":
                logger.info(
                    "[%s] interrupted: %s",
                    call_sid, data.get("utteranceUntilInterrupt", ""),
                )

            elif msg_type == "error":
                logger.error(
                    "[%s] ConversationRelay error: %s",
                    call_sid, data.get("description"),
                )

    except WebSocketDisconnect:
        logger.info("[%s] disconnected", call_sid)
    except Exception as e:
        logger.error("[%s] handler error: %s", call_sid, e, exc_info=True)
    finally:
        if call_sid and not ended:
            # Dropped mid-conversation (operator hung up, network died). Record
            # the scenario's own no-answer outcome so the row is never left
            # dangling in 'in_progress'.
            fallback = "call_dropped"
            if session is not None:
                fallback = session.outcome or session.scenario.get(
                    "no_answer_outcome", "no_response"
                )
            scenario_store.record_outcome(
                call_sid, outcome=fallback, status="disconnected"
            )
            conversation_store.end_call(call_sid, status="disconnected")
            finalize_call_log(
                call_sid,
                status="disconnected",
                outcome=fallback,
                collected=session.collected if session else None,
            )
