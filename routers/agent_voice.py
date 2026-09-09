"""Twilio ConversationRelay entrypoint for the rebuild's conversation_agent
(flow graph + tool registry + guardrails) - the piece that makes
services/agents/conversation_agent.py reachable from a real call for the
first time. Before this file existed, that module only ran inside tests
(see AUDIT.md).

Reuses the exact ConversationRelay text-in/text-out contract already proven
in routers/voice.py and routers/voice_scenario.py (setup/prompt/interrupt/
error in; {"type":"text",...} / {"type":"end",...} out) rather than adopting
Pipecat or a raw media-stream pipeline - see PHASE2_ARCHITECTURE_PROPOSAL.md
for that trade-off. Not exercised against a real Twilio call in this pass -
no Twilio credentials are configured on this machine yet (see AUDIT.md) - so
treat the exact TwiML attribute names the same way this repo already treats
services/stt/sarvam.py etc.: written to the documented shape, not smoke
tested against a live call.

Bootstrap-greeting design decision (new - no prior art elsewhere in this
repo to copy): unlike the scripted ShiftEaze scenarios, this agent's opening
line is LLM-generated from the flow graph's entry node objective, not a
fixed string, so there's no template to render before the call connects the
way scenario_engine.opening_line() does. TwiML still needs a
`welcomeGreeting` string up front, so `agent_voice_twiml` runs one run_turn()
call against a synthetic bootstrap message (CALL_CONNECTED_SENTINEL) to get
that line, and the WebSocket handler re-runs the same call on "setup" so the
model's own turn history matches what the caller actually heard (see the
comment at that call site for why it's re-run rather than threaded through
TwiML).

Multi-agent handoff (decision="handoff") is not implemented yet - flagged and
ended gracefully rather than silently mishandled. See "known limitations" in
PHASE2_ARCHITECTURE_PROPOSAL.md.
"""

import json
import logging
import time
from xml.sax.saxutils import quoteattr

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from sqlalchemy import select

import models
from db import now as db_now, session_scope
from services import agent_registry
from services.agents.conversation_agent import run_turn
from services.dispatch.dialer import mark_call_ended
from services.flow.slots import SlotState

logger = logging.getLogger("agent_voice")
router = APIRouter(tags=["agent-voice"])

CALL_CONNECTED_SENTINEL = "[CALL_CONNECTED]"

_ERROR_TWIML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    "<Response><Say>Configuration error. Goodbye.</Say><Hangup/></Response>"
)


async def _load_call_context(task_id: str) -> dict | None:
    async with session_scope() as session:
        task = await session.get(models.CallTask, task_id)
        if task is None:
            return None
        contact = await session.get(models.Contact, task.contact_id)
        campaign = await session.get(models.Campaign, task.campaign_id)
        if contact is None or campaign is None:
            return None
        return {
            "task_id": task.id,
            "contact": {
                "id": contact.id,
                "name": contact.name,
                "terminal": contact.terminal,
                "designation": contact.designation,
                "phone_e164": contact.phone_e164,
            },
            "language": contact.language_pref or "hi",
            "agent_id": campaign.agent_id,
            "campaign_id": campaign.id,
        }


@router.api_route("/agent-voice/twiml", methods=["GET", "POST"])
async def agent_voice_twiml(request: Request):
    """Twilio webhook: returns TwiML connecting the call to ConversationRelay.
    `task_id` (a CallTask id) is how this endpoint knows which contact/agent/
    campaign this call is for - see services/dispatch/dialer.run_dispatch_cycle,
    which is the only code that should ever be building this URL."""
    task_id = request.query_params.get("task_id")
    if not task_id:
        return HTMLResponse(_ERROR_TWIML, media_type="application/xml")

    ctx = await _load_call_context(task_id)
    if ctx is None:
        logger.error("agent-voice TwiML requested for unknown/incomplete task_id=%s", task_id)
        return HTMLResponse(_ERROR_TWIML, media_type="application/xml")

    try:
        config = agent_registry.get_agent_config(ctx["agent_id"])
        flow_graph = agent_registry.get_flow_graph(ctx["agent_id"])
    except Exception as e:  # noqa: BLE001 - a bad/missing config must fail this call, not crash the server
        logger.error("[%s] could not load agent config/flow for agent_id=%s: %s", task_id, ctx["agent_id"], e)
        return HTMLResponse(_ERROR_TWIML, media_type="application/xml")

    voice_cfg = config.voice.get(ctx["language"]) or next(iter(config.voice.values()))

    slot_state = SlotState(current_node_id=flow_graph.entry)
    async with session_scope() as session:
        outcome = await run_turn(
            session=session,
            call_id=f"pending:{task_id}",
            agent_id=ctx["agent_id"],
            flow_graph=flow_graph,
            slot_state=slot_state,
            contact=ctx["contact"],
            language=ctx["language"],
            history=[{"role": "user", "content": CALL_CONNECTED_SENTINEL}],
        )
    greeting = outcome.text_to_speak or "Namaste."

    ws_url = f"wss://{request.url.hostname}/agent-voice/media-stream"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f"<ConversationRelay url={quoteattr(ws_url)} "
        f"welcomeGreeting={quoteattr(greeting)} "
        f'ttsProvider="Google" voice={quoteattr(voice_cfg.voice_id)} '
        'interruptible="true" debug="true">'
        f'<Parameter name="taskId" value={quoteattr(task_id)}/>'
        "</ConversationRelay>"
        "</Connect></Response>"
    )
    return HTMLResponse(content=twiml, media_type="application/xml")


@router.websocket("/agent-voice/media-stream")
async def agent_voice_media_stream(websocket: WebSocket):
    await websocket.accept()

    task_id: str | None = None
    call_id: str | None = None
    agent_span_id: int | None = None
    ctx: dict | None = None
    flow_graph = None
    slot_state: SlotState | None = None
    history: list[dict] = []
    turn_seq = 0
    ended = False

    async def _persist_turn(role: str, text: str, latency_ms: int | None = None) -> None:
        nonlocal turn_seq
        turn_seq += 1
        async with session_scope() as session:
            session.add(
                models.Turn(
                    call_id=call_id,
                    agent_span_id=agent_span_id,
                    seq=turn_seq,
                    role=role,
                    text=text,
                    latency_total_ms=latency_ms,
                )
            )

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "setup":
                provider_call_sid = data.get("callSid", "unknown")
                params = data.get("customParameters") or data.get("custom_parameters") or {}
                task_id = params.get("taskId")
                if not task_id:
                    logger.error("[%s] setup with no taskId custom param - cannot proceed", provider_call_sid)
                    await websocket.send_json(
                        {"type": "end", "handoffData": json.dumps({"outcome": "config_error"})}
                    )
                    return

                ctx = await _load_call_context(task_id)
                if ctx is None:
                    logger.error("[%s] setup: task_id=%s not found", provider_call_sid, task_id)
                    await websocket.send_json(
                        {"type": "end", "handoffData": json.dumps({"outcome": "config_error"})}
                    )
                    return

                flow_graph = agent_registry.get_flow_graph(ctx["agent_id"])
                slot_state = SlotState(current_node_id=flow_graph.entry)

                async with session_scope() as session:
                    call_row = (
                        await session.scalars(
                            select(models.Call).where(models.Call.provider_call_sid == provider_call_sid)
                        )
                    ).first()
                    if call_row is None:
                        # Not dispatched through services/dispatch/dialer.py (e.g. this
                        # endpoint was hit directly while testing) - create the Call row
                        # here so the rest of this handler still has one to write against.
                        call_row = models.Call(
                            call_task_id=task_id,
                            agent_id=ctx["agent_id"],
                            provider_call_sid=provider_call_sid,
                            to_number=ctx["contact"]["phone_e164"],
                        )
                        session.add(call_row)
                        await session.flush()
                    call_row.answered_at = call_row.answered_at or db_now()
                    call_id = call_row.id

                    agent_span = models.AgentSpan(
                        call_id=call_id,
                        agent_id=ctx["agent_id"],
                        agent_version=agent_registry.get_agent_config(ctx["agent_id"]).version,
                    )
                    session.add(agent_span)
                    await session.flush()
                    agent_span_id = agent_span.id

                # Re-derive the same greeting text produced at TwiML time so the
                # model's own turn history matches what the caller actually heard,
                # rather than threading the string through a TwiML <Parameter> -
                # ConversationRelay's <Parameter> values show up in Twilio's own
                # call logs/debug console, and a multi-sentence greeting is an
                # awkward fit for an XML attribute on some providers. One extra
                # Claude call on a path that already made this exact call once at
                # TwiML time is an acceptable latency trade for not depending on
                # that instead.
                history = [{"role": "user", "content": CALL_CONNECTED_SENTINEL}]
                t0 = time.monotonic()
                async with session_scope() as session:
                    outcome = await run_turn(
                        session=session, call_id=call_id, agent_id=ctx["agent_id"],
                        flow_graph=flow_graph, slot_state=slot_state,
                        contact=ctx["contact"], language=ctx["language"], history=history,
                    )
                latency_ms = int((time.monotonic() - t0) * 1000)
                history.append({"role": "assistant", "content": outcome.text_to_speak})
                await _persist_turn("assistant", outcome.text_to_speak, latency_ms)
                logger.info(
                    "[%s] setup complete, agent=%s task=%s (greeting latency=%dms)",
                    provider_call_sid, ctx["agent_id"], task_id, latency_ms,
                )

            elif msg_type == "prompt":
                if ctx is None or slot_state is None or ended:
                    continue
                if not data.get("last", True):
                    continue

                utterance = (data.get("voicePrompt") or "").strip()
                if not utterance:
                    continue

                history.append({"role": "user", "content": utterance})

                t0 = time.monotonic()
                async with session_scope() as session:
                    outcome = await run_turn(
                        session=session, call_id=call_id, agent_id=ctx["agent_id"],
                        flow_graph=flow_graph, slot_state=slot_state,
                        contact=ctx["contact"], language=ctx["language"], history=history,
                    )
                latency_ms = int((time.monotonic() - t0) * 1000)

                history.append({"role": "assistant", "content": outcome.text_to_speak})
                await _persist_turn("user", utterance)
                await _persist_turn("assistant", outcome.text_to_speak, latency_ms)

                await websocket.send_json({"type": "text", "token": outcome.text_to_speak, "last": True})

                for v in outcome.guardrail_violations:
                    logger.warning("[%s] guardrail violation: %s (severity=%s)", call_id, v.rule_id, v.severity)

                if outcome.decision == "handoff":
                    # Multi-agent handoff is not implemented yet (see module
                    # docstring) - end gracefully with a flagged outcome
                    # rather than continue a call the flow graph thinks
                    # should have moved to a different agent.
                    logger.warning(
                        "[%s] flow graph requested a handoff to %s - not implemented, ending call",
                        call_id, (outcome.decision_detail or {}).get("target_agent_id"),
                    )
                    ended = True
                    await _finalize_call(call_id, task_id, "handoff_not_implemented", slot_state)
                    await websocket.send_json(
                        {"type": "end", "handoffData": json.dumps({"outcome": "handoff_not_implemented"})}
                    )
                elif outcome.decision in ("end_call", "transfer_to_human"):
                    ended = True
                    outcome_name = (outcome.decision_detail or {}).get("outcome", outcome.decision)
                    await _finalize_call(call_id, task_id, outcome_name, slot_state)
                    await websocket.send_json(
                        {"type": "end", "handoffData": json.dumps({"outcome": outcome_name})}
                    )
                # else: advance/advance_terminal/reprompt - call stays open,
                # ConversationRelay listens for the next operator utterance.

            elif msg_type == "interrupt":
                logger.info("[%s] interrupted: %s", call_id, data.get("utteranceUntilInterrupt", ""))

            elif msg_type == "error":
                logger.error("[%s] ConversationRelay error: %s", call_id, data.get("description"))

    except WebSocketDisconnect:
        logger.info("[%s] disconnected", call_id)
    except Exception:
        logger.exception("[%s] agent_voice handler error", call_id)
    finally:
        if call_id and not ended:
            # Dropped mid-conversation (operator hung up, network died) -
            # record it rather than leave the CallTask stuck in "in_call"
            # forever, which would otherwise permanently occupy a
            # concurrency slot.
            await _finalize_call(call_id, task_id, "call_dropped", slot_state, status="disconnected")


async def _finalize_call(
    call_id: str | None,
    task_id: str | None,
    outcome_name: str,
    slot_state: SlotState | None,
    *,
    status: str = "completed",
) -> None:
    if call_id:
        async with session_scope() as session:
            call_row = await session.get(models.Call, call_id)
            if call_row is not None:
                call_row.status = status
                call_row.hangup_by = "agent" if status == "completed" else "system"
                session.add(
                    models.CallOutcome(
                        call_id=call_id,
                        slots_json=slot_state.filled if slot_state else {},
                        primary_outcome=outcome_name,
                        extracted_by_model="conversation_agent",
                    )
                )
    if task_id:
        async with session_scope() as session:
            task = await session.get(models.CallTask, task_id)
            if task is not None:
                task.state = "completed" if status == "completed" else "abandoned"
    await mark_call_ended()
