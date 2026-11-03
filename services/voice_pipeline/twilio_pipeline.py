"""Assembles and runs the self-hosted Pipecat pipeline for one live agent
call over raw Twilio Media Streams (or WebRTC, see webrtc_pipeline.py):

  caller <-> RNNoise -> Silero VAD -> barge-in -> latency monitor      (Layer 2)
         -> STT: Sarvam saaras:v3 (USE_SARVAM_STT=true) or local
            faster-whisper on the GPU, decoded speculatively
         -> ConversationAgentProcessor (Qwen via conversation_agent.py),
            reply streamed piece by piece as the LLM writes it        (Layer 1)
         -> OmniVoice TTS (tts_service/), one request per reply piece,
            repeated pieces served from the audio cache -> caller

Both directions run at once (full duplex): the caller is heard - and can
barge in - while the bot is still speaking.

Written to Twilio's documented Media Streams wire format (the "connected"/
"start" handshake before any "media" frames arrive) - same caveat this repo
already carries for services/telephony/*.py: correct against the
documented protocol shape, not yet exercised against a real live call on
this machine (see AUDIT.md). Verify with a real call before trusting this in
production; see tests/integration/test_voice_pipeline_harness.py for what
CAN be verified locally, without Twilio, first.
"""

import json
import logging

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

import models
from config import settings
from db import now as db_now, session_scope
from services import agent_registry
from services.tts.pipecat_service import OmniVoiceTTSService
from services.voice_pipeline.audio_frontend import build_noise_filter, front_end
from services.voice_pipeline.conversation_processor import ConversationAgentProcessor

logger = logging.getLogger("voice_pipeline.twilio_pipeline")

STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 8000
# End-of-turn silence for free-form agent calls. Was Pipecat's default 0.2s,
# which ends a turn at every comma pause (natural ones run 0.2-0.45s, see
# scenario_pipeline.VAD_STOP_SECS); 0.5s costs nothing extra in reply
# latency now that STT decodes speculatively inside this window.
VAD_STOP_SECS = 0.5

async def read_twilio_start(websocket: WebSocket) -> dict:
    """Twilio sends a "connected" event first, then "start" (which carries
    streamSid/callSid/customParameters) before any "media" events - both are
    plain JSON text frames. Must be consumed here, before
    FastAPIWebsocketTransport takes over reading from this same socket."""
    connected_raw = await websocket.receive_text()
    connected = json.loads(connected_raw)
    if connected.get("event") != "connected":
        logger.warning("expected Twilio 'connected' event first, got: %r", connected.get("event"))

    start_raw = await websocket.receive_text()
    start_msg = json.loads(start_raw)
    if start_msg.get("event") != "start":
        raise ValueError(f"expected Twilio 'start' event, got: {start_msg.get('event')!r}")
    return start_msg["start"]


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


async def _ensure_call_row(*, provider_call_sid: str, task_id: str, ctx: dict) -> tuple[str, int]:
    """Mirrors the old ConversationRelay handler's setup logic: find the
    dialer-created Call row by provider_call_sid, or create one for a direct
    test hit, then open a fresh AgentSpan for this pipeline run."""
    async with session_scope() as session:
        from sqlalchemy import select

        call_row = (
            await session.scalars(
                select(models.Call).where(models.Call.provider_call_sid == provider_call_sid)
            )
        ).first()
        if call_row is None:
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
        return call_id, agent_span.id


def _build_tts_service(agent_id: str, language: str) -> OmniVoiceTTSService:
    """OmniVoice for every reply piece the processor sends (TTSSpeakFrame).
    The voice is looked up per piece from Voice Studio's assignments
    (services/tts/voice_registry.py), so changing a voice needs no restart;
    the agent YAML's voice block is the fallback. Barge-in needs nothing
    special here: unlike a WebSocket TTS there is no connection to reset, so
    Pipecat's interruption cancels the in-flight piece and flushes at once."""
    voice_cfg = agent_registry.get_agent_config(agent_id).voice.get(language)
    return OmniVoiceTTSService(
        owner=agent_id, language=language,
        fallback_voice_id=voice_cfg.voice_id if voice_cfg else None,
        fallback_speed=voice_cfg.speed if voice_cfg else 1.0,
    )


def _build_stt_service(language: str):
    # The same STT scenario calls use (USE_SARVAM_STT): Sarvam, or the shared,
    # preloaded, speculative GPU Whisper - Pipecat's stock WhisperSTTService
    # loaded a CPU model per call and decoded on the event loop (local_stt.py).
    from services.voice_pipeline.stt_factory import build_stt

    return build_stt(language)


async def load_agent_call(task_id: str, provider_call_sid: str) -> dict | None:
    """Call context + Call/AgentSpan rows for one agent call, any transport."""
    ctx = await _load_call_context(task_id)
    if ctx is None:
        return None
    ctx["call_id"], ctx["agent_span_id"] = await _ensure_call_row(
        provider_call_sid=provider_call_sid, task_id=task_id, ctx=ctx
    )
    return ctx


async def run_agent_pipeline(transport, ctx: dict, *, stt=None, tts=None, label: str) -> ConversationAgentProcessor:
    """One agent call over any Pipecat transport (Twilio Media Streams or
    WebRTC), until it ends. `stt`/`tts` are injectable for tests."""
    call_id = ctx["call_id"]
    tts = tts or _build_tts_service(ctx["agent_id"], ctx["language"])
    processor = ConversationAgentProcessor(
        playout=getattr(tts, "playout", None),
        task_id=ctx["task_id"],
        call_id=call_id,
        agent_span_id=ctx["agent_span_id"],
        agent_id=ctx["agent_id"],
        flow_graph=agent_registry.get_flow_graph(ctx["agent_id"]),
        contact=ctx["contact"],
        language=ctx["language"],
    )
    front, _ = front_end(call_id=str(call_id), stop_secs=VAD_STOP_SECS)
    pipeline = Pipeline([
        transport.input(),
        *front,
        stt or _build_stt_service(ctx["language"]),
        processor,
        tts,
        transport.output(),
    ])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=STT_SAMPLE_RATE, audio_out_sample_rate=TTS_SAMPLE_RATE),
        enable_rtvi=False,
    )
    # handle_sigint=False: this runs inside one request handler of a FastAPI
    # server process juggling many concurrent calls - it must not install a
    # process-wide SIGINT handler per call (default True is meant for a
    # runner that owns the whole process, e.g. a standalone pipecat script).
    runner = PipelineRunner(handle_sigint=False)

    logger.info("[%s] starting agent pipeline over %s (task=%s, agent=%s)", call_id, label, ctx["task_id"], ctx["agent_id"])
    try:
        await runner.run(task)
    except Exception:
        logger.exception("[%s] voice pipeline crashed", call_id)
    finally:
        await processor.handle_disconnect()
    return processor


async def run_twilio_media_pipeline(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        start = await read_twilio_start(websocket)
    except (WebSocketDisconnect, ValueError, json.JSONDecodeError) as e:
        logger.error("Twilio media stream handshake failed: %s", e)
        return

    stream_sid = start.get("streamSid")
    call_sid = start.get("callSid")
    custom_params = start.get("customParameters", {})
    task_id = custom_params.get("taskId")

    if not task_id or not stream_sid or not call_sid:
        logger.error(
            "media stream 'start' missing taskId/streamSid/callSid (custom_params=%r)", custom_params
        )
        return

    ctx = await load_agent_call(task_id, call_sid)
    if ctx is None:
        logger.error("media stream: task_id=%s not found", task_id)
        return

    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=True),
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
            audio_in_filter=build_noise_filter(),
        ),
    )
    await run_agent_pipeline(transport, ctx, label="Twilio Media Streams")
