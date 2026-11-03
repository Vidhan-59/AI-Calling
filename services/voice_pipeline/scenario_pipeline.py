"""Assembles and runs the live pipeline for a scripted scenario call over raw
Twilio Media Streams (phone calls) or WebRTC (browser calls, webrtc_pipeline.py):

  caller audio -> RNNoise -> Silero VAD -> barge-in -> latency monitor
               -> STT (Sarvam saaras:v3 when USE_SARVAM_STT=true; else local
                       faster-whisper on the GPU, decoded speculatively)
               -> ScenarioCallProcessor (ScenarioSession + fast/Qwen classify)
               -> cached OmniVoice TTS audio -> caller

The noise/VAD/barge-in layer is audio_frontend.py.

There is no TTS service in the pipeline on purpose: every line is scripted,
so audio comes from services/voice_pipeline/tts_cache.py, prefetched while
the phone rings (see prefetch_call_lines, called from POST /scenarios/call).
"""

import asyncio
import json
import logging

from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

from config import settings
from services import conversation_store, scenario_engine, scenario_store
from services.call_logger import init_call_log
from services.voice_pipeline import tts_cache
from services.voice_pipeline.audio_frontend import build_noise_filter, front_end
from services.voice_pipeline.scenario_processor import ScenarioCallProcessor
from services.voice_pipeline.stt_factory import STT_SAMPLE_RATE, build_stt
from services.voice_pipeline.twilio_pipeline import read_twilio_start

logger = logging.getLogger("voice_pipeline.scenario_pipeline")

# End-of-turn silence. stop_secs is only the "flush STT now" signal - the
# processor adds its own grace window for anything but a clear yes/no, so
# this can stay short. 0.5s: at 0.3s VAD split "haan ji, [pause] aa jaunga"
# at the comma and the bot answered the first half while the caller was still
# talking (found by tools/simulate_scenario_call.py). Natural comma pauses are
# ~0.2-0.45s. Speculative STT decodes during this window, so the decode no
# longer adds on top of it. Start/barge-in thresholds: audio_frontend.vad_params.
VAD_STOP_SECS = 0.5


def lines_for(scenario_id: str, language: str, facts: dict, attempt: int = 1) -> list[str]:
    """Every line this scenario might speak, rendered with this call's facts."""
    cfg = scenario_engine.get_language_config(scenario_id, language)
    if not cfg:
        return []
    opening_key = "attempt_2_opening" if attempt > 1 else "opening"
    templates = [
        cfg.get(opening_key) or cfg["opening"],
        cfg.get("unclear_reprompt", ""),
        cfg.get("reason_reprompt", ""),
        cfg.get("silence_checkin", ""),
        cfg.get("silence_reply", ""),
    ]
    for branch in cfg["branches"].values():
        templates += [branch.get("reply", ""), branch.get("after_collect_reply", "")]
    return [scenario_engine.render(t, facts) for t in templates if t]


async def prefetch_call_lines(scenario_id: str, language: str, facts: dict, attempt: int = 1) -> None:
    """Synthesize every line for a call while it's ringing. The opening goes
    first on its own, since it's needed the instant the call connects."""
    lines = lines_for(scenario_id, language, facts, attempt)
    if not lines:
        return
    failed = await tts_cache.prewarm([(lines[0], language)], owner=scenario_id)
    failed += await tts_cache.prewarm([(line, language) for line in lines[1:]], owner=scenario_id)
    if failed:
        logger.warning("[%s/%s] %d line(s) failed to prefetch - will retry on demand", scenario_id, language, failed)


async def prewarm_static_lines() -> None:
    """Server-startup warmup: every line with no per-call facts in it, for
    every scenario and language, into the disk cache. One-time cost - later
    startups hit the disk cache (the key includes the text, so edited lines
    are re-synthesized automatically)."""
    total = failed = 0
    for scenario_id in scenario_engine.load_scenarios():
        # Per scenario: each can be assigned a different voice in Voice Studio.
        lines: list[tuple[str, str]] = []
        for language in ("hi", "gu", "en"):
            for attempt in (1, 2):
                for line in lines_for(scenario_id, language, {}, attempt):
                    if "{" not in line:
                        lines.append((line, language))
        total += len(set(lines))
        failed += await tts_cache.prewarm(lines, owner=scenario_id)
    logger.info("TTS static-line prewarm done: %d lines, %d failed", total, failed)


def build_stream_twiml(ws_url: str, params: dict[str, str]) -> str:
    from xml.sax.saxutils import quoteattr

    params_xml = "".join(f"<Parameter name={quoteattr(k)} value={quoteattr(v)}/>" for k, v in params.items())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Connect><Stream url={quoteattr(ws_url)}>{params_xml}</Stream></Connect></Response>"
    )


async def run_scenario_media_pipeline(websocket: WebSocket, *, auto_hang_up: bool = True, stt=None) -> ScenarioCallProcessor | None:
    """`stt` is injectable for tests; production builds it from settings.use_sarvam_stt."""
    await websocket.accept()
    try:
        start = await read_twilio_start(websocket)
    except WebSocketDisconnect:
        # Closed before sending anything - run-all.ps1's tunnel check does
        # exactly this on every startup, so it isn't an error.
        logger.info("scenario media stream closed before the Twilio handshake (reachability probe?)")
        return None
    except (ValueError, json.JSONDecodeError) as e:
        logger.error("scenario media stream handshake failed: %s", e)
        return None

    call_sid = start.get("callSid") or "unknown"
    stream_sid = start.get("streamSid")
    params = start.get("customParameters") or {}
    try:
        session = start_scenario_call(
            call_sid=call_sid,
            scenario_id=params.get("scenarioId") or "shift_confirm",
            language=params.get("language") or "hi",
            attempt=int(params.get("attempt") or 1),
            ref=params.get("ref", ""),
        )
    except ValueError as e:
        logger.error("[%s] bad scenario config: %s", call_sid, e)
        return None

    hang_up_via_twilio = bool(auto_hang_up and settings.twilio_account_sid and settings.twilio_auth_token)
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=hang_up_via_twilio),
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer,
            audio_in_filter=build_noise_filter(),
        ),
    )
    return await run_scenario_pipeline(transport, call_sid=call_sid, session=session, stt=stt, label="Twilio Media Streams")


def start_scenario_call(
    *, call_sid: str, scenario_id: str, language: str, attempt: int, ref: str = "",
) -> scenario_engine.ScenarioSession:
    """Everything a scenario call needs before audio flows, whatever the
    transport: the session, its rows, the TTS prefetch. Raises ValueError for
    a bad scenario config."""
    pending = scenario_store.get(f"pending:{ref}") if ref else None
    facts = (pending or {}).get("facts", {})
    session = scenario_engine.ScenarioSession(
        scenario_id=scenario_id, language=language, facts=facts, attempt=attempt
    )
    to_number = (pending or {}).get("to_number", "")
    scenario_store.create(
        call_sid=call_sid, scenario_id=scenario_id, language=language, to_number=to_number,
        facts=facts, attempt=attempt, rgid=(pending or {}).get("rgid"),
    )
    if pending:
        # Its facts now live on the real call's row; left behind, every call
        # showed up twice (once as a forever-"in_progress" pending:* row).
        scenario_store.delete(f"pending:{ref}")
    conversation_store.create_call(call_sid, scenario_id, "", to_number, "outbound")
    init_call_log(
        call_sid, scenario_id=scenario_id, language=language, to_number=to_number,
        from_number="", facts=facts, attempt=attempt,
    )
    # No-op when POST /scenarios/call already prefetched while ringing; makes
    # any other entry path (e.g. a directly dialled test) just as fast after
    # the first turn.
    asyncio.create_task(prefetch_call_lines(scenario_id, language, facts, attempt))
    return session


async def run_scenario_pipeline(
    transport, *, call_sid: str, session: scenario_engine.ScenarioSession, stt=None, label: str,
) -> ScenarioCallProcessor:
    """One scenario call over any Pipecat transport, until it ends."""
    processor = ScenarioCallProcessor(call_sid=call_sid, session=session, vad_stop_secs=VAD_STOP_SECS)
    front, _ = front_end(call_id=call_sid, stop_secs=VAD_STOP_SECS, can_interrupt=processor.can_be_interrupted)
    pipeline = Pipeline([
        transport.input(),
        *front,
        stt or build_stt(session.language),
        processor,
        transport.output(),
    ])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=STT_SAMPLE_RATE, audio_out_sample_rate=tts_cache.SAMPLE_RATE),
        enable_rtvi=False,
    )
    logger.info("[%s] scenario call connected: %s/%s attempt=%d (%s)",
                call_sid, session.scenario_id, session.language, session.attempt, label)
    try:
        await PipelineRunner(handle_sigint=False).run(task)
    except Exception:
        logger.exception("[%s] scenario pipeline crashed", call_sid)
    finally:
        processor.handle_disconnect()
    return processor
