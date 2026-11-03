"""Assembles and runs the live pipeline for a Studio agent's phone call:

  Twilio audio -> RNNoise -> Silero VAD -> barge-in -> latency monitor
               -> local Whisper STT -> StudioCallProcessor (instructions +
                  retrieved training + LLM) -> OmniVoice TTS (the agent's
                  voice and speed) -> Twilio

The call was placed by services/studio/calling.py, which created the
studio_calls row and passed its id through the TwiML as the `callId` stream
parameter. The id is the only credential this socket has (Twilio cannot send
our admin key), so it is accepted only for a call that is still waiting to
connect.
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

from config import settings
from services.studio import store
from services.tts.pipecat_service import OmniVoiceTTSService
from services.voice_pipeline.audio_frontend import build_noise_filter, front_end
from services.voice_pipeline.scenario_pipeline import STT_SAMPLE_RATE, build_stt
from services.voice_pipeline.studio_processor import StudioCallProcessor
from services.voice_pipeline.twilio_pipeline import TTS_SAMPLE_RATE, VAD_STOP_SECS, read_twilio_start

logger = logging.getLogger("voice_pipeline.studio_pipeline")


def build_tts(agent: dict) -> OmniVoiceTTSService:
    """The agent's own voice and speed. `owner` is unique to the agent so the
    Voice Studio assignment table (scenarios/YAML agents) can never override it;
    no voice chosen = the library default voice."""
    voice = agent.get("voice") or {}
    return OmniVoiceTTSService(
        owner=f"studio:{agent['id']}", language=agent.get("language") or "hi",
        fallback_voice_id=voice.get("voice_id") or None, fallback_speed=float(voice.get("speed") or 1.0),
    )


async def run_studio_pipeline(transport, *, agent: dict, call_id: str, stt=None, tts=None, stream_reply=None,
                              label: str = "Twilio") -> StudioCallProcessor:
    tts = tts or build_tts(agent)
    extra = {"stream_reply": stream_reply} if stream_reply else {}
    processor = StudioCallProcessor(agent=agent, call_id=call_id, playout=getattr(tts, "playout", None), **extra)
    front, _ = front_end(call_id=call_id, stop_secs=VAD_STOP_SECS)
    pipeline = Pipeline([
        transport.input(), *front, stt or build_stt(agent.get("language") or "hi"), processor, tts, transport.output(),
    ])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=STT_SAMPLE_RATE, audio_out_sample_rate=TTS_SAMPLE_RATE),
        enable_rtvi=False,
    )

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(_transport, _client):
        # The caller hung up (Twilio closed the stream). The transport only
        # reports it; without ending the task the pipeline - and the call row,
        # still "in progress" - lives on until the maximum call length.
        logger.info("[%s] caller disconnected - ending the pipeline", call_id)
        await task.cancel(reason="client disconnected")

    logger.info("[%s] studio call connected over %s: agent=%s (%s)", call_id, label, agent["name"], agent["id"])
    try:
        await PipelineRunner(handle_sigint=False).run(task)
    except Exception:
        logger.exception("[%s] studio pipeline crashed", call_id)
        store.update_call(call_id, error="The call pipeline crashed - see the server log.")
    finally:
        processor.finish("caller hung up or the stream ended")
    return processor


async def run_studio_media_pipeline(websocket: WebSocket, *, auto_hang_up: bool = True, stt=None, tts=None,
                                    stream_reply=None) -> StudioCallProcessor | None:
    await websocket.accept()
    try:
        start = await read_twilio_start(websocket)
    except WebSocketDisconnect:
        logger.info("studio media stream closed before the Twilio handshake (reachability probe?)")
        return None
    except (ValueError, json.JSONDecodeError) as e:
        logger.error("studio media stream handshake failed: %s", e)
        return None

    params = start.get("customParameters") or {}
    call_id = params.get("callId", "")
    call = store.get_call(call_id) if call_id else None
    if not call or call["status"] != "connecting" or call["kind"] != "call":
        logger.warning("studio media stream refused: unknown or already-used callId %r", call_id)
        await websocket.close(code=4403)
        return None
    agent = store.get_agent(call["agent_id"]) or call["agent_snapshot"]
    if not agent.get("name"):
        await websocket.close(code=4404)
        return None
    store.update_call(call_id, metrics={**call["metrics"], "twilio_call_sid": start.get("callSid") or call["metrics"].get("twilio_call_sid")})

    hang_up = bool(auto_hang_up and settings.twilio_account_sid and settings.twilio_auth_token)
    serializer = TwilioFrameSerializer(
        stream_sid=start.get("streamSid"), call_sid=start.get("callSid"),
        account_sid=settings.twilio_account_sid, auth_token=settings.twilio_auth_token,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=hang_up),
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False, serializer=serializer,
            audio_in_filter=build_noise_filter(),
        ),
    )
    return await run_studio_pipeline(transport, agent=agent, call_id=call_id, stt=stt, tts=tts,
                                     stream_reply=stream_reply, label="Twilio Media Streams")
