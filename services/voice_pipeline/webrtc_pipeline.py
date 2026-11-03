"""Browser calls over WebRTC (Layer 1's "full-duplex WebRTC"), running the
exact same pipelines as phone calls - only the transport differs:

  phone:   PSTN -> Twilio -> Media Streams WebSocket (8kHz mu-law) -> pipeline
  browser: mic  -> WebRTC (Opus, browser echo cancellation)        -> pipeline

Phone calls can't use WebRTC - the phone network reaches us only through
Twilio's media stream - so this is for browser-originated calls: testing
barge-in and latency from a laptop without dialling, or a "talk to the
agent" button in a web app.

Uses Pipecat's SmallWebRTC (aiortc, peer-to-peer, no media server). The
browser posts an SDP offer to routers/webrtc.py, gets the answer back, and
audio then flows directly between browser and server in both directions.
"""

import asyncio
import logging
import uuid

from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequestHandler
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from services.voice_pipeline.audio_frontend import build_noise_filter

logger = logging.getLogger("voice_pipeline.webrtc")

# Public STUN so the browser and server can find each other across NAT;
# behind a strict corporate NAT a TURN server would be needed as well.
request_handler = SmallWebRTCRequestHandler(ice_servers=[IceServer(urls="stun:stun.l.google.com:19302")])

_calls: set[asyncio.Task] = set()


def _transport(connection: SmallWebRTCConnection) -> SmallWebRTCTransport:
    return SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_filter=build_noise_filter(),
        ),
    )


def _run_in_background(coro, call_id: str) -> None:
    task = asyncio.create_task(coro, name=f"webrtc-call-{call_id}")
    _calls.add(task)
    task.add_done_callback(_calls.discard)


async def start_scenario_call(connection: SmallWebRTCConnection, *, scenario_id: str, language: str, ref: str) -> str:
    """`ref` names a pending:* row holding this call's facts (routers/webrtc.py
    writes it), the same hand-off POST /scenarios/call uses for phone calls."""
    from services.voice_pipeline import scenario_pipeline

    call_sid = f"WEBRTC_{uuid.uuid4().hex[:12]}"
    session = scenario_pipeline.start_scenario_call(
        call_sid=call_sid, scenario_id=scenario_id, language=language, attempt=1, ref=ref,
    )
    _run_in_background(
        scenario_pipeline.run_scenario_pipeline(_transport(connection), call_sid=call_sid, session=session, label="WebRTC"),
        call_sid,
    )
    return call_sid


async def start_agent_call(connection: SmallWebRTCConnection, *, task_id: str) -> str:
    from services.voice_pipeline import twilio_pipeline

    provider_call_sid = f"WEBRTC_{uuid.uuid4().hex[:12]}"
    ctx = await twilio_pipeline.load_agent_call(task_id, provider_call_sid)
    if ctx is None:
        raise ValueError(f"call task {task_id!r} not found")
    _run_in_background(twilio_pipeline.run_agent_pipeline(_transport(connection), ctx, label="WebRTC"), provider_call_sid)
    return str(ctx["call_id"])


async def close_all() -> None:
    await request_handler.close()
