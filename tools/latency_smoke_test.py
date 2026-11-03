"""Latency + barge-in smoke test for the low-latency voice architecture -
no phone, no Twilio account, no tunnel needed.

Runs the REAL agent pipeline (services/voice_pipeline/twilio_pipeline.py's
run_agent_pipeline: RNNoise -> Silero VAD -> barge-in -> speculative GPU
Whisper -> streamed Qwen reply -> Sarvam streaming TTS) in-process, behind a
fake Twilio Media Streams socket (tools/simulate_scenario_call.FakeTwilio),
with real caller speech synthesized by Sarvam in a different voice.

Checks, in one call:
  1. connect -> first greeting audio
  2. NOISE while the bot speaks must NOT barge in (Layer 2)
  3. caller SPEECH while the bot speaks must barge in: time until the
     "clear" (bot audio flushed) after the caller's first syllable
  4. per turn, time to first audio: from VAD end-of-speech (server-side,
     the <1s target) and from the caller's last word (client-side)

Then, with --scenario, the scripted-scenario simulator's cases too.

    .\\venv\\Scripts\\python.exe tools\\latency_smoke_test.py
    .\\venv\\Scripts\\python.exe tools\\latency_smoke_test.py --scenario

Uses a throwaway in-memory database and fake Redis, so nothing is written
to data/app.db. Caller audio is cached under data/smoke_audio so repeat runs
don't spend Sarvam credits on it again.
"""

import argparse
import asyncio
import audioop
import hashlib
import json
import logging
import os
import random
import sys
import time
import uuid

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
import websockets  # noqa: E402
from fastapi import FastAPI, WebSocket  # noqa: E402

import simulate_scenario_call as sim  # noqa: E402
from config import DATA_DIR, settings  # noqa: E402

TARGET_MS = 1000
SPEECH_RMS = 500
CALLER_TURNS = [
    "हाँ जी, मैं कल आ जाऊँगा",
    "मेरी शिफ्ट किस टर्मिनल पर है?",
    "ठीक है, समझ गया",
    "क्या मैं थोड़ा देर से आ सकता हूँ?",
    "नहीं, बस इतना ही, धन्यवाद",
]
BARGE_IN_LINE = "एक मिनट रुकिए, मुझे कुछ पूछना है"
CTX: dict = {}
STATE: dict = {}


# -- caller audio --------------------------------------------------------------

async def caller_audio(client: httpx.AsyncClient, text: str) -> bytes:
    path = os.path.join(DATA_DIR, "smoke_audio", hashlib.sha1(text.encode()).hexdigest() + ".ulaw")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    audio = await sim.caller_speech(client, text, "hi")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(audio)
    return audio


def speech_bounds(mulaw: bytes) -> tuple[float, float]:
    """(leading, trailing) silence in the clip, seconds - so timings count
    from the first / last real syllable, not the clip's padding."""
    pcm = audioop.ulaw2lin(mulaw, 2)
    frame = 320  # 20ms of 8kHz PCM16
    loud = [i for i in range(0, len(pcm), frame) if audioop.rms(pcm[i:i + frame], 2) > SPEECH_RMS]
    if not loud:
        return 0.0, 0.0
    total = len(pcm) / 16000
    return loud[0] / 16000, total - (loud[-1] + frame) / 16000


def background_noise(secs: float) -> bytes:
    """Fan/AC hiss plus a 100Hz machine hum plus random clatter - loud, but
    nothing like a voice."""
    rnd = random.Random(7)
    samples = []
    for n in range(int(secs * 8000)):
        hum = 1400 if (n // 40) % 2 else -1400  # 100Hz square wave
        hiss = rnd.gauss(0, 1100)
        clatter = rnd.gauss(0, 5000) if (n % 4000) < 160 else 0  # a 20ms knock every 0.5s
        samples.append(max(-32000, min(32000, int(hum + hiss + clatter))))
    pcm = b"".join(s.to_bytes(2, "little", signed=True) for s in samples)
    return audioop.lin2ulaw(pcm, 2)


# -- in-process server ----------------------------------------------------------

async def setup_db() -> None:
    import fakeredis
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    import db
    import models
    from db import Base

    from services.dispatch import dialer

    fake_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    db.get_redis = dialer.get_redis = lambda: fake_redis
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool,
                                 connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    db._engine = engine
    db._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with db.session_scope() as s:
        call = models.Call(provider="mock", provider_call_sid="CA_SMOKE", to_number="+910000000000")
        s.add(call)
        await s.flush()
        span = models.AgentSpan(call_id=call.id, agent_id="shift_confirm", agent_version=1)
        s.add(span)
        await s.flush()
        CTX.update(
            task_id="smoke-task", call_id=call.id, agent_span_id=span.id, agent_id="shift_confirm",
            campaign_id="smoke", language="hi",
            contact={"id": "c1", "name": "Amit Patel", "terminal": "CT3", "designation": "Operator",
                     "phone_e164": "+910000000000"},
        )


def build_app() -> FastAPI:
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

    from services.voice_pipeline.audio_frontend import build_noise_filter
    from services.voice_pipeline.twilio_pipeline import read_twilio_start, run_agent_pipeline

    app = FastAPI()

    @app.websocket("/media")
    async def media(websocket: WebSocket):
        await websocket.accept()
        start = await read_twilio_start(websocket)
        serializer = TwilioFrameSerializer(
            stream_sid=start["streamSid"], call_sid=start["callSid"],
            params=TwilioFrameSerializer.InputParams(auto_hang_up=False),
        )
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False,
                                          serializer=serializer, audio_in_filter=build_noise_filter()),
        )
        STATE["processor"] = await run_agent_pipeline(transport, CTX, label="smoke test")

    return app


# -- the call ------------------------------------------------------------------------

async def watch_clears(tw, out: list[float]) -> None:
    seen = 0
    while not tw.closed.is_set():
        if tw.clears > seen:
            out.extend([time.monotonic()] * (tw.clears - seen))
            seen = tw.clears
        await asyncio.sleep(0.005)


async def wait_for(cond, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(0.01)
    return False


async def run_agent_call(port: int) -> dict:
    from services.voice_pipeline import latency

    report: dict = {"pipeline": "agent (streamed LLM)", "turns": []}
    async with httpx.AsyncClient(timeout=30) as client:
        turn_audio = [await caller_audio(client, t) for t in CALLER_TURNS]
        barge_audio = await caller_audio(client, BARGE_IN_LINE)
    noise = background_noise(2.0)

    async with websockets.connect(f"ws://127.0.0.1:{port}/media") as ws:
        stream_sid = f"MZ{uuid.uuid4().hex[:10]}"
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "sequenceNumber": "1", "streamSid": stream_sid,
                                  "start": {"streamSid": stream_sid, "callSid": "CA_SMOKE", "customParameters": {},
                                            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}))
        connected = time.monotonic()
        tw = sim.FakeTwilio(ws, stream_sid)
        clears: list[float] = []
        tasks = [asyncio.create_task(tw.send_loop()), asyncio.create_task(tw.recv_loop()),
                 asyncio.create_task(watch_clears(tw, clears))]

        # 1. greeting
        if not await wait_for(lambda: tw._cur is not None or tw.segments, 60):
            report["error"] = "bot never spoke"
            return report
        report["connect_to_greeting_audio_ms"] = int((time.monotonic() - connected) * 1000)

        # 2. noise over the greeting - must not interrupt
        await asyncio.sleep(0.5)
        tw.say(noise)
        await asyncio.sleep(2.3)
        report["noise_caused_barge_in"] = bool(clears)

        stats = latency.RECENT.get(str(CTX["call_id"]))

        async def caller_turn(text: str, audio: bytes) -> None:
            """Say `audio`, then time the bot's first audio after the caller's
            last syllable (client side: what the caller would hear)."""
            _, trail = speech_bounds(audio)
            tw.say(audio)
            await wait_for(lambda: tw.caller_done_at is not None, 15)
            last_word_at = tw.caller_done_at - trail
            if await wait_for(lambda: tw._last_loud > tw.caller_done_at, 30):
                report["turns"].append({"caller": text, "from_last_word_ms": int((tw._last_loud - last_word_at) * 1000)})
            else:
                report["turns"].append({"caller": text, "from_last_word_ms": None})
            await wait_for(lambda: time.monotonic() - tw._last_loud > 1.2, 30)  # bot finished

        # 3. real speech over the greeting - must interrupt
        if time.monotonic() - tw._last_loud > 0.5:
            report["barge_in"] = "skipped: greeting already finished before the barge-in test"
        else:
            lead, _ = speech_bounds(barge_audio)
            first_syllable_at = time.monotonic() + lead
            task = asyncio.create_task(caller_turn(BARGE_IN_LINE, barge_audio))
            if await wait_for(lambda: clears, 5):
                confirmed = stats.barge_in_confirmed_at[0] if stats and stats.barge_in_confirmed_at else None
                report["barge_in"] = {
                    "first_syllable_to_audio_flushed_ms": int((clears[0] - first_syllable_at) * 1000),
                    "first_syllable_to_vad_confirmed_ms": int((confirmed - first_syllable_at) * 1000) if confirmed else None,
                    "vad_confirmed_to_flush_sent_ms": int((clears[0] - confirmed) * 1000) if confirmed else None,
                }
            else:
                report["barge_in"] = "FAILED: bot kept talking"
            await task

        # 4. ordinary turns
        for text, audio in zip(CALLER_TURNS, turn_audio):
            await asyncio.sleep(0.8)
            await caller_turn(text, audio)

        for t in tasks:
            t.cancel()

    await asyncio.sleep(0.5)
    if stats:
        report["time_to_first_audio_from_end_of_speech_ms"] = stats.turns_ms
    return report


async def run_scenario_cases(port: int, cases: list[str]) -> list[dict]:
    from services.voice_pipeline import latency

    out = []
    for name in cases:
        latency.RECENT.clear()
        try:
            r = await sim.run_case(name, port)
        except Exception as e:  # noqa: BLE001 - keep going, report it
            r = {"case": name, "passed": False, "error": f"{type(e).__name__}: {e}"}
        stats = next(iter(latency.RECENT.values()), None)
        if stats:
            r["time_to_first_audio_from_end_of_speech_ms"] = stats.turns_ms
        out.append({k: r.get(k) for k in ("case", "passed", "outcome", "connect_to_first_audio_ms",
                                          "time_to_first_audio_from_end_of_speech_ms", "error")}
                   | {"from_caller_last_frame_ms": [t.get("latency_ms") for t in r.get("turns", [])]})
    return out


def summarize(agent: dict, scenario: list[dict]) -> dict:
    ttfa = list(agent.get("time_to_first_audio_from_end_of_speech_ms") or [])
    for r in scenario:
        ttfa += r.get("time_to_first_audio_from_end_of_speech_ms") or []
    barge = agent.get("barge_in")
    return {
        "time_to_first_audio_ms": {"all": ttfa, "max": max(ttfa) if ttfa else None,
                                   "under_1s": bool(ttfa) and max(ttfa) < TARGET_MS},
        "noise_ignored": agent.get("noise_caused_barge_in") is False,
        "barge_in_works": isinstance(barge, dict),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--scenario", action="store_true", help="also run the scripted-scenario simulator cases")
    parser.add_argument("--cases", default="yes,no_with_reason", help="scenario cases (with --scenario)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    await setup_db()
    from simple_db import init_db

    init_db()

    # The same warm-up the server does at startup (main.py): STT models
    # loaded, Qwen resident - so this measures steady state, not a cold load.
    from services.voice_pipeline.local_stt import preload

    specs = [(settings.local_stt_model_for("hi"), settings.local_stt_compute_type)]
    await asyncio.to_thread(preload, specs, settings.local_stt_isolation)
    from local_ai.llm_engine import prime_prompt

    await prime_prompt("Reply with the JSON.", {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})

    app = build_app()

    @app.websocket("/scenarios/media")
    async def scenario_media(websocket: WebSocket):
        from services.voice_pipeline.scenario_pipeline import run_scenario_media_pipeline

        await run_scenario_media_pipeline(websocket, auto_hang_up=False)

    # Attached after the imports above, not before: importing pipecat and
    # uvicorn's config reset the app loggers (same as simulate_scenario_call).
    handler = logging.FileHandler(os.path.join(TOOLS, "latency_smoke_last_run.log"), encoding="utf-8", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)-34s %(message)s"))
    for name in ("voice_pipeline", "conversation_agent", "scenario_engine", "local_ai"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
        lg.propagate = False
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", log_config=None,
                                           timeout_graceful_shutdown=3))
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    try:
        agent = await run_agent_call(args.port)
    except Exception as e:  # noqa: BLE001
        import traceback

        agent = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    scenario = await run_scenario_cases(args.port, args.cases.split(",")) if args.scenario else []
    server.should_exit = True
    await serve_task
    if settings.local_stt_isolation == "process":
        from services.voice_pipeline.stt_worker import worker

        await asyncio.to_thread(worker.close)

    report = {"agent_call": agent, "scenario_calls": scenario, "summary": summarize(agent, scenario)}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
