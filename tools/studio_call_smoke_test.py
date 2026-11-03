"""Smoke test for Agent Studio phone calls - no phone, no Twilio, nothing dialled.

Creates a Studio agent (Hindi, with its own instructions, greeting and one Q&A
pair) in a throwaway in-memory database, then connects a fake Twilio socket to
the REAL Studio call pipeline (/studio/media -> services/voice_pipeline/
studio_pipeline.py): local Whisper STT, retrieval, the real LLM and OmniVoice.
The "caller" speaks a question whose answer is only in the agent's Q&A pair.

Checks: the agent's own greeting is spoken; the caller's words are transcribed;
the Q&A pair is retrieved into the prompt; the agent answers out loud; time to
first audio; the call row ends "completed" with the transcript saved.

    .\\venv\\Scripts\\python.exe tools\\studio_call_smoke_test.py

Caller audio comes from data/smoke_audio (cached by tools/latency_smoke_test.py).
"""

import asyncio
import json
import os
import sqlite3
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

import latency_smoke_test as smoke  # noqa: E402
import simulate_scenario_call as sim  # noqa: E402
from config import settings  # noqa: E402

QUESTION = "मेरी शिफ्ट किस टर्मिनल पर है?"
CLOSER = "ठीक है, समझ गया"
GREETING = "नमस्ते, मैं शिफ्टईज़ की सहायक बोल रही हूँ। मैं आपकी कैसे मदद कर सकती हूँ?"
FACT = "आपकी कल की शिफ्ट टर्मिनल सीटी थ्री पर है, सुबह आठ बजे से।"


def setup_db():
    import simple_db
    from services.studio import store

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    simple_db._connection = conn
    store._ready = False
    agent = store.create_agent(
        name="Shift Helper", language="hi",
        system_instructions="You are a polite assistant for ShiftEaze who answers workers' questions about their shift. Answer in Hindi in one or two short sentences.",
        personality={"greeting": GREETING, "role": "Shift assistant", "unknown_questions": "Say the supervisor will confirm by SMS."},
        voice={"speed": 1.0, "style": "friendly"}, basic={"objective": "Answer shift questions."},
    )
    store.add_training(agent["id"], "qa", QUESTION, FACT, {"category": "shift"})
    call_id = store.create_call(agent=agent, kind="call", language="hi")
    return agent, call_id


async def main() -> int:
    agent, call_id = setup_db()
    from services.studio import store

    handler = __import__("logging").FileHandler(os.path.join(TOOLS, "studio_call_last_run.log"), encoding="utf-8", mode="w")
    handler.setFormatter(__import__("logging").Formatter("%(asctime)s %(name)-34s %(message)s"))
    for name in ("voice_pipeline", "studio", "local_ai", "tts"):
        lg = __import__("logging").getLogger(name)
        lg.setLevel(20)
        lg.addHandler(handler)
        lg.propagate = False

    from services.voice_pipeline.local_stt import preload
    from services.voice_pipeline.studio_pipeline import run_studio_media_pipeline

    specs = [(settings.local_stt_model_for("hi"), settings.local_stt_compute_type)]
    await asyncio.to_thread(preload, specs, settings.local_stt_isolation)
    try:
        from local_ai.llm_engine import prime_prompt

        await prime_prompt("Reply with the JSON.", {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
    except Exception as e:  # noqa: BLE001
        print("LLM warm-up failed:", e)

    app = FastAPI()

    @app.websocket("/studio/media")
    async def media(websocket: WebSocket):
        await run_studio_media_pipeline(websocket, auto_hang_up=False)

    port = 8768
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", log_config=None, timeout_graceful_shutdown=3))
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            audio_q, audio_c = await smoke.caller_audio(client, QUESTION), await smoke.caller_audio(client, CLOSER)
        except Exception as e:  # noqa: BLE001 - caller audio not cached and Sarvam unavailable
            print("Caller audio unavailable (needs data/smoke_audio or Sarvam credits):", e)
            return 2

    report: dict = {"agent": agent["name"], "turns": []}
    async with websockets.connect(f"ws://127.0.0.1:{port}/studio/media") as ws:
        stream_sid = f"MZ{uuid.uuid4().hex[:10]}"
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "sequenceNumber": "1", "streamSid": stream_sid, "start": {
            "streamSid": stream_sid, "callSid": "CA_STUDIO_SMOKE", "customParameters": {"callId": call_id},
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}))
        t_connect = time.monotonic()
        tw = sim.FakeTwilio(ws, stream_sid)
        tasks = [asyncio.create_task(tw.send_loop()), asyncio.create_task(tw.recv_loop())]

        if not await smoke.wait_for(lambda: tw._cur is not None or tw.segments, 90):
            report["error"] = "agent never spoke the greeting"
        else:
            report["greeting_audio_after_connect_ms"] = int((time.monotonic() - t_connect) * 1000)
            await smoke.wait_for(lambda: time.monotonic() - tw._last_loud > 1.2, 40)  # greeting finished
            for text, audio in ((QUESTION, audio_q), (CLOSER, audio_c)):
                await asyncio.sleep(0.6)
                _, trail = smoke.speech_bounds(audio)
                tw.say(audio)
                await smoke.wait_for(lambda: tw.caller_done_at is not None, 15)
                last_word_at = tw.caller_done_at - trail
                got = await smoke.wait_for(lambda: tw._last_loud > tw.caller_done_at, 40)
                report["turns"].append({"caller": text, "agent_audio_from_last_word_ms":
                                        int((tw._last_loud - last_word_at) * 1000) if got else None})
                await smoke.wait_for(lambda: time.monotonic() - tw._last_loud > 1.2, 40)
        for t in tasks:
            t.cancel()
    t_closed = time.monotonic()
    await smoke.wait_for(lambda: store.get_call(call_id)["status"] == "completed", 60)  # pipeline closes, finish() runs
    report["call_finalized_after_hangup_secs"] = round(time.monotonic() - t_closed, 1)

    call = store.get_call(call_id)
    msgs = store.list_messages(call_id)
    traces = store.list_traces(call_id)
    kn = [t for t in traces if t["category"] == "knowledge"]
    report["call"] = {"status": call["status"], "outcome": call["outcome"], "duration_secs": call["duration_secs"], "error": call["error"]}
    report["transcript"] = [f"{m['speaker']}: {m['message']}" for m in msgs]
    report["retrieved_for_question"] = (kn[0]["data"]["matches"] if kn else None)
    report["trace_categories"] = sorted({t["category"] for t in traces})
    from services.tts.playout import DIAGNOSTICS
    from services.voice_pipeline import latency

    stats = next(iter(latency.RECENT.values()), None)
    report["time_to_first_audio_from_end_of_speech_ms"] = stats.turns_ms if stats else None
    report["tts_playout"] = DIAGNOSTICS.summary()

    agent_lines = [m["message"] for m in msgs if m["speaker"] == "agent"]
    report["checks"] = {
        "greeting_is_agents_own": bool(agent_lines) and agent_lines[0].startswith("नमस्ते, मैं शिफ्टईज़"),
        "question_transcribed": any(m["speaker"] == "customer" for m in msgs),
        "qa_pair_retrieved": bool(kn and any(x["source"] == "qa" for x in kn[0]["data"]["matches"])),
        "agent_answered_aloud": len(agent_lines) >= 2,
        "call_completed": call["status"] == "completed" and call["outcome"] == "completed",
    }
    server.should_exit = True
    await serve_task
    if settings.local_stt_isolation == "process":
        from services.voice_pipeline.stt_worker import worker

        await asyncio.to_thread(worker.close)
    sys.stdout.buffer.write((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
