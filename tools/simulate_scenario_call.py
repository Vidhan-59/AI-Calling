"""End-to-end phone-call simulator for the Sarvam scenario pipeline - no
phone, no Twilio account, no tunnel needed.

Acts exactly like Twilio Media Streams: connects to the pipeline's WebSocket,
sends the "connected"/"start" handshake, then streams 20ms mu-law frames in
real time. Caller answers are real Hindi/Gujarati/English speech synthesized
by Sarvam in a different voice. The bot's audio is captured, split into
utterances, timed against the moment the caller stopped speaking, and
transcribed back through Sarvam STT so the report shows what the bot
actually SAID - not just that some audio came back.

Runs the real pipeline code (services/voice_pipeline/scenario_pipeline.py)
in-process on a spare port, with Twilio auto-hang-up disabled. Test rows are
written under call SIDs starting CA_SIM and removed afterwards.

    .\\venv\\Scripts\\python.exe tools\\simulate_scenario_call.py
    .\\venv\\Scripts\\python.exe tools\\simulate_scenario_call.py --case no_with_reason
"""

import argparse
import asyncio
import audioop
import base64
import io
import json
import logging
import os
import sys
import time
import uuid
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
import websockets  # noqa: E402
from fastapi import FastAPI, WebSocket  # noqa: E402

from config import settings  # noqa: E402

FRAME_BYTES = 160  # 20ms of 8kHz mu-law
SPEECH_RMS = 350
QUIET_TO_END_SEGMENT = 0.7

CASES = {
    "no_with_reason": {
        "scenario_id": "shift_confirm", "language": "hi",
        "turns": ["नहीं, मैं कल नहीं आ पाऊँगा", "मेरी तबीयत ठीक नहीं है, बुखार है"],
        "expect_outcome": "absent_with_reason",
    },
    "yes": {
        "scenario_id": "shift_confirm", "language": "hi",
        "turns": ["हाँ जी, आ जाऊँगा"],
        "expect_outcome": "confirmed",
    },
    "no_reason_same_breath": {
        "scenario_id": "shift_confirm", "language": "hi",
        "turns": ["नहीं आ पाऊँगा, मेरी पत्नी अस्पताल में है", "जी, पत्नी अस्पताल में भर्ती है"],  # asked why anyway
        "expect_outcome": "absent_with_reason",
    },
    "hello_over_opening": {
        "scenario_id": "shift_confirm", "language": "hi",
        "say_during_opening": "हेलो",
        "turns": ["हाँ जी, आ जाऊँगा"],
        "expect_outcome": "confirmed",
    },
    "correction_means_yes": {
        "scenario_id": "shift_confirm", "language": "hi",
        "turns": ["नहीं नहीं, मैं आ जाऊँगा"],
        "expect_outcome": "confirmed",
    },
    "gujarati_no": {
        "scenario_id": "shift_confirm", "language": "gu",
        "turns": ["ના, હું કાલે નહીં આવી શકું", "મારી તબિયત સારી નથી"],
        "expect_outcome": "absent_with_reason",
    },
}
FACTS = {"employee_name": "अमित", "date": "कल", "time": "सुबह आठ", "shift_name": "डे", "terminal": "CT3"}
FACTS_GU = {"employee_name": "અમિત", "date": "કાલે", "time": "સવારે આઠ", "shift_name": "ડે", "terminal": "CT3"}

H = {"api-subscription-key": settings.sarvam_api_key or ""}
_LANG = {"hi": "hi-IN", "gu": "gu-IN", "en": "en-IN"}
VERBOSE = False
_T0 = time.monotonic()


def note(msg: str) -> None:
    if VERBOSE:
        sys.stderr.buffer.write(f"[sim {time.monotonic() - _T0:7.2f}s] {msg}\n".encode("utf-8"))
        sys.stderr.flush()


async def caller_speech(client: httpx.AsyncClient, text: str, language: str) -> bytes:
    """Caller audio as 8kHz mu-law, in a male voice (the bot is female)."""
    r = await client.post(
        "https://api.sarvam.ai/text-to-speech", headers=H,
        json={"inputs": [text], "target_language_code": _LANG[language], "speaker": "rahul",
              "model": "bulbul:v3", "speech_sample_rate": 8000},
    )
    r.raise_for_status()
    with wave.open(io.BytesIO(base64.b64decode(r.json()["audios"][0]))) as w:
        pcm = w.readframes(w.getnframes())
    return audioop.lin2ulaw(pcm, 2)


async def transcribe(client: httpx.AsyncClient, pcm8k: bytes, language: str) -> str:
    pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16k)
    r = await client.post(
        "https://api.sarvam.ai/speech-to-text", headers=H,
        files={"file": ("bot.wav", buf.getvalue(), "audio/wav")},
        data={"model": "saaras:v3", "language_code": _LANG[language]},
    )
    return r.json().get("transcript", f"<stt error {r.status_code}>") if r.status_code == 200 else f"<stt error {r.status_code}>"


class FakeTwilio:
    def __init__(self, ws, stream_sid: str):
        self.ws = ws
        self.stream_sid = stream_sid
        self.outgoing: list[bytes] = []
        self.caller_done_at: float | None = None
        self.segments: list[dict] = []  # bot utterances: start, end, pcm
        self._cur: dict | None = None
        self._last_loud = 0.0
        self.closed = asyncio.Event()
        self.clears = 0
        self.event_counts: dict[str, int] = {}
        self.max_rms = 0
        self.recent_rms: list[int] = []

    def say(self, mulaw: bytes) -> None:
        self.outgoing = [mulaw[i:i + FRAME_BYTES].ljust(FRAME_BYTES, b"\xff") for i in range(0, len(mulaw), FRAME_BYTES)]
        self.caller_done_at = None

    async def send_loop(self) -> None:
        seq, next_t = 2, time.monotonic()
        silence = b"\xff" * FRAME_BYTES
        try:
            while not self.closed.is_set():
                if self.outgoing:
                    chunk = self.outgoing.pop(0)
                    if not self.outgoing:
                        self.caller_done_at = time.monotonic()
                else:
                    chunk = silence
                await self.ws.send(json.dumps({
                    "event": "media", "sequenceNumber": str(seq), "streamSid": self.stream_sid,
                    "media": {"track": "inbound", "chunk": str(seq), "timestamp": str(seq * 20),
                              "payload": base64.b64encode(chunk).decode()},
                }))
                seq += 1
                next_t += 0.02
                await asyncio.sleep(max(0.0, next_t - time.monotonic()))
        except websockets.ConnectionClosed:
            self.closed.set()

    async def recv_loop(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                event = msg.get("event")
                self.event_counts[event] = self.event_counts.get(event, 0) + 1
                if event != "media":
                    note(f"server sent event={event}")
                if event == "clear":
                    self.clears += 1
                if event != "media":
                    continue
                pcm = audioop.ulaw2lin(base64.b64decode(msg["media"]["payload"]), 2)
                now = time.monotonic()
                rms = audioop.rms(pcm, 2)
                self.max_rms = max(self.max_rms, rms)
                self.recent_rms.append(rms)
                loud = rms > SPEECH_RMS
                if loud:
                    self._last_loud = now
                    if self._cur is None:
                        self._cur = {"start": now, "pcm": bytearray()}
                        note(f"bot utterance #{len(self.segments)} started")
                if self._cur is not None:
                    self._cur["pcm"] += pcm
                    if not loud and now - self._last_loud > QUIET_TO_END_SEGMENT:
                        self._cur["end"] = self._last_loud
                        self.segments.append(self._cur)
                        note(f"bot utterance #{len(self.segments) - 1} ended "
                             f"({self._cur['end'] - self._cur['start']:.1f}s)")
                        self._cur = None
        except websockets.ConnectionClosed:
            pass
        finally:
            if self._cur is not None:
                self._cur["end"] = self._last_loud
                self.segments.append(self._cur)
                self._cur = None
            self.closed.set()

    async def heartbeat(self) -> None:
        while not self.closed.is_set():
            await asyncio.sleep(1.0)
            n, self.recent_rms = len(self.recent_rms), []
            note(f"heartbeat: {n} media frames/s from bot, segment_open={self._cur is not None}")

    def close_stale_segment(self) -> None:
        # Close the open bot utterance by wall clock too, not only when a
        # quiet frame arrives - robust whether or not the server sends
        # silence frames between utterances.
        if self._cur is not None and time.monotonic() - self._last_loud > QUIET_TO_END_SEGMENT:
            self._cur["end"] = self._last_loud
            self.segments.append(self._cur)
            note(f"bot utterance #{len(self.segments) - 1} ended ({self._cur['end'] - self._cur['start']:.1f}s)")
            self._cur = None

    async def wait_bot_utterance(self, index: int, timeout: float = 30.0) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.close_stale_segment()
            if len(self.segments) > index:
                return self.segments[index]
            if self.closed.is_set() and len(self.segments) <= index:
                return None
            await asyncio.sleep(0.05)
        return None


async def run_case(name: str, port: int) -> dict:
    from services import scenario_store

    case = CASES[name]
    lang = case["language"]
    facts = FACTS_GU if lang == "gu" else FACTS
    ref = f"sim{uuid.uuid4().hex[:8]}"
    call_sid = f"CA_SIM_{uuid.uuid4().hex[:10]}"
    scenario_store.create(call_sid=f"pending:{ref}", scenario_id=case["scenario_id"], language=lang,
                          to_number="+910000000000", facts=facts, attempt=1)

    from services.voice_pipeline.scenario_pipeline import prefetch_call_lines

    report = {"case": name, "turns": []}
    async with httpx.AsyncClient(timeout=30) as client:
        # Same as POST /scenarios/call: synthesize every line while "ringing".
        t0 = time.monotonic()
        await prefetch_call_lines(case["scenario_id"], lang, facts, 1)
        report["prefetch_secs"] = round(time.monotonic() - t0, 2)
        note(f"prefetch done in {report['prefetch_secs']}s")
        caller_audio = [await caller_speech(client, t, lang) for t in case["turns"]]
        interjection = await caller_speech(client, case["say_during_opening"], lang) if case.get("say_during_opening") else None

        async with websockets.connect(f"ws://127.0.0.1:{port}/scenarios/media") as ws:
            stream_sid = f"MZ{uuid.uuid4().hex[:10]}"
            await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
            await ws.send(json.dumps({
                "event": "start", "sequenceNumber": "1", "streamSid": stream_sid,
                "start": {"accountSid": "ACsim", "streamSid": stream_sid, "callSid": call_sid, "tracks": ["inbound"],
                          "customParameters": {"ref": ref, "scenarioId": case["scenario_id"], "language": lang, "attempt": "1"},
                          "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}},
            }))
            connected_at = time.monotonic()
            note("handshake sent, streaming caller audio")
            tw = FakeTwilio(ws, stream_sid)
            sender = asyncio.create_task(tw.send_loop())
            receiver = asyncio.create_task(tw.recv_loop())
            beat = asyncio.create_task(tw.heartbeat())

            if interjection:
                await asyncio.sleep(1.5)
                tw.say(interjection)  # "hello?" while the opening plays

            opening = await tw.wait_bot_utterance(0, timeout=40)
            if opening is None:
                report["error"] = "bot never spoke the opening"
            else:
                report["connect_to_first_audio_ms"] = int((opening["start"] - connected_at) * 1000)
                report["opening"] = await transcribe(client, bytes(opening["pcm"]), lang)
                report["opening_interrupted"] = tw.clears > 0
                idx = 1
                for text, audio in zip(case["turns"], caller_audio):
                    await asyncio.sleep(0.6)
                    tw.say(audio)
                    note(f"caller says: {text}")
                    while tw.caller_done_at is None and not tw.closed.is_set():
                        await asyncio.sleep(0.01)
                    reply = await tw.wait_bot_utterance(idx, timeout=30)
                    if reply is None:
                        report["turns"].append({"caller": text, "bot": None, "latency_ms": None})
                        break
                    report["turns"].append({
                        "caller": text,
                        "bot": await transcribe(client, bytes(reply["pcm"]), lang),
                        "latency_ms": int((reply["start"] - tw.caller_done_at) * 1000),
                    })
                    idx += 1
            try:
                await asyncio.wait_for(tw.closed.wait(), timeout=20)
                report["bot_hung_up"] = True
            except asyncio.TimeoutError:
                report["bot_hung_up"] = False
            sender.cancel()
            receiver.cancel()
            beat.cancel()
            report["events_from_server"] = tw.event_counts
            report["max_bot_rms"] = tw.max_rms

    await asyncio.sleep(0.5)
    row = scenario_store.get(call_sid) or {}
    report["outcome"] = row.get("outcome")
    report["collected"] = row.get("collected") or row.get("collected_json")
    report["expected_outcome"] = case["expect_outcome"]
    # Every scripted caller turn must get a reply: an outcome can come out
    # "right" while the flow is wrong (e.g. reason never asked, so the refusal
    # itself got saved as the reason - this check is what caught that).
    all_answered = len(report["turns"]) == len(case["turns"]) and all(t["bot"] for t in report["turns"])
    report["passed"] = report["outcome"] == case["expect_outcome"] and report.get("bot_hung_up", False) and all_answered

    # Post-call analysis runs after hang-up (services/call_analysis.py); let
    # it finish, then record what landed on the call's DB row.
    from services import call_analysis

    if call_analysis._tasks:
        await asyncio.gather(*list(call_analysis._tasks), return_exceptions=True)
    row = scenario_store.get(call_sid) or {}
    report["db_row"] = {k: row.get(k) for k in (
        "absence_reason", "reason_category", "reason_english", "sentiment", "sentiment_score", "emotion",
        "summary", "follow_up_required", "follow_up_note", "turn_count", "duration_secs",
        "avg_turn_latency_ms", "analysis_status")}
    report["pending_row_left"] = scenario_store.get(f"pending:{ref}") is not None
    _cleanup(call_sid, ref)
    return report


def _cleanup(call_sid: str, ref: str) -> None:
    from services.call_logger import _log_path
    from simple_db import get_cursor

    with get_cursor() as cur:
        for table, col in (("scenario_calls", "call_sid"), ("messages", "call_sid"), ("calls", "call_sid")):
            try:
                cur.execute(f"DELETE FROM {table} WHERE {col} IN (?, ?)", (call_sid, f"pending:{ref}"))
            except Exception:
                pass
    try:
        os.remove(_log_path(call_sid))
    except OSError:
        pass


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--out", default=None, help="write the JSON report here (UTF-8)")
    parser.add_argument("--verbose", action="store_true", help="print a live timeline to stderr")
    args = parser.parse_args()
    global VERBOSE
    VERBOSE = args.verbose

    from services.voice_pipeline.scenario_pipeline import run_scenario_media_pipeline

    # Attached to the app's own loggers (after imports) rather than root:
    # importing pipecat reconfigures the root logger.
    handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), "simulate_last_run.log"),
                                  encoding="utf-8", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)-34s %(message)s"))
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(logging.Formatter("[app] %(name)s: %(message)s"))
    for name in ("voice_pipeline", "scenario_engine", "local_ai", "call_analysis"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
        if args.verbose:
            lg.addHandler(err_handler)
        lg.propagate = False

    app = FastAPI()

    @app.websocket("/scenarios/media")
    async def media(websocket: WebSocket):
        await run_scenario_media_pipeline(websocket, auto_hang_up=False)

    from config import settings as app_settings
    from simple_db import init_db

    init_db()  # as main.py does at startup: adds any new scenario_calls columns

    if app_settings.stt_engine == "local":
        # Same as the server's startup warm-up (main.py): models load once, up
        # front, so the first simulated call doesn't measure a model load.
        from services.voice_pipeline.local_stt import preload

        langs = [l.strip() for l in app_settings.local_stt_preload.split(",") if l.strip()]
        specs = [(app_settings.local_stt_model_for(l), app_settings.local_stt_compute_type) for l in langs]
        await asyncio.to_thread(preload, specs, app_settings.local_stt_isolation)
        note(f"local STT preloaded for {langs}")

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=3))
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    names = list(CASES) if args.case == "all" else [args.case]
    reports = []
    for name in names:
        try:
            reports.append(await run_case(name, args.port))
        except Exception as e:  # noqa: BLE001 - one bad case (e.g. a network blip) mustn't lose the rest
            import traceback; reports.append({"case": name, "passed": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:], "turns": []})
    server.should_exit = True
    await serve_task

    text = json.dumps(reports, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
