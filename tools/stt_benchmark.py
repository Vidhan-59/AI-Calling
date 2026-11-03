"""Benchmark open-source STT models for the live calling pipeline.

Scores each model on what matters for these calls, on the actual GPU, with
the Qwen classifier resident in VRAM alongside it (as on a live call):
  - CER   : character error rate vs the reference text (Hindi / Gujarati),
            on clean telephone audio and on noisy telephone audio
  - intent: does the transcript, run through the real classifier
            (services/scenario_engine.classify_utterance), land on the right
            branch - and is a same-breath reason detected? This is the
            number that decides whether a call goes right.
  - latency: model inference time for one VAD-segmented utterance
  - VRAM

Test audio (tools/stt_testset.json) is synthesized with Sarvam TTS in three
voices, squeezed through 8 kHz mu-law (the codec on a real call), and
duplicated with added noise. Synthetic speech is cleaner and more fluent than
real callers, so absolute CER will be optimistic - use the RANKING, and
re-run on real call recordings when you have them (--testset).

    python tools/stt_benchmark.py build
    python tools/stt_benchmark.py run --models whisper-large-v3-turbo,whisper-large-v3
    python tools/stt_benchmark.py run            # every registered model
"""

import argparse
import asyncio
import audioop
import base64
import gc
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np  # noqa: E402

AUDIO_DIR = os.path.join(ROOT, "data", "stt_testset")
# Converted models are multi-GB - kept off the repo drive by default (on this
# machine D: is a 50GB partition that converted models filled once).
MODELS_DIR = os.environ.get("STT_MODELS_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "ai-calling", "stt_models")
TESTSET = os.path.join(ROOT, "tools", "stt_testset.json")
VOICES = ["rahul", "neha", "rohan"]
LANG_CODES = {"hi": "hi-IN", "gu": "gu-IN", "en": "en-IN"}
NOISE_SNR_DB = 10.0

# name -> (source, compute_type, languages it's meant for)
REGISTRY = {
    "whisper-small": ("Systran/faster-whisper-small", "float16", ["hi", "gu"]),
    "whisper-medium": ("Systran/faster-whisper-medium", "float16", ["hi", "gu"]),
    "whisper-large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "float16", ["hi", "gu"]),
    "whisper-large-v3": ("Systran/faster-whisper-large-v3", "float16", ["hi", "gu"]),
    "indic-hi-large-v2": (os.path.join(MODELS_DIR, "whisper-hindi-large-v2"), "float16", ["hi"]),
    "indic-hi-medium": (os.path.join(MODELS_DIR, "whisper-hindi-medium"), "float16", ["hi"]),
    "indic-gu-medium": (os.path.join(MODELS_DIR, "whisper-gujarati-medium"), "float16", ["gu"]),
    "sarvam-saaras-v3": ("sarvam", None, ["hi", "gu"]),  # paid cloud baseline, for reference only
}


# --------------------------------------------------------------------------
# test audio
# --------------------------------------------------------------------------

def _telephone(pcm8k: bytes) -> np.ndarray:
    """8 kHz mu-law round trip (what a phone call does to audio), then up to
    16 kHz float32 - the same path as Twilio -> TwilioFrameSerializer."""
    pcm8k = audioop.ulaw2lin(audioop.lin2ulaw(pcm8k, 2), 2)
    pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 16000, None)
    return np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32) / 32768.0


def _add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Band-limited hiss plus low engine/crane rumble, at the given SNR."""
    n = len(x)
    hiss = rng.standard_normal(n)
    hiss = np.convolve(hiss, np.ones(4) / 4, mode="same")  # crude low-pass toward phone band
    t = np.arange(n) / 16000.0
    rumble = 0.6 * np.sin(2 * np.pi * 95 * t) + 0.3 * np.sin(2 * np.pi * 190 * t) + 0.2 * np.sin(2 * np.pi * 47 * t)
    noise = hiss + rumble * rng.uniform(0.5, 1.5)
    p_signal = np.mean(x ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12
    noise *= np.sqrt(p_signal / (p_noise * 10 ** (snr_db / 10)))
    return np.clip(x + noise, -1.0, 1.0).astype(np.float32)


def _write_wav(path: str, x: np.ndarray) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


async def build(args) -> None:
    import httpx

    from config import settings

    items = json.load(open(args.testset, encoding="utf-8"))["items"]
    os.makedirs(AUDIO_DIR, exist_ok=True)
    rng = np.random.default_rng(1234)
    made = 0
    async with httpx.AsyncClient(timeout=30) as client:
        for item in items:
            for voice in VOICES:
                clean_path = os.path.join(AUDIO_DIR, f"{item['id']}__{voice}__clean.wav")
                noisy_path = os.path.join(AUDIO_DIR, f"{item['id']}__{voice}__noisy.wav")
                if os.path.exists(clean_path) and os.path.exists(noisy_path):
                    continue
                for attempt in range(6):  # Sarvam rate-limits bursts (429) - back off and retry
                    r = await client.post(
                        "https://api.sarvam.ai/text-to-speech",
                        headers={"api-subscription-key": settings.sarvam_api_key},
                        json={"inputs": [item["text"]], "target_language_code": LANG_CODES[item["lang"]],
                              "speaker": voice, "model": "bulbul:v3", "speech_sample_rate": 8000},
                    )
                    if r.status_code != 429:
                        break
                    await asyncio.sleep(2 ** attempt)
                r.raise_for_status()
                with wave.open(io.BytesIO(base64.b64decode(r.json()["audios"][0]))) as w:
                    pcm8k = w.readframes(w.getnframes())
                x = _telephone(pcm8k)
                _write_wav(clean_path, x)
                _write_wav(noisy_path, _add_noise(x, NOISE_SNR_DB, rng))
                made += 1
    print(f"test set ready in {AUDIO_DIR}: {len(items)} utterances x {len(VOICES)} voices x 2 conditions "
          f"({made} newly synthesized)")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

_NORM_DROP = re.compile(r"[।॥,.!?;:\"'()\[\]\-–—…‌‍़઼]")


def normalize(text: str) -> str:
    """Spelling variants that sound identical shouldn't count as errors:
    chandrabindu vs anusvara (हाँ/हां), nukta, punctuation, spacing."""
    t = (text or "").lower().replace("ँ", "ं").replace("ઁ", "ં")
    t = _NORM_DROP.sub(" ", t)
    return " ".join(t.split())


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).replace(" ", ""), normalize(hyp).replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return min(1.0, prev[-1] / len(r))


def vram_used_mb() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------

class FasterWhisperEngine:
    def __init__(self, source: str, compute_type: str, beam_size: int, initial_prompt: dict | None):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(source, device="cuda", compute_type=compute_type)
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt or {}

    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        segments, _ = self.model.transcribe(
            audio, language=lang, beam_size=self.beam_size, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True,
            initial_prompt=self.initial_prompt.get(lang),
        )
        text = "".join(s.text for s in segments).strip()
        if lang == "gu":  # same post-processing as production (local_stt.py)
            from services.voice_pipeline.local_stt import devanagari_to_gujarati

            text = devanagari_to_gujarati(text)
        return text

    def close(self) -> None:
        del self.model


class SarvamEngine:
    def __init__(self):
        import httpx

        from config import settings

        self.client = httpx.Client(timeout=30)
        self.key = settings.sarvam_api_key

    def transcribe(self, audio: np.ndarray, lang: str) -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((audio * 32767).astype(np.int16).tobytes())
        for attempt in range(6):  # rate-limited (429) under benchmark load - back off, don't score it as an error
            r = self.client.post("https://api.sarvam.ai/speech-to-text", headers={"api-subscription-key": self.key},
                                 files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                                 data={"model": "saaras:v3", "language_code": LANG_CODES[lang]})
            if r.status_code != 429:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        return r.json().get("transcript", "")

    def close(self) -> None:
        self.client.close()


# Domain words, so short noisy answers resolve toward what callers actually say.
PROMPTS = {
    "hi": "हाँ जी, नहीं, ड्यूटी, शिफ्ट, टर्मिनल, सुपरवाइज़र, तबीयत, छुट्टी।",
    "gu": "હા, ના, ડ્યુટી, શિફ્ટ, ટર્મિનલ, સુપરવાઇઝર, તબિયત, રજા.",
}


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

async def _classify(text: str, lang: str):
    from services import scenario_engine

    facts = {"employee_name": "अमित", "date": "2 अगस्त", "time": "सुबह 8", "shift_name": "डे", "terminal": "T2"}
    return await scenario_engine.classify_utterance("shift_confirm", lang, text, facts)


def run(args) -> None:
    os.environ["SCENARIO_LLM_BACKEND"] = "local"
    # One loop for the whole run: local_ai.llm_engine keeps a shared HTTP
    # client bound to the loop it was first used on.
    loop = asyncio.new_event_loop()
    items = {i["id"]: i for i in json.load(open(args.testset, encoding="utf-8"))["items"]}
    clips = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    if not clips:
        raise SystemExit("no test audio - run `build` first")

    # Qwen resident first, so VRAM and latency reflect a live call.
    from local_ai.llm_engine import prime_prompt

    loop.run_until_complete(prime_prompt("ok", {"type": "object", "properties": {"ok": {"type": "boolean"}}}))
    baseline_vram = vram_used_mb()

    names = args.models.split(",") if args.models else list(REGISTRY)
    results = []
    for name in names:
        source, compute_type, langs = REGISTRY[name]
        if source != "sarvam" and os.path.isabs(source) and not os.path.isdir(source):
            print(f"skip {name}: {source} not found (convert it first)")
            continue
        print(f"\n=== {name} ({compute_type or 'cloud'}) ===", flush=True)
        t0 = time.monotonic()
        engine = SarvamEngine() if source == "sarvam" else FasterWhisperEngine(
            source, compute_type, args.beam_size, PROMPTS if args.prompt else None)
        load_s = time.monotonic() - t0
        vram = vram_used_mb()

        mine = [c for c in clips if items[c.split("__")[0]]["lang"] in langs]
        engine.transcribe(_read_wav(os.path.join(AUDIO_DIR, mine[0])), items[mine[0].split("__")[0]]["lang"])  # warm-up

        rows = []
        for clip in mine:
            item_id, voice, cond = clip[:-4].split("__")
            item = items[item_id]
            audio = _read_wav(os.path.join(AUDIO_DIR, clip))
            t = time.monotonic()
            try:
                text = engine.transcribe(audio, item["lang"])
            except Exception as e:  # noqa: BLE001
                text = f"<error {type(e).__name__}>"
            ms = (time.monotonic() - t) * 1000
            row = {"id": item_id, "voice": voice, "cond": cond, "lang": item["lang"], "ref": item["text"],
                   "hyp": text, "cer": cer(item["text"], text), "ms": ms, "codemix": item.get("codemix", False)}
            if "expect" in item:
                c = loop.run_until_complete(_classify(text, item["lang"]))
                row["intent_ok"] = c.branch == item["expect"] and (
                    "has_reason" not in item or (c.volunteered_reason is not None) == item["has_reason"])
                row["got"] = c.branch
            elif item.get("reason"):
                row["intent_ok"] = bool(text.strip()) and row["cer"] < 0.35  # reason captured recognisably
            rows.append(row)
        engine.close()
        gc.collect()

        def agg(filter_fn, key):
            vals = [r[key] for r in rows if filter_fn(r)]
            return statistics.mean(vals) if vals else None

        lat = sorted(r["ms"] for r in rows)
        summary = {
            "model": name, "compute_type": compute_type, "load_s": round(load_s, 1),
            "vram_mb": (vram - baseline_vram) if vram and baseline_vram else None,
            "cer_hi_clean": agg(lambda r: r["lang"] == "hi" and r["cond"] == "clean" and not r["codemix"], "cer"),
            "cer_hi_noisy": agg(lambda r: r["lang"] == "hi" and r["cond"] == "noisy" and not r["codemix"], "cer"),
            "cer_gu_clean": agg(lambda r: r["lang"] == "gu" and r["cond"] == "clean", "cer"),
            "cer_gu_noisy": agg(lambda r: r["lang"] == "gu" and r["cond"] == "noisy", "cer"),
            "intent_hi": agg(lambda r: r["lang"] == "hi" and "intent_ok" in r, "intent_ok"),
            "intent_gu": agg(lambda r: r["lang"] == "gu" and "intent_ok" in r, "intent_ok"),
            "lat_p50_ms": lat[len(lat) // 2], "lat_p90_ms": lat[int(len(lat) * 0.9)],
            "n": len(rows),
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        results.append({"summary": summary, "rows": rows})

    out = args.out or os.path.join(ROOT, "data", "stt_benchmark_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nfull results: {out}\n")
    print(markdown_table([r["summary"] for r in results]))


def markdown_table(summaries: list[dict]) -> str:
    def pct(v):
        return "-" if v is None else f"{v * 100:.1f}%"

    lines = ["| Model | CER hi clean | CER hi noisy | CER gu clean | CER gu noisy | Intent hi | Intent gu | Latency p50 / p90 | VRAM |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {pct(s['cer_hi_clean'])} | {pct(s['cer_hi_noisy'])} | {pct(s['cer_gu_clean'])} | "
            f"{pct(s['cer_gu_noisy'])} | {pct(s['intent_hi'])} | {pct(s['intent_gu'])} | "
            f"{s['lat_p50_ms']:.0f} / {s['lat_p90_ms']:.0f} ms | {s['vram_mb'] or '-'} MB |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--testset", default=TESTSET)
    r = sub.add_parser("run")
    r.add_argument("--testset", default=TESTSET)
    r.add_argument("--models", default=None, help=f"comma list from: {', '.join(REGISTRY)}")
    r.add_argument("--beam-size", type=int, default=1)
    r.add_argument("--prompt", action="store_true", help="bias Whisper with domain words")
    r.add_argument("--out", default=None)
    args = p.parse_args()
    if args.cmd == "build":
        asyncio.run(build(args))
    else:
        run(args)


if __name__ == "__main__":
    main()
