# local_ai/ — fully local, offline-capable voice AI pipeline

A zero-cloud-dependency alternative pipeline: **faster-whisper** (STT) → **Ollama**
(response generation) → **Piper** (TTS). Built as an *additive, swappable*
alternative alongside the existing Claude + Twilio ConversationRelay path — it
does not replace or touch that code. See "Where this fits" below.

**Read "Honest results" before deciding how to use this.** UPDATE (2026-09-30):
on a GPU (RTX 5060 Laptop, 8GB VRAM, 100%-offloaded via Ollama), this backend
is fast enough for live calls (~1s/turn measured through the real
conversation-agent code path) and is now the default live-call backend - see
`LLM_BACKEND=local` / `SCENARIO_LLM_BACKEND=local` in `.env.example`. The
CPU-only numbers below are kept for anyone running this without a GPU, where
the original conclusion (offline/dev/QA only) still holds.

---

## Where this fits

This repo has two other AI backends already:

1. **Claude + Twilio ConversationRelay** (the default, live-call path) —
   `services/claude_client.py`, `services/scenario_engine.py`. Unchanged by
   anything in this document.
2. **A separate, much larger ShiftEaze platform** built in a parallel effort
   in this same repo, using Claude + cloud STT/TTS providers (Sarvam,
   Deepgram, ElevenLabs, Azure). Also untouched.

`local_ai/` is a **third option**, wired in via exactly one config flag
(`SCENARIO_LLM_BACKEND`) touching exactly one existing file
(`services/scenario_engine.py`, one small additive hook — see the diff if
you're curious, it's about 10 lines). Nothing here deletes, rewrites, or
depends on either of the above.

---

## Hardware this was built and measured on

| | |
|---|---|
| CPU | AMD Ryzen 7 5800HS, 8 cores / 16 threads |
| RAM | ~15 GB |
| GPU | None (integrated AMD graphics only - no CUDA) |
| OS | Windows 11 |

**All numbers in this document are measured on exactly this machine, not
estimated.** If you're running on different hardware - especially a real GPU
- re-run the comparison in "Honest results" below; the accuracy numbers will
likely hold, the speed numbers will not.

---

## Setup

### 1. Install Ollama

Not a pip package - a separate application. https://ollama.com, or:

```powershell
winget install --id Ollama.Ollama --source winget
```

Start the server (leave running in its own terminal, or let it run as the
background service the installer sets up):

```powershell
ollama serve
```

### 2. Pull a model

```powershell
ollama pull qwen2.5:7b-instruct   # recommended default - see comparison below
# or, for faster iteration during dev at the cost of accuracy:
ollama pull phi4-mini
```

| Model | License | Size | Notes |
|---|---|---|---|
| `qwen2.5:7b-instruct` | Apache-2.0 | 4.7 GB | **Default.** Better Hindi/code-mixed understanding |
| `phi4-mini` | MIT | 2.5 GB | Faster, less accurate on the hard cases (see below) |

### 3. Install the Python dependencies

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

No PyTorch anywhere in this stack - `faster-whisper` uses CTranslate2,
`piper-tts` uses onnxruntime. This is why it's viable on a CPU at all.

### 4. Verify it's ready

```powershell
.\venv\Scripts\python.exe -m local_ai.cli --check
```

```json
{
  "ollama_running": true,
  "model_pulled": true,
  "available_models": ["qwen2.5:7b-instruct", "phi4-mini:latest"]
}

Ollama backend ready.
```

### 5. First-run downloads (need network once, then never again)

- **faster-whisper** downloads its model from Hugging Face on first use, then
  caches it. After that, `local_files_only=True` is used automatically (see
  "The bug this caught" below) - genuinely zero network on every run after.
- **Piper voices** likewise auto-download to `local_ai/voices/` on first use.

To pre-warm both before going fully offline:

```powershell
.\venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
.\venv\Scripts\python.exe -c "from pathlib import Path; from piper.download_voices import download_voice; from local_ai import config; download_voice(config.PIPER_VOICE_EN, Path(config.PIPER_VOICES_DIR))"
```

---

## Running it

### Plain chat (no scenario JSON needed)

```powershell
.\venv\Scripts\python.exe -m local_ai.cli --audio caller.wav --out reply.wav
```

### Through the real ShiftEaze scenario engine

```powershell
.\venv\Scripts\python.exe -m local_ai.cli --audio caller.wav --out reply.wav `
  --scenario shift_confirm --language hi `
  --facts '{"employee_name":"Rajesh","date":"2 अगस्त","time":"सुबह 8","shift_name":"Day","terminal":"CT3"}'
```

Output:

```json
{
  "transcript": "नहीं, तबीयत ठीक नहीं है",
  "detected_language": "hi",
  "language_probability": 0.94,
  "reply_text": "कृपया अपनी अनुपस्थिति का कारण बताएँ।",
  "output_audio_path": "reply.wav",
  "timings_ms": { "stt_ms": 1800.2, "response_ms": 41200.5, "normalize_ms": 0.1, "tts_ms": 900.3, "total_ms": 43901.1 }
}
```

`--scenario` mode forces `SCENARIO_LLM_BACKEND=local` for that process
regardless of what's in `.env` - the CLI is explicitly for testing the local
backend, so it should never silently fall back to Claude.

---

## Honest results (measured, not estimated)

### Accuracy: same 3 test cases used to verify the Claude path

These are the exact cases that proved the Claude backend works correctly on
a live phone call earlier in this project - including the hardest one
(negation trap: opens with "haan"/yes but means no).

| Operator said | Correct branch | Claude (live call) | phi4-mini | qwen2.5:7b-instruct |
|---|---|---|---|---|
| "हाँ भाई आ जाऊंगा" | `yes` | ✅ | ✅ | ✅ |
| "haan nahi aa paunga yaar, tabiyat kharab hai" | `no` | ✅ | ❌ got `dispute` | ✅ |
| "mujhe Suresh sir se baat karni hai" | `supervisor` | ✅ | ❌ got `dispute` | ✅ |

phi4-mini defaulted to `dispute` on both hard cases - on a real call that
misfiles a legitimate sick-leave decline and fails to route an escalation
request. qwen2.5:7b-instruct matched Claude on all three.

### Latency: this is the number that actually matters

| Model | Per-turn latency (CPU-only, Ryzen 7 5800HS) | Per-turn latency (GPU, RTX 5060 8GB) |
|---|---|---|
| Claude (cloud) | ~1-3 seconds | ~1-3 seconds |
| phi4-mini (local) | 23-33 seconds | not re-measured on GPU |
| qwen2.5:7b-instruct (local) | 33-60 seconds | **0.9-1.7 seconds** (measured through the real conversation-agent code path, 100% GPU-resident) |
| qwen3:8b (local) | not measured | 3.2-3.8s steady-state (2-4x slower than qwen2.5, likely its default "thinking mode"), **and returned empty replies on non-tool turns** - a real compatibility defect with this app's Ollama tool-calling contract, not adopted |

**On CPU, this is a hardware ceiling, not a software one** - a bigger/better
model is slower, not faster, and 20-60 seconds of dead air after every
sentence is not usable regardless of accuracy. **On a GPU, the ceiling is
gone**: qwen2.5:7b-instruct is fast enough for live calls. Don't assume a
newer/bigger model is automatically better, though - qwen3:8b was slower
*and* buggier here; re-verify any model swap the same way (through
`run_turn()`, not a bare chat ping) before trusting it on a live call.

### The bug this project's own smoke test caught

`tests/local_ai/test_e2e_offline_smoke.py` blocks all non-loopback network
access at the socket layer and then runs the real pipeline. The first time it
ran, it failed - `faster-whisper` (via `huggingface_hub`) was silently trying
to reach `huggingface.co` on every load to check for model updates, even
though the model was already cached locally. Fixed by passing
`local_files_only=True` once cached (see `local_ai/stt_engine.py`), with a
one-time fallback to allow the very first download. This is exactly the kind
of silent cloud dependency that "no imports from `anthropic`/`openai`" checks
would miss - only an actual network-level block catches it. Ollama itself
needed no such fix; it never phoned home.

---

## Recommendation

| Use case | Recommended |
|---|---|
| Live phone calls, on a GPU (this project's current setup) | **Local** (`SCENARIO_LLM_BACKEND=local`, `LLM_BACKEND=local`, the current default) with `qwen2.5:7b-instruct` - ~1s/turn, verified via `run_turn()` |
| Live phone calls, CPU-only, no GPU | **Claude** (`SCENARIO_LLM_BACKEND=claude`) - do not use local without a GPU |
| Offline/batch reprocessing of recorded calls | Local with `qwen2.5:7b-instruct` |
| Dev/QA testing scenario scripts, zero API cost | Local, either model - use `phi4-mini` for faster iteration |
| Proving the "no cloud dependency" property | Local - see `tests/local_ai/test_e2e_offline_smoke.py` |

---

## Known limitation: Gujarati has no local TTS voice

Verified against Piper's actual voice catalog (171 voices,
`rhasspy/piper-voices` on Hugging Face), not assumed: there are 3 Hindi
voices (`hi_IN-pratham-medium`, `hi_IN-priyamvada-medium`, `hi_IN-rohan-medium`)
and **zero** Gujarati voices. `PIPER_VOICE_GU` is intentionally left blank in
`.env` / `.env.example` - `TTSEngine.synthesize(..., language="gu")` raises
`VoiceNotAvailableError` rather than silently speaking Gujarati text with the
wrong language's phonemes, which would be worse than failing loudly.

**For Gujarati, keep using cloud TTS** (`TTS_PROVIDER=sarvam` etc. in the
main `.env`) regardless of what `SCENARIO_LLM_BACKEND` is set to - local STT
and the local LLM handle Gujarati *understanding* fine; only *voice output*
has no local option today.

Hindi TTS quality via Piper is also noticeably below Google/Sarvam's cloud
Hindi voices - usable for testing, not recommended as-is for a production
voice a real operator hears.

---

## Architecture

```
local_ai/
  config.py       All model names/settings, one place, env-overridable
  stt_engine.py   STTEngine interface + FasterWhisperSTT
  llm_engine.py   Ollama client - mirrors services/claude_client.py's exact
                  function signatures (classify_structured, generate_reply,
                  generate_reply_stream) so scenario_engine.py can swap
                  between them with one if-branch
  tts_engine.py   TTSEngine interface + PiperTTS
  text_norm.py    Markdown stripping, time expansion, sentence splitting -
                  rule-based, not spaCy (see file docstring for why)
  pipeline.py     LocalVoicePipeline: audio-in -> STT -> respond() -> TTS -> audio-out
  cli.py          File-in/file-out test runner
```

Every stage is swappable independently - e.g. to add Vosk as a second STT
option, implement `STTEngine` in a new file; nothing else changes.

### Config

Everything is env-var driven, no hardcoded model names in application code -
see `local_ai/config.py` for the full list and `.env.example` for the
documented block. The one existing-file touch point:

```python
# services/scenario_engine.py, classify_utterance()
backend = os.getenv("SCENARIO_LLM_BACKEND", "claude")  # unchanged unless set to "local"
```

---

## Testing

```powershell
# Fast tests - mocked, no real models needed, run in CI
.\venv\Scripts\python.exe -m pytest tests/local_ai/ -m "not slow"

# Real end-to-end test - needs Ollama running + model pulled + Piper voice
# downloaded (see Setup above). Genuinely blocks the network at the socket
# layer and proves the pipeline runs without it.
.\venv\Scripts\python.exe -m pytest tests/local_ai/ -m slow -v
```

22 tests total: unit tests per stage with the model layer mocked, plus the
real offline smoke test described above. The slow-marked tests skip
themselves (not fail) if Ollama/the model/the voice aren't set up yet.

---

## Definition of done, against what was actually asked

- [x] Zero cloud LLM/speech calls in this pipeline - proven by an actual
      network-blocking test, not just an absence of `anthropic`/`openai` imports
- [x] Runs end-to-end with the network disabled (see the smoke test)
- [x] All models open-weight, permissively licensed (MIT/Apache-2.0)
- [x] Every stage swappable via config, no hardcoded model names
- [ ] "Runs offline from a clean machine" - true for inference; the very
      first run of each model needs one-time network access to download it
      (documented above, not hidden)
- [ ] Live real-time voice assistant - **explicitly not achieved** on this
      hardware; see "Honest results". This pipeline is offline/batch/dev
      capable, not a live-call replacement, without a GPU.
