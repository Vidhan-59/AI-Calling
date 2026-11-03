# AI Calling — ShiftEaze (OmniVoice edition)

An AI phone agent that calls port-terminal operators in **Hindi, Gujarati and English** to confirm shifts, remind them to punch in/out, check on absences and request overtime. The caller's answers are understood, branched on, and logged (outcome, reason for absence, sentiment, transcript).

This edition runs the **voice stack on your own GPU** to cut cost:

| Part | What it uses | Cost |
|---|---|---|
| **Text-to-speech** | [OmniVoice](https://github.com/k2-fsa/OmniVoice) (open source, voice cloning) in its own service | no per-character fee |
| **Speech-to-text** | **Sarvam** cloud (`USE_SARVAM_STT=true`) **or** local Whisper (`false`) — one flag | ≈ ₹0.50 / minute, or ₹0 |
| **LLM ("brain")** | Qwen2.5-7B via [Ollama](https://ollama.com), local | no API fee |
| **Telephony** | Twilio Media Streams | Twilio per-minute |

A built-in **Voice Studio** web page lets you record/upload a short sample to **clone a voice** (or design one), preview it, and assign it per scenario and language.

---

## How it works

```
 Phone ─ Twilio ──(wss: Media Streams)──► Main app (FastAPI, port 8000)
                                           │  VAD ─► STT (Sarvam | local Whisper)
                                           │  scenario engine + Qwen (Ollama, :11434)
                                           │  audio cache ──► plays pre-rendered lines
                                           ▼
                                   TTS service (port 8100, own venv, GPU)
                                   OmniVoice: clone/design voices, synthesize 8 kHz audio

 Browser ─► http://localhost:8000/voice-studio   (create, preview, assign voices)
```

* Scripted lines are synthesized once and cached on disk, and the call-specific ones are prepared while the phone rings, so replies play instantly.
* The TTS service is a separate process because OmniVoice needs PyTorch + CUDA in its own Python environment.

---

## Requirements

| | |
|---|---|
| OS | **Windows 10/11** (the helper scripts are PowerShell). Linux works too — see [Linux notes](#linux-notes). |
| Python | 3.11 – 3.14 (tested on 3.14). Two virtualenvs are used. |
| GPU | **NVIDIA, 8 GB VRAM minimum** (tested on an RTX 5060 laptop). 16 GB+ recommended — see [GPU memory](#gpu-memory-the-8-gb-reality). |
| Disk | ~20 GB free (PyTorch+CUDA ≈ 6 GB, OmniVoice model 3.3 GB, Qwen 4.7 GB, Whisper models). **Not on a nearly-full drive.** |
| Accounts | **Twilio** (account SID, auth token, a phone number) · **Sarvam** API key (if `USE_SARVAM_STT=true`) |
| Tools | Git, [Ollama](https://ollama.com), and a public tunnel for Twilio (`ssh` for localhost.run, or ngrok) |

---

## Setup (Windows, step by step)

### 1. Clone

```powershell
git clone https://github.com/BhavsarSamay/AI-Calling-omni.git
cd AI-Calling-omni
```

### 2. Main app: virtualenv, packages, Ollama, `.env`

The one-time helper does all of this (venv, `pip install`, Ollama via winget, pulls Qwen, GPU check, creates `.env`):

```powershell
.\setup-new-machine.ps1
```

Manual equivalent:

```powershell
python -m venv venv
.\venv\Scripts\python -m pip install -r requirements.txt -r requirements-local.txt
winget install Ollama.Ollama
ollama pull qwen2.5:7b-instruct
Copy-Item .env.example .env
```

### 3. Fill in `.env`

Open `.env` and set at least:

```ini
# Twilio (https://console.twilio.com)
TELEPHONY_PROVIDER=twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxx
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx

# Speech-to-text: true = Sarvam cloud, false = local Whisper
USE_SARVAM_STT=true
SARVAM_API_KEY=xxxxxxxx

# Local LLM
SCENARIO_LLM_BACKEND=local
LLM_BACKEND=local
LOCAL_LLM_MODEL=qwen2.5:7b-instruct

# Protects /voices and the analytics/admin endpoints (use a long random string)
ADMIN_API_KEY=change-me-to-a-long-random-string
```

`NGROK_URL` is filled in automatically by `run-all.ps1` (step 6). Never commit `.env` — it is git-ignored.

### 4. TTS service: its own virtualenv with PyTorch + OmniVoice

The venv lives **outside the repo** (it is large): `~\.cache\ai-calling\tts-venv`.

```powershell
$v = "$env:USERPROFILE\.cache\ai-calling\tts-venv"
python -m venv $v

# NVIDIA GPU. RTX 50xx (Blackwell) needs the CUDA 12.8 wheels:
& $v\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

& $v\Scripts\python -m pip install -r tts_service\requirements.txt
& $v\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

**OmniVoice model (3.3 GB).** It downloads automatically on first start. On a slow or flaky connection fetch it once with a resumable download, then the service picks it up from `~\.cache\ai-calling\omnivoice-model`:

```powershell
$d = "$env:USERPROFILE\.cache\ai-calling\omnivoice-model"
New-Item -ItemType Directory -Force "$d\audio_tokenizer" | Out-Null
$base = "https://huggingface.co/k2-fsa/OmniVoice/resolve/main"
"config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja","model.safetensors",
"audio_tokenizer/config.json","audio_tokenizer/preprocessor_config.json","audio_tokenizer/model.safetensors" |
  ForEach-Object { curl.exe -L -C - --retry 20 --retry-all-errors -o "$d\$($_ -replace '/','\')" "$base/$_" }
```

### 5. Optional: local Whisper models (only if `USE_SARVAM_STT=false`)

The converter needs `torch` + `transformers`, which the live server does not, so it runs from a separate short-path environment (torch's DLLs fail on long paths):

```powershell
python -m venv "$env:USERPROFILE\.cache\ai-calling\cvenv"
& "$env:USERPROFILE\.cache\ai-calling\cvenv\Scripts\pip" install torch transformers ctranslate2
& "$env:USERPROFILE\.cache\ai-calling\cvenv\Scripts\python" tools\convert_stt_models.py
```

This writes the Hindi/Gujarati Whisper fine-tunes to `~\.cache\ai-calling\stt_models`. Then re-run the STT benchmark for your hardware if you like: `tools\stt_benchmark.py`.

### 6. Start everything

```powershell
.\run-all.ps1
```

This frees port 8000, starts the **TTS service**, starts the **app**, opens a public **tunnel**, writes its URL into `.env` as `NGROK_URL`, and verifies Twilio can reach it. Leave the window open; `Ctrl+C` stops everything. Options: `-Tunnel ngrok`, `-NoTts`, `-Port 8001`, `-SkipChecks`.

Start the pieces by hand instead:

```powershell
.\tts_service\run-tts.ps1                                  # http://127.0.0.1:8100/health  → "status":"ok"
.\venv\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 7. Create a voice (once)

Open **http://localhost:8000/voice-studio**, enter your `ADMIN_API_KEY`, and:

1. **Clone from audio** — record or upload 3–10 s of clear speech (type what is said for best quality) and save, **or** **Design from attributes**.
2. Click **Make default**. The first voice you create is the default automatically.
3. Optionally assign different voices per scenario / language in the table below.

> The shift scripts use **feminine** first-person Hindi/Gujarati verbs ("bol rahi hoon"). Use a **female** voice for them.
> Only clone voices you have permission to use.

### 8. Try it without a phone

```powershell
.\venv\Scripts\python tools\simulate_scenario_call.py          # full simulated calls through the real pipeline
```

Or import `postman/ShiftEaze_AI_Calling.postman_collection.json` into Postman and set `base_url` (`http://127.0.0.1:8000`) and `to_phone_number`. Run **2. Preview** and **3. Test without a phone** first; **4. Place a real call** needs the tunnel. See [POSTMAN_GUIDE.md](POSTMAN_GUIDE.md).

---

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `USE_SARVAM_STT` | `true` | `true` Sarvam cloud STT · `false` local Whisper |
| `SARVAM_API_KEY` | — | needed for Sarvam STT (and as a safety-net fallback) |
| `LOCAL_STT_FALLBACK_SARVAM` | `true` | if a local decode fails, retry that utterance on Sarvam |
| `LOCAL_STT_GPU_MODELS` | `1` | Whisper models kept on the GPU (the call's language); others wait in RAM |
| `LOCAL_STT_IDLE_OFFLOAD_SECS` | `30` | park idle Whisper in RAM to free VRAM (`0` = never) |
| `TTS_SERVICE_URL` / `TTS_SERVICE_TOKEN` | `http://127.0.0.1:8100` / empty | where the TTS service is; shared secret if it is reachable by others |
| `TTS_IDLE_OFFLOAD_SECS` | `auto` | OmniVoice parks itself in RAM when idle (auto = 8 s on ≤12 GB GPUs) |
| `TTS_LIVE_NUM_STEP` / `TTS_CACHED_NUM_STEP` | `16` / `32` | OmniVoice quality-vs-speed (live replies vs pre-rendered lines) |
| `SCENARIO_LLM_BACKEND` | `claude` | `local` = Qwen via Ollama (no Anthropic key needed) |
| `SCENARIO_VOICE_TRANSPORT` | `media` | `media` = this pipeline · `relay` = old Twilio ConversationRelay rollback |
| `ADMIN_API_KEY` | — | required outside development; protects admin/voice endpoints |

Full list with comments: [.env.example](.env.example). The TTS service's own settings: [tts_service/README.md](tts_service/README.md).

---

## GPU memory — the 8 GB reality

Measured on an 8 GB card: Qwen ≈ 4.6 GB, OmniVoice ≈ 2.1 GB, each Whisper-medium ≈ 2 GB. They cannot all be resident at once; the overflow spills to system RAM and decodes stall ("local STT failed: TimeoutError"). This repo handles it by keeping only the active language's Whisper on the GPU and parking idle OmniVoice/Whisper in RAM (≈ 0.3–0.7 s to wake). With **≥ 16 GB** set `LOCAL_STT_GPU_MODELS=2` and `TTS_IDLE_OFFLOAD_SECS=0`. A smaller LLM also frees memory. If you want zero risk on a small GPU, use `USE_SARVAM_STT=true`.

## Cost (rough)

* **TTS (OmniVoice) and LLM (Qwen): ₹0 per call** — electricity/GPU only.
* **Sarvam STT:** ₹30 per audio-hour ≈ **₹0.50 per minute** of call (check [Sarvam pricing](https://docs.sarvam.ai/api/pricing)).
* **Twilio:** its normal per-minute telephony charges.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Not placing the call: your public URL … returned HTTP 502` | The tunnel died (free localhost.run URLs also rotate). Re-run `.\run-all.ps1`, or use ngrok / a stable tunnel. Twilio says "an application error has occurred" when it can't reach your URL. |
| Voice Studio: "voice service offline" / calls have no voice | TTS service not running — `.\tts_service\run-tts.ps1`, check `http://127.0.0.1:8100/health`. |
| Bot doesn't hear you with local Whisper | GPU out of memory — see [GPU memory](#gpu-memory-the-8-gb-reality), or set `USE_SARVAM_STT=true`. |
| After saying "no" the bot doesn't ask why | Fixed in this version (the reason is always asked). Make sure you pulled the latest code and restarted. |
| `torch.cuda.is_available()` is False | Install the CUDA wheels (step 4) and update the NVIDIA driver. |
| Install fails with *No space left on device* | Free disk space; keep the TTS venv and models on a drive with ≥ 20 GB free. |
| `Could not reconcile active_call_count (Redis unreachable?)` | Harmless unless you use the dialer (`DIALER_ENABLED`), which needs Redis. |

---

## Tests

```powershell
.\venv\Scripts\python -m pytest tests                                              # main app (~240 tests, no GPU needed)
& "$env:USERPROFILE\.cache\ai-calling\tts-venv\Scripts\python" -m pytest tts_service\tests   # TTS service (mock engine)
```

## Project layout

```
main.py, config.py            FastAPI app and settings
routers/                      HTTP/WebSocket endpoints (scenarios, voices, agents, analytics…)
services/scenario_engine.py   the conversation brain: classify answers, branch, ask "why?"
services/voice_pipeline/      live call pipeline (VAD → STT → engine → cached TTS), local Whisper
services/tts/                 OmniVoice client, voice assignments, Pipecat TTS service
tts_service/                  the OmniVoice service (own venv)   → tts_service/README.md
static/voice_studio.html      Voice Studio UI
scenarios/*.json              the five scripted scenarios × Hindi / Gujarati / English
configs/, prompts/            config-driven agents and prompt templates
tools/                        call simulator, STT benchmark, model converter
postman/                      Postman collection
```

More background: [START_HERE.md](START_HERE.md) · [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md) · [LOCAL_AI_README.md](LOCAL_AI_README.md).

## Linux notes

The `.ps1` helpers are Windows-only, but nothing else is:

```bash
python -m venv venv && ./venv/bin/pip install -r requirements.txt -r requirements-local.txt
python -m venv ~/.cache/ai-calling/tts-venv
~/.cache/ai-calling/tts-venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
~/.cache/ai-calling/tts-venv/bin/pip install -r tts_service/requirements.txt
~/.cache/ai-calling/tts-venv/bin/python -m uvicorn tts_service.app:app --port 8100 &
./venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Set `TTS_MODEL`, `TTS_VOICES_DIR` as environment variables if you keep them elsewhere, and start your own tunnel (`ngrok http 8000`), then put its URL in `.env` as `NGROK_URL`.

## Responsible use

These calls are automated and use synthetic or cloned voices. Disclose that the caller is an AI where required, honour consent and calling-hour rules, and only clone voices you have the speaker's permission to use.

## License

See [LICENSE](LICENSE). OmniVoice is Apache-2.0.
