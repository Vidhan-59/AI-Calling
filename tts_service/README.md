# OmniVoice TTS service

Text-to-speech for every call, powered by [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
(Apache-2.0, zero-shot, 600+ languages incl. Hindi and Gujarati). It replaces
Sarvam Bulbul: no per-character fee, you own the voices.

## Architecture

```
 Browser ──► /voice-studio  (main app, routers/voices.py, X-API-Key)
                 │  create / clone / design / assign voices
                 ▼
 Main app  ──HTTP──►  tts_service  (this folder, port 8100, own venv + GPU model)
   services/tts/client.py          POST /v1/synthesize → PCM16 8 kHz
   services/tts/voice_registry.py  GET/POST /v1/voices…  (voice library on disk)
   services/voice_pipeline/tts_cache.py   (memory + disk cache in front)
```

* **Why a separate process**: OmniVoice needs torch (+CUDA) and its own Python
  environment, and one GPU thread must serialise synthesis. Keeping it out of
  the call-handling event loop means a slow generation can never stall audio
  for live calls, and the service can later move to a bigger GPU box by
  changing `TTS_SERVICE_URL` only.
* **Cost model**: scripted scenario lines are synthesized once and cached on disk
  (key = voice + voice version + text + language + speed) and prefetched while
  the phone rings. The GPU is only hit for lines never produced before and for
  free-form agent replies (`services/tts/pipecat_service.py`, per sentence).
* **Voices**: stored in `data/voices/<id>/` (`meta.json`, `reference.wav`,
  `prompt.pt`). A *clone* voice is built from a 3–10 s sample; a *design* voice
  (gender/age/pitch/accent) is rendered once and then cloned, so it sounds
  identical on every call. Assignments (which voice speaks for which
  scenario/agent + language) live in `data/voice_assignments.json`.

## Setup (once)

Python 3.10+; the venv is separate from the main app's and lives outside the
repo (torch + CUDA is ~6 GB): `~\.cache\ai-calling\tts-venv` (override with
`TTS_VENV`). Tested: Python 3.14, torch 2.11+cu128, RTX 5060 (CUDA available).

```powershell
$v = "$env:USERPROFILE\.cache\ai-calling\tts-venv"
python -m venv $v
# NVIDIA GPU, incl. RTX 50xx (Blackwell needs the CUDA 12.8 wheels):
& $v\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
& $v\Scripts\python -m pip install -r tts_service\requirements.txt
```

The model (~3.3 GB) downloads from Hugging Face on first start. On a slow or
flaky link, download it once into a folder and point `TTS_MODEL` at that folder.

Then in the repo `.env`: `TTS_SERVICE_URL=http://127.0.0.1:8100` (default) and,
if anything other than this app can reach the port, a shared
`TTS_SERVICE_TOKEN=<random string>` (set it for both processes).

## Run

```powershell
.\tts_service\run-tts.ps1            # real model (first start downloads it from Hugging Face)
.\tts_service\run-tts.ps1 -Mock      # tone generator, no GPU: for wiring tests only
```

Open `http://localhost:8000/voice-studio`, create or clone a voice, make it the
default, and assign it per scenario/language. The first voice created becomes the default.

Health: `GET http://127.0.0.1:8100/health` (`status: loading` until the model is on the GPU).

## Settings (environment)

| Variable | Default | Meaning |
|---|---|---|
| `TTS_ENGINE` | `omnivoice` | `mock` = no model (tests) |
| `TTS_MODEL` | `k2-fsa/OmniVoice` | HF repo or local path |
| `TTS_DEVICE` / `TTS_DTYPE` | `cuda:0` / `float16` | |
| `TTS_NUM_STEP` | `16` | fallback diffusion steps (the app sends its own: live 16, cached lines 32) |
| `TTS_IDLE_OFFLOAD_SECS` | `auto` | park the model in CPU RAM after N idle seconds (wakes in ~0.7s); `auto` = 8 on GPUs <= 12 GB, else off; `0` = never |
| `TTS_ASR_DEVICE` | `cpu` | Whisper used only when cloning without a transcript |
| `TTS_VOICES_DIR` | `data/voices` | voice library |
| `TTS_SERVICE_TOKEN` | empty | `X-TTS-Token` required on `/v1/*` when set |
| `TTS_MAX_QUEUE` | `16` | queued synth requests before HTTP 503 |

## Tests

```powershell
& "$env:USERPROFILE\.cache\ai-calling\tts-venv\Scripts\python" -m pytest tts_service\tests   # service, mock engine
venv\Scripts\python -m pytest tests                                # main app (TTS client faked)
```

## Notes and limits

* OmniVoice outputs 24 kHz; the service band-limits to 8 kHz PCM16 for telephony.
* No audio streaming: latency is per utterance. Tune `TTS_NUM_STEP` on the
  deployment GPU, and consider FlashInfer (NVIDIA, see the OmniVoice README).
* Voice *design* was trained on English/Chinese only — for Hindi/Gujarati prefer cloning a real recording.
* Only clone voices you have permission to use (the OmniVoice licence forbids impersonation/fraud).
