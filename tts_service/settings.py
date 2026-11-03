"""Environment configuration for the OmniVoice TTS service. Read once at
import; every value has a production-safe default and a TTS_* env override.
The service is deployed separately from the main app (its own venv: it needs
torch + omnivoice), so it deliberately does not import the app's config.py."""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


ENGINE = _env("TTS_ENGINE", "omnivoice")  # "omnivoice" | "mock" (no GPU; tests/CI)
MODEL = _env("TTS_MODEL", "k2-fsa/OmniVoice")
DEVICE = _env("TTS_DEVICE", "cuda:0")
DTYPE = _env("TTS_DTYPE", "float16")
# Fallback diffusion steps when a request doesn't specify (the main app always
# does). 32 is the model default, 16 the README's "faster" setting.
NUM_STEP = int(_env("TTS_NUM_STEP", "16"))
# Whisper used by OmniVoice only when a cloned voice is created WITHOUT a
# transcript. CPU by default so it never competes with the LLM for VRAM.
ASR_DEVICE = _env("TTS_ASR_DEVICE", "cpu")
# Park the model in CPU RAM after this many idle seconds (and move it back to
# the GPU on the next request, ~1s), so a small GPU shared with the LLM and
# Whisper isn't oversubscribed. "auto" = 8s on GPUs <= 12 GB, off otherwise;
# 0 = never park; or set an explicit number of seconds.
IDLE_OFFLOAD = _env("TTS_IDLE_OFFLOAD_SECS", "auto")
VOICES_DIR = Path(_env("TTS_VOICES_DIR", str(_ROOT / "data" / "voices")))
TOKEN = _env("TTS_SERVICE_TOKEN", "")  # empty = no auth (local dev only)
MAX_QUEUE = int(_env("TTS_MAX_QUEUE", "16"))  # waiting synth requests before 503
MAX_TEXT_CHARS = int(_env("TTS_MAX_TEXT_CHARS", "600"))
TELEPHONY_RATE = 8000
MODEL_RATE = 24000
MIN_REF_SECS = 2.0
MAX_REF_SECS = 20.0
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
