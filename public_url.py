"""Resolve the public HTTPS base URL Twilio uses to reach this server.

Why this is not just `from config import NGROK_URL`: that value is captured
once at import time, so every tunnel restart (and free tunnel URLs change on
*every* restart) forced a full server restart too. Forgetting that restart is
the single most common cause of a failed call - Twilio dials, fetches the
stale URL, gets a 503, and the operator hears "an application error occurred".

Reading `.env` fresh here costs one small file read per placed call, which is
nothing next to the Twilio round-trip, and means a tunnel can be replaced
while the server keeps running.
"""

import os
import threading

from config import BASE_DIR

_ENV_PATH = os.path.join(BASE_DIR, ".env")
_lock = threading.Lock()

_PLACEHOLDERS = ("your-ngrok-url", "your_ngrok_url", "xxxx")


def _read_env_value(key: str) -> str | None:
    if not os.path.isfile(_ENV_PATH):
        return None
    try:
        with open(_ENV_PATH, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == key:
                    return value.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def get_public_url() -> str | None:
    """Current public base URL, or None if not usefully configured.

    Resolution order: live `.env` value, then the process environment (so a
    container/systemd deployment that sets NGROK_URL without an .env file
    still works).
    """
    with _lock:
        value = _read_env_value("NGROK_URL") or os.environ.get("NGROK_URL")

    if not value:
        return None
    if any(stub in value.lower() for stub in _PLACEHOLDERS):
        return None
    return value.rstrip("/")
