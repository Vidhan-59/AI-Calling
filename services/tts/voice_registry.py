"""Which OmniVoice voice speaks for which scenario/agent + language.

Voices live in the TTS service (tts_service/voices.py). This module keeps the
*assignments* - data/voice_assignments.json, edited from Voice Studio - and
resolves them to a concrete (voice_id, version, speed) for a call.

Lookup order for (owner, language), owner = scenario id or agent id:
    "<owner>:<language>"  ->  "<owner>:*"  ->  "*:<language>"  ->  "*:*"
    ->  the agent YAML's voice_id (if it names a real voice)
    ->  the TTS service's default voice.

`version` comes from the voice library and goes into the audio-cache key, so
re-recording a voice automatically invalidates its cached lines.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass

from config import DATA_DIR
from services.tts import client

logger = logging.getLogger("tts.voice_registry")

_PATH = os.path.join(DATA_DIR, "voice_assignments.json")
_LOCK = threading.Lock()
_SNAPSHOT_TTL = 15.0

_snapshot: dict | None = None
_snapshot_at = 0.0


@dataclass(frozen=True)
class ResolvedVoice:
    voice_id: str
    version: int
    speed: float = 1.0


def load_assignments() -> dict[str, dict]:
    try:
        with open(_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except ValueError:
        logger.error("%s is corrupt - ignoring it (default voice will be used)", _PATH)
        return {}


def set_assignment(owner: str, language: str, voice_id: str | None, speed: float | None = None) -> dict[str, dict]:
    """voice_id=None removes the assignment. owner/language may be "*"."""
    key = f"{owner}:{language}"
    with _LOCK:
        data = load_assignments()
        if voice_id is None:
            data.pop(key, None)
        else:
            data[key] = {"voice_id": voice_id, "speed": speed}
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, _PATH)
    invalidate()
    return data


def assigned_voice_id(owner: str, language: str) -> str | None:
    """Explicitly assigned voice (no network) - for display; None = default."""
    assignments = load_assignments()
    for key in (f"{owner}:{language}", f"{owner}:*", f"*:{language}", "*:*"):
        if key in assignments:
            return assignments[key]["voice_id"]
    return None


def invalidate() -> None:
    global _snapshot_at
    _snapshot_at = 0.0


async def _voices() -> dict:
    """Voice library snapshot, cached briefly. If the service is unreachable
    the last snapshot keeps serving cached lines instead of failing calls."""
    global _snapshot, _snapshot_at
    if _snapshot is not None and time.monotonic() - _snapshot_at < _SNAPSHOT_TTL:
        return _snapshot
    try:
        data = await client.list_voices()
        _snapshot = {"by_id": {v["id"]: v for v in data["voices"]}, "default": data.get("default_voice_id")}
        _snapshot_at = time.monotonic()
    except client.TTSUnavailable:
        if _snapshot is None:
            raise
        logger.warning("TTS service unreachable - using the last known voice list")
    return _snapshot


async def resolve(
    owner: str, language: str, *, fallback_voice_id: str | None = None, fallback_speed: float = 1.0
) -> ResolvedVoice:
    snap = await _voices()
    assignments = load_assignments()
    for key in (f"{owner}:{language}", f"{owner}:*", f"*:{language}", "*:*"):
        entry = assignments.get(key)
        if entry and entry["voice_id"] in snap["by_id"]:
            v = snap["by_id"][entry["voice_id"]]
            speed = entry.get("speed")
            return ResolvedVoice(v["id"], v["version"], float(speed) if speed else fallback_speed)
        if entry:
            logger.warning("assignment %s points at deleted voice %s - falling through", key, entry["voice_id"])
    if fallback_voice_id and fallback_voice_id in snap["by_id"]:
        return ResolvedVoice(fallback_voice_id, snap["by_id"][fallback_voice_id]["version"], fallback_speed)
    default = snap["default"]
    if default and default in snap["by_id"]:
        return ResolvedVoice(default, snap["by_id"][default]["version"], fallback_speed)
    raise client.TTSUnavailable("no voice available - create one in Voice Studio (/voice-studio)")
