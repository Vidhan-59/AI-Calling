"""On-disk voice library. One directory per voice:

    <VOICES_DIR>/<voice_id>/meta.json     name, language, kind, instruct, ref_text, timestamps
                            reference.wav  the clone reference (design voices: the rendered sample)
                            prompt.pt      OmniVoice VoiceClonePrompt (the reusable voice)
    <VOICES_DIR>/_state.json              {"default_voice_id": ...}

`version` bumps whenever a voice's audio changes; the main app puts it in its
audio-cache key so re-recording a voice invalidates stale cached lines.
Writes are atomic (temp file + replace) so a crash mid-save can't leave a
half-written voice that then fails every call."""

import json
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tts_service import settings

_LOCK = threading.RLock()
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


class VoiceNotFound(KeyError):
    pass


@dataclass
class Voice:
    id: str
    name: str
    kind: str  # "clone" | "design"
    language: str
    description: str = ""
    instruct: str = ""
    ref_text: str = ""
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self, default_id: str | None = None) -> dict:
        d = asdict(self)
        d["is_default"] = self.id == default_id
        return d


def _dir(voice_id: str) -> Path:
    # The regex also blocks path traversal ("../x"): ids reach disk paths.
    if not _ID_RE.match(voice_id):
        raise VoiceNotFound(voice_id)
    return settings.VOICES_DIR / voice_id


def _atomic_write(path: Path, data: bytes | str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def new_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "voice"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def reference_path(voice_id: str) -> Path:
    return _dir(voice_id) / "reference.wav"


def prompt_path(voice_id: str) -> Path:
    return _dir(voice_id) / "prompt.pt"


def save(voice: Voice, reference_wav: bytes | None = None) -> None:
    with _LOCK:
        d = _dir(voice.id)
        d.mkdir(parents=True, exist_ok=True)
        if reference_wav is not None:
            _atomic_write(d / "reference.wav", reference_wav)
        voice.updated_at = time.time()
        _atomic_write(d / "meta.json", json.dumps(asdict(voice), ensure_ascii=False, indent=2))


def get(voice_id: str) -> Voice:
    try:
        return Voice(**json.loads((_dir(voice_id) / "meta.json").read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise VoiceNotFound(voice_id) from None


def list_voices() -> list[Voice]:
    if not settings.VOICES_DIR.exists():
        return []
    out = []
    for d in settings.VOICES_DIR.iterdir():
        if d.is_dir() and (d / "meta.json").exists():
            try:
                out.append(get(d.name))
            except Exception:  # noqa: BLE001 - one corrupt voice must not hide the rest
                continue
    return sorted(out, key=lambda v: v.created_at)


def _write_state(state: dict) -> None:
    settings.VOICES_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write(settings.VOICES_DIR / "_state.json", json.dumps(state))


def default_id() -> str | None:
    try:
        return json.loads((settings.VOICES_DIR / "_state.json").read_text(encoding="utf-8")).get("default_voice_id")
    except (FileNotFoundError, ValueError):
        return None


def set_default(voice_id: str) -> None:
    with _LOCK:
        get(voice_id)  # raises VoiceNotFound
        _write_state({"default_voice_id": voice_id})


def delete(voice_id: str) -> None:
    with _LOCK:
        d = _dir(voice_id)
        if not d.exists():
            raise VoiceNotFound(voice_id)
        shutil.rmtree(d)
        if default_id() == voice_id:
            _write_state({"default_voice_id": None})
