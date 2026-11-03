"""OmniVoice TTS with a memory + disk cache, returning raw PCM16 mono audio
at the telephony rate (8 kHz) so it can be pushed straight into the pipeline.

Why a cache in front of the TTS service: every line a scenario call speaks
is scripted. Only the handful with call facts in them (the opening, "repeat
the time") differ per call, and those are synthesized while the phone is
still ringing (see prefetch_call_lines). So by the time the caller says
"haan", the reply audio already exists - TTS contributes ~0ms to turn
latency, and each distinct line is generated on the GPU once, not once per
call (the main cost saving over a per-character cloud TTS).

The cache key includes owner/voice/voice-version/speed/language/text, so
editing a script line, switching the assigned voice, or re-recording a voice
in Voice Studio automatically produces new audio - stale audio can't be
served for changed text or a changed voice.
"""

import asyncio
import hashlib
from collections import OrderedDict
import logging
import os

from config import DATA_DIR, settings
from services.tts import client, voice_registry

logger = logging.getLogger("voice_pipeline.tts_cache")

SAMPLE_RATE = client.SAMPLE_RATE
_CACHE_DIR = os.path.join(DATA_DIR, "tts_cache")

# LRU in memory (live agent reply pieces are free text, so it must not grow
# without bound); everything is also on disk, so an evicted line costs one
# file read, not a synthesis. ~1500 lines of a few seconds = ~70 MB.
_MEMORY_MAX = 1500
_memory: OrderedDict[str, bytes] = OrderedDict()
_inflight: dict[str, asyncio.Future] = {}


async def aclose() -> None:
    await client.aclose()


def _key(voice: voice_registry.ResolvedVoice, text: str, language: str) -> str:
    raw = "|".join(["omnivoice-a2", voice.voice_id, str(voice.version), str(voice.speed), language, str(SAMPLE_RATE), text])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _disk_path(key: str) -> str:
    return os.path.join(_CACHE_DIR, f"{key}.pcm")


def _remember(key: str, pcm: bytes) -> None:
    _memory[key] = pcm
    _memory.move_to_end(key)
    while len(_memory) > _MEMORY_MAX:
        _memory.popitem(last=False)


async def get_audio(
    text: str, language: str, owner: str = "*", *, num_step: int | None = None,
    fallback_voice_id: str | None = None, fallback_speed: float = 1.0,
) -> bytes:
    """PCM16 mono 8 kHz audio for `text`. Memory, then disk, then the TTS
    service; concurrent requests for the same line share one synthesis.
    `owner` (scenario/agent id) selects the assigned voice. `num_step`
    overrides the cached-line quality (live agent pieces use fewer steps)."""
    pcm, _hit = await get_audio_ex(
        text, language, owner, num_step=num_step, fallback_voice_id=fallback_voice_id,
        fallback_speed=fallback_speed,
    )
    return pcm


async def get_audio_ex(
    text: str, language: str, owner: str = "*", *, num_step: int | None = None,
    fallback_voice_id: str | None = None, fallback_speed: float = 1.0,
) -> tuple[bytes, bool]:
    """get_audio, also reporting whether it was a cache hit (memory, disk or
    another call's in-flight synthesis) - the playout diagnostics need to
    tell a free line from a GPU render."""
    voice = await voice_registry.resolve(
        owner, language, fallback_voice_id=fallback_voice_id, fallback_speed=fallback_speed
    )
    key = _key(voice, text, language)
    if key in _memory:
        _memory.move_to_end(key)
        return _memory[key], True

    path = _disk_path(key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            pcm = f.read()
        _remember(key, pcm)
        return pcm, True

    if key in _inflight:
        return await _inflight[key], True

    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _inflight[key] = future
    try:
        pcm = await client.synthesize_pcm(
            text, language=language, voice_id=voice.voice_id, speed=voice.speed,
            num_step=num_step or settings.tts_cached_num_step,
        )
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            f.write(pcm)
        os.replace(tmp, path)
        _remember(key, pcm)
        future.set_result(pcm)
        return pcm, False
    except Exception as e:
        future.set_exception(e)
        future.exception()  # mark retrieved - the caller below gets it via raise
        raise
    finally:
        _inflight.pop(key, None)


async def prewarm(lines: list[tuple[str, str]], owner: str = "*", concurrency: int = 2) -> int:
    """Synthesize (text, language) pairs ahead of time. Returns how many
    failed - failures are logged, never raised, since prewarming is only an
    optimization (get_audio will retry on demand)."""
    sem = asyncio.Semaphore(concurrency)
    failed = 0

    async def one(text: str, language: str) -> None:
        nonlocal failed
        async with sem:
            try:
                await get_audio(text, language, owner)
            except Exception as e:  # noqa: BLE001
                failed += 1
                logger.error("prewarm failed for %r (%s): %s", text[:60], language, e)

    unique = list(dict.fromkeys((t, lang) for t, lang in lines if t and t.strip()))
    await asyncio.gather(*(one(t, lang) for t, lang in unique))
    return failed


def duration_secs(pcm: bytes) -> float:
    return len(pcm) / (2 * SAMPLE_RATE)
