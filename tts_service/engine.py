"""Synthesis engines behind one interface.

OmniVoiceEngine wraps k2-fsa/OmniVoice (zero-shot multilingual TTS: voice
cloning from a short reference clip, or voice design from attributes). All
GPU work runs on ONE dedicated worker thread: the model is not safe to call
concurrently, and a single thread gives FIFO ordering across live calls with
no lock contention on the event loop. Requests beyond MAX_QUEUE are shed
with a 503 by the caller instead of piling up latency on every live call.

MockEngine produces a deterministic tone (length proportional to the text)
so the service, the main-app client, and the Voice Studio can be exercised in
CI and on machines without a GPU. Select with TTS_ENGINE=mock.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tts_service import settings, voices

logger = logging.getLogger("tts_service.engine")


class Engine:
    name = "base"
    sample_rate = settings.MODEL_RATE
    loaded = False

    def _load(self) -> None: ...
    def _make_prompt(self, voice_id: str, ref_wav: Path, ref_text: str | None) -> str: ...
    def _render(self, text: str, voice_id: str | None, instruct: str | None, language: str,
                speed: float, num_step: int) -> np.ndarray: ...

    def __init__(self) -> None:
        # One thread: serialises GPU access (see module docstring).
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tts-{self.name}")
        self._pending = 0
        self.last_synth_ms: float | None = None

    @property
    def queue_depth(self) -> int:
        return self._pending

    async def _run(self, fn, *args):
        if self._pending >= settings.MAX_QUEUE:
            raise QueueFull()
        self._pending += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)
        finally:
            self._pending -= 1

    async def load(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._pool, self._load)
        self.loaded = True

    async def make_prompt(self, voice_id: str, ref_wav: Path, ref_text: str | None) -> str:
        """Build + persist the reusable voice prompt; returns the transcript used."""
        return await self._run(self._make_prompt, voice_id, ref_wav, ref_text)

    async def design_sample(self, text: str, instruct: str, language: str) -> np.ndarray:
        """Render one sample from attributes (no reference). Used to create a
        design voice's reference clip, which is then cloned - so a designed
        voice sounds identical on every call instead of re-rolling."""
        return await self._run(self._render, text, None, instruct, language, 1.0, settings.NUM_STEP)

    async def synthesize(self, text: str, voice_id: str, language: str, speed: float,
                         num_step: int | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        audio = await self._run(self._render, text, voice_id, None, language, speed, num_step or settings.NUM_STEP)
        self.last_synth_ms = (time.perf_counter() - t0) * 1000
        return audio

    def forget(self, voice_id: str) -> None:
        pass

    # --- GPU residency (only OmniVoiceEngine does anything) ---
    on_gpu = True
    idle_offload_secs = 0.0

    def _wake(self) -> None: ...

    async def warm(self) -> None:
        """Bring the model back onto the GPU ahead of a request (and restart
        its idle clock). The main app calls this when a call connects and when
        the caller starts talking, so the ~0.7s wake-up hides behind STT + LLM
        time instead of adding to the first audio."""
        if not self.on_gpu:
            await self._run(self._wake)
        else:
            self._last_used = time.monotonic()

    def _offload(self) -> None: ...

    async def idle_watcher(self) -> None:
        """Background task: park the model in CPU RAM once it has been idle
        for idle_offload_secs. Runs the move on the GPU worker thread, so it
        can never overlap a synthesis."""
        while True:
            await asyncio.sleep(1.0)
            if self.idle_offload_secs > 0 and self.on_gpu and self._pending == 0:
                try:
                    await asyncio.get_running_loop().run_in_executor(self._pool, self._offload)
                except Exception:  # noqa: BLE001 - parking is an optimisation
                    logger.exception("idle offload failed")


class QueueFull(RuntimeError):
    pass


class MockEngine(Engine):
    name = "mock"

    def _load(self) -> None:
        pass

    def _make_prompt(self, voice_id, ref_wav, ref_text):
        voices.prompt_path(voice_id).write_bytes(b"mock-prompt")
        return ref_text or "mock transcript"

    def _render(self, text, voice_id, instruct, language, speed, num_step):
        secs = max(0.3, len(text) * 0.06 / max(speed, 0.1))
        t = np.arange(int(secs * self.sample_rate)) / self.sample_rate
        freq = 180.0 + (sum(map(ord, voice_id or instruct or "")) % 120)
        return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class OmniVoiceEngine(Engine):
    name = "omnivoice"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._asr_loaded = False
        self.on_gpu = True
        self._last_used = time.monotonic()
        self._prompts: dict[str, tuple[int, object]] = {}  # voice_id -> (version, prompt)

    def _load(self) -> None:
        import torch
        from omnivoice import OmniVoice

        self.idle_offload_secs = _resolve_idle_offload(torch)
        dtype = getattr(torch, settings.DTYPE)
        t0 = time.perf_counter()
        self._model = OmniVoice.from_pretrained(
            settings.MODEL, device_map=settings.DEVICE, dtype=dtype, asr_device=settings.ASR_DEVICE,
        )
        logger.info("OmniVoice loaded on %s (%s) in %.1fs", settings.DEVICE, settings.DTYPE, time.perf_counter() - t0)
        # First generation compiles kernels / allocates buffers; pay it now,
        # not on the first caller's greeting.
        self._model.generate(text="Namaste.", num_step=8)

    def _ensure_gpu(self) -> None:
        """Bring the model back from CPU RAM (worker thread only)."""
        if not self.on_gpu:
            t0 = time.perf_counter()
            self._model.to(settings.DEVICE)
            self.on_gpu = True
            logger.info("OmniVoice moved back onto the GPU in %.2fs", time.perf_counter() - t0)

    def _wake(self) -> None:
        self._ensure_gpu()
        self._last_used = time.monotonic()

    def _offload(self) -> None:
        import torch

        if self.on_gpu and time.monotonic() - self._last_used >= self.idle_offload_secs:
            t0 = time.perf_counter()
            self._model.to("cpu")
            torch.cuda.empty_cache()
            self.on_gpu = False
            logger.info("OmniVoice idle %.0fs - parked in CPU RAM in %.2fs (VRAM freed)",
                        self.idle_offload_secs, time.perf_counter() - t0)

    def _make_prompt(self, voice_id: str, ref_wav: Path, ref_text: str | None) -> str:
        self._ensure_gpu()
        self._last_used = time.monotonic()
        if not ref_text and not self._asr_loaded:
            # Whisper (on CPU by default) is only loaded the first time a
            # voice is cloned without a transcript.
            self._model.load_asr_model()
            self._asr_loaded = True
        prompt = self._model.create_voice_clone_prompt(ref_audio=str(ref_wav), ref_text=ref_text or None)
        prompt.save(str(voices.prompt_path(voice_id)))
        self._prompts.pop(voice_id, None)
        return ref_text or getattr(prompt, "ref_text", "") or ""

    def _prompt_for(self, voice_id: str):
        from omnivoice import VoiceClonePrompt

        version = voices.get(voice_id).version
        cached = self._prompts.get(voice_id)
        if cached and cached[0] == version:
            return cached[1]
        prompt = VoiceClonePrompt.load(str(voices.prompt_path(voice_id)))
        self._prompts[voice_id] = (version, prompt)
        return prompt

    def _render(self, text, voice_id, instruct, language, speed, num_step):
        self._ensure_gpu()
        try:
            return self._generate(text, voice_id, instruct, language, speed, num_step)
        finally:
            self._last_used = time.monotonic()

    def _generate(self, text, voice_id, instruct, language, speed, num_step):
        kwargs = {"text": text, "num_step": num_step, "speed": speed}
        if language:
            kwargs["language"] = language
        if voice_id:
            kwargs["voice_clone_prompt"] = self._prompt_for(voice_id)
        elif instruct:
            kwargs["instruct"] = instruct
        out = self._model.generate(**kwargs)
        return np.asarray(out[0], dtype=np.float32).reshape(-1)

    def forget(self, voice_id: str) -> None:
        self._prompts.pop(voice_id, None)


def _resolve_idle_offload(torch) -> float:
    raw = settings.IDLE_OFFLOAD.strip().lower()
    if raw != "auto":
        return float(raw)
    try:
        total_gb = torch.cuda.get_device_properties(settings.DEVICE).total_memory / 1024**3
    except Exception:  # noqa: BLE001 - CPU/other device: nothing to park
        return 0.0
    return 8.0 if total_gb <= 12 else 0.0


def create_engine() -> Engine:
    if settings.ENGINE == "mock":
        return MockEngine()
    return OmniVoiceEngine()
