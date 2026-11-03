"""Open-source, self-hosted speech-to-text for live calls (faster-whisper on
GPU), the USE_SARVAM_STT=false alternative to Sarvam STT. Model choice is driven by the benchmark in
tools/stt_benchmark.py - re-run it on the deployment server and set
LOCAL_STT_MODEL / LOCAL_STT_COMPUTE_TYPE accordingly.

Why not Pipecat's stock WhisperSTTService:
  - it loads the model per service instance, i.e. per phone call - seconds
    of model load before every call could hear anything. Here one model
    per (source, compute type) is loaded once, at server startup
    (preload_models), and shared by every call.
  - it wraps faster-whisper's transcribe() in to_thread, but transcribe()
    only does feature extraction there - decoding is lazy and runs while
    the segments are iterated, which it does back on the event loop,
    stalling audio for every live call during each transcription. Here the
    whole decode runs in the worker thread.

In production the decode runs in a separate process (stt_worker.py,
isolation="process") so it doesn't compete with the event loop for the GIL;
isolation="thread" keeps it in-process (tests, quick local runs).

Segmenting is by the pipeline's Silero VAD (SegmentedSTTService): audio is
buffered while the caller speaks and transcribed in one pass when they stop.

Speculative decoding: VAD only declares the turn over after stop_secs of
silence (0.5s), and a decode takes ~0.65s - run back to back, that's over a
second of dead air before the reply can even start. So the decode starts
early, on UserSpeechPausedFrame (audio_frontend.py, ~0.15s into the pause),
and by the time VAD confirms the caller is done the transcript is nearly
ready. If the caller resumes instead, the guess is discarded and the full
turn is decoded as usual - correctness never depends on the guess.
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncGenerator

import numpy as np

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from config import settings
from services.voice_pipeline.audio_frontend import UserSpeechPausedFrame, UserSpeechResumedFrame

logger = logging.getLogger("voice_pipeline.local_stt")

_LANG_ENUM = {"hi": Language.HI_IN, "gu": Language.GU_IN, "en": Language.EN_IN}

# Domain words: short noisy answers resolve toward what callers actually say.
DOMAIN_PROMPTS = {
    "hi": "हाँ जी, नहीं, ड्यूटी, शिफ्ट, टर्मिनल, सुपरवाइज़र, तबीयत, छुट्टी।",
    "gu": "હા, ના, ડ્યુટી, શિફ્ટ, ટર્મિનલ, સુપરવાઇઝર, તબિયત, રજા.",
}

def devanagari_to_gujarati(text: str) -> str:
    """Whisper often writes Gujarati speech in Devanagari script ("मारी तबियत"
    for "મારી તબિયત") - phonetically right, wrong alphabet. The two Unicode
    blocks are laid out in parallel (Gujarati = Devanagari + 0x180), so map it
    back; otherwise the Gujarati keyword lists in scenario_engine never match.
    Devanagari-only marks with no Gujarati counterpart are left as-is."""
    out = []
    for ch in text:
        cp = ord(ch)
        if 0x0901 <= cp <= 0x0963 or 0x0966 <= cp <= 0x096F:  # letters, signs, digits (not dandas)
            g = chr(cp + 0x180)
            out.append(g if g.isprintable() and not unicodedata_category_unassigned(g) else ch)
        else:
            out.append(ch)
    return "".join(out)


def unicodedata_category_unassigned(ch: str) -> bool:
    import unicodedata

    return unicodedata.category(ch) == "Cn"


_models: dict[tuple[str, str, str], object] = {}
_models_lock = threading.Lock()
# Models currently resident on the GPU, least-recently-used first. Everything
# else stays parked in CPU RAM (CTranslate2 unload_model(to_cpu=True)), so a
# swap is a ~0.3s host->device copy rather than a disk load. See config.py.
_gpu_models: list[object] = []
_last_decode = time.monotonic()
# CTranslate2 models are safe to call from several threads, but one GPU
# shared with Qwen gains nothing from overlapping decodes - serialize them
# so per-utterance latency stays predictable under concurrent calls.
_decode_lock = threading.Lock()
# A decode is ~0.5-1s (up to ~3s right after a language swap); far past that,
# the worker is wedged. Short on purpose: the caller is waiting in silence.
DECODE_TIMEOUT_SECS = 6.0
# The GPU holds its clocks for several seconds after work; warming more often
# than this only adds load.
WARM_INTERVAL_SECS = 2.5
_last_warm = 0.0


def _handle(model):
    """The CTranslate2 model behind a faster-whisper WhisperModel (None for
    test fakes, which have no GPU residency to manage)."""
    h = getattr(model, "model", None)
    return h if hasattr(h, "unload_model") else None


def _park_extra(keep: int) -> None:
    """Park least-recently-used GPU models in CPU RAM until at most `keep`
    remain. Caller holds _decode_lock (nothing may be decoding)."""
    while len(_gpu_models) > keep:
        old = _gpu_models.pop(0)
        h = _handle(old)
        if h is not None and h.model_is_loaded:
            t0 = time.monotonic()
            h.unload_model(to_cpu=True)
            logger.info("parked an STT model in CPU RAM (%.2fs) to free VRAM", time.monotonic() - t0)


def _activate(model) -> None:
    """Make `model` GPU-resident, parking others beyond the VRAM budget.
    Caller holds _decode_lock."""
    global _last_decode
    _last_decode = time.monotonic()
    h = _handle(model)
    if h is None:
        return
    if model in _gpu_models:
        _gpu_models.remove(model)
    _park_extra(max(1, settings.local_stt_gpu_models) - 1)
    if not h.model_is_loaded:
        t0 = time.monotonic()
        h.load_model()
        logger.info("moved an STT model back onto the GPU (%.2fs)", time.monotonic() - t0)
    _gpu_models.append(model)


def offload_if_idle() -> None:
    """Park every GPU-resident Whisper model once nothing has decoded for
    `local_stt_idle_offload_secs`, so Qwen/OmniVoice get the VRAM between
    calls. Called from the STT worker's idle loop; never blocks a decode."""
    idle = settings.local_stt_idle_offload_secs
    if idle <= 0 or not _gpu_models or time.monotonic() - _last_decode < idle:
        return
    if not _decode_lock.acquire(blocking=False):
        return
    try:
        _park_extra(0)
    finally:
        _decode_lock.release()


def get_model(source: str, compute_type: str, device: str = "cuda"):
    key = (source, compute_type, device)
    with _models_lock:
        if key not in _models:
            from faster_whisper import WhisperModel

            if device == "cuda":
                # A new model lands on the GPU at construction - make room first.
                with _decode_lock:
                    _park_extra(max(1, settings.local_stt_gpu_models) - 1)
            t0 = time.monotonic()
            _models[key] = WhisperModel(source, device=device, compute_type=compute_type)
            if device == "cuda":
                _gpu_models.append(_models[key])
            logger.info("loaded STT model %s (%s, %s) in %.1fs", source, compute_type, device, time.monotonic() - t0)
        return _models[key]


def preload_models(specs: list[tuple[str, str]]) -> None:
    """Load + warm each (source, compute_type) once in THIS process."""
    silence = np.zeros(16000, dtype=np.float32)
    for source, compute_type in specs:
        model = get_model(source, compute_type)
        with _decode_lock:
            _activate(model)
            list(model.transcribe(silence, language="hi", beam_size=1, without_timestamps=True)[0])


def warm_gpu(source: str, compute_type: str) -> None:
    """A throwaway decode of a moment of silence, to pull the GPU out of its
    idle clocks before a real utterance arrives. Traced on the RTX 5060
    laptop: between turns the GPU sits idle for 10s+ (cached TTS needs no
    GPU), drops to P8 at 180 MHz, and the next decode ran while the clocks
    were still ramping - ~1.1s instead of ~0.65s."""
    model = get_model(source, compute_type)
    with _decode_lock:
        _activate(model)  # a swap back from CPU RAM happens here, off the critical path
        list(model.transcribe(np.zeros(1600, dtype=np.float32), language="hi", beam_size=1,
                              without_timestamps=True, max_new_tokens=1)[0])


def preload(specs: list[tuple[str, str]], isolation: str = "thread") -> None:
    """Load + warm the models wherever calls will decode - call at server
    startup, off the event loop, so the first call of the day isn't slow."""
    if isolation == "process":
        from services.voice_pipeline.stt_worker import worker

        worker.load(specs)
    else:
        preload_models(specs)


def transcribe_with(
    model, audio: np.ndarray, *, language: str, beam_size: int, prompt: str | None, no_speech_threshold: float,
) -> tuple[str, float]:
    """One utterance -> (text, decode ms). Shared by the in-process path and
    the worker process."""
    with _decode_lock:
        _activate(model)
        t0 = time.monotonic()
        segments, _ = model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            temperature=0.0,
            vad_filter=False,  # the pipeline's VAD already segmented this
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=prompt,
        )
        # Whisper invents text for noise - from a stray "धन्यवाद" up to whole
        # news sentences (seen in tools/stt_benchmark.py on noisy audio:
        # "उन्होंने बताया कि प्रदेश में ... टीके लगाये"). Drop segments it
        # thinks contain no speech, or that it decoded with low confidence /
        # repetitively - faster-whisper's own hallucination thresholds.
        text = "".join(
            s.text for s in segments
            if s.no_speech_prob < no_speech_threshold
            and s.avg_logprob > -1.0
            and s.compression_ratio < 2.4
        )
        if language == "gu":
            text = devanagari_to_gujarati(text)
        return text.strip(), (time.monotonic() - t0) * 1000


_SARVAM_LANG = {"hi": "hi-IN", "gu": "gu-IN", "en": "en-IN"}
_sarvam_client = None


async def sarvam_fallback(audio_pcm16: bytes, language: str) -> str:
    """One utterance through Sarvam's REST STT (16 kHz mono PCM16 in). Only
    used when the local decode failed - see local_stt_fallback_sarvam."""
    import io
    import wave

    import httpx

    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = httpx.AsyncClient(timeout=10.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(audio_pcm16)
    resp = await _sarvam_client.post(
        "https://api.sarvam.ai/speech-to-text",
        headers={"api-subscription-key": settings.sarvam_api_key or ""},
        files={"file": ("utterance.wav", buf.getvalue(), "audio/wav")},
        data={"model": settings.sarvam_stt_model, "language_code": _SARVAM_LANG.get(language, "hi-IN")},
    )
    resp.raise_for_status()
    return (resp.json().get("transcript") or "").strip()


class _Speculation:
    def __init__(self, n_bytes: int, task: asyncio.Task):
        self.n_bytes = n_bytes
        self.task = task
        self.started_at = time.monotonic()


class LocalWhisperSTT(SegmentedSTTService):
    def __init__(
        self,
        *,
        source: str,
        compute_type: str,
        language: str,
        beam_size: int = 1,
        use_domain_prompt: bool = True,
        no_speech_threshold: float = 0.6,
        isolation: str = "thread",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._source, self._compute_type = source, compute_type
        self._isolation = isolation
        # In process mode the worker owns the model; this process loads none.
        self._model = get_model(source, compute_type) if isolation == "thread" else None
        self._lang = language if language in _LANG_ENUM else "hi"
        self._beam_size = beam_size
        self._prompt = DOMAIN_PROMPTS.get(self._lang) if use_domain_prompt else None
        self._no_speech_threshold = no_speech_threshold
        self._spec: _Speculation | None = None
        self.speculative_hits = 0
        self.speculative_misses = 0

    @property
    def wants_wav_segments(self) -> bool:
        return False  # raw 16-bit PCM, not a WAV upload

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, UserSpeechPausedFrame):
            self._speculate()
        elif isinstance(frame, (UserSpeechResumedFrame, VADUserStartedSpeakingFrame)):
            self._discard_speculation()
        await super().process_frame(frame, direction)
        # The bot finishing a line is when the caller is about to answer; the
        # caller starting to talk leaves >=0.75s (VAD start + stop windows)
        # before their audio can reach the decoder. Either way, warm the GPU.
        if isinstance(frame, (BotStoppedSpeakingFrame, VADUserStartedSpeakingFrame)):
            self._warm_gpu()

    def _warm_gpu(self) -> None:
        global _last_warm
        now = time.monotonic()
        if now - _last_warm < WARM_INTERVAL_SECS:
            return
        _last_warm = now
        try:
            if self._isolation == "process":
                from services.voice_pipeline.stt_worker import worker

                worker.warm(self._source, self._compute_type)
            else:
                self.create_task(asyncio.to_thread(warm_gpu, self._source, self._compute_type), "stt_warm")
        except Exception as e:  # noqa: BLE001 - a missed warm-up is slower, not broken
            logger.debug("STT warm-up skipped: %r", e)

    def _speculate(self) -> None:
        # One decode in flight at a time: a second one would only queue
        # behind the first on the single GPU worker and delay the real turn.
        if self._spec is not None and not self._spec.task.done():
            return
        audio = bytes(self._audio_buffer)
        if len(audio) // 2 < 1600:
            return
        self._spec = _Speculation(len(audio), self.create_task(self._decode_quietly(audio), "stt_speculative"))

    def _discard_speculation(self) -> None:
        if self._spec is not None:
            self.speculative_misses += 1
            self._spec = None  # its decode finishes in the worker; the result is just ignored
            logger.info("speculative STT guess discarded - caller resumed")

    async def _decode_quietly(self, audio: bytes) -> tuple[str, float] | None:
        try:
            return await self._decode(audio)
        except Exception as e:  # noqa: BLE001 - the end-of-turn decode retries it
            logger.warning("speculative STT failed: %r", e)
            return None

    def _options(self) -> dict:
        return {"language": self._lang, "beam_size": self._beam_size, "prompt": self._prompt,
                "no_speech_threshold": self._no_speech_threshold}

    async def _decode(self, audio: bytes) -> tuple[str, float]:
        if self._isolation == "process":
            from services.voice_pipeline.stt_worker import worker

            fut = worker.transcribe(self._source, self._compute_type, audio, **self._options())
            try:
                return await asyncio.wait_for(asyncio.wrap_future(fut), DECODE_TIMEOUT_SECS)
            except asyncio.TimeoutError:
                worker.restart()  # wedged (GPU fault?) - a fresh worker serves the next turn
                raise
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        return await asyncio.to_thread(transcribe_with, self._model, samples, **self._options())

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        n_samples = len(audio) // 2  # 16-bit PCM
        if n_samples < 1600:  # <0.1s - a click, not speech
            yield None
            return
        await self.start_processing_metrics()
        t0 = time.monotonic()
        result, how = None, "decode"
        spec, self._spec = self._spec, None
        # Only audio after the pause can be missing from the guess, and VAD
        # just called all of it silence - so the guess is the whole turn.
        if spec is not None and spec.n_bytes <= len(audio):
            try:
                result = await asyncio.wait_for(asyncio.shield(spec.task), DECODE_TIMEOUT_SECS)
            except Exception:  # noqa: BLE001 - a failed/timed-out guess just means decode for real
                result = None
            if result is not None:
                self.speculative_hits += 1
                how = f"speculative, started {(t0 - spec.started_at) * 1000:.0f}ms before end-of-turn"
        if result is None:
            try:
                result = await self._decode(audio)
            except Exception as e:  # noqa: BLE001
                logger.error("local STT failed: %r", e)
                if settings.local_stt_fallback_sarvam and settings.sarvam_api_key:
                    try:
                        t1 = time.monotonic()
                        text = await sarvam_fallback(audio, self._lang)
                        result = (text, (time.monotonic() - t1) * 1000)
                        how = "Sarvam fallback"
                        logger.warning("local STT failed - this utterance was transcribed by Sarvam instead (%.0fms)", result[1])
                    except Exception as e2:  # noqa: BLE001
                        logger.error("Sarvam STT fallback failed too: %r", e2)
                if result is None:
                    await self.stop_processing_metrics()
                    yield ErrorFrame(error=f"local STT failed: {e!r}")
                    return
        text, ms = result
        await self.stop_processing_metrics()
        logger.info("STT %.0fms decode, ready %.0fms after end-of-turn (%s, %.1fs audio): %r",
                    ms, (time.monotonic() - t0) * 1000, how, n_samples / 16000, text)
        if text:
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), _LANG_ENUM[self._lang])
        else:
            yield None
