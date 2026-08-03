"""Speech-to-text behind a small, swappable interface.

faster-whisper is the only implementation today. To add another (e.g. Vosk
for a much smaller footprint, or NVIDIA Parakeet if a GPU is later added),
implement STTEngine and nothing else in this file changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from local_ai import config


@dataclass
class TranscriptionResult:
    text: str
    detected_language: str
    language_probability: float
    segments: list[dict] = field(default_factory=list)


class STTEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        """language: ISO 639-1 hint ('hi', 'gu', 'en') or None to auto-detect."""
        raise NotImplementedError


class FasterWhisperSTT(STTEngine):
    """CTranslate2-backed Whisper. No PyTorch dependency, int8 quantized by
    default - this is what keeps it fast on a CPU-only machine."""

    _model = None  # loaded lazily and shared across instances - a few hundred
    # MB and several seconds to load, not something to redo per call.

    def __init__(
        self,
        model_size: str = config.WHISPER_MODEL_SIZE,
        device: str = config.WHISPER_DEVICE,
        compute_type: str = config.WHISPER_COMPUTE_TYPE,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def _get_model(self):
        if FasterWhisperSTT._model is None:
            from faster_whisper import WhisperModel

            # local_files_only=True is what actually makes this offline once
            # the model is cached - without it, faster-whisper (via
            # huggingface_hub) reaches out to huggingface.co on every load to
            # check for updates, even when nothing needs downloading. Caught
            # by tests/local_ai/test_e2e_offline_smoke.py's network guard
            # during development - this is not a hypothetical concern.
            try:
                FasterWhisperSTT._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    local_files_only=True,
                )
            except Exception:
                # Not cached yet - allow exactly this one download, then every
                # subsequent load in this process (and on this machine) is
                # local_files_only=True again.
                FasterWhisperSTT._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    local_files_only=False,
                )
        return FasterWhisperSTT._model

    def transcribe(self, audio_path: str, language: str | None = None) -> TranscriptionResult:
        model = self._get_model()
        segments, info = model.transcribe(audio_path, language=language, vad_filter=True)
        segments = list(segments)  # the library returns a generator
        text = "".join(s.text for s in segments).strip()
        return TranscriptionResult(
            text=text,
            detected_language=info.language,
            language_probability=info.language_probability,
            segments=[
                {"start": s.start, "end": s.end, "text": s.text} for s in segments
            ],
        )


def get_default_engine() -> STTEngine:
    return FasterWhisperSTT()
