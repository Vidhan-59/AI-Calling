"""Text-to-speech behind a small, swappable interface.

Piper is the only implementation today. See LOCAL_AI_README.md "Known
limitation" before relying on this for Hindi, and do not rely on it for
Gujarati at all - see the note on PIPER_VOICE_GU in config.py.
"""

import os
import wave
from abc import ABC, abstractmethod
from pathlib import Path

from local_ai import config

_LANGUAGE_TO_VOICE = {
    "en": config.PIPER_VOICE_EN,
    "hi": config.PIPER_VOICE_HI,
    "gu": config.PIPER_VOICE_GU,
}


class TTSEngine(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_wav_path: str, language: str = "en") -> str:
        """Writes a WAV file to output_wav_path, returns that same path."""
        raise NotImplementedError


class VoiceNotAvailableError(RuntimeError):
    """No usable voice model for the requested language. Raised instead of
    silently substituting a different language's voice, which would speak
    the wrong language's phonemes over the actual text - worse than failing
    loudly, especially for a compliance-sensitive HR call."""


class PiperTTS(TTSEngine):
    _voices: dict[str, object] = {}  # loaded lazily per language, shared

    def __init__(self, voices_dir: str = config.PIPER_VOICES_DIR):
        self.voices_dir = voices_dir

    def _voice_name_for(self, language: str) -> str:
        name = _LANGUAGE_TO_VOICE.get(language)
        if not name:
            raise VoiceNotAvailableError(
                f"No Piper voice configured for language '{language}'. "
                f"Configured: {[k for k, v in _LANGUAGE_TO_VOICE.items() if v]}. "
                "See LOCAL_AI_README.md - Gujarati has no usable open-source "
                "Piper voice at the time of writing; use cloud TTS for it."
            )
        return name

    def _get_voice(self, language: str):
        name = self._voice_name_for(language)
        if name not in PiperTTS._voices:
            from piper import PiperVoice
            from piper.download_voices import download_voice

            model_path = os.path.join(self.voices_dir, f"{name}.onnx")
            if not os.path.isfile(model_path):
                os.makedirs(self.voices_dir, exist_ok=True)
                # download_voice requires a Path, not a str - passing a str
                # fails inside the library with a confusing "unsupported
                # operand type(s) for /" TypeError.
                download_voice(name, Path(self.voices_dir))
            PiperTTS._voices[name] = PiperVoice.load(model_path)
        return PiperTTS._voices[name]

    def synthesize(self, text: str, output_wav_path: str, language: str = "en") -> str:
        voice = self._get_voice(language)
        os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)) or ".", exist_ok=True)
        with wave.open(output_wav_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        return output_wav_path


def get_default_engine() -> TTSEngine:
    return PiperTTS()
