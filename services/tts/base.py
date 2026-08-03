"""TTS provider ABC. Takes a stream of text (sentence-chunked by
services/pipeline/chunker.py) and yields a stream of audio bytes, so the
first clause can start playing before the model has finished the sentence."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class TTSProvider(ABC):
    name: str

    @abstractmethod
    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        language: str,
        voice_id: str,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        """Yield mu-law 8kHz audio chunks as they become available. Must
        support being cancelled mid-stream (barge-in kills this immediately -
        see services/pipeline/interruption.py)."""
