"""STT provider ABC. One method: take a stream of raw audio frames, yield a
stream of transcript results (interim, then final). The turn manager
(services/pipeline/turn_manager.py) is the only caller - it decides what to
do with interim vs. final text (early LLM prefetch on high-confidence
interim, per the latency budget in the architecture doc)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class STTResult:
    text: str
    is_final: bool
    confidence: float
    language: str | None = None


class STTProvider(ABC):
    name: str

    @abstractmethod
    async def stream_transcribe(
        self,
        audio_frames: AsyncIterator[bytes],
        *,
        language_hint: str | None = None,
    ) -> AsyncIterator[STTResult]:
        """`audio_frames` is mu-law 8kHz 20ms packets from the telephony
        WebSocket (see 01_MASTER_ARCHITECTURE.md Sec 2). Yield STTResult as
        soon as anything is available - interim results should come faster
        than you'd like, final results should come as soon as you're sure."""
