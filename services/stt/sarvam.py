"""Sarvam Saarika streaming STT (Hindi / Gujarati / code-mixed).

The reason this project needs Sarvam at all: Saarika handles Hindi-English
and Gujarati-Hindi code-mixing (the Mundra operator default, not an edge
case) noticeably better than English-first STT models. Sarvam's streaming
websocket API shape (api.sarvam.ai) is implemented here from documented
behaviour but NOT verified against a live key in this build - confirm the
exact websocket path and frame envelope in Sarvam's current docs console
before a pilot; the REST batch endpoint is stable enough to trust more than
the streaming one, which is newer.
"""

from typing import AsyncIterator

import httpx

from config import settings
from services.stt.base import STTProvider, STTResult

SARVAM_STREAMING_WS_URL = "wss://api.sarvam.ai/speech-to-text/streaming"


class SarvamSTTProvider(STTProvider):
    name = "sarvam"

    def __init__(self) -> None:
        if not settings.sarvam_api_key:
            raise RuntimeError("STT_PROVIDER=sarvam requires SARVAM_API_KEY.")

    async def stream_transcribe(
        self,
        audio_frames: AsyncIterator[bytes],
        *,
        language_hint: str | None = None,
    ) -> AsyncIterator[STTResult]:
        # NOTE: real implementation opens a websocket to SARVAM_STREAMING_WS_URL
        # with header `api-subscription-key: {settings.sarvam_api_key}`, sends
        # a config frame {"language_code": language_hint or "hi-IN",
        # "model": "saarika:v2"}, then forwards each mu-law frame as a binary
        # message and parses back {"transcript": ..., "is_final": ...} JSON
        # frames. Needs `websockets` or httpx's experimental WS support and a
        # live key to verify the exact frame envelope - left as the concrete
        # Phase 2 task.
        raise NotImplementedError(
            "Sarvam streaming wiring needs a live API key to verify the frame "
            "envelope - see comment above. httpx client is otherwise ready:"
        )
        yield  # pragma: no cover

    async def transcribe_batch(self, audio_bytes: bytes, *, language_hint: str = "hi-IN") -> STTResult:
        """REST batch fallback (used by e.g. voicemail/SMS-audio transcription,
        not the live call path). This endpoint shape is stable."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": settings.sarvam_api_key or ""},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"language_code": language_hint, "model": "saarika:v2"},
            )
            resp.raise_for_status()
            data = resp.json()
            return STTResult(
                text=data.get("transcript", ""), is_final=True, confidence=1.0, language=language_hint
            )
