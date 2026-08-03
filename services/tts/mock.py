"""No-op TTS: yields the UTF-8 bytes of the text itself instead of real
audio. Lets the text-mode conversation harness and unit tests exercise
streaming/chunking/cancellation logic without a vendor key."""

from typing import AsyncIterator

from services.tts.base import TTSProvider


class MockTTSProvider(TTSProvider):
    name = "mock"

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        language: str,
        voice_id: str,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        async for chunk in text_stream:
            if chunk:
                yield chunk.encode("utf-8")
