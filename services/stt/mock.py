"""No-op STT for local dev/tests without STT credentials. Treats each
"frame" as UTF-8 text rather than real audio, so unit tests and the
text-mode conversation harness (services/agents/conversation_agent.py's
run_text_turn) can exercise the full turn pipeline without a real codec."""

from typing import AsyncIterator

from services.stt.base import STTProvider, STTResult


class MockSTTProvider(STTProvider):
    name = "mock"

    async def stream_transcribe(
        self,
        audio_frames: AsyncIterator[bytes],
        *,
        language_hint: str | None = None,
    ) -> AsyncIterator[STTResult]:
        buffer = b""
        async for frame in audio_frames:
            buffer += frame
        text = buffer.decode("utf-8", errors="ignore").strip()
        if text:
            yield STTResult(text=text, is_final=True, confidence=0.95, language=language_hint)
