"""Deepgram streaming STT (English).

Uses Deepgram's websocket streaming endpoint via the official `deepgram-sdk`
package rather than hand-rolling the socket protocol, so wire-format drift
is the SDK's problem, not ours. Add `deepgram-sdk` to requirements.txt and
verify the live-transcription options (model, smart_format, interim_results)
against your Deepgram project's current defaults before going live - this
is written to the documented v3 SDK shape, not smoke-tested against a real
API key in this build.
"""

from typing import AsyncIterator

from config import settings
from services.stt.base import STTProvider, STTResult


class DeepgramSTTProvider(STTProvider):
    name = "deepgram"

    def __init__(self) -> None:
        if not settings.deepgram_api_key:
            raise RuntimeError("STT_PROVIDER=deepgram requires DEEPGRAM_API_KEY.")
        try:
            from deepgram import DeepgramClient  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "deepgram-sdk is not installed. Run `pip install deepgram-sdk` "
                "(not in requirements.txt by default - only STT_PROVIDER=deepgram needs it)."
            ) from e
        self._client = DeepgramClient(settings.deepgram_api_key)

    async def stream_transcribe(
        self,
        audio_frames: AsyncIterator[bytes],
        *,
        language_hint: str | None = None,
    ) -> AsyncIterator[STTResult]:
        # NOTE: real implementation opens self._client.listen.asyncwebsocket.v("1")
        # with LiveOptions(model="nova-2", encoding="mulaw", sample_rate=8000,
        # interim_results=True, language=language_hint or "en"), pumps
        # `audio_frames` into it via .send(), and yields STTResult from the
        # on(LiveTranscriptionEvents.Transcript) callback. Wiring an async
        # generator around Deepgram's callback-based SDK needs a queue
        # bridge - left as the concrete Phase 2 task since it can only be
        # verified against a live Deepgram connection.
        raise NotImplementedError(
            "Deepgram streaming wiring needs a live API key and a queue bridge "
            "around the SDK's callback interface - see comment above."
        )
        yield  # pragma: no cover - makes this an async generator
