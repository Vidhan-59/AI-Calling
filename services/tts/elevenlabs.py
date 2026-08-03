"""ElevenLabs Flash (English), via the official `elevenlabs` SDK's
streaming endpoint. Flash-tier model chosen specifically for TTFB, per the
latency budget - do not swap to a higher-quality/higher-latency model
without re-measuring the p50/p95 gate."""

from typing import AsyncIterator

from config import settings
from services.tts.base import TTSProvider


class ElevenLabsTTSProvider(TTSProvider):
    name = "elevenlabs"

    def __init__(self) -> None:
        if not settings.elevenlabs_api_key:
            raise RuntimeError("TTS_PROVIDER=elevenlabs requires ELEVENLABS_API_KEY.")
        try:
            from elevenlabs.client import AsyncElevenLabs  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "`elevenlabs` package is not installed. Run `pip install elevenlabs` "
                "(not in requirements.txt by default - only TTS_PROVIDER=elevenlabs needs it)."
            ) from e
        self._client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        language: str,
        voice_id: str,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        async for sentence in text_stream:
            if not sentence.strip():
                continue
            audio_stream = self._client.text_to_speech.convert_as_stream(
                voice_id=voice_id,
                model_id="eleven_flash_v2_5",
                text=sentence,
                output_format="ulaw_8000",
                voice_settings={"speed": speed},
            )
            async for chunk in audio_stream:
                yield chunk
