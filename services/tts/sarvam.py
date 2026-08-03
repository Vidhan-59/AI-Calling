"""Sarvam Bulbul TTS (Hindi). REST endpoint shape documented but not
verified against a live key in this build - Bulbul's streaming websocket
variant exists per Sarvam's docs but this implementation uses the simpler
REST synth-then-chunk approach, which costs a little first-byte latency
versus true streaming synthesis. Swap to the websocket variant once a key
is available to verify the frame envelope, same caveat as services/stt/sarvam.py.
"""

from typing import AsyncIterator

import httpx

from config import settings
from services.tts.base import TTSProvider


class SarvamTTSProvider(TTSProvider):
    name = "sarvam"

    def __init__(self) -> None:
        if not settings.sarvam_api_key:
            raise RuntimeError("TTS_PROVIDER=sarvam requires SARVAM_API_KEY.")

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        language: str,
        voice_id: str,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            async for sentence in text_stream:
                if not sentence.strip():
                    continue
                resp = await client.post(
                    "https://api.sarvam.ai/text-to-speech",
                    headers={"api-subscription-key": settings.sarvam_api_key or ""},
                    json={
                        "inputs": [sentence],
                        "target_language_code": language,
                        "speaker": voice_id,
                        "speech_sample_rate": 8000,
                        "pace": speed,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                import base64

                for audio_b64 in data.get("audios", []):
                    yield base64.b64decode(audio_b64)
