"""Sarvam Bulbul TTS (Hindi/Gujarati/English). REST endpoint shape verified
against a live key (2026-09-24): the request/response envelope here is
correct (POST /text-to-speech, {"inputs":[...],"target_language_code":...,
"speaker":...}, response {"audios": [base64 wav, ...]}), latency measured
at ~1.5-2s per call for a one-sentence request. Bulbul's streaming
websocket variant exists per Sarvam's docs but this implementation uses
the simpler REST synth-then-chunk approach, which costs that first-byte
latency versus true streaming synthesis - worth revisiting if TTS ever
sits on this repo's live-call critical path (today it doesn't - see
routers/analytics.py's sibling doc TECHNICAL_REFERENCE.md §5: every live
call path uses Twilio ConversationRelay's own TTS, this provider is only
reachable via services/tts/registry.py, which nothing currently calls).

Speaker names are model-specific and do change across Bulbul versions -
"meera" (previously used in configs/agents/*.yaml) is not a valid speaker
on bulbul:v3 as of this verification; valid speakers were returned by the
API's own 400-error message and configs were updated to "priya".
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
