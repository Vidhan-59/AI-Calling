from functools import lru_cache

from config import settings
from services.tts.base import TTSProvider


@lru_cache
def get_tts_provider() -> TTSProvider:
    provider = settings.tts_provider
    if provider == "mock":
        from services.tts.mock import MockTTSProvider

        return MockTTSProvider()
    if provider == "sarvam":
        from services.tts.sarvam import SarvamTTSProvider

        return SarvamTTSProvider()
    if provider == "elevenlabs":
        from services.tts.elevenlabs import ElevenLabsTTSProvider

        return ElevenLabsTTSProvider()
    if provider == "azure":
        from services.tts.azure import AzureTTSProvider

        return AzureTTSProvider()
    raise ValueError(f"Unknown TTS_PROVIDER: {provider}")
