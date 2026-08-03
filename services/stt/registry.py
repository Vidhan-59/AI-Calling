from functools import lru_cache

from config import settings
from services.stt.base import STTProvider


@lru_cache
def get_stt_provider() -> STTProvider:
    provider = settings.stt_provider
    if provider == "mock":
        from services.stt.mock import MockSTTProvider

        return MockSTTProvider()
    if provider == "sarvam":
        from services.stt.sarvam import SarvamSTTProvider

        return SarvamSTTProvider()
    if provider == "deepgram":
        from services.stt.deepgram import DeepgramSTTProvider

        return DeepgramSTTProvider()
    raise ValueError(f"Unknown STT_PROVIDER: {provider}")
