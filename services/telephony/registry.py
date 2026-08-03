"""Provider selection - the acceptance test for the whole ABC layer is that
this is the ONLY place that branches on TELEPHONY_PROVIDER."""

from functools import lru_cache

from config import settings
from services.telephony.base import TelephonyProvider


@lru_cache
def get_telephony_provider() -> TelephonyProvider:
    provider = settings.telephony_provider
    if provider == "mock":
        from services.telephony.mock import MockTelephonyProvider

        return MockTelephonyProvider()
    if provider == "twilio":
        from services.telephony.twilio_provider import TwilioProvider

        return TwilioProvider()
    if provider == "exotel":
        from services.telephony.exotel import ExotelProvider

        return ExotelProvider()
    if provider == "plivo":
        from services.telephony.plivo import PlivoProvider

        return PlivoProvider()
    raise ValueError(f"Unknown TELEPHONY_PROVIDER: {provider}")
