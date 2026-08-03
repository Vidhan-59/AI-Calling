"""Telephony provider ABC. Swapping Exotel for Plivo for Twilio must be a
config change (TELEPHONY_PROVIDER env var) - see services/telephony/registry.py.

A provider's job is narrow and deliberately does NOT include media handling:
originate a call, tell the carrier how to connect it to our media WebSocket,
and offer the handful of mid-call control operations (transfer, hangup) that
tools need. Everything about the audio itself is Pipecat's job.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OriginatedCall:
    provider_call_sid: str
    provider: str
    raw: dict


class TelephonyProvider(ABC):
    name: str

    @abstractmethod
    async def originate_call(
        self,
        *,
        to_number: str,
        from_number: str,
        answer_webhook_url: str,
        status_callback_url: str | None = None,
        record: bool = True,
    ) -> OriginatedCall:
        """Place an outbound call. `answer_webhook_url` is where the carrier
        will ask us how to connect the call once answered - the response is
        whatever `build_connect_response` returns."""

    @abstractmethod
    def build_connect_response(
        self,
        *,
        media_ws_url: str,
        greeting: str,
        voice_id: str,
        custom_params: dict[str, str],
    ) -> tuple[str, str]:
        """Return (body, content_type) for the answer webhook: provider-native
        markup that connects the call to our media WebSocket. Twilio wants
        TwiML <Connect><Stream>/<ConversationRelay>; Exotel/Plivo have their
        own XML dialects - see each provider's module docstring."""

    @abstractmethod
    async def terminate_call(self, provider_call_sid: str) -> None:
        """Hang up a live call - used by the global kill-switch and by
        transfer_to_human once the warm handoff completes."""

    @abstractmethod
    async def transfer_call(self, provider_call_sid: str, target_number: str) -> None:
        """Warm transfer (SIP REFER or provider equivalent) to a human."""
