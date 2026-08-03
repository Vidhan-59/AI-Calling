"""In-memory telephony provider. No credentials, no network calls - this is
what TELEPHONY_PROVIDER=mock (the default) uses, so the rest of the stack
(flow engine, guardrails, tools, multi-agent handoff) can be built and
tested against a real call lifecycle before any vendor contract is signed."""

import logging
import uuid

from services.telephony.base import OriginatedCall, TelephonyProvider

logger = logging.getLogger("ai_calling.telephony.mock")


class MockTelephonyProvider(TelephonyProvider):
    name = "mock"

    def __init__(self) -> None:
        self._live_calls: dict[str, dict] = {}

    async def originate_call(
        self,
        *,
        to_number: str,
        from_number: str,
        answer_webhook_url: str,
        status_callback_url: str | None = None,
        record: bool = True,
    ) -> OriginatedCall:
        sid = f"mock-call-{uuid.uuid4().hex[:12]}"
        self._live_calls[sid] = {"to": to_number, "from": from_number, "status": "ringing"}
        logger.info("mock originate_call sid=%s to=%s from=%s", sid, to_number, from_number)
        return OriginatedCall(provider_call_sid=sid, provider=self.name, raw={"to": to_number})

    def build_connect_response(
        self,
        *,
        media_ws_url: str,
        greeting: str,
        voice_id: str,
        custom_params: dict[str, str],
    ) -> tuple[str, str]:
        import json

        return (
            json.dumps(
                {
                    "connect": {"media_ws_url": media_ws_url, "voice_id": voice_id},
                    "greeting": greeting,
                    "params": custom_params,
                }
            ),
            "application/json",
        )

    async def terminate_call(self, provider_call_sid: str) -> None:
        if provider_call_sid in self._live_calls:
            self._live_calls[provider_call_sid]["status"] = "terminated"
        logger.info("mock terminate_call sid=%s", provider_call_sid)

    async def transfer_call(self, provider_call_sid: str, target_number: str) -> None:
        logger.info("mock transfer_call sid=%s target=%s", provider_call_sid, target_number)
