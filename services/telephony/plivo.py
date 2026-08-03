"""Plivo, via the official `plivo` SDK.

Better streaming ergonomics than Exotel per the architecture doc's own
comparison table. Answer webhook returns Plivo XML with a <Stream> element
pointing at our raw mu-law media WebSocket - confirm the exact <Stream>
attribute set (bidirectional, audioTrack, contentType) against Plivo's
current Voice XML docs before a live pilot; the shape below is the
documented v1 contract as of this build."""

import plivo as plivo_sdk

from config import settings
from services.telephony.base import OriginatedCall, TelephonyProvider


class PlivoProvider(TelephonyProvider):
    name = "plivo"

    def __init__(self) -> None:
        if not (settings.plivo_auth_id and settings.plivo_auth_token):
            raise RuntimeError(
                "TELEPHONY_PROVIDER=plivo requires PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN."
            )
        self._client = plivo_sdk.RestClient(settings.plivo_auth_id, settings.plivo_auth_token)

    async def originate_call(
        self,
        *,
        to_number: str,
        from_number: str,
        answer_webhook_url: str,
        status_callback_url: str | None = None,
        record: bool = True,
    ) -> OriginatedCall:
        response = self._client.calls.create(
            from_=from_number or settings.plivo_phone_number,
            to_=to_number,
            answer_url=answer_webhook_url,
            answer_method="POST",
            hangup_url=status_callback_url,
        )
        return OriginatedCall(
            provider_call_sid=response.request_uuid, provider=self.name, raw=response.__dict__
        )

    def build_connect_response(
        self,
        *,
        media_ws_url: str,
        greeting: str,
        voice_id: str,
        custom_params: dict[str, str],
    ) -> tuple[str, str]:
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'<Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">'
            f"{media_ws_url}"
            "</Stream>"
            "</Response>"
        )
        return xml, "application/xml"

    async def terminate_call(self, provider_call_sid: str) -> None:
        self._client.calls.hangup(call_uuid=provider_call_sid)

    async def transfer_call(self, provider_call_sid: str, target_number: str) -> None:
        self._client.calls.transfer(
            call_uuid=provider_call_sid,
            legs="aleg",
            aleg_url=f"https://api.plivo.com/dial/{target_number}",
        )
