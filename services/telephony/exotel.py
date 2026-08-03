"""Exotel.

NOT verified against a live Exotel account in this build - Exotel's Voice
API is stable (basic-auth REST, https://api.exotel.com/v1/Accounts/{Sid}/...)
but some accounts are provisioned on a regional host
(api.in.exotel.com / api.exotel.com) and the Voicebot Applet's WebSocket
media-stream message schema is the part most likely to have moved since.
Before pointing real traffic at this: confirm your account's API host and
the exact Voicebot Applet stream frame format in the current Exotel docs
console, and update EXOTEL_API_BASE_URL / build_connect_response accordingly.
"""

import base64
import json

import httpx

from config import settings
from services.telephony.base import OriginatedCall, TelephonyProvider

EXOTEL_API_BASE_URL = "https://api.exotel.com/v1"


class ExotelProvider(TelephonyProvider):
    name = "exotel"

    def __init__(self) -> None:
        if not (settings.exotel_api_key and settings.exotel_api_token and settings.exotel_sid):
            raise RuntimeError(
                "TELEPHONY_PROVIDER=exotel requires EXOTEL_API_KEY, EXOTEL_API_TOKEN, "
                "EXOTEL_SID to be set."
            )
        auth = (settings.exotel_api_key, settings.exotel_api_token)
        self._client = httpx.AsyncClient(auth=auth, timeout=10.0)

    async def originate_call(
        self,
        *,
        to_number: str,
        from_number: str,
        answer_webhook_url: str,
        status_callback_url: str | None = None,
        record: bool = True,
    ) -> OriginatedCall:
        url = f"{EXOTEL_API_BASE_URL}/Accounts/{settings.exotel_sid}/Calls/connect.json"
        resp = await self._client.post(
            url,
            data={
                "From": to_number,
                "CallerId": from_number or settings.exotel_virtual_number,
                "Url": answer_webhook_url,
                "StatusCallback": status_callback_url,
                "Record": "true" if record else "false",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        sid = data.get("Call", {}).get("Sid", "")
        return OriginatedCall(provider_call_sid=sid, provider=self.name, raw=data)

    def build_connect_response(
        self,
        *,
        media_ws_url: str,
        greeting: str,
        voice_id: str,
        custom_params: dict[str, str],
    ) -> tuple[str, str]:
        # Exotel's Voicebot Applet takes a <Response><Voicebot url=".../>
        # element pointing at the raw-audio WebSocket. Greeting/voice here
        # are carried as base64 custom params for our own resolver to read
        # on the "setup" message, mirroring Twilio's <Parameter> mechanism.
        params_b64 = base64.b64encode(json.dumps(custom_params).encode()).decode()
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Response><Voicebot url="{media_ws_url}?params={params_b64}" /></Response>'
        )
        return xml, "application/xml"

    async def terminate_call(self, provider_call_sid: str) -> None:
        url = f"{EXOTEL_API_BASE_URL}/Accounts/{settings.exotel_sid}/Calls/{provider_call_sid}.json"
        await self._client.post(url, data={"Status": "completed"})

    async def transfer_call(self, provider_call_sid: str, target_number: str) -> None:
        raise NotImplementedError(
            "Exotel warm transfer needs a Connect Applet flow configured in the "
            "Exotel dashboard - wire the specific applet id here once provisioned."
        )
