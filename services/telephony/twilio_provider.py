"""Twilio, via ConversationRelay.

This is the one provider in this file that is battle-tested in this repo -
it's a lift of the logic that already lived in routers/calls.py, which has
actually placed and answered real calls. ConversationRelay is a Twilio
public-beta product: it does STT/TTS/VAD *for* us and speaks plain text/JSON
over the media WebSocket, which is a materially different (and simpler)
contract than the raw mu-law frame stream Exotel/Plivo give Pipecat. If
ConversationRelay changes its message schema, this is the file that breaks -
see routers/voice.py for the WS side of the contract.
"""

from xml.sax.saxutils import quoteattr

from twilio.rest import Client

from config import settings
from services.telephony.base import OriginatedCall, TelephonyProvider


class TwilioProvider(TelephonyProvider):
    name = "twilio"

    def __init__(self) -> None:
        if not (settings.twilio_account_sid and settings.twilio_auth_token):
            raise RuntimeError(
                "TELEPHONY_PROVIDER=twilio requires TWILIO_ACCOUNT_SID and "
                "TWILIO_AUTH_TOKEN to be set."
            )
        self._client = Client(settings.twilio_account_sid, settings.twilio_auth_token)

    async def originate_call(
        self,
        *,
        to_number: str,
        from_number: str,
        answer_webhook_url: str,
        status_callback_url: str | None = None,
        record: bool = True,
    ) -> OriginatedCall:
        call = self._client.calls.create(
            url=answer_webhook_url,
            to=to_number,
            from_=from_number or settings.twilio_phone_number,
            record=record,
            recording_status_callback=status_callback_url,
            recording_status_callback_method="POST" if status_callback_url else None,
        )
        return OriginatedCall(provider_call_sid=call.sid, provider=self.name, raw={"sid": call.sid})

    def build_connect_response(
        self,
        *,
        media_ws_url: str,
        greeting: str,
        voice_id: str,
        custom_params: dict[str, str],
    ) -> tuple[str, str]:
        params_xml = "".join(
            f'<Parameter name={quoteattr(k)} value={quoteattr(v)}/>' for k, v in custom_params.items()
        )
        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect>"
            f"<ConversationRelay url={quoteattr(media_ws_url)} "
            f"welcomeGreeting={quoteattr(greeting)} "
            f"voice={quoteattr(voice_id)} "
            'ttsProvider="Google" interruptible="true">'
            f"{params_xml}"
            "</ConversationRelay>"
            "</Connect></Response>"
        )
        return twiml, "application/xml"

    async def terminate_call(self, provider_call_sid: str) -> None:
        self._client.calls(provider_call_sid).update(status="completed")

    async def transfer_call(self, provider_call_sid: str, target_number: str) -> None:
        # Warm transfer via TwiML <Dial> redirect - requires a small TwiML
        # endpoint that returns <Response><Dial>{target_number}</Dial></Response>.
        # Wire `url=` below to that endpoint once it exists (Phase 5/7).
        self._client.calls(provider_call_sid).update(
            twiml=f"<Response><Dial>{target_number}</Dial></Response>"
        )
