"""SMS fallback - used when a live call can't get through or can't finish
(TELEPHONY silence, tool timeout degrade, no-answer retry backoff). Provider
abstracted the same way telephony/STT/TTS are."""

import logging

from config import settings

logger = logging.getLogger("ai_calling.sms")

_TEMPLATES = {
    "shift_reminder": "ShiftEaze: Aapki agli shift {date} ko {shift} hai, {terminal} par. Reply karke confirm karein.",
    "roster_unavailable": "ShiftEaze: Hum aapka roster abhi confirm nahi kar paaye. Jald hi dobara sampark karenge.",
}


async def send_sms(*, rgid: str, template_id: str, params: dict) -> bool:
    template = _TEMPLATES.get(template_id)
    if template is None:
        raise ValueError(f"Unknown SMS template_id: {template_id!r}")
    message = template.format(**params)

    if settings.sms_provider == "mock":
        logger.info("mock SMS to rgid=%s: %s", rgid, message)
        return True

    if settings.sms_provider == "twilio":
        if not (settings.twilio_account_sid and settings.twilio_auth_token):
            raise RuntimeError("SMS_PROVIDER=twilio requires TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN.")
        from twilio.rest import Client

        # NOTE: needs the contact's phone_e164, not rgid - the caller
        # (comms_tools.send_sms_fallback) should resolve rgid -> phone via
        # the Contact table before this line in a real deployment; left as
        # a lookup TODO since this module has no DB session of its own.
        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        client.messages.create(body=message, from_=settings.twilio_phone_number, to=rgid)
        return True

    raise ValueError(f"Unknown SMS_PROVIDER: {settings.sms_provider}")
