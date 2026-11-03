"""Placing a live phone call with a Studio agent (Twilio).

The "Call now" section of the agent workspace dials ONE number, set in .env as
STUDIO_CALL_NUMBER (E.164, e.g. +919876543210). The call is a normal Twilio
outbound call whose TwiML connects a media stream to /studio/media, where
services/voice_pipeline/studio_pipeline.py runs the agent: speech-to-text, the
agent's saved instructions + retrieved training material through the LLM,
and the agent's OmniVoice voice back to the phone.

The number is read from .env on every call (like NGROK_URL, see public_url.py),
so changing it needs no server restart. The number is never sent to the browser
in full - only a masked form.
"""

import asyncio
import os
import re
from datetime import datetime, timezone

from config import settings
from public_url import _read_env_value, get_public_url
from services.studio import store

ENV_NAME = "STUDIO_CALL_NUMBER"
_E164 = re.compile(r"^\+[1-9]\d{7,14}$")
ACTIVE = ("connecting", "in_progress")
# A call that has not connected this long after it was placed was never answered.
NO_ANSWER_AFTER_SECS = 150
# Refuse a second call for the same agent while one is plausibly still ringing.
DOUBLE_CLICK_GUARD_SECS = 90


class CallSetupError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


_INDIA_MOBILE = re.compile(r"^(?:\+?91|0)?([6-9]\d{9})$")


def normalize(raw: str) -> str | None:
    """E.164 form of what was typed in .env, or None. A bare Indian mobile
    number (9426005911, 09426005911, 919426005911) gets the +91 prefix - this
    app's contacts are all +91 - anything else must already be international."""
    value = re.sub(r"[\s\-().]", "", raw or "")
    m = _INDIA_MOBILE.match(value)
    if m:
        return "+91" + m.group(1)
    return value if _E164.match(value) else None


def destination() -> str | None:
    """The configured number, read fresh from .env (then the process
    environment), or None when unset / not a usable phone number."""
    return normalize(_read_env_value(ENV_NAME) or os.environ.get(ENV_NAME) or "")


def _raw_destination() -> str:
    return (_read_env_value(ENV_NAME) or os.environ.get(ENV_NAME) or "").strip()


def mask(number: str | None) -> str | None:
    if not number:
        return None
    return number[:3] + "•" * max(0, len(number) - 6) + number[-3:]


def _twilio_ready() -> bool:
    return bool(settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_phone_number)


def config_status() -> dict:
    """What the Call now section needs to show - and what is missing."""
    problems = []
    raw = _raw_destination()
    number = destination()
    if not raw:
        problems.append(f"Set {ENV_NAME} in .env to the phone number to call (international format, e.g. +919876543210).")
    elif not number:
        problems.append(f"{ENV_NAME} in .env is not a valid phone number (use international format like +919876543210, or a 10-digit Indian mobile number).")
    if not _twilio_ready():
        problems.append("Twilio is not configured: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER in .env.")
    if not get_public_url():
        problems.append("NGROK_URL in .env is not set to a public https URL that Twilio can reach.")
    return {"number_masked": mask(number), "number_set": bool(number), "twilio_ready": _twilio_ready(),
            "public_url_set": bool(get_public_url()), "problems": problems, "ready": not problems}


def _age_secs(iso: str) -> float:
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    except (TypeError, ValueError):
        return 0.0


def expire_unanswered(call: dict) -> dict:
    """A call row still 'connecting' long after placement was never answered:
    mark it so the UI stops showing 'ringing' forever."""
    if call["status"] == "connecting" and _age_secs(call["started_at"]) > NO_ANSWER_AFTER_SECS:
        store.update_call(call["id"], status="failed", outcome="no_answer", ended_at=_now(),
                          error="The call was not answered or did not connect.")
        return store.get_call(call["id"])
    return call


def _now() -> str:
    from simple_db import now_iso

    return now_iso()


def _check_tunnel(base_url: str) -> str | None:
    from routers.scenarios import _check_tunnel_reachable

    return _check_tunnel_reachable(base_url)


def _twilio_client():
    from twilio.rest import Client

    return Client(settings.twilio_account_sid, settings.twilio_auth_token)


async def place_call(agent: dict) -> dict:
    """Dial STUDIO_CALL_NUMBER with `agent`. Raises CallSetupError with a
    message fit for the UI; returns {call_id, call_sid, to}."""
    status = config_status()
    if status["problems"]:
        raise CallSetupError(409, " ".join(status["problems"]))
    for c in store.list_calls(agent_id=agent["id"], limit=5):
        if c["kind"] == "call" and c["status"] in ACTIVE and _age_secs(c["started_at"]) < DOUBLE_CLICK_GUARD_SECS:
            raise CallSetupError(409, "A call for this agent is already in progress.")
    public_url = get_public_url()
    problem = await asyncio.to_thread(_check_tunnel, public_url)
    if problem:
        raise CallSetupError(
            502,
            f"Not placing the call: your public URL {public_url} {problem}. Start a fresh tunnel and put the new "
            "URL in NGROK_URL in .env (no restart needed), or re-run run-all.ps1.",
        )

    to_number = destination()
    call_id = store.create_call(agent=agent, kind="call", language=agent["language"])
    twiml_url = f"{public_url}/studio/voice?call_id={call_id}"
    try:
        call = await asyncio.to_thread(
            lambda: _twilio_client().calls.create(url=twiml_url, to=to_number, from_=settings.twilio_phone_number)
        )
    except Exception as e:  # noqa: BLE001 - Twilio's message is the useful part
        store.update_call(call_id, status="failed", outcome="failed", ended_at=_now(), error=str(e))
        raise CallSetupError(502, f"Twilio could not place the call: {e}") from None
    store.update_call(call_id, metrics={"twilio_call_sid": call.sid})
    return {"call_id": call_id, "call_sid": call.sid, "to": mask(to_number)}


async def hang_up(call: dict) -> None:
    sid = (call.get("metrics") or {}).get("twilio_call_sid")
    if not sid:
        raise CallSetupError(409, "This call has no Twilio call to end.")
    try:
        await asyncio.to_thread(lambda: _twilio_client().calls(sid).update(status="completed"))
    except Exception as e:  # noqa: BLE001
        raise CallSetupError(502, f"Twilio could not end the call: {e}") from None
