"""Fallback comms when the live call itself can't finish the job: schedule
a retry, or send the details by SMS instead."""

from db import get_redis
from services.integrations.sms_client import send_sms
from services.tools.registry import tool


@tool(idempotent=True)
async def schedule_callback(rgid: str, at: str, reason: str) -> dict:
    """Pushes a retry marker into Redis for the dispatcher to pick up
    (services/dispatch/scheduler.py reads keys under this prefix). Real
    campaign-level retry state lives in CallTask.next_attempt_at; this is
    the lightweight ad-hoc version a live call can request mid-conversation."""
    redis = get_redis()
    key = f"callback:{rgid}:{at}"
    await redis.set(key, reason, ex=7 * 24 * 3600)
    return {"status": "scheduled", "rgid": rgid, "at": at}


@tool(idempotent=True, timeout_ms=1200)
async def send_sms_fallback(rgid: str, template_id: str, params: dict) -> dict:
    result = await send_sms(rgid=rgid, template_id=template_id, params=params)
    return {"status": "sent" if result else "failed", "template_id": template_id}
