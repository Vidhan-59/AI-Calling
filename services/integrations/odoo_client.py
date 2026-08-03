"""Outbound webhook to the shifteaze_ai_calling Odoo module, HMAC-SHA256
signed. At-least-once delivery: every attempt is recorded in
WebhookDelivery regardless of outcome, so a dead-letter can always be
replayed by workers/postcall.py's retry sweep (see that module for the
backoff schedule) instead of silently vanishing.
"""

import hashlib
import hmac
import json
from datetime import datetime, timezone

import httpx

from config import settings


def sign_payload(payload_bytes: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


async def send_event(
    session,
    *,
    call_id: str | None,
    event: str,
    payload: dict,
) -> "models.WebhookDelivery":  # noqa: F821 - see import below
    import models

    body = {
        "event": event,
        "call_id": call_id,
        "payload": payload,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    payload_bytes = json.dumps(body, sort_keys=True).encode()
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    signature = sign_payload(payload_bytes, settings.odoo_webhook_hmac_secret)

    delivery = models.WebhookDelivery(
        call_id=call_id,
        endpoint=settings.odoo_webhook_url or "unconfigured",
        payload_hash=payload_hash,
        hmac_sig=signature,
        attempts=0,
        status="pending",
    )
    session.add(delivery)
    await session.flush()

    if not settings.odoo_webhook_url:
        # No Odoo endpoint configured (local dev). Record the attempt as
        # dead-lettered rather than silently dropping it, so it's visible
        # in the audit trail that nothing was actually sent.
        delivery.status = "dead_letter"
        delivery.last_error = "ODOO_WEBHOOK_URL not configured"
        delivery.attempts = 1
        return delivery

    delivery.attempts += 1
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                settings.odoo_webhook_url,
                content=payload_bytes,
                headers={
                    "Content-Type": "application/json",
                    "X-ShiftEaze-Signature": signature,
                },
            )
            resp.raise_for_status()
        delivery.status = "delivered"
        delivery.delivered_at = datetime.now(timezone.utc)
    except Exception as e:
        delivery.status = "pending"  # eligible for the retry sweep
        delivery.last_error = str(e)
    return delivery
