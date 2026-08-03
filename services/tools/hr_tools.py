"""Write tools that record what an operator told us and forward it to
Odoo. None of these approve or deny anything - see no_leave_approval /
no_disciplinary_threat guardrails - they only record and notify.
"""

from services.integrations.odoo_client import send_event
from services.tools.registry import tool


@tool(idempotent=True, requires_slots=("identity_verified",))
async def confirm_shift(
    session,
    call_id: str,
    rgid: str,
    roster_line_id: str,
    intent: str,  # "confirm" | "decline"
    reason: str | None = None,
) -> dict:
    delivery = await send_event(
        session,
        call_id=call_id,
        event="call.shift_confirmed",
        payload={"rgid": rgid, "roster_line_id": roster_line_id, "intent": intent, "reason": reason},
    )
    return {"status": "recorded", "webhook_status": delivery.status}


@tool(idempotent=True)
async def mark_absence_intent(
    session,
    call_id: str,
    rgid: str,
    roster_line_id: str,
    reason: str,
    category: str,  # e.g. personal | family | health | transport | other
) -> dict:
    delivery = await send_event(
        session,
        call_id=call_id,
        event="call.absence_intent",
        payload={
            "rgid": rgid,
            "roster_line_id": roster_line_id,
            "reason": reason,
            "category": category,
        },
    )
    return {"status": "recorded", "webhook_status": delivery.status}


@tool(idempotent=True)
async def record_ot_response(
    session,
    call_id: str,
    rgid: str,
    roster_line_id: str,
    response: str,  # "accept" | "decline" | "conditional"
    condition: str | None = None,
) -> dict:
    delivery = await send_event(
        session,
        call_id=call_id,
        event="call.ot_response",
        payload={
            "rgid": rgid,
            "roster_line_id": roster_line_id,
            "response": response,
            "condition": condition,
        },
    )
    return {"status": "recorded", "webhook_status": delivery.status}


@tool(idempotent=True)
async def log_attendance_exception(
    session,
    call_id: str,
    rgid: str,
    exception_type: str,  # "forgot_punch" | "geofence" | "left_early" | "overrun" | "unclear"
    notes: str,
) -> dict:
    delivery = await send_event(
        session,
        call_id=call_id,
        event="call.attendance_exception",
        payload={"rgid": rgid, "exception_type": exception_type, "notes": notes},
    )
    return {"status": "recorded", "webhook_status": delivery.status}
