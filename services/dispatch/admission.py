"""Pre-dial compliance gate - the single choke point every outbound dial must
pass through before the telephony API is ever called (services/dispatch/dialer.py
is the only caller on the real dialing path; services/guardrails/input.is_callable
does a second, cheaper belt-and-suspenders check at call-connect time in case
consent was revoked in the gap between admission and the call actually
connecting).

TRAI/TCCCPR context: DLT Principal Entity registration and the 1600/140x
number series are a manual/legal process, not something code can do - see
AUDIT.md and PHASE2_ARCHITECTURE_PROPOSAL.md. What *is* code's job, and what
this module is, is making sure nothing downstream of this function can ever
originate a call for a contact who hasn't granted consent, is on the DNC
list, or is outside the permitted calling window - and that a mid-call
opt-out request reliably lands here before the next dial attempt for that
contact, not just in a log somewhere.
"""

from dataclasses import dataclass
from datetime import datetime, time as time_type, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import models
from config import settings
from db import get_redis


@dataclass(frozen=True)
class AdmissionResult:
    allowed: bool
    reason: Optional[str] = None
    # "consent" | "dnc" | "calling_window" | "kill_switch" | None (allowed)
    reason_code: Optional[str] = None


async def is_kill_switch_engaged() -> bool:
    """Global stop-all-dialing switch. Deliberately read fresh from Redis on
    every check (not cached) so flipping it takes effect on the very next
    admission check, not after a process restart - this is the "something is
    visibly wrong, stop calling people right now" control, so it must never
    be one restart away from working."""
    redis = get_redis()
    value = await redis.get(settings.global_kill_switch_redis_key)
    return value not in (None, "", "0", "false", "False")


def _within_calling_window(campaign: "models.Campaign", *, now: datetime) -> bool:
    tz = ZoneInfo(campaign.calling_window_tz)
    local_now = now.astimezone(tz).time()
    start = time_type.fromisoformat(campaign.calling_window_start)
    end = time_type.fromisoformat(campaign.calling_window_end)
    if start <= end:
        return start <= local_now <= end
    # A window that wraps past midnight (e.g. 22:00-06:00) - not used by any
    # of the five ShiftEaze agent configs today (all are 07:00-21:00), but
    # the CallingWindow model doesn't forbid it, so handle it correctly
    # rather than silently miscomputing it if a future agent config sets one.
    return local_now >= start or local_now <= end


async def can_dial(
    contact: "models.Contact",
    campaign: Optional["models.Campaign"] = None,
    *,
    now: Optional[datetime] = None,
) -> AdmissionResult:
    """The gate. Called once per contact per dispatch attempt, immediately
    before services/telephony/*'s originate_call - never earlier (consent
    can be revoked between when a CallTask was created and when it's picked
    up) and never skipped for a "trusted" campaign; there is no bypass flag,
    on purpose.

    `campaign` is optional so this is also usable for the ad-hoc,
    non-campaign call paths (routers/calls.py's /make-call, routers/scenarios.py's
    /scenarios/call) if/when those are wired to pass a Contact through this
    gate too - see PHASE2_ARCHITECTURE_PROPOSAL.md. Without a campaign there
    is no calling-window to enforce, so that check is skipped, not bypassed.
    """
    now = now or datetime.now(timezone.utc)

    if await is_kill_switch_engaged():
        return AdmissionResult(False, "global kill switch is engaged", "kill_switch")

    if contact.dnc:
        return AdmissionResult(False, "contact is on the do-not-call list", "dnc")

    if contact.consent_status != "granted":
        return AdmissionResult(
            False,
            f"contact consent_status={contact.consent_status!r}, not 'granted'",
            "consent",
        )

    if campaign is not None and not _within_calling_window(campaign, now=now):
        return AdmissionResult(
            False,
            (
                f"outside calling window {campaign.calling_window_start}-"
                f"{campaign.calling_window_end} {campaign.calling_window_tz} "
                f"(checked at {now.isoformat()})"
            ),
            "calling_window",
        )

    return AdmissionResult(True)


async def record_opt_out(session, contact: "models.Contact", *, source: str = "call") -> None:
    """Mid-call (or API-driven) opt-out. Sets dnc AND consent_status together
    - dnc alone would satisfy can_dial's dnc check but a future consent-status
    report would still show "granted", which is confusing and, worse, is
    exactly the kind of inconsistency that lets a contact get re-dialed by
    code that only checks one of the two fields. `source` is recorded on the
    ConsentRecord audit row (e.g. "call", "sms", "api") so the revocation
    trail says how, not just that.

    Because every future admission check re-reads contact.consent_status /
    contact.dnc from the database (can_dial takes a live Contact row, not a
    cached one), a revocation recorded here is picked up by the very next
    dispatch cycle for this contact - "propagates to future dial lists" is a
    property of always checking live state, not a separate sync step.
    """
    contact.dnc = True
    contact.consent_status = "revoked"
    session.add(models.ConsentRecord(contact_id=contact.id, event="revoked", source=source))
