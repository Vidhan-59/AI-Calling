"""Queue-backed outbound dialer.

The queue is `models.CallTask` (state / next_attempt_at / idempotency_key /
priority) - that table already existed, unused, before this module (see
AUDIT.md). Using it as the actual queue means there is exactly one place that
knows "what state is this call in", instead of a CallTask row *and* a
separate Redis Streams entry that can drift out of sync with each other.
Redis is used for the two things that genuinely need to be shared and fast
across processes: the live concurrency counter, and the kill switch
(services/dispatch/admission.py). This is the trade-off explained in
PHASE2_ARCHITECTURE_PROPOSAL.md - re-read that before changing this module's
storage model.

`run_dispatch_cycle` does exactly one pass and returns; it does not own an
event loop or a sleep. That's what makes it callable from a test, a one-shot
admin endpoint, and a background asyncio task in main.py with the same
function and the same guarantees.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

from sqlalchemy import select

import models
from config import settings
from db import get_redis
from services.dispatch.admission import can_dial
from services.telephony.base import TelephonyProvider
from services.telephony.registry import get_telephony_provider

logger = logging.getLogger("dialer")

ACTIVE_CALLS_REDIS_KEY = "ai_calling:active_call_count"

# Fallback retry schedule if an agent config can't be loaded for some reason
# (bad YAML, agent_id typo on the campaign) - a config error should degrade
# to "try again later on a sane default schedule", not lose the task.
_DEFAULT_BACKOFF = ["15m", "2h", "6h"]
_UNIT_TO_TIMEDELTA = {"m": "minutes", "h": "hours", "d": "days"}


def _parse_backoff(token: str) -> timedelta:
    unit = token[-1]
    amount = int(token[:-1])
    kwarg = _UNIT_TO_TIMEDELTA.get(unit)
    if kwarg is None:
        raise ValueError(f"Unrecognized backoff token {token!r} (expected e.g. '15m', '2h', '6h')")
    return timedelta(**{kwarg: amount})


async def active_call_count() -> int:
    redis = get_redis()
    value = await redis.get(ACTIVE_CALLS_REDIS_KEY)
    return int(value) if value else 0


async def reconcile_active_call_count(session) -> int:
    """Recompute ACTIVE_CALLS_REDIS_KEY from the database instead of trusting
    whatever Redis held across a restart. An ungraceful shutdown (crash,
    kill -9, OOM) can skip mark_call_ended's decrement, leaking a slot
    forever - see that function's docstring. Call once at startup, before
    the dialer loop or any admission check runs, so a fresh boot always
    starts from ground truth (count of CallTask rows still `in_call`) rather
    than a stale counter. Returns the reconciled value."""
    from sqlalchemy import func, select

    import models

    count = await session.scalar(
        select(func.count(models.CallTask.id)).where(models.CallTask.state == "in_call")
    )
    count = count or 0
    redis = get_redis()
    await redis.set(ACTIVE_CALLS_REDIS_KEY, count)
    return count


async def _adjust_active_call_count(delta: int) -> None:
    redis = get_redis()
    new_value = await redis.incrby(ACTIVE_CALLS_REDIS_KEY, delta)
    if new_value < 0:
        # Should never happen (decrement without a matching increment - a
        # double `finally` firing, most likely) - clamp rather than let the
        # counter go negative and silently over-admit forever afterwards.
        await redis.set(ACTIVE_CALLS_REDIS_KEY, 0)


async def mark_call_ended() -> None:
    """Call this exactly once per call that was actually originated (i.e.
    once per matching call to run_dispatch_cycle's originate step), from the
    WebSocket handler's `finally` block - see routers/agent_voice.py. If this
    is ever skipped on some exit path, the concurrency counter leaks a slot
    and the dialer silently under-dials forever until the process restarts;
    there is no self-healing lease/TTL on this counter today - see
    PHASE2_ARCHITECTURE_PROPOSAL.md's "known limitations" for the follow-up."""
    await _adjust_active_call_count(-1)


def _schedule_retry_or_fail(task: "models.CallTask", agent_id: Optional[str]) -> str:
    """Mutates task.next_attempt_at in place and returns the new state
    ("pending" to retry, "failed" if attempts are exhausted)."""
    backoff = _DEFAULT_BACKOFF
    max_attempts = len(_DEFAULT_BACKOFF)
    if agent_id:
        try:
            from services import agent_registry

            retry_cfg = agent_registry.get_agent_config(agent_id).retry
            backoff = retry_cfg.backoff or _DEFAULT_BACKOFF
            max_attempts = retry_cfg.attempts or len(backoff)
        except Exception as e:  # noqa: BLE001 - config error degrades, doesn't crash the dispatch cycle
            logger.warning("could not load retry config for agent_id=%s: %s", agent_id, e)

    if task.attempt_no >= max_attempts:
        return "failed"

    step = backoff[min(task.attempt_no - 1, len(backoff) - 1)] if backoff else "1h"
    try:
        delay = _parse_backoff(step)
    except ValueError:
        delay = timedelta(hours=1)
    task.next_attempt_at = datetime.now(timezone.utc) + delay
    return "pending"


async def run_dispatch_cycle(
    session,
    *,
    answer_webhook_url_for: Callable[[str], str],
    status_callback_url_for: Optional[Callable[[str], str]] = None,
    max_concurrent: Optional[int] = None,
    batch_size: int = 20,
    provider: Optional[TelephonyProvider] = None,
) -> dict:
    """One dispatch pass: pick up to `batch_size` eligible, due CallTask rows
    (respecting the concurrency ceiling), admission-gate each one, and
    originate the ones that pass. Does not commit the session - the caller
    owns the transaction (main.py's background loop and the tests both use
    `async with session_scope() as session:`), so a crash mid-cycle rolls
    back cleanly instead of leaving some tasks half-updated.

    `answer_webhook_url_for(call_task_id)` and `status_callback_url_for` are
    injected rather than built from settings.ngrok_url here, so tests (and a
    one-shot admin endpoint that already knows the request's own host) don't
    need a real public tunnel URL to exercise this function.
    """
    provider = provider or get_telephony_provider()
    max_concurrent = settings.max_concurrent_calls if max_concurrent is None else max_concurrent

    capacity = max(0, max_concurrent - await active_call_count())
    if capacity == 0:
        return {"picked": 0, "originated": 0, "skipped": 0, "capacity": 0}

    now = datetime.now(timezone.utc)
    rows = (
        await session.scalars(
            select(models.CallTask)
            .where(models.CallTask.state == "pending")
            .where(
                (models.CallTask.next_attempt_at.is_(None))
                | (models.CallTask.next_attempt_at <= now)
            )
            .order_by(models.CallTask.priority.desc(), models.CallTask.created_at.asc())
            .limit(min(batch_size, capacity))
        )
    ).all()

    originated = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}

    for task in rows:
        contact = await session.get(models.Contact, task.contact_id)
        campaign = await session.get(models.Campaign, task.campaign_id)

        if contact is None or campaign is None:
            task.state = "failed"
            skipped += 1
            skip_reasons["missing_contact_or_campaign"] = skip_reasons.get("missing_contact_or_campaign", 0) + 1
            logger.error(
                "[%s] contact or campaign missing (contact_id=%s campaign_id=%s) - marking failed",
                task.id, task.contact_id, task.campaign_id,
            )
            continue

        admission = await can_dial(contact, campaign, now=now)
        if not admission.allowed:
            skipped += 1
            skip_reasons[admission.reason_code or "unknown"] = skip_reasons.get(admission.reason_code or "unknown", 0) + 1
            logger.info("[%s] admission refused: %s", task.id, admission.reason)
            if admission.reason_code == "kill_switch":
                # Transient by definition - leave state as "pending" with no
                # next_attempt_at change so it's retried as soon as the
                # switch is lifted, not stuck behind a backoff timer.
                continue
            # consent/dnc/calling_window are not retryable by waiting - a
            # human decision (re-consent, campaign reschedule) is needed, so
            # this does not consume one of the agent's retry attempts.
            task.state = "abandoned"
            continue

        task.attempt_no += 1
        task.state = "originating"
        try:
            originated_call = await provider.originate_call(
                to_number=contact.phone_e164,
                from_number=settings.twilio_phone_number or "",
                answer_webhook_url=answer_webhook_url_for(task.id),
                status_callback_url=status_callback_url_for(task.id) if status_callback_url_for else None,
                record=True,
            )
        except Exception as e:  # noqa: BLE001 - a provider error must not kill the whole dispatch cycle
            logger.error("[%s] origination failed: %s", task.id, e)
            task.state = _schedule_retry_or_fail(task, campaign.agent_id)
            skipped += 1
            skip_reasons["origination_error"] = skip_reasons.get("origination_error", 0) + 1
            continue

        call_row = models.Call(
            call_task_id=task.id,
            agent_id=campaign.agent_id,
            provider=provider.name,
            provider_call_sid=originated_call.provider_call_sid,
            to_number=contact.phone_e164,
            from_number=settings.twilio_phone_number,
        )
        session.add(call_row)
        task.state = "in_call"
        await _adjust_active_call_count(1)
        originated += 1

    # Flush (not commit - the caller owns the transaction, see docstring)
    # so every task/call mutation above is visible to any query run against
    # this session for the rest of its transaction, not just after the
    # eventual commit. Without this, a second run_dispatch_cycle call - or a
    # test - sharing this session would see stale pre-mutation rows.
    await session.flush()

    return {
        "picked": len(rows),
        "originated": originated,
        "skipped": skipped,
        "skip_reasons": skip_reasons,
        "capacity": capacity,
    }


async def handle_call_status_callback(
    session,
    *,
    provider_call_sid: str,
    provider_status: str,
    provider_name: str = "twilio",
) -> Optional[str]:
    """Twilio (or another provider)'s call-status webhook lands here -
    routers/campaigns.py's /dialer/call-status wraps this. Maps the
    provider's terminal statuses onto CallTask outcomes and schedules a retry
    per the agent's backoff policy for no-answer/busy/failed; anything else
    (in-progress, ringing, ...) is a no-op here since it isn't terminal.

    Twilio's actual status strings (queued/ringing/in-progress/completed/
    busy/failed/no-answer/canceled) are documented on their status-callback
    reference - this maps the ones that matter for retry logic and passes
    through the raw string into Call.status so nothing is silently dropped
    even if a status shows up that this function doesn't specifically branch
    on. Not verified against a live Twilio webhook in this pass - no Twilio
    credentials are configured yet (see AUDIT.md) - so treat the exact status
    vocabulary as "written to the documented shape", the same caveat this
    repo already applies to services/stt/deepgram.py and friends.
    """
    call = (
        await session.scalars(
            select(models.Call).where(models.Call.provider_call_sid == provider_call_sid)
        )
    ).first()
    if call is None:
        logger.warning("call-status callback for unknown provider_call_sid=%s", provider_call_sid)
        return None

    call.status = provider_status
    terminal_retryable = {"busy", "no-answer", "failed", "canceled"}
    terminal_done = {"completed"}

    if provider_status not in terminal_retryable and provider_status not in terminal_done:
        return call.status  # not terminal yet (queued/ringing/in-progress/...)

    await mark_call_ended()

    task = await session.get(models.CallTask, call.call_task_id) if call.call_task_id else None
    if task is None:
        await session.flush()
        return call.status

    if provider_status in terminal_done:
        task.state = "completed"
        await session.flush()
        return call.status

    campaign = await session.get(models.Campaign, task.campaign_id)
    task.state = _schedule_retry_or_fail(task, campaign.agent_id if campaign else None)
    await session.flush()
    return call.status
