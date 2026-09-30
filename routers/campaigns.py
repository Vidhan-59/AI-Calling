"""Contacts, campaigns, call tasks, and the dialer's admin surface (manual
dispatch trigger, Twilio call-status webhook, global kill switch).

This is what makes models.py's Contact/Campaign/CallTask/ConsentRecord
tables - defined but never written to before this file existed, see
AUDIT.md - actually reachable from an API. Every dial this router can lead
to goes through services/dispatch/admission.can_dial() inside
services/dispatch/dialer.run_dispatch_cycle(); there is no path from here
that calls a telephony provider directly.

Every endpoint below except /dialer/call-status requires X-API-Key when
ADMIN_API_KEY is configured (services/auth.py::require_admin_key) - see that
module's docstring for why the Twilio webhook is the one exception. Running
with ADMIN_API_KEY unset (the default) is still unauthenticated - see
PHASE2_ARCHITECTURE_PROPOSAL.md, "known limitations", and set it before a
real deploy.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import models
from config import settings
from db import get_redis, get_session, now as db_now
from public_url import get_public_url
from schemas import CampaignCreateRequest, ContactCreateRequest, KillSwitchRequest
from services.auth import require_admin_key
from services.dispatch.admission import is_kill_switch_engaged, record_opt_out
from services.dispatch.dialer import handle_call_status_callback, run_dispatch_cycle

logger = logging.getLogger("campaigns")
router = APIRouter(tags=["dialer"])
_admin = Depends(require_admin_key)


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def _contact_dict(c: "models.Contact") -> dict:
    return {
        "id": c.id,
        "rgid": c.rgid,
        "name": c.name,
        "phone_e164": c.phone_e164,
        "language_pref": c.language_pref,
        "terminal": c.terminal,
        "designation": c.designation,
        "consent_status": c.consent_status,
        "dnc": c.dnc,
        "consent_captured_at": c.consent_captured_at.isoformat() if c.consent_captured_at else None,
        "last_contacted_at": c.last_contacted_at.isoformat() if c.last_contacted_at else None,
    }


@router.post("/contacts", dependencies=[_admin])
async def create_contact(payload: ContactCreateRequest, session: AsyncSession = Depends(get_session)):
    existing = await session.scalar(select(models.Contact).where(models.Contact.rgid == payload.rgid))
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Contact with rgid={payload.rgid!r} already exists")

    contact = models.Contact(
        rgid=payload.rgid,
        name=payload.name,
        phone_e164=payload.phone_e164,
        language_pref=payload.language_pref,
        terminal=payload.terminal,
        designation=payload.designation,
        consent_status=payload.consent_status,
        consent_captured_at=db_now() if payload.consent_status == "granted" else None,
    )
    session.add(contact)
    await session.flush()
    if payload.consent_status in ("granted", "revoked"):
        session.add(models.ConsentRecord(contact_id=contact.id, event=payload.consent_status, source="api"))
    return _contact_dict(contact)


@router.get("/contacts", dependencies=[_admin])
async def list_contacts(limit: int = 100, session: AsyncSession = Depends(get_session)):
    rows = (
        await session.scalars(
            select(models.Contact).order_by(models.Contact.created_at.desc()).limit(limit)
        )
    ).all()
    return [_contact_dict(c) for c in rows]


@router.get("/contacts/{contact_id}", dependencies=[_admin])
async def get_contact(contact_id: str, session: AsyncSession = Depends(get_session)):
    contact = await session.get(models.Contact, contact_id)
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    return _contact_dict(contact)


@router.post("/contacts/{contact_id}/opt-out", dependencies=[_admin])
async def opt_out_contact(
    contact_id: str, source: str = "api", session: AsyncSession = Depends(get_session)
):
    """Mid-call or manual opt-out. Goes through
    services/dispatch/admission.record_opt_out rather than setting dnc/
    consent_status here directly, because that is the one function every
    write to those two fields must go through - see its docstring for why
    both fields are set together."""
    contact = await session.get(models.Contact, contact_id)
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    await record_opt_out(session, contact, source=source)
    logger.info("[%s] opted out (source=%s)", contact_id, source)
    return _contact_dict(contact)


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------


def _campaign_dict(c: "models.Campaign") -> dict:
    return {
        "id": c.id,
        "agent_id": c.agent_id,
        "agent_version": c.agent_version,
        "name": c.name,
        "scenario": c.scenario,
        "status": c.status,
        "calling_window_start": c.calling_window_start,
        "calling_window_end": c.calling_window_end,
        "calling_window_tz": c.calling_window_tz,
        "concurrency_cap": c.concurrency_cap,
        "scheduled_at": c.scheduled_at.isoformat() if c.scheduled_at else None,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


@router.post("/campaigns", dependencies=[_admin])
async def create_campaign(payload: CampaignCreateRequest, session: AsyncSession = Depends(get_session)):
    from services import agent_registry

    try:
        agent_config = agent_registry.get_agent_config(payload.agent_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown agent_id {payload.agent_id!r} - no configs/agents/{payload.agent_id}.yaml",
        )

    campaign = models.Campaign(
        agent_id=payload.agent_id,
        agent_version=agent_config.version,
        name=payload.name,
        scenario=payload.scenario,
        status="scheduled",
        calling_window_start=payload.calling_window_start,
        calling_window_end=payload.calling_window_end,
        calling_window_tz=payload.calling_window_tz,
        concurrency_cap=payload.concurrency_cap,
        scheduled_at=payload.scheduled_at,
    )
    session.add(campaign)
    await session.flush()

    created_tasks = 0
    skipped: list[dict] = []
    for contact_id in dict.fromkeys(payload.contact_ids):  # de-dup, preserve order
        contact = await session.get(models.Contact, contact_id)
        if contact is None:
            skipped.append({"contact_id": contact_id, "reason": "contact_not_found"})
            continue
        session.add(
            models.CallTask(
                campaign_id=campaign.id,
                contact_id=contact_id,
                state="pending",
                idempotency_key=f"{campaign.id}:{contact_id}:1",
            )
        )
        created_tasks += 1

    return {
        "campaign": _campaign_dict(campaign),
        "tasks_created": created_tasks,
        "skipped_contacts": skipped,
    }


@router.get("/campaigns", dependencies=[_admin])
async def list_campaigns(limit: int = 100, session: AsyncSession = Depends(get_session)):
    rows = (
        await session.scalars(
            select(models.Campaign).order_by(models.Campaign.created_at.desc()).limit(limit)
        )
    ).all()
    return [_campaign_dict(c) for c in rows]


@router.get("/campaigns/{campaign_id}", dependencies=[_admin])
async def get_campaign(campaign_id: str, session: AsyncSession = Depends(get_session)):
    campaign = await session.get(models.Campaign, campaign_id)
    if campaign is None:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return _campaign_dict(campaign)


@router.get("/campaigns/{campaign_id}/tasks", dependencies=[_admin])
async def list_campaign_tasks(campaign_id: str, session: AsyncSession = Depends(get_session)):
    rows = (
        await session.scalars(
            select(models.CallTask)
            .where(models.CallTask.campaign_id == campaign_id)
            .order_by(models.CallTask.created_at.asc())
        )
    ).all()
    return [
        {
            "id": t.id,
            "contact_id": t.contact_id,
            "state": t.state,
            "attempt_no": t.attempt_no,
            "next_attempt_at": t.next_attempt_at.isoformat() if t.next_attempt_at else None,
            "priority": t.priority,
        }
        for t in rows
    ]


# ---------------------------------------------------------------------------
# Dialer admin
# ---------------------------------------------------------------------------


@router.post("/kill-switch", dependencies=[_admin])
async def set_kill_switch(payload: KillSwitchRequest):
    """Flip the global stop-all-dialing switch. Read fresh from Redis by
    every admission check (services/dispatch/admission.is_kill_switch_engaged),
    so this takes effect on the very next dispatch cycle, not after a
    restart."""
    redis = get_redis()
    if payload.enabled:
        await redis.set(settings.global_kill_switch_redis_key, "1")
        logger.warning("KILL SWITCH ENGAGED: %s", payload.reason)
    else:
        await redis.delete(settings.global_kill_switch_redis_key)
        logger.warning("kill switch released: %s", payload.reason)
    return {"kill_switch_engaged": payload.enabled, "reason": payload.reason}


@router.get("/kill-switch", dependencies=[_admin])
async def get_kill_switch():
    return {"kill_switch_engaged": await is_kill_switch_engaged()}


@router.post("/dialer/run-once", dependencies=[_admin])
async def run_dialer_once(session: AsyncSession = Depends(get_session)):
    """Trigger exactly one dispatch cycle right now, without waiting for the
    background loop's poll interval. Useful for testing the pipeline end to
    end, or as a manual override in an environment where dialer_enabled is
    deliberately left False (see config.py)."""
    public_url = get_public_url()
    if not public_url:
        raise HTTPException(
            status_code=500,
            detail="NGROK_URL is not configured to a real public URL - cannot build an "
            "answer webhook URL Twilio could reach. See .env.example.",
        )
    return await run_dispatch_cycle(
        session,
        answer_webhook_url_for=lambda task_id: f"{public_url}/agent-voice/twiml?task_id={task_id}",
        status_callback_url_for=lambda task_id: f"{public_url}/dialer/call-status",
    )


@router.post("/dialer/call-status")
async def dialer_call_status(request: Request, session: AsyncSession = Depends(get_session)):
    """Twilio's call-status webhook. Expected fields (per Twilio's
    documented status-callback shape): CallSid, CallStatus. Not exercised
    against a real Twilio callback in this pass - no Twilio credentials are
    configured yet (see AUDIT.md) - verify the exact field names against a
    live callback before relying on this in production."""
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")
    if not call_sid or not call_status:
        raise HTTPException(status_code=400, detail="Missing CallSid/CallStatus in callback payload")

    new_status = await handle_call_status_callback(
        session, provider_call_sid=str(call_sid), provider_status=str(call_status)
    )
    if new_status is None:
        logger.warning("call-status callback for unknown CallSid=%s", call_sid)
    return {"status": "received", "call_status": new_status}
