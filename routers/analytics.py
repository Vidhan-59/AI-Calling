"""Read/reporting surface for the rebuild's call data (models.Call/Turn/
AgentSpan/CallOutcome) - the richest of the three call-tracking stores (see
AUDIT.md), and until this file existed the only one with no read endpoint
at all: routers/agent_voice.py writes these rows but nothing served them
back.

Every endpoint requires X-API-Key when ADMIN_API_KEY is configured (see
services/auth.py) - transcripts and outcomes are sensitive.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import models
from db import get_session
from services.auth import require_admin_key

logger = logging.getLogger("analytics")
router = APIRouter(tags=["analytics"], dependencies=[Depends(require_admin_key)])


def _call_dict(c: "models.Call", outcome: "models.CallOutcome | None") -> dict:
    return {
        "id": c.id,
        "agent_id": c.agent_id,
        "call_task_id": c.call_task_id,
        "provider": c.provider,
        "provider_call_sid": c.provider_call_sid,
        "direction": c.direction,
        "to_number": c.to_number,
        "status": c.status,
        "started_at": c.started_at.isoformat() if c.started_at else None,
        "answered_at": c.answered_at.isoformat() if c.answered_at else None,
        "ended_at": c.ended_at.isoformat() if c.ended_at else None,
        "duration_sec": c.duration_sec,
        "hangup_by": c.hangup_by,
        "outcome": (
            {
                "primary_outcome": outcome.primary_outcome,
                "secondary_outcomes": outcome.secondary_outcomes,
                "slots_json": outcome.slots_json,
                "confidence": outcome.confidence,
            }
            if outcome is not None
            else None
        ),
    }


@router.get("/agent-voice/calls")
async def list_calls(
    agent_id: Optional[str] = None,
    campaign_id: Optional[str] = None,
    status: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
):
    """Paginated call list, richest read path in the app - joins CallOutcome
    so a caller doesn't need a second request per call to see how it ended."""
    query = select(models.Call, models.CallOutcome).outerjoin(
        models.CallOutcome, models.CallOutcome.call_id == models.Call.id
    )
    if agent_id:
        query = query.where(models.Call.agent_id == agent_id)
    if status:
        query = query.where(models.Call.status == status)
    if since:
        query = query.where(models.Call.started_at >= since)
    if until:
        query = query.where(models.Call.started_at <= until)
    if campaign_id:
        query = query.join(models.CallTask, models.CallTask.id == models.Call.call_task_id).where(
            models.CallTask.campaign_id == campaign_id
        )
    query = query.order_by(models.Call.started_at.desc()).limit(limit)

    rows = (await session.execute(query)).all()
    return [_call_dict(call, outcome) for call, outcome in rows]


@router.get("/agent-voice/calls/{call_id}")
async def get_call_detail(call_id: str, session: AsyncSession = Depends(get_session)):
    """One call's full detail: the Call row, every Turn in order (with
    sentiment), the AgentSpan(s) that handled it, and its CallOutcome."""
    call = await session.get(models.Call, call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")

    outcome = await session.scalar(
        select(models.CallOutcome).where(models.CallOutcome.call_id == call_id)
    )
    turns = (
        await session.scalars(
            select(models.Turn).where(models.Turn.call_id == call_id).order_by(models.Turn.seq.asc())
        )
    ).all()
    spans = (
        await session.scalars(
            select(models.AgentSpan)
            .where(models.AgentSpan.call_id == call_id)
            .order_by(models.AgentSpan.entered_at.asc())
        )
    ).all()

    return {
        **_call_dict(call, outcome),
        "agent_spans": [
            {
                "agent_id": s.agent_id,
                "agent_version": s.agent_version,
                "entered_at": s.entered_at.isoformat() if s.entered_at else None,
                "exited_at": s.exited_at.isoformat() if s.exited_at else None,
                "exit_reason": s.exit_reason,
            }
            for s in spans
        ],
        "turns": [
            {
                "seq": t.seq,
                "role": t.role,
                "text": t.text,
                "latency_total_ms": t.latency_total_ms,
                "sentiment_label": t.sentiment_label,
                "sentiment_score": t.sentiment_score,
                "emotion": t.emotion,
                "escalate": t.escalate,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in turns
        ],
    }


@router.get("/analytics/summary")
async def analytics_summary(
    agent_id: Optional[str] = None,
    campaign_id: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    session: AsyncSession = Depends(get_session),
):
    """Aggregate stats over the filtered window: call volume, completion
    rate, outcome breakdown, sentiment breakdown, average duration/latency.
    Plain SQLAlchemy count/group_by - no separate aggregation store needed
    at this data volume."""

    def _scope(query):
        if agent_id:
            query = query.where(models.Call.agent_id == agent_id)
        if since:
            query = query.where(models.Call.started_at >= since)
        if until:
            query = query.where(models.Call.started_at <= until)
        if campaign_id:
            query = query.join(
                models.CallTask, models.CallTask.id == models.Call.call_task_id
            ).where(models.CallTask.campaign_id == campaign_id)
        return query

    total_calls = await session.scalar(_scope(select(func.count(models.Call.id))))
    completed_calls = await session.scalar(
        _scope(select(func.count(models.Call.id))).where(models.Call.status == "completed")
    )
    avg_duration = await session.scalar(_scope(select(func.avg(models.Call.duration_sec))))

    outcome_rows = (
        await session.execute(
            _scope(
                select(models.CallOutcome.primary_outcome, func.count(models.CallOutcome.id)).join(
                    models.Call, models.Call.id == models.CallOutcome.call_id
                )
            ).group_by(models.CallOutcome.primary_outcome)
        )
    ).all()

    sentiment_rows = (
        await session.execute(
            _scope(
                select(models.Turn.sentiment_label, func.count(models.Turn.id)).join(
                    models.Call, models.Call.id == models.Turn.call_id
                )
            )
            .where(models.Turn.sentiment_label.is_not(None))
            .group_by(models.Turn.sentiment_label)
        )
    ).all()

    avg_turn_latency = await session.scalar(
        _scope(select(func.avg(models.Turn.latency_total_ms)).join(models.Call, models.Call.id == models.Turn.call_id))
    )

    return {
        "total_calls": total_calls or 0,
        "completed_calls": completed_calls or 0,
        "completion_rate": (completed_calls / total_calls) if total_calls else None,
        "avg_duration_sec": avg_duration,
        "avg_turn_latency_ms": avg_turn_latency,
        "outcome_breakdown": {outcome: count for outcome, count in outcome_rows},
        "sentiment_breakdown": {label: count for label, count in sentiment_rows},
    }
