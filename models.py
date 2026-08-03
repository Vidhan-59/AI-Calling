"""SQLAlchemy ORM models - the data model from 01_MASTER_ARCHITECTURE.md Sec 6.

Pydantic request/response schemas live in schemas.py; this file is
persistence only. IDs are UUID strings for anything referenced across
services or by an external system (Odoo, dispatcher, webhooks); plain
autoincrement ints for append-only log/detail rows.
"""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base, now


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Agent registry - config-driven personas (Sec 3.1 / 6)
# ---------------------------------------------------------------------------


class Agent(Base):
    """One row per agent persona id (shift_confirm, punch_in_remind, ...).

    Rows are kept in sync with configs/agents/*.yaml by
    services/agent_registry.sync_configs_to_db() - the YAML file is the
    source of truth, this table exists so calls/campaigns can FK to a
    stable id and so publishing is auditable.
    """

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(255))
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), default="published")  # draft|published|disabled
    created_at: Mapped[datetime] = mapped_column(default=now)
    updated_at: Mapped[datetime] = mapped_column(default=now, onupdate=now)

    versions: Mapped[list["AgentVersion"]] = relationship(back_populates="agent")


class AgentVersion(Base):
    """Immutable snapshot of an agent's config at a point in time.

    Publishing a new version never mutates an old one - Call/AgentSpan rows
    reference the exact version that ran, so a dispute or a regression can
    always be traced back to the config that was live at the time.
    """

    __tablename__ = "agent_versions"
    __table_args__ = (UniqueConstraint("agent_id", "version", name="uq_agent_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    version: Mapped[int] = mapped_column(Integer)
    config_yaml: Mapped[str] = mapped_column(Text)
    persona_hash: Mapped[str] = mapped_column(String(64))
    flow_hash: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[str] = mapped_column(String(255), default="system")
    published_at: Mapped[Optional[datetime]] = mapped_column(default=now)
    created_at: Mapped[datetime] = mapped_column(default=now)

    agent: Mapped["Agent"] = relationship(back_populates="versions")


class Prompt(Base):
    """A single versioned, immutable prompt fragment (role = base_persona |
    voice_output_rules | language_policy | guardrails | persona)."""

    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    version: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(64))
    body: Mapped[str] = mapped_column(Text)
    checksum: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=now)


class FlowGraph(Base):
    """The compiled flows/<agent>.yaml, stored as JSON alongside the version
    that published it."""

    __tablename__ = "flow_graphs"
    __table_args__ = (UniqueConstraint("agent_id", "version", name="uq_flow_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    version: Mapped[int] = mapped_column(Integer)
    definition_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=now)


# ---------------------------------------------------------------------------
# Contacts, consent, campaigns, dispatch
# ---------------------------------------------------------------------------


class Contact(Base):
    __tablename__ = "contacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    rgid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    phone_e164: Mapped[str] = mapped_column(String(32))
    language_pref: Mapped[str] = mapped_column(String(8), default="hi")
    terminal: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    designation: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    consent_status: Mapped[str] = mapped_column(String(32), default="unknown")  # granted|revoked|unknown
    consent_captured_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    dnc: Mapped[bool] = mapped_column(Boolean, default=False)
    last_contacted_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=now)


class ConsentRecord(Base):
    """Append-only audit trail of consent capture/revocation events.
    Contact.consent_status is the current projection of this log."""

    __tablename__ = "consent_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    contact_id: Mapped[str] = mapped_column(ForeignKey("contacts.id"))
    event: Mapped[str] = mapped_column(String(32))  # granted|revoked
    source: Mapped[str] = mapped_column(String(64))  # onboarding|call|sms|odoo_sync
    recorded_at: Mapped[datetime] = mapped_column(default=now)


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    agent_version: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(255))
    scenario: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="scheduled")  # scheduled|running|paused|completed|cancelled
    calling_window_start: Mapped[str] = mapped_column(String(8), default="07:00")
    calling_window_end: Mapped[str] = mapped_column(String(8), default="21:00")
    calling_window_tz: Mapped[str] = mapped_column(String(64), default="Asia/Kolkata")
    concurrency_cap: Mapped[int] = mapped_column(Integer, default=10)
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), default="system")
    created_at: Mapped[datetime] = mapped_column(default=now)


class CallTask(Base):
    __tablename__ = "call_tasks"
    __table_args__ = (Index("ix_call_tasks_state_next_attempt", "state", "next_attempt_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    campaign_id: Mapped[str] = mapped_column(ForeignKey("campaigns.id"))
    contact_id: Mapped[str] = mapped_column(ForeignKey("contacts.id"))
    state: Mapped[str] = mapped_column(String(32), default="pending")
    # pending|admitted|originating|in_call|completed|failed|abandoned
    attempt_no: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=now)
    updated_at: Mapped[datetime] = mapped_column(default=now, onupdate=now)


# ---------------------------------------------------------------------------
# Calls, spans, turns, tools
# ---------------------------------------------------------------------------


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    call_task_id: Mapped[Optional[str]] = mapped_column(ForeignKey("call_tasks.id"), nullable=True)
    agent_id: Mapped[Optional[str]] = mapped_column(ForeignKey("agents.id"), nullable=True)
    provider: Mapped[str] = mapped_column(String(32), default="mock")
    provider_call_sid: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    direction: Mapped[str] = mapped_column(String(16), default="outbound")
    from_number: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    to_number: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="in_progress")
    started_at: Mapped[datetime] = mapped_column(default=now)
    answered_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    duration_sec: Mapped[Optional[int]] = mapped_column(nullable=True)
    hangup_by: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # agent|operator|system
    disposition: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    cost_paise: Mapped[Optional[int]] = mapped_column(nullable=True)
    recording_uri: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    consent_disclosed: Mapped[bool] = mapped_column(Boolean, default=False)


class AgentSpan(Base):
    """One row per agent that participated in a call - multi-agent trace."""

    __tablename__ = "agent_spans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"))
    agent_version: Mapped[int] = mapped_column(Integer)
    entered_at: Mapped[datetime] = mapped_column(default=now)
    exited_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    handoff_target: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class Turn(Base):
    __tablename__ = "turns"
    __table_args__ = (Index("ix_turns_call_seq", "call_id", "seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"))
    agent_span_id: Mapped[Optional[int]] = mapped_column(ForeignKey("agent_spans.id"), nullable=True)
    seq: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))  # user|assistant
    text: Mapped[str] = mapped_column(Text)
    language_detected: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    audio_offset_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    asr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    was_interrupted: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_vad_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    latency_stt_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    latency_llm_ttft_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    latency_tts_ttfb_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    latency_total_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    tokens_in: Mapped[Optional[int]] = mapped_column(nullable=True)
    tokens_out: Mapped[Optional[int]] = mapped_column(nullable=True)
    # sentiment carried over from the pre-existing single-agent build
    sentiment_label: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    emotion: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    escalate: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=now)


class ToolInvocation(Base):
    __tablename__ = "tool_invocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"))
    turn_id: Mapped[Optional[int]] = mapped_column(ForeignKey("turns.id"), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(64))
    args_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    result_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|ok|error|timeout
    duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(default=now)


# ---------------------------------------------------------------------------
# Post-call: outcomes, QA, webhooks
# ---------------------------------------------------------------------------


class CallOutcome(Base):
    __tablename__ = "call_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"), unique=True)
    slots_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    primary_outcome: Mapped[str] = mapped_column(String(64))
    secondary_outcomes: Mapped[list[str]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    extracted_by_model: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=now)


class QAScore(Base):
    __tablename__ = "qa_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id"), unique=True)
    score: Mapped[int] = mapped_column(Integer)
    guardrail_violations_json: Mapped[list[Any]] = mapped_column(JSON, default=list)
    rubric_version: Mapped[str] = mapped_column(String(32))
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=now)


class WebhookDelivery(Base):
    __tablename__ = "webhook_deliveries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[Optional[str]] = mapped_column(ForeignKey("calls.id"), nullable=True)
    endpoint: Mapped[str] = mapped_column(String(512))
    payload_hash: Mapped[str] = mapped_column(String(64))
    hmac_sig: Mapped[str] = mapped_column(String(128))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|delivered|dead_letter
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=now)
