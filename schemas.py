"""Pydantic request/response schemas for the HTTP API boundary.

This is what used to live in models.py before models.py became the
SQLAlchemy ORM layer (see models.py docstring). Kept as a separate module
per the project's own standard: "Pydantic models at every boundary. No bare
dicts crossing module lines."
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from config import settings
from services.guardrails.input import contains_prompt_injection


def _check_knowledge_base_for_injection(items: Optional[list[str]]) -> Optional[list[str]]:
    """knowledge_base entries are woven verbatim into this agent's system
    prompt on every call (see services/agent_manager.build_system_prompt) -
    exactly the "free-text CRM field becomes a prompt-injection vector" risk,
    since these facts are the kind of thing that could originate from an
    external system rather than be hand-typed by whoever configured the
    agent. Reject at the API boundary rather than silently save it."""
    for item in items or []:
        if contains_prompt_injection(item):
            raise ValueError(
                f"knowledge_base entry {item!r} looks like a prompt-injection "
                "attempt (matches a pattern like 'ignore previous instructions', "
                "'you are now', 'system:', ...) and was refused. This field is "
                "woven directly into the agent's system prompt on every call."
            )
    return items


class AgentCreateRequest(BaseModel):
    """Legacy free-form agent creation, retained for the /chat/test demo
    path. The five ShiftEaze scenarios are defined declaratively in
    configs/agents/*.yaml instead - see services/agent_registry.py."""

    agent_id: Optional[str] = Field(None, description="Optional custom id/slug.")
    name: str
    system_prompt: str
    welcome_greeting: str = "Hello! How can I help you today?"
    voice: str = "Google.en-US-Neural2-C"
    model: str = settings.default_claude_model
    effort: str = Field("medium", description="low | medium | high")
    knowledge_base: list[str] = Field(default_factory=list)

    _check_knowledge_base = field_validator("knowledge_base")(_check_knowledge_base_for_injection)


class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    system_prompt: Optional[str] = None
    welcome_greeting: Optional[str] = None
    voice: Optional[str] = None
    model: Optional[str] = None
    effort: Optional[str] = None
    knowledge_base: Optional[list[str]] = None

    _check_knowledge_base = field_validator("knowledge_base")(_check_knowledge_base_for_injection)


class CallRequest(BaseModel):
    to_phone_number: str
    agent_id: str = "default"
    contact_id: Optional[str] = None


class ChatTestRequest(BaseModel):
    agent_id: str = "default"
    session_id: str = Field(..., description="Any stable string identifying this test conversation.")
    message: str


# ---------------------------------------------------------------------------
# Campaigns / dispatch
# ---------------------------------------------------------------------------


class CampaignCreateRequest(BaseModel):
    agent_id: str
    name: str
    scenario: str
    contact_ids: list[str] = Field(default_factory=list)
    calling_window_start: str = "07:00"
    calling_window_end: str = "21:00"
    calling_window_tz: str = "Asia/Kolkata"
    concurrency_cap: int = 10
    scheduled_at: Optional[datetime] = None


class ContactCreateRequest(BaseModel):
    rgid: str
    name: str
    phone_e164: str
    language_pref: str = "hi"
    terminal: Optional[str] = None
    designation: Optional[str] = None
    consent_status: Literal["granted", "revoked", "unknown"] = "unknown"


# ---------------------------------------------------------------------------
# Webhooks (outbound to Odoo)
# ---------------------------------------------------------------------------


class OdooWebhookPayload(BaseModel):
    event: str  # call.completed | call.outcome | call.escalated
    call_id: str
    rgid: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime


# ---------------------------------------------------------------------------
# Admin / safety
# ---------------------------------------------------------------------------


class KillSwitchRequest(BaseModel):
    enabled: bool
    reason: str
