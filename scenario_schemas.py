"""Pydantic schemas for the ShiftEaze scenario-calling endpoints.

Kept in its own module (rather than added to schemas.py) so the scenario
feature is self-contained and does not collide with the schemas.py owned by
the broader platform rebuild.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from services.guardrails.input import contains_prompt_injection

Language = Literal["en", "hi", "gu"]


class CallFacts(BaseModel):
    """The shift facts substituted into the scripted lines.

    Whatever the scenario's `required_facts` lists must be present, otherwise
    the call is rejected up front - an operator hearing "your duty is on {date}"
    is worse than no call at all. This is the hallucination guard: the agent can
    only ever say facts that arrived here.

    These values are exactly the "contact name, notes field" prompt-injection
    risk called out for this project: employee_name/supervisor_name etc. are
    rendered not only into the spoken TwiML lines but into the branch
    classifier's system prompt on every turn of the call (see
    scenario_engine._build_classifier_prompt's "THE INITIAL QUESTION:" line).
    A caller-name field originating from an upstream CRM/roster sync is
    exactly the kind of free text that must not double as an instruction to
    the model, so every string value here (including extras, since this model
    allows them) is scanned before the request is accepted at all - reject
    loudly at intake rather than silently carry an injection attempt into a
    live call.
    """

    employee_name: str = Field(..., description="Operator's name, as spoken. e.g. 'Amit'")
    date: Optional[str] = Field(None, description="Spoken form, e.g. '2 August' / '2 अगस्त'")
    time: Optional[str] = Field(None, description="Spoken form, e.g. '8 in the morning' / 'सुबह 8 बजे'")
    shift_name: Optional[str] = Field(None, description="e.g. 'Day' / 'Night'")
    terminal: Optional[str] = Field(None, description="e.g. 'CT3'")
    supervisor_name: Optional[str] = Field(None, description="Named in escalation replies")
    shortfall_count: Optional[str] = Field(None, description="OT only: how many people short, e.g. '3'")
    duration: Optional[str] = Field(None, description="OT only: shift length, e.g. '8 hours'")

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _reject_prompt_injection(self) -> "CallFacts":
        for field_name, value in self.__dict__.items():
            if isinstance(value, str) and contains_prompt_injection(value):
                raise ValueError(
                    f"facts.{field_name}={value!r} looks like a prompt-injection "
                    "attempt (matches a pattern like 'ignore previous instructions', "
                    "'you are now', 'system:', ...) and this call was refused before "
                    "placing it. If this is a genuine name/value that happens to "
                    "match, rephrase it - e.g. spell out unusual phrasing rather "
                    "than using words like 'ignore' or 'system' verbatim."
                )
        return self


class ScenarioCallRequest(BaseModel):
    scenario_id: str = Field(
        ...,
        description="shift_confirm | punch_in_remind | absence_priority | ot_request | punch_out_remind",
    )
    language: Language = "hi"
    to_phone_number: str = Field(..., description="E.164, e.g. +919426005911")
    facts: CallFacts
    attempt: int = Field(1, ge=1, le=2, description="1 = first call, 2 = follow-up wording")
    rgid: Optional[str] = None


class ScenarioPreviewRequest(BaseModel):
    """Render the exact lines a call would speak, without placing it."""

    scenario_id: str
    language: Language = "hi"
    facts: CallFacts
    attempt: int = Field(1, ge=1, le=2)


class ScenarioTestTurnRequest(BaseModel):
    """Drive a scenario conversation over HTTP - no phone/Twilio/tunnel needed.

    Reuse the same session_id across calls to continue one conversation; the
    engine keeps attempt/branch/collection state per session_id.
    """

    scenario_id: str
    language: Language = "hi"
    session_id: str = Field(..., description="Any stable string identifying this test conversation.")
    message: str = Field(..., description="What the operator says. Empty string simulates silence.")
    facts: Optional[CallFacts] = None
    attempt: int = Field(1, ge=1, le=2)


class ScenarioTestBatchRequest(BaseModel):
    """Run a scripted list of operator utterances through one scenario in one
    request - the fast way to sanity-check every branch after editing a JSON."""

    scenario_id: str
    language: Language = "hi"
    utterances: list[str]
    facts: Optional[CallFacts] = None
    attempt: int = Field(1, ge=1, le=2)
