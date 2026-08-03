"""The guardrail catalog: one row per rule id used across all three layers
(prompt text in guardrails.j2, the fast output classifier in output.py, and
the post-call QA audit in services/agents/qa_agent.py). Keeping the catalog
in one place means a new guardrail is defined once, not three times.

Detection here is deliberately simple keyword/regex matching, not a model
call - it runs on every outgoing sentence before TTS and has to stay well
under the latency budget. It will have false negatives on cleverly-phrased
violations; that's what the QA audit (an LLM call, off the critical path) is
for. Treat this layer as a fast, imperfect net, not the only net.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    BLOCK = "block"  # must not reach TTS - substitute the deflection line
    WARN = "warn"  # let it through, flag for QA review


@dataclass(frozen=True)
class GuardrailRule:
    id: str
    description: str
    severity: Severity
    patterns: tuple[re.Pattern, ...]
    deflection: str


def _p(*patterns: str) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


GUARDRAIL_CATALOG: dict[str, GuardrailRule] = {
    "no_wage_discussion": GuardrailRule(
        id="no_wage_discussion",
        description="Never discuss salary, wages, payment, arrears, or OT rates.",
        severity=Severity.BLOCK,
        patterns=_p(
            r"\bsalary\b", r"\bwages?\b", r"\bpayment\b", r"\barrears?\b",
            r"\bot rate\b", r"\bpay\s?(rate|slip)\b", r"\btankha\b",
        ),
        deflection=(
            "Payment ke baare mein HR team aapko batayegi, main sirf "
            "shift ke liye call kar raha hoon."
        ),
    ),
    "no_leave_approval": GuardrailRule(
        id="no_leave_approval",
        description='Never say a leave request is "approved" or "rejected".',
        severity=Severity.BLOCK,
        patterns=_p(
            r"\bapproved\b", r"\brejected\b", r"\bmanzoor\b", r"\bho jayega\b",
            r"\byour leave is\b", r"\bleave (granted|denied)\b",
        ),
        deflection=(
            "Main aapka reason record kar raha hoon, Roster Planner review karenge."
        ),
    ),
    "no_third_party_employee_data": GuardrailRule(
        id="no_third_party_employee_data",
        description="Never disclose another employee's roster, absence, or contact info.",
        severity=Severity.BLOCK,
        patterns=_p(
            r"\b(he|she|they) (is|are) (also |)(absent|on leave|not coming)\b",
            r"\bother operator'?s?\b.*\b(shift|number|phone)\b",
        ),
        deflection="Main sirf aapki shift ke baare mein baat kar sakta hoon.",
    ),
    "no_disciplinary_threat": GuardrailRule(
        id="no_disciplinary_threat",
        description="Never mention consequences, warnings, or action against the operator.",
        severity=Severity.BLOCK,
        patterns=_p(
            r"\bwarning\b", r"\bconsequences?\b", r"\bchetavani\b",
            r"\baction (will be|liya)\b", r"\bdeduct(ion|ed)?\b",
            r"\brecord against you\b",
        ),
        deflection="Main sirf yaad dilane ke liye call kar raha hoon, koi problem nahi hai.",
    ),
    "no_medical_advice": GuardrailRule(
        id="no_medical_advice",
        description="Never give a medical opinion - deflect to safety officer or doctor.",
        severity=Severity.WARN,
        patterns=_p(r"\byou should take\b.*\b(medicine|tablet|dose)\b"),
        deflection="Aap safety officer ya doctor se zaroor baat kariye.",
    ),
    "rgid_full_number_leak": GuardrailRule(
        id="rgid_full_number_leak",
        description="Never read a full RGID aloud - only the last four digits.",
        severity=Severity.BLOCK,
        patterns=_p(r"\b\d{8,}\b"),
        deflection="Aapki ID ke aakhri char digit maine confirm kar liye hain.",
    ),
    "claims_to_be_human": GuardrailRule(
        id="claims_to_be_human",
        description="Must never claim to be a human when asked.",
        severity=Severity.BLOCK,
        patterns=_p(r"\bi am a human\b", r"\bmain insaan hoon\b", r"\bnot a bot\b"),
        deflection="Main ShiftEaze ka automated calling assistant hoon.",
    ),
}


def get_active_rules(guardrail_ids: list[str]) -> list[GuardrailRule]:
    """Rules for the guardrails actually configured on this agent, plus the
    handful that are non-negotiable for every ShiftEaze agent regardless of
    per-agent config (RGID leakage, humanity claims)."""
    always_on = {"rgid_full_number_leak", "claims_to_be_human"}
    ids = set(guardrail_ids) | always_on
    return [GUARDRAIL_CATALOG[i] for i in ids if i in GUARDRAIL_CATALOG]
