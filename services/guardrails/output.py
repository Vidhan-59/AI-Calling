"""Layer 2 of 3: a fast regex classifier on the outgoing agent text, run
BEFORE it reaches TTS. This is what makes guardrails.j2 (layer 1, a prompt
instruction) enforceable rather than just requested - an instruction is a
strong prior, not a guarantee, and Claude can still slip.

Layer 3 (post-call QA audit) lives in services/agents/qa_agent.py and is
the one allowed to be slow and thorough; this one runs on the critical path
so it stays cheap.
"""

from dataclasses import dataclass

from services.guardrails.rules import GuardrailRule, Severity, get_active_rules


_HALLUCINATION_GUARD_DEFLECTION = (
    "Main sirf roster mein jo dikh raha hai wahi bata sakta hoon, abhi confirm "
    "karne ke liye ek second dijiye."
)


@dataclass
class GuardrailViolation:
    rule_id: str
    severity: Severity
    matched_text: str
    description: str
    deflection: str | None = None
    """Set only for violations synthesized outside GUARDRAIL_CATALOG (e.g.
    hallucination_guard below, which has no static rule text to check
    against - it's a roster_available/keyword heuristic, not a regex). A
    catalog-backed violation leaves this None and enforce() looks its
    deflection up from GUARDRAIL_CATALOG instead, same as before."""


def check_output_text(
    text: str,
    *,
    guardrail_ids: list[str],
    roster_available: bool = True,
) -> list[GuardrailViolation]:
    violations: list[GuardrailViolation] = []
    for rule in get_active_rules(guardrail_ids):
        for pattern in rule.patterns:
            match = pattern.search(text)
            if match:
                violations.append(
                    GuardrailViolation(
                        rule_id=rule.id,
                        severity=rule.severity,
                        matched_text=match.group(0),
                        description=rule.description,
                    )
                )
                break

    if "hallucination_guard" in guardrail_ids and not roster_available:
        if _mentions_shift_details(text):
            violations.append(
                GuardrailViolation(
                    rule_id="hallucination_guard",
                    severity=Severity.BLOCK,
                    matched_text=text,
                    description=(
                        "Mentions shift/date/time details with no lookup_roster "
                        "result available this call - can't have come from real data."
                    ),
                    deflection=_HALLUCINATION_GUARD_DEFLECTION,
                )
            )
    return violations


_SHIFT_DETAIL_HINTS = (
    "shift", "terminal", "reporting time", "baje", "tareek", "roster",
    "day shift", "night shift", "din ki shift", "raat ki shift",
)


def _mentions_shift_details(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _SHIFT_DETAIL_HINTS)


def enforce(
    text: str,
    *,
    guardrail_ids: list[str],
    roster_available: bool = True,
) -> tuple[str, list[GuardrailViolation]]:
    """Returns (safe_text_to_speak, violations_for_logging). If any BLOCK
    violation fires, the entire utterance is replaced with that rule's
    canned deflection - we do not attempt to surgically edit out just the
    offending clause, since a partially-edited LLM sentence tends to read
    as broken or contradictory once spoken aloud."""
    violations = check_output_text(
        text, guardrail_ids=guardrail_ids, roster_available=roster_available
    )
    blocking = [v for v in violations if v.severity == Severity.BLOCK]
    if not blocking:
        return text, violations

    top_violation = blocking[0]
    if top_violation.deflection is not None:
        return top_violation.deflection, violations

    from services.guardrails.rules import GUARDRAIL_CATALOG

    rule: GuardrailRule | None = GUARDRAIL_CATALOG.get(top_violation.rule_id)
    if rule is None:
        # Defensive fallback, not expected to be reachable today: every
        # catalog-sourced violation's rule_id comes straight from
        # get_active_rules(), which only ever returns catalog entries. Speak
        # a generic deflection rather than crash the turn on an
        # unrecognized rule_id (e.g. a future rule added to guardrails.j2's
        # prompt text without a matching rules.py entry).
        return "Main is baare mein abhi baat nahi kar sakta.", violations
    return rule.deflection, violations
