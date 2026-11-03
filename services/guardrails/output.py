"""Layer 2 of 3: a fast regex classifier on the outgoing agent text, run
BEFORE it reaches TTS. This is what makes guardrails.j2 (layer 1, a prompt
instruction) enforceable rather than just requested - an instruction is a
strong prior, not a guarantee, and Claude can still slip.

Layer 3 (post-call QA audit) lives in services/agents/qa_agent.py and is
the one allowed to be slow and thorough; this one runs on the critical path
so it stays cheap.
"""

import re
from dataclasses import dataclass

from services.guardrails.rules import GuardrailRule, Severity, get_active_rules


_HALLUCINATION_GUARD_DEFLECTION = (
    "Main sirf roster mein jo dikh raha hai wahi bata sakta hoon, abhi confirm "
    "karne ke liye ek second dijiye."
)

# Presence check, not a forbidden-phrase check (unlike every rule in
# GUARDRAIL_CATALOG) - so it needs its own keyword list per language and its
# own prepend-if-missing constant, not a regex the normal loop can use.
_DISCLOSURE_HINTS = ("automated", "record", "assistant", "bot", "ai ")
_DISCLOSURE_PREPEND = {
    "hi": "Namaste, main ShiftEaze ka automated calling assistant hoon, yeh call record ho sakti hai. ",
    "gu": "Namaste, hun ShiftEaze no automated calling assistant chu, aa call record thai shake chhe. ",
    "en": "Hello, this is ShiftEaze's automated calling assistant, and this call may be recorded. ",
}


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


def _discloses_ai_and_recording(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _DISCLOSURE_HINTS)


def check_output_text(
    text: str,
    *,
    guardrail_ids: list[str],
    roster_available: bool = True,
    is_first_turn: bool = False,
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

    if "disclose_ai_and_recording" in guardrail_ids and is_first_turn:
        if not _discloses_ai_and_recording(text):
            # WARN, not BLOCK: enforce() prepends the disclosure line rather
            # than discarding the model's actual opening content - see there.
            # This violation exists so QA can see the prompt instruction
            # (guardrails.j2) isn't being followed and needs tuning.
            violations.append(
                GuardrailViolation(
                    rule_id="disclose_ai_and_recording",
                    severity=Severity.WARN,
                    matched_text="",
                    description=(
                        "Opening turn did not disclose AI assistant / call "
                        "recording - guardrails.j2 instruction was not followed."
                    ),
                )
            )

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


_SHIFT_DETAIL_RE = re.compile(r"\b(?:" + "|".join(re.escape(h) for h in _SHIFT_DETAIL_HINTS) + r")\b")


def _mentions_shift_details(text: str) -> bool:
    # Whole words: a plain substring check matched "shift" inside the
    # company name "ShiftEaze", so every opening that named the company -
    # including the mandatory disclosure line - was swapped for the
    # deflection before any roster lookup had happened.
    return bool(_SHIFT_DETAIL_RE.search(text.lower()))


def enforce(
    text: str,
    *,
    guardrail_ids: list[str],
    roster_available: bool = True,
    is_first_turn: bool = False,
    language: str = "hi",
) -> tuple[str, list[GuardrailViolation]]:
    """Returns (safe_text_to_speak, violations_for_logging). If any BLOCK
    violation fires, the entire utterance is replaced with that rule's
    canned deflection - we do not attempt to surgically edit out just the
    offending clause, since a partially-edited LLM sentence tends to read
    as broken or contradictory once spoken aloud.

    disclose_ai_and_recording is the one exception to that replace-the-whole-
    utterance approach: there's no useful "deflection" for a missing
    disclosure (the greeting itself is the content worth keeping), so a
    missing disclosure gets the canned line PREPENDED instead - structurally
    guaranteeing the legal disclosure happens even if the model forgets,
    without discarding the actual opening line."""
    violations = check_output_text(
        text,
        guardrail_ids=guardrail_ids,
        roster_available=roster_available,
        is_first_turn=is_first_turn,
    )
    blocking = [v for v in violations if v.severity == Severity.BLOCK]

    if not blocking:
        safe_text = text
    else:
        top_violation = blocking[0]
        if top_violation.deflection is not None:
            safe_text = top_violation.deflection
        else:
            from services.guardrails.rules import GUARDRAIL_CATALOG

            rule: GuardrailRule | None = GUARDRAIL_CATALOG.get(top_violation.rule_id)
            if rule is None:
                # Defensive fallback, not expected to be reachable today: every
                # catalog-sourced violation's rule_id comes straight from
                # get_active_rules(), which only ever returns catalog entries.
                # Speak a generic deflection rather than crash the turn on an
                # unrecognized rule_id (e.g. a future rule added to
                # guardrails.j2's prompt text without a matching rules.py entry).
                safe_text = "Main is baare mein abhi baat nahi kar sakta."
            else:
                safe_text = rule.deflection

    if (
        is_first_turn
        and "disclose_ai_and_recording" in guardrail_ids
        and any(v.rule_id == "disclose_ai_and_recording" for v in violations)
        and not _discloses_ai_and_recording(safe_text)
    ):
        safe_text = _DISCLOSURE_PREPEND.get(language, _DISCLOSURE_PREPEND["hi"]) + safe_text

    return safe_text, violations
