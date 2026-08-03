"""ShiftEaze scenario engine - the AI layer over the scripted call flows.

The five ShiftEaze calling scenarios are defined declaratively as JSON
knowledge bases in scenarios/*.json (one file per case, all three languages
inside). Adding or editing a scenario, a script line, or a keyword list is a
JSON edit - no Python change.

What the AI actually adds over a keyword-matching IVR (this is the whole
point): the operator does not have to say a magic word. The classifier reads
the utterance in context - code-mixed Hinglish/Gujlish, negation, paraphrase,
a reason volunteered before the question was asked - and maps it onto exactly
one scripted branch. The *reply* is then the exact scripted line from the
document, so compliance is preserved: the model chooses the branch, it never
writes the words. That combination is what makes this safe for HR-sensitive
calls at 1,300+ operators while still sounding like a conversation.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from config import BASE_DIR
from services import claude_client

logger = logging.getLogger("scenario_engine")

SCENARIOS_DIR = os.path.join(BASE_DIR, "scenarios")

# Utterances shorter than this after stripping are treated as noise, not speech.
_MIN_UTTERANCE_CHARS = 2

# How many times we re-ask on an unclear answer before giving up on the turn.
MAX_REPROMPTS_PER_ATTEMPT = 2

# Max total conversational turns before force-ending (safety cap for runaway loops).
MAX_TOTAL_TURNS = 10

_cache: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Knowledge-base loading
# ---------------------------------------------------------------------------


def load_scenarios(force: bool = False) -> dict[str, dict]:
    """Load and cache every scenarios/*.json.

    Cached because it is read on every call turn; pass force=True (or hit
    POST /scenarios/reload) after editing a JSON file to pick up changes
    without restarting the server.
    """
    global _cache
    if _cache and not force:
        return _cache

    loaded: dict[str, dict] = {}
    if not os.path.isdir(SCENARIOS_DIR):
        raise RuntimeError(f"scenarios directory not found: {SCENARIOS_DIR}")

    for filename in sorted(os.listdir(SCENARIOS_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(SCENARIOS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _validate_scenario(data, filename)
        loaded[data["id"]] = data

    _cache = loaded
    return _cache


def _validate_scenario(data: dict, filename: str) -> None:
    """Fail loudly at load time rather than mid-call on a live phone call."""
    for key in ("id", "display_name", "languages"):
        if key not in data:
            raise ValueError(f"{filename}: missing required key '{key}'")
    for lang, cfg in data["languages"].items():
        for key in ("opening", "voice", "tts_language", "transcription_language", "branches", "silence_reply"):
            if key not in cfg:
                raise ValueError(f"{filename}: language '{lang}' missing '{key}'")
        for branch_key, branch in cfg["branches"].items():
            for key in ("description", "reply", "outcome"):
                if key not in branch:
                    raise ValueError(
                        f"{filename}: language '{lang}' branch '{branch_key}' missing '{key}'"
                    )


def list_scenarios() -> list[dict]:
    scenarios = load_scenarios()
    return [
        {
            "id": s["id"],
            "case_no": s.get("case_no"),
            "display_name": s["display_name"],
            "trigger": s.get("trigger"),
            "flow_type": s.get("flow_type"),
            "max_attempts": s.get("max_attempts", 2),
            "languages": sorted(s["languages"].keys()),
            "required_facts": s.get("required_facts", []),
            "branches": sorted(next(iter(s["languages"].values()))["branches"].keys()),
        }
        for s in scenarios.values()
    ]


def get_scenario(scenario_id: str) -> Optional[dict]:
    return load_scenarios().get(scenario_id)


def get_language_config(scenario_id: str, language: str) -> Optional[dict]:
    scenario = get_scenario(scenario_id)
    if not scenario:
        return None
    return scenario["languages"].get(language)


# ---------------------------------------------------------------------------
# Fact substitution
# ---------------------------------------------------------------------------

DEFAULT_FACTS = {
    "employee_name": "Operator",
    "date": "",
    "time": "",
    "shift_name": "",
    "terminal": "",
    "supervisor_name": "your supervisor",
    "shortfall_count": "",
    "duration": "",
}


def render(template: str, facts: dict[str, Any]) -> str:
    """Fill {placeholders} from the call payload.

    Deliberately not str.format: a missing key must not raise mid-call, and
    unknown placeholders are left visible so a misconfigured scenario is
    obvious in the transcript rather than silently dropping information.
    """
    merged = {**DEFAULT_FACTS, **{k: v for k, v in (facts or {}).items() if v not in (None, "")}}

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        value = merged.get(key)
        return str(value) if value not in (None, "") else match.group(0)

    return re.sub(r"\{(\w+)\}", _sub, template)


def missing_facts(scenario_id: str, facts: dict[str, Any]) -> list[str]:
    scenario = get_scenario(scenario_id)
    if not scenario:
        return []
    required = scenario.get("required_facts", [])
    return [f for f in required if not (facts or {}).get(f)]


# ---------------------------------------------------------------------------
# AI intent classification
# ---------------------------------------------------------------------------


def _classification_schema(branch_keys: list[str], collects_reason: bool) -> dict:
    properties: dict[str, Any] = {
        "branch": {
            "type": "string",
            "enum": [*branch_keys, "unclear"],
            "description": (
                "The single branch that best matches what the operator said. "
                "Use 'unclear' ONLY if the utterance is genuinely unintelligible "
                "or completely unrelated to the question - not merely because the "
                "operator phrased it unusually."
            ),
        },
        "confidence": {
            "type": "number",
            "description": "0.0 to 1.0 confidence in the chosen branch.",
        },
        "detected_language": {
            "type": "string",
            "enum": ["en", "hi", "gu", "mixed", "unknown"],
            "description": "Language the operator actually spoke in.",
        },
        "rationale": {
            "type": "string",
            "description": "One short sentence explaining the choice, in English, for the audit log.",
        },
    }
    required = ["branch", "confidence", "detected_language", "rationale"]

    if collects_reason:
        # Plain string with "" for absent rather than a nullable union - keeps
        # the schema inside the guaranteed-supported structured-output subset.
        properties["volunteered_reason"] = {
            "type": "string",
            "description": (
                "If the operator already gave a reason for not attending in this same "
                "utterance (e.g. 'no, my wife is in hospital'), the reason verbatim. "
                "Otherwise an empty string."
            ),
        }
        required.append("volunteered_reason")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _build_classifier_prompt(
    scenario: dict,
    lang_cfg: dict,
    facts: dict,
    conversation_history: list[dict] | None = None,
) -> str:
    """Build the system prompt for the classifier.

    Includes conversation history for multi-turn context: the classifier needs
    to know what was already said to correctly interpret a follow-up utterance
    like 'haan' (yes) after the agent asked 'can you come?', versus after the
    agent asked 'please tell us the reason'.
    """
    lines = [
        "You classify what a port equipment operator said during an automated "
        "ShiftEaze workforce phone call, onto exactly one scripted branch.",
        "",
        f"CALL TYPE: {scenario['display_name']} ({scenario.get('trigger', '')})",
        f"LANGUAGE THE AGENT SPOKE: {lang_cfg.get('language_name')}",
        "",
        "THE INITIAL QUESTION:",
        render(lang_cfg["opening"], facts),
        "",
    ]

    # Add conversation history for multi-turn context
    if conversation_history:
        lines.append("CONVERSATION SO FAR (most recent last):")
        for turn in conversation_history[-6:]:  # last 6 turns max
            role_label = "AGENT" if turn["role"] == "assistant" else "OPERATOR"
            lines.append(f"  {role_label}: {turn['content']}")
        lines.append("")
        lines.append("THE OPERATOR'S LATEST UTTERANCE IS WHAT YOU MUST CLASSIFY.")
        lines.append("")

    lines.append("AVAILABLE BRANCHES:")
    for key, branch in lang_cfg["branches"].items():
        kws = ", ".join(branch.get("keywords", [])[:8])
        lines.append(f"- {key}: {branch['description']}")
        if kws:
            lines.append(f"    example phrasings: {kws}")

    lines += [
        "",
        "RULES:",
        "- Operators speak code-mixed Hindi/Gujarati/English, often over crane and "
        "container noise, so the transcript may be imperfect. Judge intent, not exact words.",
        "- The example phrasings are hints, NOT a required match. A paraphrase, a "
        "regional variant, or an answer wrapped in extra words still maps to its branch.",
        "- Watch negation carefully: 'haan nahi aa paunga' means NO, not yes.",
        "- If the operator answers the question AND adds context, classify on the answer.",
        "- Pick exactly one branch. Never invent a branch that is not listed.",
        "- IMPORTANT: Gujarati speech may be transcribed through a Hindi speech model, "
        "so Gujarati words may appear transliterated or approximated in Devanagari. "
        "Example: 'હા' (haa/yes in Gujarati) may appear as 'हा'. Classify on meaning.",
        "- Code-mixed utterances (e.g. 'no nahi aavish' mixing English, Hindi, Gujarati) "
        "are common. Extract the core intent regardless of language mixing.",
        "- If the conversation history shows the agent just asked for a reason (after "
        "a 'no'), and the operator is now giving a reason, that is NOT a new 'no' - "
        "it is the reason being collected. In that case classify as 'unclear' so the "
        "collection flow captures it verbatim.",
    ]
    return "\n".join(lines)


@dataclass
class Classification:
    branch: str
    confidence: float
    detected_language: str
    rationale: str
    volunteered_reason: Optional[str] = None


async def classify_utterance(
    scenario_id: str,
    language: str,
    utterance: str,
    facts: dict[str, Any],
    conversation_history: list[dict] | None = None,
) -> Classification:
    """Map a caller utterance onto one scripted branch using Claude.

    conversation_history is passed so the classifier understands multi-turn
    context (e.g. distinguishing a standalone 'no' from a reason given after
    the agent asked 'please tell us the reason').
    """
    scenario = get_scenario(scenario_id)
    lang_cfg = get_language_config(scenario_id, language)
    if not scenario or not lang_cfg:
        raise ValueError(f"unknown scenario/language: {scenario_id}/{language}")

    branch_keys = list(lang_cfg["branches"].keys())
    collects_reason = any("collect" in b for b in lang_cfg["branches"].values())

    prompt = _build_classifier_prompt(
        scenario, lang_cfg, facts, conversation_history=conversation_history,
    )

    # Additive hook: SCENARIO_LLM_BACKEND=local (see local_ai/config.py) routes
    # classification through Ollama instead of Claude. Default is "claude" -
    # unset, this line behaves exactly as before. See local_ai/README for why
    # this exists (a fully local, offline-capable alternative pipeline) and
    # local_ai/llm_engine.py for the Ollama-side implementation, which mirrors
    # claude_client.classify_structured's exact signature so this is a
    # same-shape swap, not a forked code path.
    backend = os.getenv("SCENARIO_LLM_BACKEND", "claude")
    classify_fn = claude_client.classify_structured
    if backend == "local":
        from local_ai import llm_engine as local_llm_engine

        classify_fn = local_llm_engine.classify_structured

    result = await classify_fn(
        system_prompt=prompt,
        user_text=utterance,
        schema=_classification_schema(branch_keys, collects_reason),
    )
    return Classification(
        branch=result.get("branch", "unclear"),
        confidence=float(result.get("confidence", 0.0)),
        detected_language=result.get("detected_language", "unknown"),
        rationale=result.get("rationale", ""),
        volunteered_reason=result.get("volunteered_reason") or None,
    )


# ---------------------------------------------------------------------------
# Call state machine
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """What the agent should say next, and whether the call is over."""

    reply: str
    end_call: bool
    outcome: Optional[str] = None
    branch: Optional[str] = None
    classification: Optional[Classification] = None
    collected: dict[str, str] = field(default_factory=dict)
    reprompted: bool = False


@dataclass
class ScenarioSession:
    """Drives one live call through its scenario's branch graph.

    Deliberately holds no I/O: the WebSocket handler (routers/voice_scenario.py)
    and the text-test endpoint both feed it utterances and act on TurnResult,
    which is what lets the exact same flow logic be exercised over HTTP with
    no phone, Twilio, or tunnel involved.
    """

    scenario_id: str
    language: str
    facts: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1

    collecting: Optional[str] = None
    _pending_branch: Optional[dict] = None
    reprompts: int = 0
    total_turns: int = 0
    finished: bool = False
    outcome: Optional[str] = None
    collected: dict[str, str] = field(default_factory=dict)
    # Conversation history for multi-turn context in classification.
    conversation_history: list[dict] = field(default_factory=list)
    # Kept on the session so the final TurnResult still reports it after a
    # two-step reason capture, where the closing turn is a verbatim capture
    # with no classification of its own.
    last_classification: Optional[Classification] = None

    def __post_init__(self) -> None:
        self.scenario = get_scenario(self.scenario_id)
        self.lang_cfg = get_language_config(self.scenario_id, self.language)
        if not self.scenario or not self.lang_cfg:
            raise ValueError(f"unknown scenario/language: {self.scenario_id}/{self.language}")

    # -- opening line -------------------------------------------------------

    def opening_line(self) -> str:
        key = "opening" if self.attempt <= 1 else "attempt_2_opening"
        template = self.lang_cfg.get(key) or self.lang_cfg["opening"]
        line = render(template, self.facts)
        # Record the opening in conversation history for classifier context.
        if not self.conversation_history:
            self.conversation_history.append({"role": "assistant", "content": line})
        return line

    # -- silence / no-answer ------------------------------------------------

    def on_silence(self) -> TurnResult:
        self.finished = True
        self.outcome = self.scenario.get("no_answer_outcome", "no_response")
        return TurnResult(
            reply=render(self.lang_cfg["silence_reply"], self.facts),
            end_call=True,
            outcome=self.outcome,
            branch="silence",
        )

    # -- main turn ---------------------------------------------------------

    async def handle_utterance(self, utterance: str) -> TurnResult:
        text = (utterance or "").strip()

        if self.finished:
            return TurnResult(reply="", end_call=True, outcome=self.outcome)

        if len(text) < _MIN_UTTERANCE_CHARS:
            return self.on_silence()

        self.total_turns += 1

        # Safety cap: prevent infinite conversation loops.
        if self.total_turns > MAX_TOTAL_TURNS:
            logger.warning(
                "[%s/%s] max turns (%d) exceeded, force-ending",
                self.scenario_id, self.language, MAX_TOTAL_TURNS,
            )
            return self.on_silence()

        # Record operator utterance in conversation history.
        self.conversation_history.append({"role": "user", "content": text})

        # Mid-flow free-text capture (e.g. the absence reason after a "no").
        # Captured verbatim and NOT re-classified - "because my wife is in
        # hospital" must not be re-read as a new branch.
        if self.collecting:
            slot = self.collecting
            branch = self._pending_branch or {}
            self.collected[slot] = text
            self.collecting = None
            self._pending_branch = None
            self.finished = True
            self.outcome = branch.get("outcome")
            reply = render(branch.get("after_collect_reply", ""), self.facts)
            self.conversation_history.append({"role": "assistant", "content": reply})
            logger.info(
                "[%s/%s] collected %s='%s' -> outcome=%s",
                self.scenario_id, self.language, slot, text, self.outcome,
            )
            return TurnResult(
                reply=reply,
                end_call=True,
                outcome=self.outcome,
                branch=branch.get("_key"),
                classification=self.last_classification,
                collected=dict(self.collected),
            )

        classification = await classify_utterance(
            self.scenario_id, self.language, text, self.facts,
            conversation_history=self.conversation_history,
        )
        self.last_classification = classification

        if classification.branch == "unclear" or classification.branch not in self.lang_cfg["branches"]:
            if self.reprompts < MAX_REPROMPTS_PER_ATTEMPT:
                self.reprompts += 1
                reprompt_reply = render(
                    self.lang_cfg.get("unclear_reprompt", self.lang_cfg["opening"]),
                    self.facts,
                )
                self.conversation_history.append({"role": "assistant", "content": reprompt_reply})
                return TurnResult(
                    reply=reprompt_reply,
                    end_call=False,
                    outcome="awaiting_response",
                    branch="unclear",
                    classification=classification,
                    reprompted=True,
                )
            result = self.on_silence()
            result.classification = classification
            return result

        branch = dict(self.lang_cfg["branches"][classification.branch])
        branch["_key"] = classification.branch

        # Branch collects free text (reason for absence) and the operator has
        # not already volunteered it in the same breath.
        if branch.get("collect") and not classification.volunteered_reason:
            self.collecting = branch["collect"]
            self._pending_branch = branch
            reply = render(branch["reply"], self.facts)
            self.conversation_history.append({"role": "assistant", "content": reply})
            logger.info(
                "[%s/%s] branch=%s -> collecting '%s', call stays open",
                self.scenario_id, self.language,
                classification.branch, branch["collect"],
            )
            return TurnResult(
                reply=reply,
                end_call=False,
                outcome="awaiting_" + branch["collect"],
                branch=classification.branch,
                classification=classification,
            )

        # Operator answered and volunteered the reason at once - skip the
        # extra question rather than asking for something they just told us.
        if branch.get("collect") and classification.volunteered_reason:
            self.collected[branch["collect"]] = classification.volunteered_reason
            self.finished = True
            self.outcome = branch["outcome"]
            reply = render(branch.get("after_collect_reply", branch["reply"]), self.facts)
            self.conversation_history.append({"role": "assistant", "content": reply})
            return TurnResult(
                reply=reply,
                end_call=True,
                outcome=self.outcome,
                branch=classification.branch,
                classification=classification,
                collected=dict(self.collected),
            )

        end_call = bool(branch.get("end_call", True))
        reply = render(branch["reply"], self.facts)
        self.conversation_history.append({"role": "assistant", "content": reply})

        if end_call:
            self.finished = True
            self.outcome = branch["outcome"]
        else:
            # Re-ask branches (repeat / asks_pay / asks_hours / why_calling)
            # keep the call open; reset the unclear counter since the operator
            # is engaged, just not answered yet.
            self.reprompts = 0
            logger.info(
                "[%s/%s] branch=%s (end_call=False), conversation continues",
                self.scenario_id, self.language, classification.branch,
            )

        return TurnResult(
            reply=reply,
            end_call=end_call,
            outcome=branch["outcome"],
            branch=classification.branch,
            classification=classification,
            collected=dict(self.collected),
        )
