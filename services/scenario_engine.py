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

from config import BASE_DIR, settings
from services import claude_client

logger = logging.getLogger("scenario_engine")

SCENARIOS_DIR = os.path.join(BASE_DIR, "scenarios")

# Utterances shorter than this after stripping are treated as noise, not speech.
_MIN_UTTERANCE_CHARS = 2

# How many times we re-ask on an unclear answer before giving up on the turn.
MAX_REPROMPTS_PER_ATTEMPT = 2

# Max total conversational turns before force-ending (safety cap for runaway loops).
MAX_TOTAL_TURNS = 10
# After a "no" the agent asks why, then re-asks up to this many times if the
# answer has no reason in it (a dangling "nahi aa paunga", a pause-split
# fragment). Then it records what it has - it can never loop forever.
MAX_REASON_REASKS = 2

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


def _classification_schema(branch_keys: list[str], collects_reason: bool, lean: bool = False) -> dict:
    """`lean=True` is the local-model (Qwen) schema: only the fields that
    drive a decision. Measured 2026-09-30 on qwen2.5:7b-instruct: with the
    free-text `rationale` field the model writes essays until it hits the
    token cap - 400 tokens, ~17s, and the JSON gets truncated mid-string
    (the "Unterminated string ... line 5" error live calls hit). Without it:
    ~50 tokens, ~2s, valid every time. detected_language is dropped too -
    it's analytics only, and the call's own language is used instead."""
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
    }
    required = ["branch"]
    if lean:
        # Only a boolean for the reason, never the reason text: having the
        # model re-type the caller's words cost ~25 Devanagari tokens (~0.8s)
        # and is how it produced a fake "reason" from the opening's own date
        # phrase on a Gujarati call. The caller's utterance is saved verbatim
        # instead (see classify_utterance). Measured: ~16 output tokens, 0.4-0.7s.
        if collects_reason:
            properties["gave_reason"] = {
                "type": "boolean",
                "description": (
                    "True only if the operator stated WHY they cannot attend in this "
                    "same utterance (e.g. illness, family emergency). A bare refusal "
                    "('nahi aa paunga') is false."
                ),
            }
            required.append("gave_reason")
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}

    properties["confidence"] = {"type": "number", "description": "0.0 to 1.0 confidence in the chosen branch."}
    required.append("confidence")
    properties["detected_language"] = {
        "type": "string",
        "enum": ["en", "hi", "gu", "mixed", "unknown"],
        "description": "Language the operator actually spoke in.",
    }
    properties["rationale"] = {
        "type": "string",
        "description": "One short sentence explaining the choice, in English, for the audit log.",
    }
    required += ["detected_language", "rationale"]

    if collects_reason:
        # Plain string with "" for absent rather than a nullable union - keeps
        # the schema inside the guaranteed-supported structured-output subset.
        properties["volunteered_reason"] = {
            "type": "string",
            "description": (
                "ONLY the cause of the absence, if the operator stated one in this same "
                "utterance - e.g. for 'no, my wife is in hospital' return 'my wife is in "
                "hospital'. A restatement of the refusal ('I can't come', 'nahi aa "
                "paunga') is NOT a reason - return an empty string for it."
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
    # Order matters for latency: everything that is the same for every call of
    # this scenario+language comes first, per-call content (facts, history)
    # last. Ollama reuses the KV cache for an identical prompt prefix, so this
    # layout lets ~1000 tokens be cached across calls instead of re-processed
    # (measured: 870-930ms prompt eval uncached vs ~60-100ms cached).
    lines = [
        "You classify what a port equipment operator said during an automated "
        "ShiftEaze workforce phone call, onto exactly one scripted branch.",
        "",
        f"CALL TYPE: {scenario['display_name']} ({scenario.get('trigger', '')})",
        f"LANGUAGE THE AGENT SPOKE: {lang_cfg.get('language_name')}",
        "",
    ]

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
        "- A correction reverses it: 'nahi nahi, main aa jaunga' ('no no, I will come') "
        "means YES - the operator is correcting a misunderstanding, not refusing.",
        "- A refusal followed by a personal reason ('na, mari tabiyat sari nathi' / "
        "'nahi, tabiyat kharab hai' - no, my health is not good) is a NO with a reason. "
        "'dispute' is ONLY when the operator says the shift details themselves are "
        "wrong or not theirs ('yeh meri duty nahi hai').",
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
        "",
        "THE INITIAL QUESTION:",
        render(lang_cfg["opening"], facts),
        "",
    ]

    if conversation_history:
        lines.append("CONVERSATION SO FAR (most recent last):")
        for turn in conversation_history[-6:]:
            role_label = "AGENT" if turn["role"] == "assistant" else "OPERATOR"
            lines.append(f"  {role_label}: {turn['content']}")
        lines.append("")
    lines.append("THE OPERATOR'S LATEST UTTERANCE (the user message) IS WHAT YOU MUST CLASSIFY.")
    return "\n".join(lines)


@dataclass
class Classification:
    branch: str
    confidence: float
    detected_language: str
    rationale: str
    volunteered_reason: Optional[str] = None
    source: str = "llm"  # "fast" (keyword fast path) | "llm"


# ---------------------------------------------------------------------------
# Fast path: plain yes/no answers, classified by keyword with no LLM call
# ---------------------------------------------------------------------------
# Most real answers to "aa rahe hain?" are two or three words ("haan ji",
# "nahi aa paunga"). Sending those to the LLM costs ~1-2s of dead air for no
# gain. Anything this can't classify with certainty - an unknown word, both
# polarities at once ("haan nahi aa paunga"), a question, a reason - falls
# through to the LLM, so the fast path can only ever be skipped, never wrong
# in a way the LLM would have caught.

# Apostrophes are kept so "can't" stays one token (curly ones are normalized first).
_PUNCT_RE = re.compile(r"[।॥,.!?;:\"()\[\]\-–—…]+")

# Devanagari "ना" is deliberately absent: in Hindi it is also a tag particle
# ("aa raha hoon na") that means the opposite. Gujarati "ના" is unambiguous.
_NEG = {
    "नहीं", "नही", "नहि", "नई", "मत", "नो",
    "nahi", "nahin", "nahee", "nai", "no", "nope", "not", "cannot", "can't", "cant", "won't", "wont",
    "નહીં", "નહિ", "ના", "નથી", "nathi",
}
_AFF = {
    "हाँ", "हां", "हा", "हाँजी", "हांजी", "ठीक", "बिल्कुल", "बिलकुल", "पक्का", "ज़रूर", "जरूर", "यस", "ओके",
    "haan", "han", "haa", "ha", "haanji", "hanji", "yes", "yeah", "yep", "ok", "okay", "sure",
    "theek", "thik", "bilkul", "pakka", "zaroor", "jarur",
    "હા", "હાં", "બરાબર", "ચોક્કસ",
}
# Attendance verbs: affirmative alone ("aa jaunga"), negated when a refusal
# word is in scope ("nahi aaunga").
_ATTEND = {
    "आऊँगा", "आऊंगा", "आऊँगी", "आऊंगी", "जाऊँगा", "जाऊंगा", "जाऊँगी", "जाऊंगी", "आएंगे", "आएँगे",
    "आऊँ", "आऊं", "जाऊँ", "जाऊं", "आएगा", "आएगी", "आता", "आती", "आते",
    "रहा", "रही", "रहे", "पहुंचूंगा", "पहुँचूँगा", "पहुंचूंगी", "पहुँचूँगी",
    "aaunga", "aaungi", "aunga", "jaunga", "jaungi", "aata", "aati", "raha", "rahi",
    "coming", "come", "attend", "make",
    "આવીશ", "આવીશું", "આવું", "આવુ", "આવી", "આવતો", "આવતી", "આવતા", "આવશે", "આવવાનો", "આવવાની",
    "પહોંચીશ", "aavish", "aavu", "aavto",
}
# Auxiliaries that keep a preceding refusal in scope: "nahi [aa paunga]".
_SCOPE_KEEP = {
    "आ", "पाऊँगा", "पाऊंगा", "पाऊँगी", "पाऊंगी", "सकता", "सकती", "सकूंगा", "सकूँगा",
    "पाऊँ", "पाऊं", "सकूंगी", "सकूँगी", "पाएगा", "पाएगी",
    "a", "aa", "paunga", "paaunga", "paungi", "sakta", "sakti", "be", "able", "to",
    "શકું", "શકીશ", "શકીશું",
}
# "can / be able to": after an attendance verb these mean "I can('t) come", so
# a doubled refusal with one of them is a REFUSAL, not a "no no, I'll come"
# correction (see fast_classify).
_CAN = {
    "पाऊँगा", "पाऊंगा", "पाऊँगी", "पाऊंगी", "सकता", "सकती", "सकूंगा", "सकूँगा", "पाऊँ", "पाऊं",
    "सकूंगी", "सकूँगी", "पाएगा", "पाएगी", "paunga", "paaunga", "paungi", "sakta", "sakti", "able",
    "શકું", "શકીશ", "શકીશું",
}
# No yes/no meaning; also end a refusal's scope ("nahi nahi, [main] aa jaunga").
_NEUTRAL = {
    "जी", "मैं", "में", "हूँ", "हूं", "है", "हैं", "तो", "भी", "कल", "आज", "पर", "पे", "ड्यूटी", "शिफ्ट",
    "भाई", "सर", "साहब", "मैडम", "ना", "हम", "ही", "यार", "सॉरी", "माफ़", "माफ", "कीजिए", "कीजिये", "सच", "देखिए", "देखिये", "देखो",
    "ji", "main", "mai", "mein", "hoon", "hu", "hai", "bhi", "kal", "aaj", "par", "pe", "duty", "shift",
    "bhai", "sir", "madam", "na", "i", "am", "i'm", "im", "will", "tomorrow", "today", "the", "it", "i'll", "ill",
    "yaar", "yar", "sorry", "maaf", "kijiye", "dekhiye", "dekho",
    "હું", "છું", "છે", "કાલે", "આજે", "જી", "ભાઈ", "સાહેબ", "ડ્યુટી", "શિફ્ટ", "રોજ", "ને", "માફ", "કરજો", "છો", "હતો", "તો",
}
_FILLER = _NEG | _AFF | _ATTEND | _SCOPE_KEEP | _NEUTRAL
_FAST_PATH_MAX_WORDS = 6


def _tokens(text: str) -> list[str]:
    text = (text or "").lower().replace("’", "'").replace("‘", "'")
    return [t.strip("'") for t in _PUNCT_RE.sub(" ", text).split() if t.strip("'")]


def fast_classify(utterance: str, branch_keys: list[str]) -> str | None:
    """A branch key for a short, unambiguous answer; None means "ask the
    LLM". yes/no scenarios get "yes"/"no"; reminder scenarios (an
    "acknowledged" branch, no yes/no) get "acknowledged" for a plain
    "theek hai" - anything negative there needs the LLM to pick between
    already_punched / technical_issue / still_working."""
    has_yes_no = "yes" in branch_keys and "no" in branch_keys
    if not has_yes_no:
        if "acknowledged" not in branch_keys:
            return None
        polarity = fast_classify(utterance, ["yes", "no"])
        return "acknowledged" if polarity == "yes" else None
    tokens = _tokens(utterance)
    if not tokens or len(tokens) > _FAST_PATH_MAX_WORDS:
        return None
    if tokens in (["जी"], ["ji"], ["જી"]):
        return "yes"

    # "nahi nahi, main aa jaunga" ("no no, I'll come") is a correction meaning
    # YES. Found by tools/stt_benchmark.py: local STT drops the comma Sarvam
    # used to add, and without it Qwen read the correction as a refusal - on
    # every voice. A doubled refusal word opening the answer, followed by an
    # un-negated attendance verb, is unambiguous enough to decide here.
    # Simulator finding: STT also drops one of the two words ("नहीं मैं आ
    # जाऊंगा"), so a single refusal word counts too - but only when a
    # subject/filler word ("मैं", "कल", "हું") separates it from an attendance
    # verb, which keeps "नहीं आऊंगा" (won't come) a refusal.
    n_neg = 0
    while n_neg < len(tokens) and tokens[n_neg] in _NEG:
        n_neg += 1
    rest = tokens[n_neg:]
    # A "can/able" auxiliary (शकું / सकता / पाऊँगा / paunga) after the attendance
    # verb means "I can('t) come" - live failure: Gujarati "ના, નહીં આવી શકું"
    # ("no, I can't come") matched this correction rule and was recorded as
    # CONFIRMED, so the reason question was never asked. Those go to the
    # normal scope logic below, where the refusal stays in scope -> "no".
    if (
        n_neg
        and len(rest) >= 2
        and not any(t in _NEG for t in rest)
        and all(t in _FILLER for t in rest)
        and not any(t in _CAN for t in rest)
    ):
        if n_neg >= 2 and any(t in _ATTEND or t in _AFF for t in rest):
            return "yes"
        if n_neg == 1 and rest[0] in _NEUTRAL and any(t in _ATTEND for t in rest):
            return "yes"

    saw_neg = saw_aff = False
    neg_in_scope = False
    for tok in tokens:
        if tok in _NEG:
            saw_neg = neg_in_scope = True
        elif tok in _AFF:
            saw_aff, neg_in_scope = True, False
        elif tok in _ATTEND:
            if not neg_in_scope:
                saw_aff = True
            neg_in_scope = False
        elif tok in _SCOPE_KEEP:
            pass
        elif tok in _NEUTRAL:
            neg_in_scope = False
        else:
            return None  # a word we don't know - could be a reason, a question, a dispute

    if saw_neg and not saw_aff:
        return "no"
    if saw_aff and not saw_neg:
        return "yes"
    if not saw_neg and not saw_aff:
        # Every word is filler ("हूँ", "मैं", "કાલે") - nothing that answers
        # the question. Seen with local STT on noisy audio: a fragment like
        # "હું" ("I"), which Qwen then guessed into a confident "no". Asking
        # again is right; a guess records a wrong outcome.
        return "unclear"
    return None  # both polarities ("haan nahi aa paunga") - LLM decides


def is_real_reason(text: str | None) -> bool:
    """False for a restatement of the refusal ("kal nahi aa paunga") that a
    model returned as a "volunteered reason" - observed live on qwen2.5,
    and it silently skipped the reason question. True once there's any word
    that isn't refusal/attendance filler ("bukhar hai", "wife hospital")."""
    return any(tok not in _FILLER for tok in _tokens(text or ""))


async def warm_classifier(session: "ScenarioSession") -> None:
    """Process this call's classifier prompt into Ollama's KV cache while the
    opening plays, so the first LLM turn only pays for the caller's words
    (~60-100ms) instead of the whole ~1200-token prompt (~900ms, measured).
    Best-effort and local-backend only; never raises."""
    if os.getenv("SCENARIO_LLM_BACKEND", "claude") != "local":
        return
    try:
        from local_ai import llm_engine as local_llm_engine

        branch_keys = list(session.lang_cfg["branches"].keys())
        collects_reason = any("collect" in b for b in session.lang_cfg["branches"].values())
        prompt = _build_classifier_prompt(
            session.scenario, session.lang_cfg, session.facts,
            conversation_history=session.conversation_history,
        )
        await local_llm_engine.prime_prompt(
            prompt, _classification_schema(branch_keys, collects_reason, lean=True)
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("classifier warm-up failed (non-fatal): %s", e)


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

    fast = fast_classify(utterance, branch_keys)
    if fast is not None:
        return Classification(
            branch=fast,
            confidence=1.0,
            detected_language=language,
            rationale="fast-path keyword match",
            volunteered_reason=None,
            source="fast",
        )

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

    lean = backend == "local"
    extra = {"max_tokens": 40} if lean else {}
    result = await classify_fn(
        system_prompt=prompt,
        user_text=utterance,
        schema=_classification_schema(branch_keys, collects_reason, lean=lean),
        **extra,
    )

    if lean:
        # The caller's own words, verbatim, when the model says a reason was
        # given - and only if there's more in them than a restated refusal.
        volunteered_reason = utterance.strip() if result.get("gave_reason") else None
    else:
        volunteered_reason = result.get("volunteered_reason") or None
    if volunteered_reason is not None and not is_real_reason(volunteered_reason):
        logger.info(
            "[%s/%s] ignoring volunteered_reason=%r - a restatement of the refusal, not a "
            "reason; the reason question will be asked",
            scenario_id, language, volunteered_reason,
        )
        volunteered_reason = None

    return Classification(
        branch=result.get("branch", "unclear"),
        confidence=float(result.get("confidence", 0.0)),
        detected_language=result.get("detected_language", language),
        rationale=result.get("rationale", ""),
        volunteered_reason=volunteered_reason,
        source="llm",
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
    _reason_asks: int = 0  # re-asks of the reason question so far
    _prelim_reason: Optional[str] = None  # reason volunteered with the "no", kept as a fallback
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
            # Simulator finding: after a pause, the rest of the refusal sentence
            # ("ના, ... હું કાલે નહીં આવી શકું") arrived just after the bot asked
            # why, and was saved as the reason - then the real reason came after
            # the call had closed. A reply with no reason content gets one gentle
            # re-ask; the second time it's recorded as-is so it can't loop.
            if "reason" in slot and not is_real_reason(text) and self._reason_asks < MAX_REASON_REASKS:
                self._reason_asks += 1
                reply = render(self.lang_cfg.get("reason_reprompt") or branch.get("reply", ""), self.facts)
                self.conversation_history.append({"role": "assistant", "content": reply})
                logger.info("[%s/%s] %r has no reason in it - asking once more", self.scenario_id, self.language, text)
                return TurnResult(
                    reply=reply, end_call=False, outcome="awaiting_response",
                    branch="reason_reprompt", classification=self.last_classification,
                )
            if "reason" in slot and not is_real_reason(text) and self._prelim_reason:
                text = self._prelim_reason  # they did say why with the "no"; the answer added nothing
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
        if branch.get("collect") and (settings.scenario_always_ask_reason or not classification.volunteered_reason):
            self.collecting = branch["collect"]
            self._pending_branch = branch
            self._prelim_reason = classification.volunteered_reason
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
