"""Builds the system prompt for a Studio agent from its saved configuration.

The saved config is applied for real: every field below that the user fills in
ends up in the prompt the LLM receives on each turn (and the trace panel shows
which sections were loaded).
"""

from services.pipeline.chunker import SpeechChunker  # noqa: F401  (re-export for the processor)

LANGUAGE_NAMES = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}

# Voice-call rules are not user-editable: a phone/browser voice call breaks if
# the model answers with markdown, lists, emoji or long monologues.
VOICE_RULES = (
    "You are speaking on a live voice call. Reply in short, natural spoken sentences (usually 1-3). "
    "Never use markdown, bullet points, numbered lists, emoji, or stage directions. "
    "Write numbers and dates the way they are spoken. Do not read out URLs or symbols."
)

PERSONALITY_FIELDS = (
    ("role", "Role"),
    ("personality", "Personality"),
    ("communication_style", "Communication style"),
    ("objective", "Primary objective"),
    ("rules", "Rules and restrictions"),
    ("guidelines", "Conversation guidelines"),
    ("fallback", "Fallback responses"),
    ("escalation", "Escalation instructions"),
    ("unknown_questions", "When you do not know the answer"),
    ("ending", "How to end the conversation"),
)


def sections_loaded(agent: dict) -> list[str]:
    p = agent.get("personality") or {}
    names = [label for key, label in PERSONALITY_FIELDS if (p.get(key) or "").strip()]
    if (agent.get("system_instructions") or "").strip():
        names.insert(0, "System instructions")
    return names


def build_system_prompt(agent: dict, retrieved: list[dict] | None = None, *, language: str | None = None) -> str:
    lang = language or agent.get("language") or "hi"
    basic = agent.get("basic") or {}
    parts = [f"You are {agent['name']}, an AI voice agent."]
    if agent.get("description"):
        parts.append(agent["description"].strip())
    if basic.get("purpose"):
        parts.append(f"Purpose: {basic['purpose'].strip()}")
    if basic.get("objective"):
        parts.append(f"Conversation objective: {basic['objective'].strip()}")
    if (agent.get("system_instructions") or "").strip():
        parts.append(agent["system_instructions"].strip())
    p = agent.get("personality") or {}
    for key, label in PERSONALITY_FIELDS:
        val = (p.get(key) or "").strip()
        if val:
            parts.append(f"{label}:\n{val}")
    voice_cfg = agent.get("voice") or {}
    if voice_cfg.get("style"):
        parts.append(f"Speaking style: {voice_cfg['style']}.")
    if voice_cfg.get("tone"):
        parts.append(f"Emotional tone: {voice_cfg['tone']}.")
    parts.append(f"Speak in {LANGUAGE_NAMES.get(lang, lang)} unless the caller clearly switches language.")
    parts.append(VOICE_RULES)
    if retrieved:
        facts = "\n\n".join(f"[{r['source']}] {r['text']}" for r in retrieved)
        parts.append(
            "Reference material from your training. When it answers the caller, state its facts EXACTLY as written "
            "- names, places, numbers, times and prices must not be changed, translated or guessed - and keep your "
            "answer short. "
            "Do not mention that you were given it. If it does not cover the question, follow the instructions "
            "above for unknown questions instead of guessing:\n\n" + facts
        )
    else:
        parts.append("No reference material matched this question; answer from your instructions only and do not invent facts.")
    return "\n\n".join(parts)


def greeting(agent: dict) -> str:
    """Opening line: the user's Opening script if present, else a neutral default."""
    opening = ((agent.get("personality") or {}).get("greeting") or "").strip()
    if opening:
        return opening
    lang = agent.get("language") or "hi"
    name = agent["name"]
    return {
        "hi": f"नमस्ते, मैं {name} बोल रही हूँ। मैं आपकी कैसे मदद कर सकती हूँ?",
        "gu": f"નમસ્તે, હું {name} બોલી રહી છું. હું તમારી કેવી રીતે મદદ કરી શકું?",
    }.get(lang, f"Hello, this is {name}. How can I help you today?")
