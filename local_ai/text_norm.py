"""Text normalization between LLM output and TTS input.

Deliberately rule-based rather than a spaCy pipeline: the intent set here is
small (strip markdown, expand a few number/time patterns, split sentences),
and a full NLP dependency would add real weight and startup latency for
almost no benefit at this scope. If the intent/entity extraction needs grow
beyond simple regex rules, that is the point to bring in spaCy - not before.
"""

import re

_MARKDOWN_PATTERNS = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),  # **bold**
    (re.compile(r"\*(.+?)\*"), r"\1"),  # *italic*
    (re.compile(r"`(.+?)`"), r"\1"),  # `code`
    (re.compile(r"^#{1,6}\s*", re.MULTILINE), ""),  # # headings
    (re.compile(r"^[-*+]\s+", re.MULTILINE), ""),  # - bullet
    (re.compile(r"\[(.+?)\]\(.+?\)"), r"\1"),  # [text](url)
]

# 24h time -> spoken form, English only today. "14:00" -> "2 PM", "09:30" -> "9:30 AM".
_TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def strip_markdown(text: str) -> str:
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


def _expand_time(match: re.Match) -> str:
    hour, minute = int(match.group(1)), match.group(2)
    period = "AM" if hour < 12 else "PM"
    spoken_hour = hour % 12 or 12
    return f"{spoken_hour} {period}" if minute == "00" else f"{spoken_hour}:{minute} {period}"


def expand_times(text: str, language: str = "en") -> str:
    """24h clock times -> spoken form. English only - Hindi/Gujarati verbal
    time expressions (see scenarios/*.json's `time` facts) are supplied
    pre-spoken by the caller rather than normalized here; see
    services/scenario_engine.py's `render()` and POSTMAN_GUIDE.md's "two
    rules for writing facts" for why that responsibility sits with the
    caller, not this function."""
    if language != "en":
        return text
    return _TIME_PATTERN.sub(_expand_time, text)


def split_sentences(text: str) -> list[str]:
    """Sentence-level chunks, for TTS engines that synthesize better per
    sentence than on one long paragraph. Handles '.', '?', '!' plus the
    Hindi/Gujarati sentence terminator '।' (devanagari danda)."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?।])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def normalize_for_speech(text: str, language: str = "en") -> str:
    """The full pass: strip markdown, expand times, collapse whitespace.
    Call this on any LLM output before handing it to a TTSEngine."""
    text = strip_markdown(text)
    text = expand_times(text, language)
    text = re.sub(r"\s+", " ", text).strip()
    return text
