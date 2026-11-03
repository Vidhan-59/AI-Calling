"""What the voice controls in the Agent Studio can REALLY do, derived from what
the OmniVoice integration exposes (tts_service/app.py, services/tts/client.py).

OmniVoice takes, per request: text, language (hi/gu/en), a voice from the
library, and speed (0.5-2.0). A voice's gender / age / pitch / accent are
chosen when the voice is *designed* (a design voice is rendered once from
those attributes and then cloned, so it sounds identical on every call).
It has no stability / clarity / expressiveness / emotion / dialect parameter.

The UI shows this list next to the controls so a setting that does nothing is
labelled as such instead of being a slider that pretends to work.
"""

SUPPORTED = "supported"             # changes the audio on every request
CREATION_ONLY = "creation_only"     # fixed when a voice is created, not per call
VIA_INSTRUCTIONS = "via_instructions"  # shapes the agent's wording (LLM prompt), not the audio
UNSUPPORTED = "unsupported"         # OmniVoice has no such parameter

CAPABILITIES = [
    {"key": "voice", "label": "Voice", "status": SUPPORTED,
     "detail": "Any voice in the OmniVoice library (cloned from a sample, or designed from attributes)."},
    {"key": "language", "label": "Language", "status": SUPPORTED,
     "detail": "Hindi, Gujarati and English."},
    {"key": "speed", "label": "Speaking speed", "status": SUPPORTED,
     "detail": "0.5x to 2.0x, applied to every request."},
    {"key": "gender", "label": "Gender", "status": CREATION_ONLY,
     "detail": "Chosen when a voice is designed (male / female). Create a new voice to change it."},
    {"key": "age", "label": "Age", "status": CREATION_ONLY,
     "detail": "young adult / middle-aged / elderly, chosen when a voice is designed."},
    {"key": "pitch", "label": "Pitch", "status": CREATION_ONLY,
     "detail": "Five levels from very low to very high, chosen when a voice is designed - not adjustable per call."},
    {"key": "accent", "label": "Accent", "status": CREATION_ONLY,
     "detail": "English voices only (Indian / British / American / Australian), chosen when a voice is designed."},
    {"key": "style", "label": "Speaking style", "status": VIA_INSTRUCTIONS,
     "detail": "Shapes how the agent words its replies (friendly, calm, authoritative...). The synthesized audio itself is not restyled."},
    {"key": "tone", "label": "Emotional tone", "status": VIA_INSTRUCTIONS,
     "detail": "Same: guides the wording of replies. OmniVoice has no emotion control."},
    {"key": "stability", "label": "Voice stability", "status": UNSUPPORTED,
     "detail": "OmniVoice exposes no stability parameter."},
    {"key": "clarity", "label": "Voice clarity", "status": UNSUPPORTED,
     "detail": "No such parameter. (Synthesis quality is a server-wide setting, TTS_LIVE_NUM_STEP, traded against latency.)"},
    {"key": "expressiveness", "label": "Expressiveness", "status": UNSUPPORTED,
     "detail": "OmniVoice exposes no expressiveness parameter."},
    {"key": "regional_pronunciation", "label": "Regional pronunciation", "status": UNSUPPORTED,
     "detail": "No dialect control; pronunciation follows the language and the voice's own sample."},
]

STYLES = ("professional", "friendly", "conversational", "energetic", "calm", "authoritative", "empathetic")
TONES = ("neutral", "warm", "upbeat", "reassuring", "serious", "apologetic")

# Vocabulary OmniVoice accepts for voice design (see static/voice_studio.html).
DESIGN_OPTIONS = {
    "gender": ["female", "male"],
    "age": ["young adult", "middle-aged", "elderly"],
    "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
    "accent": ["indian accent", "british accent", "american accent", "australian accent"],
}


def design_instruct(gender: str | None, age: str | None, pitch: str | None, accent: str | None, language: str) -> str:
    """The `instruct` string for POST /voices/design. Only vocabulary OmniVoice
    accepts; accent is English-only, so it is dropped for other languages."""
    parts = []
    for value, allowed in ((gender, "gender"), (age, "age"), (pitch, "pitch")):
        if value:
            if value not in DESIGN_OPTIONS[allowed]:
                raise ValueError(f"{allowed} must be one of {DESIGN_OPTIONS[allowed]}")
            parts.append(value)
    if accent and language == "en":
        if accent not in DESIGN_OPTIONS["accent"]:
            raise ValueError(f"accent must be one of {DESIGN_OPTIONS['accent']}")
        parts.append(accent)
    if not parts:
        raise ValueError("choose at least one attribute")
    return ", ".join(parts)
