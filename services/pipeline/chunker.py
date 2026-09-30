"""Sentence-boundary chunking for text-to-speech output.

services/claude_client.py::generate_conversation_turn is non-streaming (the
Conversation Agent needs the full response before it can decide whether a
tool call needs a resume round - see that function's docstring), so the
final text is already complete by the time routers/agent_voice.py has it.
Sending it to ConversationRelay as one WebSocket message means Twilio's TTS
doesn't start synthesizing until the entire reply has been generated *and*
transmitted. Splitting on sentence boundaries and sending each chunk as its
own message lets TTS start speaking the first sentence while later ones are
still being sent, the same token-arrives-as-spoken behavior
routers/voice.py already gets for free from Claude's native token stream.
"""

import re

# Danda (।) and double danda (॥) are the Hindi/Gujarati sentence-final
# punctuation marks; kept alongside the Latin '.', '!', '?' so chunking works
# the same regardless of which language the persona is speaking.
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+")


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentence-ish chunks on whitespace that follows a
    sentence-final punctuation mark. Not full NLP - a deliberately simple
    heuristic that never blocks on find a chunk boundary: text with no
    sentence-final punctuation at all is returned as a single chunk."""
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_END.split(text)]
    return [p for p in parts if p]
