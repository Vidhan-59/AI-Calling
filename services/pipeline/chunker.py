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


# Where a streamed reply may be cut after the first chunk: sentence ends, plus
# commas/semicolons/colons - a clause is a natural place to breathe, so the
# TTS cut there doesn't sound like one.
_CLAUSE_END = re.compile(r"[.!?।॥,;:]\s")


class SpeechChunker:
    """Turns an LLM token stream into pieces to speak, as early as possible
    (Layer 1, "first 3-4 tokens -> audio starts").

    The FIRST piece is released as soon as it has `first_chunk_words` whole
    words (or hits a clause/sentence end first) - TTS starts on "Namaste! Main
    aapki kaise" while the LLM is still writing "madad kar sakti hoon?". Every
    later piece is cut at a clause or sentence boundary, so the rest of the
    reply keeps natural phrasing. Whitespace-delimited words, not model
    tokens: a Devanagari word can be 3-6 tokens, and half a word can't be
    spoken.
    """

    def __init__(self, first_chunk_words: int = 4, max_chunk_chars: int = 140, min_chunk_words: int = 1,
                 budget_words=None):
        self._first_words = max(1, first_chunk_words)
        # Pieces after the first are merged up to this many words before being
        # released: every piece boundary is a chance for a gap (OmniVoice
        # synthesizes ~in real time, see services/tts/playout.py), so fewer,
        # longer pieces keep the audio continuous. A sentence end still
        # releases at half this, so a short sentence isn't held back long.
        self._min_words = max(1, min_chunk_words)
        # Optional () -> int | None: the most words the next piece may have for
        # its synthesis to finish before the audio already queued runs out
        # (PlayoutPlanner.next_budget_words). None = this piece starts a reply.
        self._budget = budget_words
        self._max_chars = max_chunk_chars
        self._buf = ""
        self._emitted_any = False

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            cut = self._next_cut()
            if cut is None:
                return out
            piece, self._buf = self._buf[:cut].strip(), self._buf[cut:].lstrip()
            if piece:
                out.append(piece)
                self._emitted_any = True

    def flush(self) -> list[str]:
        piece, self._buf = self._buf.strip(), ""
        if piece:
            self._emitted_any = True
            return [piece]
        return []

    def _next_cut(self) -> int | None:
        cap = self._budget() if self._budget else None
        min_words = self._min_words if self._emitted_any else 1
        if cap is not None:
            min_words = min(min_words, cap)
        for m in _CLAUSE_END.finditer(self._buf):
            n = len(self._buf[:m.end()].split())
            if cap is not None and n > cap:
                break
            final = self._buf[m.start()] in ".!?।॥"
            if n >= min_words or (final and n >= max(2, min_words // 2)):
                return m.end()
        if cap is not None:
            words = self._buf.split()
            complete = len(words) if self._buf[-1:].isspace() else len(words) - 1
            if complete >= cap:
                return _end_of_nth_word(self._buf, cap)
        if not self._emitted_any:
            # Only complete words: the last one counts once whitespace follows it.
            words = self._buf.split()
            complete = len(words) if self._buf[-1:].isspace() else len(words) - 1
            if complete >= self._first_words:
                return _end_of_nth_word(self._buf, self._first_words)
        if len(self._buf) > self._max_chars:
            # A runaway clause with no punctuation: cut at the last space.
            space = self._buf.rfind(" ", 0, self._max_chars)
            return space + 1 if space > 0 else None
        return None


def _end_of_nth_word(text: str, n: int) -> int:
    for i, m in enumerate(re.finditer(r"\S+", text), start=1):
        if i == n:
            return m.end()
    return len(text)
