"""Playout planning and diagnostics for streamed OmniVoice replies.

OmniVoice synthesizes a whole utterance per request, on one GPU worker. Cost
is affine in the audio length: synth = FIXED + SLOPE x audio, with a large
fixed part (measured on this laptop's RTX 5060, warm: ~0.75s + 0.12 x audio at
10 diffusion steps, ~0.7s + 0.7 x audio at 16). A streamed reply is a series of
pieces; piece N+1 is only ready that long after piece N finished synthesizing.
If that exceeds the audio still queued from piece N, the caller hears silence
in the middle of the answer - the "voice breaks".

Because of the fixed part, many tiny pieces are the WORST case (each pays the
full fixed cost for ~1s of audio) and a few long ones the best: so the planner
sizes pieces by what the buffer can afford - a 3-word first piece for latency,
then each next piece as long as its synthesis still lands in time, which
grows quickly (the buffer gains audio - synth per piece).

PlayoutPlanner models the caller-side buffer (audio synthesized but not yet
played) and picks the diffusion step count per piece so the next piece lands
before the buffer runs dry, falling back to the cheapest setting rather than
letting it underrun. It also records what actually happened (underruns, gap
lengths, synth time per piece, time to first audio) for the diagnostics API.

Nothing here touches the network or the GPU: it is driven by the TTS service
wrapper (pipecat_service.py) and unit-testable with a fake clock.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

WORDS_PER_SEC = 2.6  # Hindi at the default pace: 7 words -> ~2.1-2.7s measured

# Quality ladder, best first. 10 is the floor the app already uses for short
# pieces; going lower audibly degrades OmniVoice, so a piece that still can't
# keep up at the floor underruns and is reported instead of degraded further.
STEP_LADDER = (16, 12, 10)
# New reply if the previous one has been fully played for this long.
REPLY_IDLE_SECS = 1.0
# (fixed secs, secs per second of audio) per step count, until real
# measurements exist: probe results on the target GPU, warm and otherwise idle.
# Fitted online from every real synthesis (see _Fit), so contention with the
# LLM on the same GPU shows up in the estimates within a few pieces.
_SEED = {16: (0.7, 0.7), 12: (0.75, 0.35), 10: (0.75, 0.12)}
_FIT_WINDOW = 24
# Safety margin: require predicted synth time <= buffered audio * this.
_HEADROOM = 0.85


@dataclass
class PieceRecord:
    ts: float
    owner: str
    chars: int
    steps: int
    synth_ms: float
    audio_ms: float
    cached: bool
    first_of_reply: bool
    gap_ms: float  # silence before this piece began playing (0 = continuous)
    buffer_ms: float  # audio still queued when this piece was ready


@dataclass
class Diagnostics:
    pieces: deque = field(default_factory=lambda: deque(maxlen=500))
    replies: int = 0
    underruns: int = 0
    interruptions: int = 0
    dropped_pieces: int = 0
    tts_errors: int = 0
    first_audio_ms: deque = field(default_factory=lambda: deque(maxlen=200))

    def summary(self) -> dict:
        live = [p for p in self.pieces if not p.cached]
        synth = sorted(p.synth_ms for p in live)

        def pct(q: float) -> float | None:
            return round(synth[min(len(synth) - 1, int(q * len(synth)))], 1) if synth else None

        gaps = [p.gap_ms for p in self.pieces if p.gap_ms > 0 and not p.first_of_reply]
        first = sorted(self.first_audio_ms)
        return {
            "replies": self.replies,
            "pieces": len(self.pieces),
            "cache_hits": sum(1 for p in self.pieces if p.cached),
            "underruns": self.underruns,
            "mean_gap_ms": round(sum(gaps) / len(gaps), 1) if gaps else 0.0,
            "max_gap_ms": round(max(gaps), 1) if gaps else 0.0,
            "interruptions": self.interruptions,
            "dropped_pieces": self.dropped_pieces,
            "tts_errors": self.tts_errors,
            "synth_ms_p50": pct(0.5),
            "synth_ms_p95": pct(0.95),
            "first_audio_ms_p50": round(first[len(first) // 2], 1) if first else None,
            "avg_steps": round(sum(p.steps for p in live) / len(live), 1) if live else None,
        }


DIAGNOSTICS = Diagnostics()


class _Fit:
    """synth_secs ~= fixed + slope * audio_secs, from recent real synthesis."""

    def __init__(self, seed: tuple[float, float]):
        self.fixed, self.slope = seed
        self._seed = seed
        self._obs: deque[tuple[float, float]] = deque(maxlen=_FIT_WINDOW)

    def add(self, audio: float, synth: float) -> None:
        self._obs.append((audio, synth))
        n = len(self._obs)
        if n < 4:
            return
        mx = sum(a for a, _ in self._obs) / n
        my = sum(y for _, y in self._obs) / n
        var = sum((a - mx) ** 2 for a, _ in self._obs)
        if var > 0.25 * n:  # enough spread in lengths to fit a line
            slope = sum((a - mx) * (y - my) for a, y in self._obs) / var
            self.slope = min(1.5, max(0.02, slope))
            self.fixed = min(3.0, max(0.1, my - self.slope * mx))
        else:  # all about the same length: keep the seed's fixed cost, learn the slope from the mean
            self.fixed = self._seed[0]
            self.slope = min(1.5, max(0.02, (my - self.fixed) / max(mx, 0.3)))

    def estimate(self, audio: float) -> float:
        return self.fixed + self.slope * audio


class PlayoutPlanner:
    """One per TTS service instance (= one per call)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, diagnostics: Diagnostics | None = None):
        self._clock = clock
        self._diag = diagnostics if diagnostics is not None else DIAGNOSTICS
        self._fit = {steps: _Fit(seed) for steps, seed in _SEED.items()}
        self._play_end = 0.0  # when the audio queued so far finishes playing
        self._first_text_at: float | None = None
        self._cur_first = True  # is the piece being synthesized the first of its reply?
        # Pieces handed to TTS (release) but not yet synthesized, oldest first:
        # (estimated audio secs, estimated synth secs). The head is the one in flight.
        self._queue: deque[tuple[float, float]] = deque()
        self._head_started: float | None = None

    # -- planning ---------------------------------------------------------

    def buffered_secs(self) -> float:
        return max(0.0, self._play_end - self._clock())

    def estimate_audio_secs(self, text: str) -> float:
        return max(0.6, len(text.split()) / WORDS_PER_SEC)

    # -- piece sizing (chunker side) ----------------------------------------

    def release(self, text: str) -> None:
        """A reply piece was handed to the TTS service (not yet synthesized)."""
        audio = self.estimate_audio_secs(text)
        self._queue.append((audio, self.estimate_synth_secs(STEP_LADDER[-1], audio) * 1.15))

    def estimate_synth_secs(self, steps: int, audio_secs: float) -> float:
        return self._fit[steps].estimate(audio_secs)

    def next_budget_words(self) -> int | None:
        """How many words the NEXT piece may have so that, synthesized right
        after everything already released, it is ready before the audio
        queued ahead of it runs out. None = nothing is queued or playing, i.e.
        the next piece starts a reply (sized by the first-piece rule instead).

        Piece k+1 can be synthesized in at most the audio that remains
        buffered when the (serial) synthesizer frees up, so a reply grows
        geometrically - 3 words, then a few more, then longer - instead of a
        fixed size that underruns right behind a short first piece."""
        now = self._clock()
        if not self._queue and self._play_end <= now:
            return None
        play_end = max(self._play_end, now)
        synth_free = now
        for i, (audio, synth) in enumerate(self._queue):
            if i == 0 and self._head_started is not None:
                synth = max(0.05, synth - (now - self._head_started))
            synth_end = synth_free + synth
            if synth_end > play_end:
                play_end = synth_end
            play_end += audio
            synth_free = synth_end
        buffer = max(0.0, play_end - synth_free)
        fit = self._fit[STEP_LADDER[-1]]
        fixed, slope = fit.fixed * 1.15, max(fit.slope * 1.15, 0.05)
        room = buffer * _HEADROOM - fixed  # buffered audio left over after the fixed cost
        if room <= 0:
            # Even a one-word piece can't land in time, so a gap is coming
            # whatever its size: make it ONE gap, with a piece long enough to
            # then rebuild the buffer, rather than a string of small ones.
            return 12
        return max(2, min(40, int(room / slope * WORDS_PER_SEC)))

    def is_new_reply(self) -> bool:
        return self._clock() > self._play_end + REPLY_IDLE_SECS

    def choose_steps(self, text: str, *, default_steps: int, quality_floor: int = STEP_LADDER[-1]) -> int:
        """Highest quality (<= default_steps) whose predicted synth time still
        lands inside the buffered audio. The first piece of a reply has no
        buffer to protect, so it takes the default; the caller waits for it."""
        if self._cur_first:
            return default_steps
        buffered = self.buffered_secs()
        est = self.estimate_audio_secs(text)
        for steps in STEP_LADDER:
            if steps > default_steps or steps < quality_floor:
                continue
            if self.estimate_synth_secs(steps, est) <= buffered * _HEADROOM:
                return steps
        return min(default_steps, max(quality_floor, STEP_LADDER[-1]))

    # -- bookkeeping ----------------------------------------------------------

    def begin_piece(self, text: str) -> None:
        """Called when a piece's text arrives: starts the clock for the
        reply's time-to-first-audio (only the first piece of a reply uses it)."""
        self._cur_first = self.is_new_reply()
        self._head_started = self._clock()
        if self._cur_first:
            self._first_text_at = self._clock()

    def piece_ready(self, *, owner: str, text: str, steps: int, synth_secs: float, audio_secs: float, cached: bool) -> float:
        """Record a finished piece; returns the silence (seconds) the caller
        will hear before it, 0 when it joins the buffer seamlessly."""
        now = self._clock()
        if self._queue:
            self._queue.popleft()
        self._head_started = None
        first = self._cur_first
        buffer_before = self.buffered_secs()
        gap = 0.0
        if first:
            self._diag.replies += 1
            if self._first_text_at is not None:
                self._diag.first_audio_ms.append((now - self._first_text_at) * 1000)
            self._play_end = now
        elif now > self._play_end:
            gap = now - self._play_end
            self._diag.underruns += 1
            self._play_end = now
        self._play_end += audio_secs
        if not cached and audio_secs > 0 and steps in self._fit:
            self._fit[steps].add(audio_secs, synth_secs)
        self._diag.pieces.append(PieceRecord(
            ts=time.time(), owner=owner, chars=len(text), steps=steps, synth_ms=synth_secs * 1000,
            audio_ms=audio_secs * 1000, cached=cached, first_of_reply=first, gap_ms=gap * 1000,
            buffer_ms=buffer_before * 1000,
        ))
        if first:
            self._first_text_at = None
        return gap

    def reset(self, *, interrupted: bool = False) -> None:
        """Interruption / new call: whatever was queued is gone."""
        self._play_end = 0.0
        self._first_text_at = None
        self._queue.clear()
        self._head_started = None
        if interrupted:
            self._diag.interruptions += 1

    def model(self) -> dict[int, tuple[float, float]]:
        """Current (fixed secs, secs per audio second) per step count."""
        return {steps: (f.fixed, f.slope) for steps, f in self._fit.items()}
