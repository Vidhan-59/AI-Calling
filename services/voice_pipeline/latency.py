"""Per-turn latency, measured where it matters: from the moment VAD declares
the caller finished (VADUserStoppedSpeakingFrame) to the moment the output
transport starts playing the bot's reply (BotStartedSpeakingFrame) - the
"time to first audio" the whole pipeline is tuned for (target: under 1s).

The caller's actual last syllable was VAD stop_secs earlier still; that is
logged too ("from last word"), so neither number hides the other.

Also measures barge-in: caller speech confirmed -> bot audio stopped.

Results are kept per call in RECENT (bounded) so tools/latency_smoke_test.py
and the simulators, which run the pipeline in-process, can read the
server-side numbers without parsing logs.
"""

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger("voice_pipeline.latency")

TARGET_MS = 1000
_MAX_CALLS = 200


@dataclass
class CallLatency:
    vad_stop_secs: float
    turns_ms: list[int] = field(default_factory=list)
    barge_in_stop_ms: list[int] = field(default_factory=list)
    barge_in_confirmed_at: list[float] = field(default_factory=list)  # time.monotonic() of each


RECENT: "OrderedDict[str, CallLatency]" = OrderedDict()


class LatencyMonitor(FrameProcessor):
    def __init__(self, *, call_id: str, vad_stop_secs: float, **kwargs):
        super().__init__(**kwargs)
        self._call_id = call_id
        self._vad_stop_secs = vad_stop_secs
        self._stats = CallLatency(vad_stop_secs=vad_stop_secs)
        RECENT[call_id] = self._stats
        while len(RECENT) > _MAX_CALLS:
            RECENT.popitem(last=False)
        self._turn_ended_at: float | None = None
        self._bot_speaking = False
        self._barge_in_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        now = time.monotonic()
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._turn_ended_at = now
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._turn_ended_at = None  # still talking - the turn isn't over
            if self._bot_speaking:
                self._barge_in_at = now
                self._stats.barge_in_confirmed_at.append(now)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if self._turn_ended_at is not None:
                ms = int((now - self._turn_ended_at) * 1000)
                self._turn_ended_at = None
                self._stats.turns_ms.append(ms)
                logger.info(
                    "[%s] time to first audio: %dms from end-of-speech (%dms from last word) %s",
                    self._call_id, ms, ms + int(self._vad_stop_secs * 1000),
                    "OK" if ms < TARGET_MS else f"OVER {TARGET_MS}ms TARGET",
                )
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._barge_in_at is not None:
                ms = int((now - self._barge_in_at) * 1000)
                self._barge_in_at = None
                self._stats.barge_in_stop_ms.append(ms)
                logger.info("[%s] barge-in: bot audio stopped %dms after caller speech confirmed", self._call_id, ms)

        await self.push_frame(frame, direction)
