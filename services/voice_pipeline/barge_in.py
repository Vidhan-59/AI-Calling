"""Barge-in controller (Layer 2): lets the caller cut the bot off, like a
real call - and only the caller, never background noise.

Sits right after the VAD. By the time VADUserStartedSpeakingFrame reaches
it, Silero has already held vad_start_secs (150ms) of above-threshold speech
on noise-suppressed audio, so that frame alone is the "human is talking"
decision. If the bot is mid-sentence at that moment, it broadcasts an
interruption, which:

  - makes the TTS service drop any audio it hasn't sent yet,
  - flushes the output transport's audio buffer, and
  - for Twilio, sends a "clear" so Twilio drops what it has buffered too
    (pipecat's TwilioFrameSerializer turns InterruptionFrame into "clear").

The call processor downstream sees the same InterruptionFrame and stops
generating the rest of the interrupted reply.

`can_interrupt` lets the call logic protect a line - the scripted opening,
which people talk over with "Hello?" while it starts, and the goodbye.
"""

import logging
import time
from collections.abc import Callable

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger("voice_pipeline.barge_in")


class BargeInController(FrameProcessor):
    def __init__(self, *, call_id: str, can_interrupt: Callable[[], bool] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._call_id = call_id
        self._can_interrupt = can_interrupt or (lambda: True)
        self._bot_speaking = False
        self.barge_ins = 0
        self.ignored = 0
        self.last_barge_in_at: float | None = None

    def set_can_interrupt(self, can_interrupt: Callable[[], bool]) -> None:
        """For pipelines whose call processor is built after this controller."""
        self._can_interrupt = can_interrupt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False

        await self.push_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame) and self._bot_speaking:
            if not self._can_interrupt():
                self.ignored += 1
                logger.info("[%s] caller spoke over a protected line - not interrupting", self._call_id)
                return
            self.barge_ins += 1
            self.last_barge_in_at = time.monotonic()
            logger.info("[%s] barge-in: caller speech held %.0fms - flushing bot audio",
                        self._call_id, getattr(frame, "start_secs", 0.0) * 1000)
            # The VAD frame goes downstream first so the call processor knows
            # the caller is talking before the interruption lands.
            await self.broadcast_interruption()
