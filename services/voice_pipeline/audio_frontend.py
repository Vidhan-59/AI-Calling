"""The caller-audio front end shared by every live-call pipeline (Layer 2):

  caller audio -> NOISE SUPPRESSION (RNNoise, neural)
               -> SILERO VAD (speech must last vad_start_secs, 150ms)
               -> BARGE-IN controller (services/voice_pipeline/barge_in.py)
               -> AUDIO BUFFER (the output transport's queue, flushed on barge-in)

Noise suppression runs as the input transport's audio filter, so VAD, STT
and recording all see the cleaned signal. Background noise is attenuated
before Silero scores it, and Silero then needs vad_confidence held for
vad_start_secs - a passing truck or a dropped tool never adds up to that,
real speech does within a syllable or two.

SpeculativeVADProcessor is stock Pipecat VAD plus two extra signals the
local STT uses to decode during the end-of-turn window instead of after it
(see local_stt.py): UserSpeechPausedFrame once the caller has been quiet for
speculate_after_secs, and UserSpeechResumedFrame if they start again before
VAD's stop_secs elapses.
"""

import logging
from dataclasses import dataclass

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    SystemFrame,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection

from config import settings

logger = logging.getLogger("voice_pipeline.audio_frontend")


@dataclass
class UserSpeechPausedFrame(SystemFrame):
    """The caller has been quiet for speculate_after_secs mid-turn. A
    SystemFrame so it isn't queued behind audio on its way to the STT."""


@dataclass
class UserSpeechResumedFrame(SystemFrame):
    """The caller started talking again before VAD declared the turn over -
    any speculative transcript taken at the pause is now stale."""


def vad_params(*, stop_secs: float) -> VADParams:
    """stop_secs (end-of-turn silence) stays per pipeline - scripted calls
    and free-form agent calls have different turn-taking needs - while the
    barge-in/noise thresholds come from settings, one place for both."""
    return VADParams(
        confidence=settings.vad_confidence,
        start_secs=settings.vad_start_secs,
        stop_secs=stop_secs,
        min_volume=settings.vad_min_volume,
    )


def build_noise_filter():
    """RNNoise on the input transport, or None when disabled/unavailable."""
    if not settings.noise_suppression:
        return None
    try:
        import pyrnnoise  # noqa: F401 - only checking it's installed

        from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
    except ImportError:
        logger.warning("NOISE_SUPPRESSION is on but pyrnnoise isn't installed - pip install 'pipecat-ai[rnnoise]'")
        return None
    return RNNoiseFilter()


def front_end(*, call_id: str, stop_secs: float, can_interrupt=None):
    """The processors that go between the input transport and the STT, in
    order: VAD -> barge-in -> latency monitor. Returns (processors,
    barge_in) - the controller is returned too so a pipeline whose call
    processor is built later can hand it a can_interrupt predicate."""
    from services.voice_pipeline.barge_in import BargeInController
    from services.voice_pipeline.latency import LatencyMonitor

    barge_in = BargeInController(call_id=call_id, can_interrupt=can_interrupt)
    processors = [
        SpeculativeVADProcessor(params=vad_params(stop_secs=stop_secs)),
        barge_in,
        LatencyMonitor(call_id=call_id, vad_stop_secs=stop_secs),
    ]
    return processors, barge_in


class SpeculativeVADProcessor(VADProcessor):
    def __init__(self, *, params: VADParams, speculate_after_secs: float | None = None, **kwargs):
        self._analyzer = SileroVADAnalyzer(params=params)
        super().__init__(vad_analyzer=self._analyzer, **kwargs)
        self._speculate_after_secs = (
            settings.speculate_after_secs if speculate_after_secs is None else speculate_after_secs
        )
        self._speculate = settings.speculative_stt and self._speculate_after_secs < params.stop_secs
        self._paused = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and self._speculate:
            await self._track_pause()

    async def _track_pause(self) -> None:
        # In a turn = the controller has reported speech started and not yet
        # stopped. (Not tracked from VADUserStarted/StoppedSpeakingFrame: this
        # processor broadcasts those itself, it never receives them.)
        if self._vad_controller._vad_state != VADState.SPEAKING:
            self._paused = False
            return
        # The analyzer's own state machine, read after it has scored this
        # frame: STOPPING = below threshold but not yet for stop_secs. Pipecat
        # only reports the QUIET/SPEAKING edges, so the count is read directly.
        state = self._analyzer._vad_state
        if state == VADState.STOPPING:
            quiet_secs = self._analyzer._vad_stopping_count * self._analyzer.num_frames_required() / self._analyzer.sample_rate
            if not self._paused and quiet_secs >= self._speculate_after_secs:
                self._paused = True
                await self.push_frame(UserSpeechPausedFrame())
        elif state == VADState.SPEAKING and self._paused:
            self._paused = False
            await self.push_frame(UserSpeechResumedFrame())
