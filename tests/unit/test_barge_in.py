"""BargeInController (services/voice_pipeline/barge_in.py): interrupt only
when the bot is speaking, and never during a protected line."""

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, VADUserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection

from services.voice_pipeline.barge_in import BargeInController


def _controller(can_interrupt=None) -> tuple[BargeInController, list]:
    c = BargeInController(call_id="t", can_interrupt=can_interrupt)
    events: list = []

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        events.append(type(frame).__name__)

    async def interrupt():
        events.append("INTERRUPT")

    c.push_frame = push
    c.broadcast_interruption = interrupt
    return c, events


async def test_speech_while_bot_talks_interrupts_after_forwarding_vad_frame():
    c, events = _controller()
    await c.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await c.process_frame(VADUserStartedSpeakingFrame(start_secs=0.15), FrameDirection.DOWNSTREAM)
    assert events[-2:] == ["VADUserStartedSpeakingFrame", "INTERRUPT"]
    assert c.barge_ins == 1


async def test_speech_while_bot_silent_does_not_interrupt():
    c, events = _controller()
    await c.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await c.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    await c.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert "INTERRUPT" not in events
    assert c.barge_ins == 0


async def test_protected_line_is_not_interrupted():
    c, events = _controller(can_interrupt=lambda: False)
    await c.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await c.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert "INTERRUPT" not in events
    assert c.ignored == 1


async def test_every_frame_is_forwarded():
    c, events = _controller()
    await c.process_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    await c.process_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert events == ["BotStartedSpeakingFrame", "BotStoppedSpeakingFrame"]
