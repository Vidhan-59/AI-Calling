"""Pipecat TTSService backed by the OmniVoice service, for the config-driven
conversation agent where the reply text is only known at call time.

OmniVoice generates a whole utterance per request (no audio streaming), so
latency is per piece: the conversation processor streams the LLM reply and
sends the first piece after a few words (STREAM_FIRST_CHUNK_WORDS), later
pieces at clause boundaries, so the first piece plays while the rest are
still being written and synthesized. Pieces go through the same audio cache
as scripted lines (services/voice_pipeline/tts_cache.py): a phrase the agent
has said before - the tool filler, a greeting, a closing - plays instantly.
Scripted scenario calls skip this service and play pre-rendered audio.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

from config import settings
from services.tts import client
from services.tts.playout import DIAGNOSTICS, PlayoutPlanner
from services.voice_pipeline import tts_cache

import time


@dataclass
class OmniVoiceTTSSettings(TTSSettings):
    pass


class OmniVoiceTTSService(TTSService):
    Settings = OmniVoiceTTSSettings
    _settings: Settings

    def __init__(self, *, owner: str, language: str, fallback_voice_id: str | None = None,
                 fallback_speed: float = 1.0, **kwargs):
        """`owner` is the agent/scenario id used to look up the assigned voice;
        the fallback_* values come from the agent's YAML `voice:` block."""
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=client.SAMPLE_RATE,
            settings=self.Settings(model=None, voice=None, language=language),
            **kwargs,
        )
        self._owner = owner
        self._lang = language
        self._fallback_voice_id = fallback_voice_id
        self._fallback_speed = fallback_speed
        self.playout = PlayoutPlanner()

    async def _handle_interruption(self, frame, direction):
        # Whatever was queued for playback is gone; the next piece is the
        # first of a new reply (no buffer to protect, default quality).
        self.playout.reset(interrupted=True)
        await super()._handle_interruption(frame, direction)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_ttfb_metrics()
            piece = text.strip()
            default_steps = (
                settings.tts_short_piece_num_step
                if len(piece.split()) <= settings.tts_short_piece_words
                else settings.tts_live_num_step
            )
            self.playout.begin_piece(piece)
            # Quality only as high as keeps the NEXT piece ahead of playback
            # (see services/tts/playout.py): continuity beats the last notch
            # of diffusion quality.
            steps = self.playout.choose_steps(piece, default_steps=default_steps)
            t0 = time.monotonic()
            pcm, cached = await tts_cache.get_audio_ex(
                piece, self._lang, self._owner, num_step=steps,
                fallback_voice_id=self._fallback_voice_id, fallback_speed=self._fallback_speed,
            )
            synth_secs = time.monotonic() - t0
            await self.stop_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            if len(pcm) % 2:
                pcm = pcm[:-1]
            buffered = self.playout.buffered_secs()
            audio_secs = tts_cache.duration_secs(pcm)
            gap = self.playout.piece_ready(
                owner=self._owner, text=piece, steps=steps, synth_secs=synth_secs,
                audio_secs=audio_secs, cached=cached,
            )
            level = logger.warning if gap > 0.15 else logger.info
            level(
                f"TTS piece: {len(piece.split())} words steps={steps} {'cache' if cached else 'synth'}="
                f"{synth_secs * 1000:.0f}ms audio={audio_secs * 1000:.0f}ms buffered={buffered * 1000:.0f}ms"
                + (f" UNDERRUN gap={gap * 1000:.0f}ms" if gap > 0.15 else "")
            )
            yield TTSAudioRawFrame(pcm, client.SAMPLE_RATE, 1, context_id=context_id)
        except Exception as e:  # noqa: BLE001 - a TTS failure must not kill the call pipeline
            DIAGNOSTICS.tts_errors += 1
            logger.error(f"OmniVoice TTS failed for {text[:60]!r}: {e}")
            yield ErrorFrame(error=f"TTS failed: {e}")
        finally:
            await self.stop_ttfb_metrics()
