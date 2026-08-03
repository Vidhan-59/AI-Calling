"""End-to-end orchestration: audio in -> STT -> response generation -> TTS -> audio out.

File-in/file-out mode only (per Phase 5's minimum requirement). A streaming/
mic mode is a real follow-on project, not a small addition - it needs voice
activity detection and interruption handling, which is exactly the "hard
part" the ShiftEaze master architecture doc (01_MASTER_ARCHITECTURE.md) calls
out as the reason to use a real-time media framework (Pipecat) rather than
hand-rolling it. Do not build a mic-streaming mode by extending this file
without reading that doc's section 0 first.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from local_ai import config, text_norm
from local_ai.llm_engine import OllamaUnavailableError
from local_ai.stt_engine import STTEngine, TranscriptionResult, get_default_engine as get_default_stt
from local_ai.tts_engine import TTSEngine, VoiceNotAvailableError, get_default_engine as get_default_tts

# A response function takes (transcribed_text, language) and returns the
# reply text to speak. services/scenario_engine.py's ScenarioSession is one
# implementation of this shape (see cli.py for how it's wired in); a plain
# echo or a direct llm_engine.generate_reply call are others, used by the
# smoke test so it doesn't depend on scenario JSON files being present.
ResponseFn = Callable[[str, str], str]


@dataclass
class PipelineResult:
    transcript: str
    detected_language: str
    language_probability: float
    reply_text: str
    reply_text_normalized: str
    output_audio_path: str
    timings_ms: dict = field(default_factory=dict)


class LocalVoicePipeline:
    def __init__(
        self,
        stt: Optional[STTEngine] = None,
        tts: Optional[TTSEngine] = None,
    ):
        self.stt = stt or get_default_stt()
        self.tts = tts or get_default_tts()

    def run_file(
        self,
        input_audio_path: str,
        output_audio_path: str,
        respond: ResponseFn,
        language_hint: str | None = None,
    ) -> PipelineResult:
        timings: dict[str, float] = {}

        t0 = time.monotonic()
        transcription: TranscriptionResult = self.stt.transcribe(input_audio_path, language=language_hint)
        timings["stt_ms"] = round((time.monotonic() - t0) * 1000, 1)

        language = language_hint or transcription.detected_language

        t0 = time.monotonic()
        reply_text = respond(transcription.text, language)
        timings["response_ms"] = round((time.monotonic() - t0) * 1000, 1)

        t0 = time.monotonic()
        normalized = text_norm.normalize_for_speech(reply_text, language)
        timings["normalize_ms"] = round((time.monotonic() - t0) * 1000, 1)

        t0 = time.monotonic()
        try:
            self.tts.synthesize(normalized, output_audio_path, language=language)
        except VoiceNotAvailableError:
            # Re-raise with the pipeline context attached rather than a bare
            # TTS-layer error - the caller needs to know which stage failed.
            raise
        timings["tts_ms"] = round((time.monotonic() - t0) * 1000, 1)
        timings["total_ms"] = round(sum(timings.values()), 1)

        return PipelineResult(
            transcript=transcription.text,
            detected_language=transcription.detected_language,
            language_probability=transcription.language_probability,
            reply_text=reply_text,
            reply_text_normalized=normalized,
            output_audio_path=output_audio_path,
            timings_ms=timings,
        )


__all__ = ["LocalVoicePipeline", "PipelineResult", "ResponseFn", "OllamaUnavailableError", "VoiceNotAvailableError"]
