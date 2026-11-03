"""Runs one scripted ShiftEaze call (services/scenario_engine.ScenarioSession)
over live audio: STT transcripts in, cached OmniVoice TTS audio out.

All call decisions stay in ScenarioSession - the same engine
POST /scenarios/test-turn drives over HTTP - so this file only handles the
parts that exist because it's a phone call: turn-taking, silence, hang-up
timing, and latency logging. Barge-in itself (stopping playback when the
caller talks over the bot) is the pipeline's BargeInController
(barge_in.py); this processor only says which lines may be cut off
(can_be_interrupted).

Turn-taking: a short, unambiguous answer ("haan ji", "nahi aa paunga") is
acted on the instant its transcript arrives. Anything longer waits a brief
grace window for the caller to continue first - longer while collecting a
reason, since people pause mid-sentence ("meri tabiyat... theek nahi hai")
and the first fragment must not be taken as the whole reason.

The opening line can't be interrupted: people answer the phone with
"Hello?" while it starts, and letting that cut it off means they never hear
the shift details. Greetings/backchannel said over the opening ("hello",
"ji", "haan ji") are dropped; anything more is answered once it finishes.
"""

import asyncio
import logging
import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from services import call_analysis, conversation_store, scenario_engine, scenario_store
from services.call_logger import finalize_call_log, log_turn
from services.voice_pipeline import tts_cache

logger = logging.getLogger("voice_pipeline.scenario")

# On top of VAD's 0.5s end-of-speech window. The fast-path grace is tiny but
# nonzero: it gives VAD time to notice the caller resuming after a comma
# pause, so the bot doesn't answer half a sentence over them.
FAST_PATH_GRACE_SECS = 0.2
QUESTION_GRACE_SECS = 0.4
REASON_GRACE_SECS = 0.6
# A plain "no" where the bot's next line is "why?": callers often give the
# reason in the same breath after a pause ("नहीं आ पाऊँगा, ... पत्नी अस्पताल
# में है" - ~1s in the simulator). Waiting longer here costs little (the
# question comes either way) and avoids talking over the reason.
REFUSAL_GRACE_SECS = 0.9
# A bare "ना"/"नहीं" on its own is nearly always followed by more ("ના, હું
# કાલે નહીં આવી શકું" split at the comma in the simulator).
BARE_REFUSAL_GRACE_SECS = 1.2
MIN_GRACE_SECS = 0.1
SILENCE_TIMEOUT_SECS = 7.0
MAX_CALL_SECS = 150.0

_BACKCHANNEL = {
    "hello", "helo", "hallo", "hi", "हेलो", "हैलो", "हलो", "हाय", "હેલો", "હલો",
    "जी", "हाँ", "हां", "हाँजी", "हांजी", "ji", "haan", "han", "hanji", "haanji", "જી", "હા",
    "ok", "okay", "ओके", "hmm", "हम्म", "अच्छा", "achha", "acha",
}


def _is_backchannel(text: str) -> bool:
    tokens = scenario_engine._tokens(text)
    return bool(tokens) and all(t in _BACKCHANNEL for t in tokens)


class ScenarioCallProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        call_sid: str,
        session: scenario_engine.ScenarioSession,
        silence_timeout: float = SILENCE_TIMEOUT_SECS,
        max_call_secs: float = MAX_CALL_SECS,
        vad_stop_secs: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._call_sid = call_sid
        self._session = session
        self._language = session.language
        self._scenario_id = session.scenario_id
        self._silence_timeout = silence_timeout
        self._max_call_secs = max_call_secs
        # VAD reports end-of-speech this long after the caller actually
        # stopped; added back so logged latency is from real speech end.
        self._vad_stop_secs = vad_stop_secs

        self._opening_playing = False
        self._opening_buffer: list[str] = []
        self._bot_speaking = False
        self._user_speaking = False
        self._end_after_speech = False
        self._ended = False
        self._finalized = False
        self._last_branch: str | None = None

        self._pending: list[str] = []
        self._first_transcript_at: float | None = None
        self._vad_stop_at: float | None = None
        self._grace_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._backstop_task: asyncio.Task | None = None
        self._turn_lock = asyncio.Lock()
        self._checked_in = False
        self.turn_latencies_ms: list[int] = []

    # -- frame routing ----------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self.create_task(self._open_call(), "open_call")
            self._watchdog_task = self.create_task(self._watchdog(), "watchdog")
            return

        if isinstance(frame, TranscriptionFrame):
            self._on_transcript(frame.text)
            return  # consumed - nothing downstream needs transcripts

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._on_user_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            self._vad_stop_at = time.monotonic()
            if self._pending and not self._ended and not self._end_after_speech:
                # Held text from before this speech burst; if the burst
                # yields no transcript (noise, a cough), still answer it.
                # A real transcript arriving first reschedules this.
                self._cancel(self._grace_task)
                self._grace_task = self.create_task(self._start_turn_after(0.8), "turn_grace")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._cancel(self._idle_task)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._opening_playing:
                self._opening_finished()
            if self._end_after_speech:
                await self._hang_up()
            elif not self._ended and not self._pending and not self._turn_lock.locked():
                self._arm_idle_timer()

        await self.push_frame(frame, direction)

    # -- call lifecycle -----------------------------------------------------

    async def _open_call(self) -> None:
        opening = self._session.opening_line()
        self.create_task(scenario_engine.warm_classifier(self._session), "warm_classifier")
        conversation_store.add_message(self._call_sid, "assistant", opening)
        log_turn(self._call_sid, role="assistant", content=opening)
        self._opening_playing = True
        t0 = time.monotonic()
        await self._speak(opening)
        logger.info("[%s] opening playing (audio ready in %dms)", self._call_sid, (time.monotonic() - t0) * 1000)

    def _opening_finished(self) -> None:
        self._opening_playing = False
        said = " ".join(self._opening_buffer).strip()
        self._opening_buffer.clear()
        if not said:
            return
        if _is_backchannel(said):
            logger.info('[%s] ignoring "%s" said over the opening', self._call_sid, said)
            return
        self._on_transcript(said)

    async def _speak(self, text: str, *, end_call: bool = False) -> None:
        if not text:
            if end_call:
                await self._hang_up()
            return
        try:
            pcm = await tts_cache.get_audio(text, self._language, self._scenario_id)
        except Exception:
            logger.exception("[%s] TTS failed for %r", self._call_sid, text[:60])
            if end_call:
                await self._hang_up()
            return
        self._end_after_speech = end_call
        # TTS-typed frames on purpose, even though no TTS service is in the
        # pipeline: Pipecat's output transport only tracks bot speech (and
        # emits BotStarted/StoppedSpeakingFrame, which turn-taking and the
        # hang-up timing depend on) for TTSAudioRawFrame. Plain
        # OutputAudioRawFrame plays fine but is invisible to that tracking -
        # found by the call simulator: the opening never "ended", so every
        # caller answer was held back as if spoken over the opening.
        # TTSStoppedFrame is queued behind the audio, so BotStoppedSpeaking
        # fires when playback actually finishes.
        context_id = f"line-{time.monotonic_ns()}"
        await self.push_frame(TTSStartedFrame(context_id=context_id))
        await self.push_frame(
            TTSAudioRawFrame(audio=pcm, sample_rate=tts_cache.SAMPLE_RATE, num_channels=1, context_id=context_id)
        )
        await self.push_frame(TTSStoppedFrame(context_id=context_id))
        if end_call:
            # BotStoppedSpeakingFrame normally triggers the hang-up once the
            # goodbye has finished playing; this is only a backstop so a call
            # can never be left open if that frame doesn't arrive.
            self._backstop_task = self.create_task(
                self._hang_up_after(tts_cache.duration_secs(pcm) + 3.0), "hangup_backstop"
            )

    async def _hang_up_after(self, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._hang_up()

    async def _hang_up(self) -> None:
        if self._ended:
            return
        self._ended = True
        for task in (self._idle_task, self._grace_task, self._watchdog_task, self._backstop_task):
            self._cancel(task)
        self._finalize(status="completed")
        # Ask the pipeline task to end, rather than injecting an EndFrame
        # here: an EndFrame pushed mid-pipeline only reaches what's
        # downstream of this processor, so the task never finished and the
        # stream hung open (caught by tools/simulate_scenario_call.py). The
        # task then sends EndFrame from the top, stopping every stage in
        # order - including the Twilio serializer's hang-up.
        await self.push_frame(EndWorkerFrame(reason="call complete"))

    def _finalize(self, *, status: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        s = self._session
        if status == "completed":
            outcome = s.outcome or "unknown"
        else:
            outcome = s.outcome or s.scenario.get("no_answer_outcome", "no_response")
        detected = s.last_classification.detected_language if s.last_classification else None
        scenario_store.record_outcome(
            self._call_sid, outcome=outcome, status=status, matched_branch=self._last_branch,
            collected=s.collected, detected_language=detected,
        )
        conversation_store.end_call(self._call_sid, status=status)
        finalize_call_log(
            self._call_sid, status=status, outcome=outcome, matched_branch=self._last_branch,
            collected=s.collected, detected_language=detected,
        )
        avg = int(sum(self.turn_latencies_ms) / len(self.turn_latencies_ms)) if self.turn_latencies_ms else None
        try:
            if call_analysis.store_details(self._call_sid, avg_turn_latency_ms=avg):
                call_analysis.schedule_analysis(self._call_sid)  # after hang-up: no live latency cost
        except Exception:  # noqa: BLE001 - the outcome itself is already saved
            logger.exception("[%s] storing call details failed", self._call_sid)
        logger.info(
            "[%s] call finished: status=%s outcome=%s collected=%s avg_turn_latency=%sms",
            self._call_sid, status, outcome, s.collected, avg,
        )

    def handle_disconnect(self) -> None:
        """The caller hung up / the stream dropped before a scripted end."""
        if not self._ended:
            self._ended = True
            self._finalize(status="disconnected")

    # -- caller speech ------------------------------------------------------

    def can_be_interrupted(self) -> bool:
        """For the pipeline's BargeInController (barge_in.py), which makes the
        actual stop-playback decision: the opening and the goodbye play out."""
        return not self._opening_playing and not self._end_after_speech and not self._ended

    async def _on_user_started(self) -> None:
        self._user_speaking = True
        self._checked_in = False
        self._cancel(self._idle_task)
        self._cancel(self._grace_task)  # still talking - the pending fragment isn't the whole turn

    def _on_transcript(self, text: str) -> None:
        text = (text or "").strip()
        if not text or self._ended or self._end_after_speech:
            return  # the goodbye is already playing - nothing left to decide
        if self._opening_playing:
            self._opening_buffer.append(text)
            return

        if not self._pending:
            self._first_transcript_at = time.monotonic()
        self._pending.append(text)
        self._cancel(self._grace_task)

        combined = " ".join(self._pending)
        branches = self._session.lang_cfg["branches"]
        fast = scenario_engine.fast_classify(combined, list(branches))
        if self._session.collecting:
            delay = REASON_GRACE_SECS
        elif fast == "no" and branches.get("no", {}).get("collect"):
            delay = BARE_REFUSAL_GRACE_SECS if len(combined.split()) == 1 else REFUSAL_GRACE_SECS
        elif fast not in (None, "unclear"):
            # A clear answer. A filler-only fragment ("मैं...") classifies as
            # "unclear" but is usually the caller just starting a sentence -
            # it gets the normal window, not an instant "please repeat".
            delay = FAST_PATH_GRACE_SECS
        else:
            delay = QUESTION_GRACE_SECS
        # The grace counts from when the caller stopped talking, not from when
        # the transcript arrived. Local Whisper takes ~0.7s to decode, and had
        # the caller resumed during that time, VADUserStartedSpeaking would
        # already have cancelled this turn - waiting the full grace again on
        # top of the decode was pure dead air. The floor covers a resume VAD
        # hasn't confirmed yet (it needs start_secs of speech to fire).
        if self._vad_stop_at and not self._user_speaking:
            delay = max(MIN_GRACE_SECS, delay - (time.monotonic() - self._vad_stop_at))
        self._grace_task = self.create_task(self._start_turn_after(delay), "turn_grace")

    async def _start_turn_after(self, delay: float) -> None:
        # Only this wait is cancellable (by the caller continuing to speak).
        # The turn itself runs in its own task so a new transcript arriving
        # mid-turn can never abort a reply that's already being prepared -
        # it just queues behind _turn_lock.
        if delay:
            await asyncio.sleep(delay)
        if self._user_speaking:
            return  # still talking; their next transcript reschedules this
        self.create_task(self._run_turn(), "turn")

    async def _run_turn(self) -> None:
        async with self._turn_lock:
            if self._ended or not self._pending:
                return
            text = " ".join(self._pending)
            self._pending.clear()
            heard_at = (self._vad_stop_at - self._vad_stop_secs) if self._vad_stop_at else (self._first_transcript_at or time.monotonic())
            stt_ms = max(0, int(((self._first_transcript_at or heard_at) - heard_at) * 1000))

            conversation_store.add_message(self._call_sid, "user", text)
            log_turn(self._call_sid, role="user", content=text)

            t0 = time.monotonic()
            try:
                result = await self._session.handle_utterance(text)
            except Exception:
                logger.exception("[%s] turn failed for %r - re-asking", self._call_sid, text)
                cfg = self._session.lang_cfg
                result = scenario_engine.TurnResult(
                    reply=scenario_engine.render(cfg.get("unclear_reprompt", cfg["opening"]), self._session.facts),
                    end_call=False, outcome="awaiting_response", branch="error_reprompt",
                )
            decide_ms = int((time.monotonic() - t0) * 1000)

            await self._speak(result.reply, end_call=result.end_call)
            total_ms = int((time.monotonic() - heard_at) * 1000)
            self.turn_latencies_ms.append(total_ms)
            self._last_branch = result.branch

            c = result.classification
            source = c.source if (c and result.branch == c.branch) else "engine"
            logger.info(
                '[%s] heard "%s" -> %s (via %s) | stt=%dms decide=%dms speech-end->reply=%dms%s',
                self._call_sid, text, result.branch, source, stt_ms, decide_ms, total_ms,
                " | ending call" if result.end_call else "",
            )
            conversation_store.add_message(self._call_sid, "assistant", result.reply)
            log_turn(
                self._call_sid, role="assistant", content=result.reply, branch=result.branch,
                outcome=result.outcome, end_call=result.end_call,
                classification=(
                    {"branch": c.branch, "confidence": c.confidence, "source": c.source,
                     "volunteered_reason": c.volunteered_reason} if c else None
                ),
                timings_ms={"stt": stt_ms, "decide": decide_ms, "speech_end_to_reply": total_ms},
            )

    # -- silence ------------------------------------------------------------

    def _arm_idle_timer(self) -> None:
        self._cancel(self._idle_task)
        self._idle_task = self.create_task(self._idle_timeout(), "idle")

    async def _idle_timeout(self) -> None:
        await asyncio.sleep(self._silence_timeout)
        if self._ended or self._bot_speaking or self._user_speaking or self._turn_lock.locked() or self._pending:
            return
        if not self._checked_in and self._session.lang_cfg.get("silence_checkin"):
            self._checked_in = True
            logger.info("[%s] silence - checking in", self._call_sid)
            await self._speak(self._session.lang_cfg["silence_checkin"])
            return
        logger.info("[%s] still silent - closing the call", self._call_sid)
        await self._close_for_silence()

    async def _watchdog(self) -> None:
        await asyncio.sleep(self._max_call_secs)
        if not self._ended:
            logger.warning("[%s] max call duration reached - closing", self._call_sid)
            await self._close_for_silence()

    async def _close_for_silence(self) -> None:
        result = self._session.on_silence()
        self._last_branch = result.branch
        conversation_store.add_message(self._call_sid, "assistant", result.reply)
        log_turn(self._call_sid, role="assistant", content=result.reply, branch=result.branch,
                 outcome=result.outcome, end_call=True)
        await self._speak(result.reply, end_call=True)

    def _cancel(self, task: asyncio.Task | None) -> None:
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
