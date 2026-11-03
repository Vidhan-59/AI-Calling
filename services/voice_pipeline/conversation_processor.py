"""Bridges Pipecat's frame pipeline to the flow-graph conversation engine
(services/agents/conversation_agent.py). Deliberately does NOT use Pipecat's
generic LLM+context-aggregator pattern - the flow-graph/tool/guardrail/
slot-filling logic in run_turn() must stay the single source of truth, since
it's already backend-agnostic (Claude/Qwen) and heavily tested elsewhere.
This processor's only job is translating between Pipecat frames and
run_turn()'s plain history-list contract, and persisting Call/Turn/
CallOutcome rows - the same responsibilities routers/agent_voice.py's retired
ConversationRelay WebSocket handler had, just triggered by frames instead of
JSON messages.

Layer 1 (token streaming): with settings.stream_llm_replies on, run_turn is
given an on_speech callback and the reply goes to TTS piece by piece while
the LLM is still generating - the first piece after a few words, so the
caller hears audio ~0.3s into generation instead of after the whole reply
plus a full TTS round trip. Each piece is its own TTSSpeakFrame; Sarvam's
streaming TTS starts synthesizing it the moment it arrives.

Turns run in their own task, not inside process_frame: an InterruptionFrame
(barge-in, services/voice_pipeline/barge_in.py) cancels whatever a processor
is doing in its frame task, and a turn must never be cancelled half-way
(slots and tool calls would be left half-applied). Instead an interruption
bumps _speech_epoch, and pieces from the interrupted reply stop being sent.

Latency fields: latency_stt_ms (VAD-stop to final transcript),
latency_llm_ttft_ms (transcript to first speakable piece) and
latency_total_ms (VAD-stop to first piece handed to TTS) are measured
directly. latency_tts_ttfb_ms stays None - TTS runs downstream, in Sarvam's
service; services/voice_pipeline/latency.py measures first audio out.
"""

import asyncio
import logging
import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import models
from config import settings
from db import session_scope
from services.agents.conversation_agent import (
    CALL_CONNECTED_SENTINEL,
    LLMCall,
    TurnAbandoned,
    prewarm_next_turn,
    run_turn,
)
from services.flow.slots import SlotState
from services.pipeline.chunker import split_sentences
from services.tts import client as tts_client

logger = logging.getLogger("voice_pipeline.conversation_processor")

_TERMINAL_DECISIONS = ("end_call", "transfer_to_human", "handoff")


# Neutral on purpose (either voice gender, any reply): said only when the real
# reply is late - see settings.late_filler_ms.
_LATE_FILLER = {"hi": "जी।", "gu": "જી.", "en": "Mm-hm."}


class ConversationAgentProcessor(FrameProcessor):
    """One instance per live call. Constructed with the call context already
    resolved (task_id/contact/agent_id/flow_graph/Call+AgentSpan row ids) by
    the router, mirroring what the old WebSocket handler's "setup" branch
    used to do before the pipeline existed - see routers/agent_voice.py."""

    def __init__(
        self,
        *,
        task_id: str,
        call_id: str,
        agent_span_id: int,
        agent_id: str,
        flow_graph,
        contact: dict,
        language: str,
        llm_call: LLMCall | None = None,
        stream_llm_call=None,
        stream_replies: bool | None = None,
        playout=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # The TTS service's PlayoutPlanner (services/tts/playout.py), when the
        # pipeline has one: sizes each reply piece so its synthesis finishes
        # before the audio queued ahead of it runs out, and is told when a
        # piece is handed over.
        self._playout = playout
        self._task_id = task_id
        self._call_id = call_id
        self._agent_span_id = agent_span_id
        self._agent_id = agent_id
        self._flow_graph = flow_graph
        self._contact = contact
        self._language = language
        self._llm_call = llm_call
        """Optional explicit backend override, same escape hatch run_turn()
        itself offers - lets tests inject a FakeLLM without monkeypatching
        conversation_agent's module-level backend resolution. None in
        production, which makes run_turn() resolve LLM_BACKEND normally."""
        self._stream_llm_call = stream_llm_call
        self._stream_replies = settings.stream_llm_replies if stream_replies is None else stream_replies

        self._slot_state = SlotState(current_node_id=flow_graph.entry)
        self._history: list[dict] = []
        self._turn_seq = 0
        self._bot_speaking = False
        self._pending_interruption = False
        self._vad_stop_at: float | None = None
        self._ended = False
        self._turn_lock = asyncio.Lock()
        self._speech_epoch = 0
        self._first_piece_at: float | None = None
        self._turn_in_flight = False
        self._last_tts_warm = 0.0
        self._turn_spoken: list[str] = []
        self.turn_latencies_ms: list[int] = []

    def _warm_tts(self) -> None:
        """Best-effort, throttled hint to the TTS service (see client.warm)."""
        now = time.monotonic()
        if now - self._last_tts_warm < 2.0:
            return
        self._last_tts_warm = now
        try:
            asyncio.get_running_loop().create_task(tts_client.warm())
        except RuntimeError:  # no running loop (some unit tests) - skip
            pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # StartFrame must reach downstream processors (TTS, output
            # transport) before anything is pushed after it - they drop
            # frames until they've been started.
            self._warm_tts()
            self._prime_filler()
            await self.push_frame(frame, direction)
            self.create_task(self._bootstrap_greeting(), "greeting")
            return

        if isinstance(frame, InterruptionFrame):
            # Barge-in: whatever the current reply still has to say is dropped.
            self._speech_epoch += 1
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # The caller is talking: STT + the LLM's first words take longer
            # than OmniVoice needs to come back onto the GPU, so wake it now.
            self._warm_tts()
            if self._bot_speaking:
                self._pending_interruption = True
            elif self._turn_in_flight:
                # The caller is still talking (a mid-sentence pause ended
                # their "turn" too early) and the bot hasn't started the reply
                # yet: drop that reply and anything already queued for TTS -
                # the rest of the sentence gets merged into one turn.
                self._speech_epoch += 1
                if self._turn_spoken:
                    await self.broadcast_interruption()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_stop_at = time.monotonic()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            self.create_task(self._handle_transcript(frame.text), "turn")
            return

        await self.push_frame(frame, direction)

    async def _persist_turn(
        self,
        role: str,
        text: str,
        *,
        latency_stt_ms: int | None = None,
        latency_llm_ttft_ms: int | None = None,
        latency_total_ms: int | None = None,
        was_interrupted: bool = False,
    ) -> None:
        self._turn_seq += 1
        async with session_scope() as session:
            session.add(
                models.Turn(
                    call_id=self._call_id,
                    agent_span_id=self._agent_span_id,
                    seq=self._turn_seq,
                    role=role,
                    text=text,
                    language_detected=self._language,
                    was_interrupted=was_interrupted,
                    latency_stt_ms=latency_stt_ms,
                    latency_llm_ttft_ms=latency_llm_ttft_ms,
                    latency_total_ms=latency_total_ms,
                )
            )

    async def _speak(self, text: str) -> None:
        for chunk in split_sentences(text) or [text]:
            await self._speak_piece(chunk, self._speech_epoch)

    async def _speak_piece(self, piece: str, epoch: int) -> None:
        if epoch != self._speech_epoch:
            return  # the caller barged in on this reply - don't keep talking
        if self._first_piece_at is None:
            self._first_piece_at = time.monotonic()
        self._turn_spoken.append(piece)
        if self._playout is not None:
            self._playout.release(piece)
        await self.push_frame(TTSSpeakFrame(piece), FrameDirection.DOWNSTREAM)

    async def _late_filler(self, epoch: int, delay: float) -> None:
        await asyncio.sleep(delay)
        if epoch == self._speech_epoch and self._first_piece_at is None and not self._turn_spoken and self._turn_in_flight:
            logger.info("[%s] reply still not started - playing the cached acknowledgement", self._call_id)
            await self._speak_piece(_LATE_FILLER.get(self._language, _LATE_FILLER["hi"]), epoch)

    def _prime_filler(self) -> None:
        """Render the filler into the audio cache now (call start), so it can
        play with zero synthesis delay when it is needed."""
        if settings.late_filler_ms <= 0:
            return
        from services import agent_registry
        from services.voice_pipeline import tts_cache

        async def go() -> None:
            try:
                cfg = agent_registry.get_agent_config(self._agent_id).voice.get(self._language)
                await tts_cache.get_audio(
                    _LATE_FILLER.get(self._language, _LATE_FILLER["hi"]), self._language, self._agent_id,
                    num_step=settings.tts_short_piece_num_step,
                    fallback_voice_id=cfg.voice_id if cfg else None, fallback_speed=cfg.speed if cfg else 1.0,
                )
            except Exception as e:  # noqa: BLE001 - only an optimisation
                logger.debug("filler prime skipped: %r", e)

        try:
            asyncio.get_running_loop().create_task(go())
        except RuntimeError:
            pass

    async def _run_agent_turn(self, filler_after: float | None = None):
        """One run_turn, streamed to TTS when enabled. Returns (outcome,
        cut_off), or (None, True) if the caller cut in before it finished -
        the LLM stream is stopped then (TurnAbandoned), not left to run.
        `filler_after`: seconds until the late-reply acknowledgement plays."""
        epoch = self._speech_epoch
        self._first_piece_at = None
        self._turn_spoken = []
        filler_task = None
        if filler_after is not None and self._stream_replies:
            filler_task = asyncio.get_running_loop().create_task(self._late_filler(epoch, filler_after))

        async def on_speech(piece: str) -> None:
            if epoch != self._speech_epoch:
                raise TurnAbandoned()
            await self._speak_piece(piece, epoch)

        self._turn_in_flight = True
        try:
            async with session_scope() as session:
                outcome = await run_turn(
                    session=session,
                    call_id=self._call_id,
                    agent_id=self._agent_id,
                    flow_graph=self._flow_graph,
                    slot_state=self._slot_state,
                    contact=self._contact,
                    language=self._language,
                    history=self._history,
                    llm_call=self._llm_call,
                    on_speech=on_speech if self._stream_replies else None,
                    stream_llm_call=self._stream_llm_call,
                    piece_budget=self._playout.next_budget_words if self._playout is not None else None,
                )
        except TurnAbandoned:
            return None, True
        finally:
            self._turn_in_flight = False
            if filler_task is not None:
                filler_task.cancel()
        return outcome, epoch != self._speech_epoch

    async def _record_abandoned_reply(self) -> None:
        """What the bot got out before the caller cut in is part of the
        conversation; a reply abandoned before any of it was spoken is not -
        the caller's words then wait to be merged with what they say next."""
        if self._turn_spoken:
            said = " ".join(self._turn_spoken)
            self._history.append({"role": "assistant", "content": said})
            await self._persist_turn("assistant", said, was_interrupted=True)

    async def _bootstrap_greeting(self) -> None:
        async with self._turn_lock:
            self._history = [{"role": "user", "content": CALL_CONNECTED_SENTINEL}]
            t0 = time.monotonic()
            outcome, _ = await self._run_agent_turn()
            if outcome is None:
                await self._record_abandoned_reply()
                logger.info("[%s] greeting cut short by the caller", self._call_id)
                return
            latency_ms = int(((self._first_piece_at or time.monotonic()) - t0) * 1000)
            self._history.append({"role": "assistant", "content": outcome.text_to_speak})
            await self._persist_turn("assistant", outcome.text_to_speak, latency_total_ms=latency_ms)
            logger.info("[%s] greeting: first audio queued %dms after connect", self._call_id, latency_ms)
            if not outcome.spoken_while_streaming:
                await self._speak(outcome.text_to_speak)
            self._start_prewarm()

    def _start_prewarm(self) -> None:
        """While the bot talks, have the LLM pre-process the next turn's
        prompt (conversation_agent.prewarm_next_turn). Skipped when a test
        injected its own LLM - there's no Ollama to warm."""
        if self._ended or self._llm_call is not None or self._stream_llm_call is not None:
            return
        self.create_task(self._prewarm(), "llm_prewarm")

    async def _prewarm(self) -> None:
        try:
            await prewarm_next_turn(
                agent_id=self._agent_id, flow_graph=self._flow_graph, slot_state=self._slot_state,
                contact=self._contact, language=self._language, history=list(self._history),
            )
        except Exception as e:  # noqa: BLE001 - only an optimization
            logger.debug("[%s] LLM prewarm skipped: %r", self._call_id, e)

    async def _handle_transcript(self, utterance: str) -> None:
        utterance = (utterance or "").strip()
        if not utterance or self._ended:
            return

        async with self._turn_lock:
            if self._ended:
                return
            heard_at = time.monotonic()
            vad_stop_at = self._vad_stop_at
            latency_stt_ms = int((heard_at - vad_stop_at) * 1000) if vad_stop_at else None
            was_interrupted, self._pending_interruption = self._pending_interruption, False

            await self._persist_turn(
                "user", utterance, latency_stt_ms=latency_stt_ms, was_interrupted=was_interrupted
            )
            last = self._history[-1] if self._history else None
            if last and last["role"] == "user" and last["content"] != CALL_CONNECTED_SENTINEL:
                # The previous fragment's reply was abandoned unspoken (the
                # caller kept talking) - answer the whole sentence as one turn.
                last["content"] = f'{last["content"]} {utterance}'
                utterance = last["content"]
            else:
                self._history.append({"role": "user", "content": utterance})

            filler_after = None
            if settings.late_filler_ms > 0:
                # Measured from the caller's end-of-speech, not from now: STT
                # has already used part of the budget.
                filler_after = max(0.1, settings.late_filler_ms / 1000 - (latency_stt_ms or 0) / 1000)
            outcome, cut_off = await self._run_agent_turn(filler_after=filler_after)
            if outcome is None:
                await self._record_abandoned_reply()
                logger.info('[%s] reply to "%s" abandoned - caller kept talking', self._call_id, utterance)
                return
            done_at = time.monotonic()
            first_at = self._first_piece_at or done_at
            ttft_ms = int((first_at - heard_at) * 1000)
            total_ms = int((first_at - (vad_stop_at or heard_at)) * 1000)
            self.turn_latencies_ms.append(total_ms)

            self._history.append({"role": "assistant", "content": outcome.text_to_speak})
            await self._persist_turn(
                "assistant", outcome.text_to_speak,
                latency_llm_ttft_ms=ttft_ms, latency_total_ms=total_ms, was_interrupted=cut_off,
            )
            logger.info(
                '[%s] heard "%s" | stt=%sms first-piece=%dms end-of-speech->TTS=%dms full-reply=%dms%s',
                self._call_id, utterance, latency_stt_ms, ttft_ms, total_ms,
                int((done_at - heard_at) * 1000), " | cut off by caller" if cut_off else "",
            )

            for v in outcome.guardrail_violations:
                logger.warning(
                    "[%s] guardrail violation: %s (severity=%s)", self._call_id, v.rule_id, v.severity
                )

            if not outcome.spoken_while_streaming:
                await self._speak(outcome.text_to_speak)

            if outcome.decision not in _TERMINAL_DECISIONS:
                self._start_prewarm()
            else:
                self._ended = True
                outcome_name = (outcome.decision_detail or {}).get("outcome", outcome.decision)
                if outcome.decision == "handoff":
                    logger.warning(
                        "[%s] flow graph requested a handoff to %s - not implemented, ending call",
                        self._call_id, (outcome.decision_detail or {}).get("target_agent_id"),
                    )
                    outcome_name = "handoff_not_implemented"
                await self._finalize(outcome_name, status="completed")
                # Ask the task to end (EndWorkerFrame) rather than pushing an
                # EndFrame from here, which only reaches what is downstream and
                # left the stream open (see scenario_processor._hang_up).
                await self.push_frame(EndWorkerFrame(reason="call complete"))

    async def _finalize(self, outcome_name: str, *, status: str) -> None:
        # Local import: routers/agent_voice.py imports this module too (to
        # build the pipeline), so importing it back at module load time would
        # be circular - only needed once, at call end.
        from routers.agent_voice import _finalize_call

        await _finalize_call(self._call_id, self._task_id, outcome_name, self._slot_state, status=status)

    async def handle_disconnect(self) -> None:
        """Called by the router's WebSocket handler if the pipeline ends
        without a terminal decision (operator hung up, network dropped) -
        mirrors the old handler's `finally` block so a dropped call still
        finalizes instead of leaking a concurrency slot forever."""
        if not self._ended:
            self._ended = True
            await self._finalize("call_dropped", status="disconnected")
