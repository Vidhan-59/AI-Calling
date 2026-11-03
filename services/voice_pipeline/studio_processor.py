"""Runs one live call for a Studio agent (services/studio/): the agent's saved
instructions and personality, plus the training material retrieved for what the
caller just said, drive an LLM whose reply is spoken in the agent's voice.

Per turn: caller transcript -> BM25 retrieval over the agent's Q&A / scripts /
demo conversations / documents -> system prompt (services/studio/prompt.py) ->
streamed LLM reply -> SpeechChunker pieces (sized by the TTS playout planner,
see services/tts/playout.py) -> TTSSpeakFrame -> OmniVoice -> phone.

It is deliberately much smaller than ConversationAgentProcessor: Studio agents
have no flow graph, slots or tools - just a persona and knowledge. Barge-in,
turn merging and the streamed-reply pattern are the same.

Everything observable is written to the Studio call tables (studio_messages,
studio_traces) through CallTrace, which is what call history will read.
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable

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

from config import settings
from local_ai import text_norm
from services.pipeline.chunker import SpeechChunker, split_sentences
from services.studio import knowledge, store
from services.studio.prompt import build_system_prompt, greeting, sections_loaded
from services.studio.trace import CallTrace
from simple_db import now_iso

logger = logging.getLogger("voice_pipeline.studio")

CONNECTED = "(The call has just connected.)"  # first user turn: Claude requires the history to start with one
MAX_HISTORY_MESSAGES = 16
# A caller question that is (nearly) word for word one of the agent's Q&A questions is answered with the
# stored answer itself: the author wrote exactly what should be said, and a small LLM paraphrasing it
# can garble names and numbers. Fraction of shared words (Jaccard) needed to count as "the same question".
QA_VERBATIM_OVERLAP = 0.7


def exact_qa_answer(utterance: str, retrieved: list[dict]) -> dict | None:
    """The top retrieved Q&A pair if the caller asked essentially that question."""
    if not retrieved or retrieved[0]["source"] != "qa":
        return None
    asked = set(knowledge.tokenize(utterance))
    question = set(knowledge.tokenize(retrieved[0]["label"]))
    if not asked or not question or len(asked | question) == 0:
        return None
    if len(asked & question) / len(asked | question) < QA_VERBATIM_OVERLAP:
        return None
    return retrieved[0]


async def default_stream_reply(system_prompt: str, messages: list[dict]) -> AsyncIterator[str]:
    """Token stream from whichever LLM backs the app (local Qwen or Claude)."""
    from services import llm_router

    from services import claude_client

    engine = llm_router.engine()
    extra = {} if engine is claude_client else {"temperature": 0.3}  # local Qwen: stay on the given facts
    async for piece in engine.generate_reply_stream(
        system_prompt, messages, model=llm_router.model_kwarg(settings.conversation_model),
        max_tokens=settings.studio_reply_max_tokens, **extra,
    ):
        yield piece


class StudioCallProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        agent: dict,
        call_id: str,
        playout=None,
        stream_reply: Callable[[str, list[dict]], AsyncIterator[str]] = default_stream_reply,
        retrieve: Callable[[str, str], list[dict]] = knowledge.retrieve,
        max_call_secs: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._agent = agent
        self._call_id = call_id
        self._language = agent.get("language") or "hi"
        self._playout = playout
        self._stream_reply = stream_reply
        self._retrieve = retrieve
        self._max_call_secs = settings.studio_max_call_secs if max_call_secs is None else max_call_secs
        self._trace = CallTrace(call_id)

        self._history: list[dict] = [{"role": "user", "content": CONNECTED}]
        self._turn_lock = asyncio.Lock()
        self._epoch = 0
        self._bot_speaking = False
        self._turn_in_flight = False
        self._turn_spoken: list[str] = []
        self._first_piece_at: float | None = None
        self._ended = False
        self._started_at = time.monotonic()
        self._vad_stop_at: float | None = None
        self.customer_turns = 0

    # -- frames -----------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self._trace.event("session", "Call connected", f"Agent: {self._agent['name']}", data={
                "language": self._language, "voice": (self._agent.get("voice") or {}),
                "instruction_sections": sections_loaded(self._agent),
            })
            self._trace.state("connected")
            store.update_call(self._call_id, status="in_progress")
            self.create_task(self._greet(), "greeting")
            if self._max_call_secs:
                self.create_task(self._watchdog(), "watchdog")
            return

        if isinstance(frame, InterruptionFrame):
            self._epoch += 1  # barge-in: whatever the current reply still has to say is dropped
            self._trace.event("audio_output", "Caller interrupted", "Playback stopped", status="interrupted")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._trace.state("listening")
            if not self._bot_speaking and self._turn_in_flight:
                # The caller kept talking after VAD closed their turn and the bot hasn't started
                # speaking: drop the pending reply; the fragments are answered together.
                self._epoch += 1
                if self._turn_spoken:
                    await self.broadcast_interruption()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_stop_at = time.monotonic()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._trace.state("speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._trace.state("listening")

        if isinstance(frame, TranscriptionFrame):
            self.create_task(self._on_transcript(frame.text), "turn")
            return

        await self.push_frame(frame, direction)

    # -- speaking ------------------------------------------------------------

    async def _speak_piece(self, piece: str, epoch: int) -> None:
        if epoch != self._epoch:
            return  # barged in on: stop talking
        text = text_norm.normalize_for_speech(piece, self._language)
        if not text:
            return
        if self._first_piece_at is None:
            self._first_piece_at = time.monotonic()
        self._turn_spoken.append(text)
        if self._playout is not None:
            self._playout.release(text)
        await self.push_frame(TTSSpeakFrame(text), FrameDirection.DOWNSTREAM)

    async def _greet(self) -> None:
        async with self._turn_lock:
            text = greeting(self._agent)
            epoch = self._epoch
            self._turn_spoken = []
            for piece in split_sentences(text) or [text]:
                await self._speak_piece(piece, epoch)
            said = " ".join(self._turn_spoken) or text
            self._history.append({"role": "assistant", "content": said})
            self._record("agent", said, {"kind": "greeting"})
            self._trace.event("response", "Opening greeting", said)
            self._start_prewarm()

    def _start_prewarm(self) -> None:
        """While the agent talks, have the local LLM read the agent's fixed
        prompt (persona, instructions, history) so the next turn only has the
        caller's new words and the retrieved facts left to process."""
        from local_ai import config as local_ai_config

        if self._stream_reply is not default_stream_reply or local_ai_config.LLM_BACKEND != "local" or self._ended:
            return

        async def go() -> None:
            try:
                from local_ai.llm_engine import prime_conversation_turn

                await prime_conversation_turn(
                    build_system_prompt(self._agent, None, language=self._language),
                    self._history[-MAX_HISTORY_MESSAGES:], tools=[],
                )
            except Exception as e:  # noqa: BLE001 - only an optimisation
                logger.debug("[%s] LLM prewarm skipped: %r", self._call_id, e)

        self.create_task(go(), "llm_prewarm")

    # -- a turn -----------------------------------------------------------------

    async def _on_transcript(self, utterance: str) -> None:
        utterance = (utterance or "").strip()
        if not utterance or self._ended:
            return
        async with self._turn_lock:
            if self._ended:
                return
            heard_at = time.monotonic()
            stt_ms = int((heard_at - self._vad_stop_at) * 1000) if self._vad_stop_at else None
            self.customer_turns += 1
            self._record("customer", utterance)
            self._trace.event("speech_recognition", "Transcription", utterance, duration_ms=stt_ms)
            last = self._history[-1]
            if last["role"] == "user" and last["content"] != CONNECTED:
                last["content"] = f"{last['content']} {utterance}"  # previous reply was dropped unspoken
                utterance = last["content"]
            else:
                self._history.append({"role": "user", "content": utterance})
            self._trace.state("processing")

            retrieved = await asyncio.to_thread(self._retrieve, self._agent["id"], utterance)
            self._trace.event(
                "knowledge", "Knowledge retrieval", f"{len(retrieved)} item(s) matched",
                data={"matches": [{"source": r["source"], "label": r["label"], "score": r["score"]} for r in retrieved]},
            )
            prompt = build_system_prompt(self._agent, retrieved, language=self._language)
            self._trace.event("prompt", "System prompt built", f"{len(prompt)} characters", data={
                "sections": sections_loaded(self._agent), "retrieved": len(retrieved)})

            exact = exact_qa_answer(utterance, retrieved)
            if exact:
                answer = exact["text"].split("\nA:", 1)[-1].strip()
                self._trace.event("response", "Answered from the Q&A pair", answer, data={"verbatim": True, "question": exact["label"]})
                reply = await self._say_verbatim(answer)
            else:
                reply = await self._reply(prompt, heard_at)
            if reply is None:
                if self._turn_spoken:
                    said = " ".join(self._turn_spoken)
                    self._history.append({"role": "assistant", "content": said})
                    self._record("agent", said, {"interrupted": True})
                return
            self._history.append({"role": "assistant", "content": reply})
            self._record("agent", reply)
            self._start_prewarm()
            if len(self._history) > MAX_HISTORY_MESSAGES + 1:
                self._history = self._history[:1] + self._history[-MAX_HISTORY_MESSAGES:]

    async def _say_verbatim(self, text: str) -> str | None:
        epoch = self._epoch
        self._turn_spoken, self._first_piece_at = [], None
        for piece in split_sentences(text) or [text]:
            await self._speak_piece(piece, epoch)
        return None if epoch != self._epoch else text

    async def _reply(self, prompt: str, heard_at: float) -> str | None:
        """Stream one reply to speech. Returns what was said, or None if the
        caller cut in before it finished."""
        epoch = self._epoch
        self._turn_spoken, self._first_piece_at = [], None
        chunker = SpeechChunker(
            first_chunk_words=settings.stream_first_chunk_words, min_chunk_words=settings.stream_min_chunk_words,
            budget_words=self._playout.next_budget_words if self._playout is not None else None,
        )
        self._turn_in_flight = True
        first_token_ms = None
        full: list[str] = []
        t0 = time.monotonic()
        try:
            self._trace.event("llm", "LLM request", f"{len(self._history)} message(s) in context")
            stream = self._stream_reply(prompt, self._history[-MAX_HISTORY_MESSAGES:])
            try:
                async for delta in stream:
                    if epoch != self._epoch:
                        return None
                    if first_token_ms is None:
                        first_token_ms = int((time.monotonic() - t0) * 1000)
                    full.append(delta)
                    for piece in chunker.feed(delta):
                        await self._speak_piece(piece, epoch)
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose:
                    await aclose()
            for piece in chunker.flush():
                await self._speak_piece(piece, epoch)
        except Exception as e:  # noqa: BLE001 - a failed LLM must not kill the call
            logger.exception("[%s] reply failed", self._call_id)
            self._trace.event("error", "LLM failed", str(e), status="error")
            fallback = ((self._agent.get("personality") or {}).get("fallback") or "").strip() or {
                "hi": "माफ़ कीजिए, मुझे दोबारा बताइए।", "gu": "માફ કરશો, ફરીથી કહેશો?"}.get(self._language, "Sorry, could you say that again?")
            await self._speak_piece(fallback, epoch)
            return fallback
        finally:
            self._turn_in_flight = False
        if epoch != self._epoch:
            return None
        text = "".join(full).strip()
        to_first_ms = int(((self._first_piece_at or time.monotonic()) - heard_at) * 1000)
        self._trace.event("llm", "LLM first token", "", duration_ms=first_token_ms)
        self._trace.event("response", "Response generated", text, duration_ms=int((time.monotonic() - t0) * 1000),
                          data={"first_audio_queued_ms_after_transcript": to_first_ms})
        logger.info("[%s] reply: first-token=%sms first-piece=%dms", self._call_id, first_token_ms, to_first_ms)
        return text

    # -- bookkeeping --------------------------------------------------------------

    def _record(self, speaker: str, text: str, metadata: dict | None = None) -> None:
        self._trace.message(speaker, text, metadata)

    async def _watchdog(self) -> None:
        await asyncio.sleep(self._max_call_secs)
        if not self._ended:
            self._trace.event("session", "Maximum call length reached", f"{int(self._max_call_secs)}s", status="warning")
            await self.push_frame(EndWorkerFrame(reason="max call length"))

    def finish(self, reason: str = "completed") -> None:
        """Pipeline over (caller hung up / we ended it): close the call row."""
        if self._ended:
            return
        self._ended = True
        duration = round(time.monotonic() - self._started_at, 1)
        outcome = "completed" if self.customer_turns else "no_conversation"
        store.update_call(self._call_id, status="completed", ended_at=now_iso(), duration_secs=duration, outcome=outcome)
        self._trace.event("session", "Call ended", reason, duration_ms=duration * 1000,
                          data={"customer_turns": self.customer_turns, "outcome": outcome})
        self._trace.state("ended")
        logger.info("[%s] studio call ended: %ss, %d caller turn(s), outcome=%s", self._call_id, duration, self.customer_turns, outcome)
