"""Local harness for services/voice_pipeline/conversation_processor.py -
verifies the NEW glue code (history threading, guardrail application, Turn/
CallOutcome persistence, TTSSpeakFrame chunking, terminal-decision handling)
without Twilio, real audio, or a live Qwen call. Pipecat's own VAD/STT/
serializer/transport machinery is exercised by pipecat's own test suite, not
re-tested here - this only covers services/voice_pipeline/'s own logic.

Calls the processor's internal methods directly (_bootstrap_greeting,
_handle_transcript) rather than routing synthetic frames through a real
Pipecat Pipeline/PipelineTask, and replaces push_frame with a plain recorder
- process_frame's full StartFrame/pipeline-lifecycle bookkeeping is Pipecat's
own concern, not this module's.
"""

import fakeredis
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import db
import models  # noqa: F401 - registers ORM tables
from db import Base
from services import agent_registry
from services.claude_client import ConversationTurnResult, ToolCallRequest
from services.voice_pipeline.conversation_processor import ConversationAgentProcessor

CONTACT = {"name": "Amit Patel", "terminal": "CT3", "phone_e164": "+919999999999"}


class FakeLLM:
    def __init__(self, responses: list[ConversationTurnResult]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(self, system_prompt, messages, *, model, tools, max_tokens, temperature):
        self.calls.append({"system_prompt": system_prompt, "messages": messages})
        return self._responses.pop(0)


def _turn_result(text: str, tool_calls: list[ToolCallRequest] | None = None) -> ConversationTurnResult:
    return ConversationTurnResult(
        text=text, tool_calls=tool_calls or [], stop_reason="end_turn",
        raw_content=[{"type": "text", "text": text}], tokens_in=10, tokens_out=5,
    )


@pytest.fixture
async def db_session(monkeypatch):
    fake_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(db, "get_redis", lambda: fake_redis)

    # StaticPool: every checkout shares the same underlying connection, so
    # this fixture's own session and any session_scope() call the processor
    # makes internally (a second, independent session object) see the same
    # in-memory database - without it, aiosqlite's `:memory:` gives each new
    # connection a fresh, empty database.
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    monkeypatch.setattr(db, "_engine", engine)
    monkeypatch.setattr(db, "_session_factory", factory)

    async with factory() as session:
        yield session
    await engine.dispose()


class FrameRecorder:
    def __init__(self):
        self.frames: list = []

    async def __call__(self, frame, direction=None):
        self.frames.append(frame)


async def _make_call_row(session) -> tuple[str, int]:
    call = models.Call(provider="mock", provider_call_sid="CA_TEST", to_number=CONTACT["phone_e164"])
    session.add(call)
    await session.flush()
    span = models.AgentSpan(call_id=call.id, agent_id="shift_confirm", agent_version=1)
    session.add(span)
    await session.flush()
    await session.commit()
    return call.id, span.id


async def test_bootstrap_greeting_speaks_and_persists_turn(db_session):
    call_id, agent_span_id = await _make_call_row(db_session)
    flow_graph = agent_registry.get_flow_graph("shift_confirm")
    llm = FakeLLM([_turn_result("Namaste, main ShiftEaze ka automated calling assistant hoon.")])

    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=flow_graph, contact=CONTACT,
        language="hi", llm_call=llm,
    )
    recorder = FrameRecorder()
    processor.push_frame = recorder

    await processor._bootstrap_greeting()

    assert len(llm.calls) == 1
    assert any(getattr(f, "text", None) for f in recorder.frames)
    assert processor._history[-1]["role"] == "assistant"

    turns = (await db_session.scalars(select(models.Turn).where(models.Turn.call_id == call_id))).all()
    assert len(turns) == 1
    assert turns[0].role == "assistant"
    assert turns[0].latency_total_ms is not None


async def test_handle_transcript_mid_flow_reprompts_without_ending_call(db_session):
    """Plain-text reply, no record_slot tool call -> attendance_intent stays
    unfilled -> ask_intent's transitions all fail to match -> reprompt.
    Confirms the ordinary (non-terminal) turn path: history threading, Turn
    persistence, and TTSSpeakFrame chunking - without needing a real tool
    call wired up."""
    call_id, agent_span_id = await _make_call_row(db_session)
    flow_graph = agent_registry.get_flow_graph("shift_confirm")
    # Deliberately no "shift"/"terminal"/etc. in this reply - ask_intent only
    # asks intent, it doesn't restate shift details, and roster_available is
    # False here (no lookup_roster call happened), which would trip the real
    # hallucination_guard if it did.
    llm = FakeLLM([_turn_result("Kya aap kal aane wale hain?")])

    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=flow_graph, contact=CONTACT,
        language="hi", llm_call=llm,
    )
    processor._slot_state.current_node_id = "ask_intent"
    # Bypassing _bootstrap_greeting (not needed for this test) leaves
    # turn_count at 0, which would make run_turn's guardrail check treat this
    # as the call's first turn (disclose_ai_and_recording) - bump it to
    # reflect that a greeting already happened, same fix as
    # tests/integration/test_conversation_agent.py's fixture.
    processor._slot_state.turn_count = 1
    recorder = FrameRecorder()
    processor.push_frame = recorder

    await processor._handle_transcript("Pata nahi abhi.")

    turns = (
        await db_session.scalars(
            select(models.Turn).where(models.Turn.call_id == call_id).order_by(models.Turn.seq)
        )
    ).all()
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[0].was_interrupted is False
    assert processor._ended is False
    assert turns[1].text == "Kya aap kal aane wale hain?"

    tts_frames = [f for f in recorder.frames if hasattr(f, "text")]
    assert any("kal aane" in f.text.lower() for f in tts_frames)


async def test_handle_transcript_ends_call_on_end_call_tool(db_session, monkeypatch):
    """A real end_call tool call (terminal=True in the registry) must finalize
    the Call/CallOutcome and end the pipeline - the one branch of
    _handle_transcript that isn't exercised by the reprompt test above."""
    call_id, agent_span_id = await _make_call_row(db_session)
    flow_graph = agent_registry.get_flow_graph("shift_confirm")
    llm = FakeLLM([
        _turn_result(
            "", tool_calls=[ToolCallRequest(id="t1", name="end_call", input={"outcome": "confirmed", "notes": ""})]
        ),
        _turn_result("Dhanyavaad, aapka din shubh ho."),
    ])

    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=flow_graph, contact=CONTACT,
        language="hi", llm_call=llm,
    )
    processor._slot_state.current_node_id = "confirm_and_close"
    processor._slot_state.turn_count = 1  # not this call's first turn - see the reprompt test above
    recorder = FrameRecorder()
    processor.push_frame = recorder

    finalize_calls = []

    async def fake_finalize_call(call_id_, task_id_, outcome_name, slot_state, *, status="completed"):
        finalize_calls.append((call_id_, task_id_, outcome_name, status))

    import routers.agent_voice as agent_voice_module
    monkeypatch.setattr(agent_voice_module, "_finalize_call", fake_finalize_call)

    await processor._handle_transcript("Haan main aa raha hoon.")

    from pipecat.frames.frames import EndWorkerFrame

    assert processor._ended is True
    # decision_detail's "outcome" key ({"outcome": "confirmed", ...}) wins
    # over the raw decision name ("end_call") - see _handle_transcript.
    assert finalize_calls == [(call_id, "task-1", "confirmed", "completed")]
    assert isinstance(recorder.frames[-1], EndWorkerFrame)
    tts_frames = [f for f in recorder.frames if hasattr(f, "text")]
    assert any("dhanyavaad" in f.text.lower() for f in tts_frames)


async def test_barge_in_marks_next_user_turn_interrupted(db_session):
    from pipecat.frames.frames import VADUserStartedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection

    call_id, agent_span_id = await _make_call_row(db_session)
    flow_graph = agent_registry.get_flow_graph("shift_confirm")
    llm = FakeLLM([_turn_result("Theek hai.")])

    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=flow_graph, contact=CONTACT,
        language="hi", llm_call=llm,
    )
    processor._slot_state.turn_count = 1  # not this call's first turn - see the reprompt test above
    recorder = FrameRecorder()
    processor.push_frame = recorder

    # Simulate: bot is mid-reply, caller starts talking over it (barge-in).
    processor._bot_speaking = True
    await processor.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
    assert processor._pending_interruption is True

    await processor._handle_transcript("Ruko ruko.")

    turns = (
        await db_session.scalars(
            select(models.Turn).where(models.Turn.call_id == call_id, models.Turn.role == "user")
        )
    ).all()
    assert turns[0].was_interrupted is True
    # Flag is consumed, not sticky, for the next turn.
    assert processor._pending_interruption is False


class FakeStreamLLM:
    """Streaming stand-in for local_ai.llm_engine.generate_conversation_turn_stream:
    feeds `text` to on_text in small token-like pieces, recording when each
    speech piece came out relative to the tokens."""

    def __init__(self, text: str):
        self._text = text
        self.tokens_sent = 0

    async def __call__(self, system_prompt, messages, *, model, tools, max_tokens, temperature, on_text):
        for i in range(0, len(self._text), 4):
            self.tokens_sent += 1
            await on_text(self._text[i:i + 4])
        return _turn_result(self._text)


async def test_streamed_reply_starts_speaking_before_llm_finishes(db_session):
    """Layer 1: the first TTS piece is pushed after a few words, while the
    LLM is still producing tokens - not after the whole reply."""
    from pipecat.frames.frames import TTSSpeakFrame

    call_id, agent_span_id = await _make_call_row(db_session)
    reply = "Theek hai, main note kar leti hoon ki aap kal aa rahe hain. Dhanyavaad!"
    llm = FakeStreamLLM(reply)
    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=agent_registry.get_flow_graph("shift_confirm"),
        contact=CONTACT, language="hi", stream_llm_call=llm, stream_replies=True,
    )
    processor._slot_state.current_node_id = "ask_intent"
    processor._slot_state.turn_count = 1  # not the opening turn (that one is held for the disclosure check)

    pushed_at_token: list[int] = []
    pieces: list[str] = []

    async def record(frame, direction=None):
        if isinstance(frame, TTSSpeakFrame):
            pieces.append(frame.text)
            pushed_at_token.append(llm.tokens_sent)

    processor.push_frame = record
    await processor._handle_transcript("Haan main aaunga.")

    assert " ".join(pieces) == reply
    assert len(pieces) >= 3
    total_tokens = -(-len(reply) // 4)
    assert pushed_at_token[0] < total_tokens / 3  # first audio long before the reply is done
    assistant = (await db_session.scalars(
        select(models.Turn).where(models.Turn.call_id == call_id, models.Turn.role == "assistant")
    )).all()
    assert assistant[-1].text == reply
    assert assistant[-1].latency_llm_ttft_ms is not None


async def test_streamed_reply_blocked_mid_stream_speaks_deflection_and_stops(db_session):
    """A guardrail BLOCK on a later piece: what was already said stays said,
    the rule's deflection replaces the rest, nothing after it is spoken."""
    from pipecat.frames.frames import TTSSpeakFrame

    from services.guardrails.output import _HALLUCINATION_GUARD_DEFLECTION

    call_id, agent_span_id = await _make_call_row(db_session)
    # roster_available is False (no lookup_roster this call), so naming a
    # shift time trips hallucination_guard.
    reply = "Ji bilkul samajh gayi. Aapki night shift raat das baje hai. Aur kuch?"
    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=agent_registry.get_flow_graph("shift_confirm"),
        contact=CONTACT, language="hi", stream_llm_call=FakeStreamLLM(reply), stream_replies=True,
    )
    processor._slot_state.current_node_id = "ask_intent"
    processor._slot_state.turn_count = 1
    pieces: list[str] = []

    async def record(frame, direction=None):
        if isinstance(frame, TTSSpeakFrame):
            pieces.append(frame.text)

    processor.push_frame = record
    await processor._handle_transcript("Meri shift kab hai?")

    # The first piece went out before the violation was visible; the piece that
    # contained it (merged with the rest of its clause) was replaced whole.
    assert pieces[0] == "Ji bilkul samajh"
    assert pieces[-1] == _HALLUCINATION_GUARD_DEFLECTION
    assert not any("das baje" in p for p in pieces)


async def test_interruption_stops_the_rest_of_a_streamed_reply(db_session):
    """Barge-in mid-reply: pieces generated after the InterruptionFrame are
    dropped, and the assistant turn is marked interrupted."""
    from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame
    from pipecat.processors.frame_processor import FrameDirection

    call_id, agent_span_id = await _make_call_row(db_session)
    reply = "Ji, main aapko batati hoon, kal ki shift ke baare mein, aur uske baad, aage ki baat karte hain."
    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=agent_registry.get_flow_graph("shift_confirm"),
        contact=CONTACT, language="hi", stream_llm_call=FakeStreamLLM(reply), stream_replies=True,
    )
    processor._slot_state.current_node_id = "ask_intent"
    processor._slot_state.turn_count = 1
    pieces: list[str] = []

    async def record(frame, direction=None):
        if isinstance(frame, TTSSpeakFrame):
            pieces.append(frame.text)
            if len(pieces) == 1:  # caller barges in right after the first piece
                await processor.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    processor.push_frame = record
    await processor._handle_transcript("Haan bolo.")

    assert len(pieces) == 1
    assistant = (await db_session.scalars(
        select(models.Turn).where(models.Turn.call_id == call_id, models.Turn.role == "assistant")
    )).all()
    assert assistant[-1].was_interrupted is True


class FakeStreamToolLLM:
    """Round 1: a lookup_roster tool call, no text. Round 2: streamed text."""

    def __init__(self, text: str):
        self._text = text
        self.rounds = 0

    async def __call__(self, system_prompt, messages, *, model, tools, max_tokens, temperature, on_text):
        self.rounds += 1
        if self.rounds == 1:
            return _turn_result("", tool_calls=[ToolCallRequest(id="t1", name="lookup_roster", input={})])
        for i in range(0, len(self._text), 4):
            await on_text(self._text[i:i + 4])
        return _turn_result(self._text)


async def test_tool_round_speaks_a_filler_before_the_answer(db_session, monkeypatch):
    from pipecat.frames.frames import TTSSpeakFrame

    import services.agents.conversation_agent as agent_module

    async def fake_invoke_tool(name, *, call_id, session, args):
        return {"terminal": "CT3"}

    monkeypatch.setattr(agent_module, "invoke_tool", fake_invoke_tool)
    call_id, agent_span_id = await _make_call_row(db_session)
    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=agent_registry.get_flow_graph("shift_confirm"),
        contact=CONTACT, language="hi", stream_llm_call=FakeStreamToolLLM("Aapka terminal CT3 hai."),
        stream_replies=True,
    )
    processor._slot_state.current_node_id = "ask_intent"
    processor._slot_state.turn_count = 1
    pieces: list[str] = []

    async def record(frame, direction=None):
        if isinstance(frame, TTSSpeakFrame):
            pieces.append(frame.text)

    processor.push_frame = record
    await processor._handle_transcript("Mera terminal kaunsa hai?")

    assert pieces[0] == "जी, एक सेकंड।"
    assert "CT3" in " ".join(pieces[1:])


async def test_opening_streams_once_disclosure_is_said(db_session):
    """The opening turn holds pieces only until the AI/recording disclosure
    has been spoken - the first piece goes out well before the end."""
    from pipecat.frames.frames import TTSSpeakFrame

    call_id, agent_span_id = await _make_call_row(db_session)
    reply = ("Namaste, main ShiftEaze ka automated calling assistant hoon, yeh call record ho sakti hai. "
             "Kya main Amit ji se baat kar rahi hoon? Aapse ek chhoti si baat karni thi.")
    llm = FakeStreamLLM(reply)
    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=agent_registry.get_flow_graph("shift_confirm"),
        contact=CONTACT, language="hi", stream_llm_call=llm, stream_replies=True,
    )
    pushed_at: list[int] = []

    async def record(frame, direction=None):
        if isinstance(frame, TTSSpeakFrame):
            pushed_at.append(llm.tokens_sent)

    processor.push_frame = record
    await processor._bootstrap_greeting()

    assert pushed_at and pushed_at[0] < -(-len(reply) // 4)
    assert processor._history[-1]["content"] == reply


async def test_opening_without_disclosure_gets_it_prepended(db_session):
    from pipecat.frames.frames import TTSSpeakFrame

    call_id, agent_span_id = await _make_call_row(db_session)
    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=agent_registry.get_flow_graph("shift_confirm"),
        contact=CONTACT, language="hi", stream_llm_call=FakeStreamLLM("Namaste Amit ji, kaise hain aap?"),
        stream_replies=True,
    )
    pieces: list[str] = []

    async def record(frame, direction=None):
        if isinstance(frame, TTSSpeakFrame):
            pieces.append(frame.text)

    processor.push_frame = record
    await processor._bootstrap_greeting()

    assert "automated" in pieces[0]
    assert " ".join(pieces).endswith("Namaste Amit ji, kaise hain aap?")


async def test_fragment_whose_reply_was_abandoned_unspoken_is_merged(db_session):
    """Caller pauses mid-sentence, VAD ends the turn, the caller resumes
    before any reply was spoken: that reply is dropped and the next
    transcript is answered together with the first fragment."""
    from pipecat.frames.frames import VADUserStartedSpeakingFrame
    from pipecat.processors.frame_processor import FrameDirection

    call_id, agent_span_id = await _make_call_row(db_session)

    class ResumingLLM:
        def __init__(self):
            self.calls: list[list[dict]] = []

        async def __call__(self, system_prompt, messages, *, model, tools, max_tokens, temperature, on_text):
            self.calls.append([dict(m) for m in messages])
            if len(self.calls) == 1:
                # The caller starts talking again while this reply is generating.
                await processor.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            for word in "Ji, theek hai, main samajh gayi.".split(" "):
                await on_text(word + " ")
            return _turn_result("Ji, theek hai, main samajh gayi.")

    llm = ResumingLLM()
    processor = ConversationAgentProcessor(
        task_id="task-1", call_id=call_id, agent_span_id=agent_span_id,
        agent_id="shift_confirm", flow_graph=agent_registry.get_flow_graph("shift_confirm"),
        contact=CONTACT, language="hi", stream_llm_call=llm, stream_replies=True,
    )
    processor._slot_state.current_node_id = "ask_intent"
    processor._slot_state.turn_count = 1
    processor._history = [{"role": "assistant", "content": "Kya aap kal aayenge?"}]

    async def record(frame, direction=None):
        pass

    processor.push_frame = record
    await processor._handle_transcript("Haan ji,")
    await processor._handle_transcript("main kal aa jaunga.")

    assert len(llm.calls) == 2
    assert llm.calls[1][-1] == {"role": "user", "content": "Haan ji, main kal aa jaunga."}
    assert [m["role"] for m in processor._history] == ["assistant", "user", "assistant"]
