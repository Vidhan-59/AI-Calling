import fakeredis
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import models  # noqa: F401 - registers ORM tables
from db import Base
from services import agent_registry
from services.agents.conversation_agent import run_turn
from services.claude_client import ConversationTurnResult, ToolCallRequest
from services.flow.slots import SlotState


class FakeLLM:
    """Returns queued ConversationTurnResult objects in order, one per call -
    lets a test script exactly what the model "said" on each turn/round
    without a network call."""

    def __init__(self, responses: list[ConversationTurnResult]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(self, system_prompt, messages, *, model, tools, max_tokens, temperature):
        self.calls.append({"system_prompt": system_prompt, "messages": messages, "tools": tools})
        return self._responses.pop(0)


def _turn_result(text: str, tool_calls: list[ToolCallRequest] | None = None) -> ConversationTurnResult:
    return ConversationTurnResult(
        text=text,
        tool_calls=tool_calls or [],
        stop_reason="end_turn",
        raw_content=[{"type": "text", "text": text}],
        tokens_in=10,
        tokens_out=5,
    )


@pytest.fixture
async def session(monkeypatch):
    fake_redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr("db.get_redis", lambda: fake_redis)
    monkeypatch.setattr("services.tools.roster_tools.get_redis", lambda: fake_redis)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


CONTACT = {"name": "Amit Patel", "terminal": "CT3"}


async def test_record_slot_fills_state_and_advances_node(session):
    flow_graph = agent_registry.get_flow_graph("shift_confirm")
    state = SlotState(current_node_id=flow_graph.entry)
    llm = FakeLLM(
        [
            _turn_result(
                "Namaste, kya main Amit se baat kar raha hoon?",
                [ToolCallRequest(id="t1", name="record_slot", input={"slot": "identity_verified", "value": "true"})],
            )
        ]
    )

    outcome = await run_turn(
        session=session,
        call_id="call-1",
        agent_id="shift_confirm",
        flow_graph=flow_graph,
        slot_state=state,
        contact=CONTACT,
        language="hi",
        history=[{"role": "user", "content": "Haan, main Amit bol raha hoon."}],
        llm_call=llm,
    )

    assert outcome.slot_state.filled["identity_verified"] is True
    assert outcome.decision == "advance"
    assert outcome.decision_detail == {"next_node_id": "state_shift"}
    assert outcome.slot_state.current_node_id == "state_shift"


async def test_no_matching_slot_triggers_reprompt(session):
    flow_graph = agent_registry.get_flow_graph("shift_confirm")
    state = SlotState(current_node_id=flow_graph.entry)
    llm = FakeLLM([_turn_result("Sorry, awaaz clear nahi aayi, phir se bolenge?")])

    outcome = await run_turn(
        session=session,
        call_id="call-1",
        agent_id="shift_confirm",
        flow_graph=flow_graph,
        slot_state=state,
        contact=CONTACT,
        language="hi",
        history=[{"role": "user", "content": "..."}],
        llm_call=llm,
    )

    assert outcome.decision == "reprompt"
    assert outcome.slot_state.reprompt_count_for("greet") == 1
    assert outcome.slot_state.current_node_id == "greet"


async def test_real_tool_call_round_trips_and_sets_roster_available(session):
    from services.tools.roster_tools import seed_mock_roster, RosterInfo

    await seed_mock_roster(
        [
            RosterInfo(
                rgid="RG1", name="Amit Patel", date="2026-08-03", shift_code="D",
                terminal="CT3", reporting_time="08:00", roster_line_id="RL-1",
            )
        ]
    )

    flow_graph = agent_registry.get_flow_graph("shift_confirm")
    # turn_count=1: this fixture represents a turn after the opening greeting
    # (history already has a caller reply) - turn_count=0 would make run_turn
    # treat it as the call's first turn and check for the disclose_ai_and_
    # recording opening line, which is unrelated to what this test covers.
    state = SlotState(current_node_id="state_shift", turn_count=1)
    llm = FakeLLM(
        [
            _turn_result(
                "",
                [ToolCallRequest(id="t1", name="lookup_roster", input={"rgid": "RG1", "date": "2026-08-03"})],
            ),
            _turn_result("Aapki shift kal subah 8 baje CT3 par hai. Kya aap aa rahe hain?"),
        ]
    )

    outcome = await run_turn(
        session=session,
        call_id="call-1",
        agent_id="shift_confirm",
        flow_graph=flow_graph,
        slot_state=state,
        contact=CONTACT,
        language="hi",
        history=[{"role": "user", "content": "Haan bataiye."}],
        llm_call=llm,
    )

    assert outcome.slot_state.roster_available is True
    assert outcome.tool_calls[0].tool == "lookup_roster"
    assert outcome.tool_calls[0].status == "ok"
    assert "CT3" in outcome.text_to_speak
    assert outcome.guardrail_violations == []  # roster_available=True so hallucination_guard doesn't fire


async def test_terminal_tool_call_produces_terminal_decision(session):
    flow_graph = agent_registry.get_flow_graph("shift_confirm")
    state = SlotState(current_node_id="confirm_and_close")
    llm = FakeLLM(
        [
            _turn_result(
                "",
                [ToolCallRequest(id="t1", name="end_call", input={"outcome": "confirmed", "notes": "shift confirmed"})],
            ),
            _turn_result("Dhanyavaad, aapka din shubh ho."),
        ]
    )

    outcome = await run_turn(
        session=session,
        call_id="call-1",
        agent_id="shift_confirm",
        flow_graph=flow_graph,
        slot_state=state,
        contact=CONTACT,
        language="hi",
        history=[{"role": "user", "content": "Haan main aa raha hoon."}],
        llm_call=llm,
    )

    assert outcome.decision == "end_call"
    assert outcome.decision_detail == {"outcome": "confirmed", "notes": "shift confirmed"}


async def test_guardrail_violation_replaces_output_text(session):
    flow_graph = agent_registry.get_flow_graph("shift_confirm")
    state = SlotState(current_node_id="greet")
    llm = FakeLLM([_turn_result("Aapki salary is mahine badh gayi hai.")])

    outcome = await run_turn(
        session=session,
        call_id="call-1",
        agent_id="shift_confirm",
        flow_graph=flow_graph,
        slot_state=state,
        contact=CONTACT,
        language="hi",
        history=[{"role": "user", "content": "Mera salary kitna hai?"}],
        llm_call=llm,
    )

    assert len(outcome.guardrail_violations) > 0
    assert "salary" not in outcome.text_to_speak.lower()
    assert "HR" in outcome.text_to_speak
