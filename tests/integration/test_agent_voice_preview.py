"""POST /agent-voice/preview (routers/agent_voice.py) - the dry-run endpoint
for authoring a new config-driven agent without a live call. Fakes the
Anthropic client at services.claude_client._get_client (called fresh on
every generate_conversation_turn call, so monkeypatching it here - unlike
patching run_turn's llm_call default, which is bound at def time - actually
takes effect) so this runs with no ANTHROPIC_API_KEY configured, matching
how the rest of this pass was verified (see PHASE2_ARCHITECTURE_PROPOSAL.md).
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from db import Base
from routers.agent_voice import PreviewRequest, agent_voice_preview


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text

    def model_dump(self):
        return {"type": "text", "text": self.text}


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]
        self.stop_reason = "end_turn"
        self.usage = SimpleNamespace(input_tokens=10, output_tokens=5)


class _FakeMessages:
    def __init__(self, texts):
        self._texts = list(texts)

    async def create(self, **kwargs):
        text = self._texts.pop(0) if self._texts else "Okay."
        return _FakeResponse(text)


class _FakeClient:
    def __init__(self, texts):
        self.messages = _FakeMessages(texts)


@pytest.fixture(autouse=True)
def _force_claude_backend(monkeypatch):
    # These tests fake the Anthropic client specifically to exercise the
    # Claude code path deterministically. If LLM_BACKEND=local (the default
    # in this project's own .env - see local_ai/config.py),
    # conversation_agent._default_llm_call() would otherwise route to the
    # real Ollama engine instead of the fake, making these tests slow,
    # network-dependent, and not actually testing what they claim to.
    monkeypatch.setattr("local_ai.config.LLM_BACKEND", "claude")


@pytest.fixture
async def in_memory_session_factory(monkeypatch):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        import models  # noqa: F401 - registers ORM tables

        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("routers.agent_voice.get_session_factory", lambda: factory)
    yield factory
    await engine.dispose()


def _fake_claude(monkeypatch, texts):
    client = _FakeClient(texts)
    monkeypatch.setattr("services.claude_client._get_client", lambda: client)


async def test_preview_returns_bootstrap_greeting_and_scripted_turns(monkeypatch, in_memory_session_factory):
    _fake_claude(monkeypatch, ["Namaste, kya main Amit se baat kar raha hoon?", "Dhanyavaad, shift confirm ho gayi."])

    result = await agent_voice_preview(
        PreviewRequest(agent_id="shift_confirm", language="hi", contact={"name": "Amit"}, utterances=["Haan"])
    )

    assert result["agent_id"] == "shift_confirm"
    assert len(result["turns"]) == 2
    assert result["turns"][0]["utterance"] is None  # bootstrap greeting
    assert "Namaste" in result["turns"][0]["text_to_speak"]
    assert result["turns"][1]["utterance"] == "Haan"


async def test_preview_does_not_persist_any_rows(monkeypatch, in_memory_session_factory):
    _fake_claude(monkeypatch, ["Namaste."])

    await agent_voice_preview(PreviewRequest(agent_id="shift_confirm", contact={"name": "Test User"}, utterances=[]))

    import models
    from sqlalchemy import func, select

    async with in_memory_session_factory() as session:
        call_count = await session.scalar(select(func.count(models.Call.id)))
        turn_count = await session.scalar(select(func.count(models.Turn.id)))
    assert call_count == 0
    assert turn_count == 0


async def test_preview_404s_on_unknown_agent_id(in_memory_session_factory):
    with pytest.raises(HTTPException) as exc_info:
        await agent_voice_preview(PreviewRequest(agent_id="does_not_exist", utterances=[]))
    assert exc_info.value.status_code == 404


async def test_preview_stops_after_terminal_decision(monkeypatch, in_memory_session_factory):
    # Every response is plain text with no tool_use, so the flow graph's own
    # transitions (not a tool call) drive the decision - this just confirms
    # the loop stops issuing further turns once a terminal decision is
    # reached, rather than asserting a specific decision value.
    _fake_claude(monkeypatch, ["Namaste."] + ["Okay."] * 10)

    result = await agent_voice_preview(
        PreviewRequest(agent_id="shift_confirm", contact={"name": "Test User"}, utterances=["Haan"] * 10)
    )

    terminal = {"end_call", "transfer_to_human", "handoff"}
    decisions = [t["decision"] for t in result["turns"]]
    if any(d in terminal for d in decisions):
        terminal_index = next(i for i, d in enumerate(decisions) if d in terminal)
        assert terminal_index == len(decisions) - 1
