"""Which LLM engine each call site resolves to when no explicit llm_call is
given, based on local_ai.config.LLM_BACKEND - services/agents/
conversation_agent.py::_default_llm_call (tool-calling capable) and
services/llm_router.py::engine (Track A's plain reply/sentiment calls).
"""

from local_ai import llm_engine as local_llm_engine
from services import claude_client, llm_router
from services.agents import conversation_agent


def test_default_llm_call_resolves_to_claude_by_default(monkeypatch):
    monkeypatch.setattr("local_ai.config.LLM_BACKEND", "claude")
    assert conversation_agent._default_llm_call() is claude_client.generate_conversation_turn


def test_default_llm_call_resolves_to_local_when_configured(monkeypatch):
    monkeypatch.setattr("local_ai.config.LLM_BACKEND", "local")
    assert conversation_agent._default_llm_call() is local_llm_engine.generate_conversation_turn


def test_llm_router_engine_resolves_to_claude_by_default(monkeypatch):
    monkeypatch.setattr("local_ai.config.LLM_BACKEND", "claude")
    assert llm_router.engine() is claude_client


def test_llm_router_engine_resolves_to_local_when_configured(monkeypatch):
    monkeypatch.setattr("local_ai.config.LLM_BACKEND", "local")
    assert llm_router.engine() is local_llm_engine


def test_model_kwarg_passes_through_only_for_claude(monkeypatch):
    monkeypatch.setattr("local_ai.config.LLM_BACKEND", "claude")
    assert llm_router.model_kwarg("claude-opus-5") == "claude-opus-5"

    monkeypatch.setattr("local_ai.config.LLM_BACKEND", "local")
    assert llm_router.model_kwarg("claude-opus-5") is None
