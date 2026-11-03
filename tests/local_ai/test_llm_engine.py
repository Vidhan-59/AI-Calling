"""Unit tests for local_ai/llm_engine.py with httpx mocked - no real Ollama
server or model required to run these."""

import json

import httpx
import pytest

from local_ai import llm_engine


class _FakeResponse:
    def __init__(self, json_body=None, status_code=200):
        self._json_body = json_body or {}
        self.status_code = status_code

    def json(self):
        return self._json_body

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://fake")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


class _FakeAsyncClient:
    """Minimal async context-manager stand-in for httpx.AsyncClient."""

    def __init__(self, response=None, raise_connect_error=False):
        self._response = response
        self._raise_connect_error = raise_connect_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, path, json=None):
        if self._raise_connect_error:
            raise httpx.ConnectError("connection refused")
        return self._response


@pytest.mark.asyncio
async def test_classify_structured_parses_json_response(monkeypatch):
    fake_resp = _FakeResponse({"message": {"content": '{"branch": "yes", "confidence": 0.9}'}})
    monkeypatch.setattr(llm_engine, "_client", lambda: _FakeAsyncClient(fake_resp))

    result = await llm_engine.classify_structured("system", "haan", {"type": "object"})

    assert result == {"branch": "yes", "confidence": 0.9}


@pytest.mark.asyncio
async def test_classify_structured_salvages_json_wrapped_in_prose(monkeypatch):
    # Small local models occasionally wrap JSON in prose despite the format
    # constraint - this is the fallback path that handles it.
    content = 'Sure, here is the answer: {"branch": "no", "confidence": 0.8} Hope that helps!'
    fake_resp = _FakeResponse({"message": {"content": content}})
    monkeypatch.setattr(llm_engine, "_client", lambda: _FakeAsyncClient(fake_resp))

    result = await llm_engine.classify_structured("system", "text", {"type": "object"})

    assert result == {"branch": "no", "confidence": 0.8}


class _SequentialFakeAsyncClient(_FakeAsyncClient):
    """Returns a different queued response on each successive post() call -
    for testing the one-retry-on-malformed-JSON path, which a single fixed
    _FakeAsyncClient response can't exercise."""

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)

    async def post(self, path, json=None):
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_classify_structured_retries_once_on_malformed_json(monkeypatch):
    # Observed live: Ollama occasionally returns invalid JSON despite the
    # format constraint (e.g. right after a model swap in VRAM) - one retry
    # should recover instead of crashing the whole call turn.
    responses = [
        _FakeResponse({"message": {"content": '{"branch": "yes", "confidence": 0.9'}}),  # truncated
        _FakeResponse({"message": {"content": '{"branch": "yes", "confidence": 0.9}'}}),
    ]
    client = _SequentialFakeAsyncClient(responses)
    monkeypatch.setattr(llm_engine, "_client", lambda: client)

    result = await llm_engine.classify_structured("system", "haan", {"type": "object"})

    assert result == {"branch": "yes", "confidence": 0.9}
    assert len(client._responses) == 0  # both queued responses were consumed


@pytest.mark.asyncio
async def test_classify_structured_falls_back_to_unclear_when_retry_also_fails(monkeypatch):
    # Observed live: qwen2.5:7b-instruct reproducibly (not just once) returned
    # invalid JSON for real caller speech. Must not raise here - the caller
    # (services/scenario_engine.py) used to treat that exception as "the
    # operator said nothing" (on_silence()), which is wrong: it heard them,
    # it just couldn't parse the classification. "unclear" is the honest
    # fallback and lets the flow's own re-ask/clarify branch handle it.
    responses = [
        _FakeResponse({"message": {"content": '{"branch": "yes"'}}),
        _FakeResponse({"message": {"content": '{"branch": "yes"'}}),
    ]
    monkeypatch.setattr(llm_engine, "_client", lambda: _SequentialFakeAsyncClient(responses))

    result = await llm_engine.classify_structured("system", "haan", {"type": "object"})

    assert result["branch"] == "unclear"
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_classify_structured_raises_when_ollama_unreachable(monkeypatch):
    monkeypatch.setattr(llm_engine, "_client", lambda: _FakeAsyncClient(raise_connect_error=True))

    with pytest.raises(llm_engine.OllamaUnavailableError, match="Is `ollama serve` running"):
        await llm_engine.classify_structured("system", "text", {"type": "object"})


@pytest.mark.asyncio
async def test_generate_reply_returns_stripped_content(monkeypatch):
    fake_resp = _FakeResponse({"message": {"content": "  Hello there!  "}})
    monkeypatch.setattr(llm_engine, "_client", lambda: _FakeAsyncClient(fake_resp))

    result = await llm_engine.generate_reply("system", [{"role": "user", "content": "hi"}])

    assert result == "Hello there!"


@pytest.mark.asyncio
async def test_health_check_reports_down_when_unreachable(monkeypatch):
    class _RaisingClient(_FakeAsyncClient):
        async def get(self, path):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(llm_engine, "_client", lambda: _RaisingClient())

    health = await llm_engine.health_check()

    assert health == {"ollama_running": False, "model_pulled": False}


# --- generate_conversation_turn (Ollama tool-calling) -----------------------
# Request/response shapes here match a live probe against a real
# qwen2.5:7b-instruct model via `ollama serve` (2026-09-24): tool_calls
# come back as [{"function": {"name", "arguments"}}] with no "id" field,
# and a turn that calls a tool returns empty `content` - the actual spoken
# reply only shows up once the tool result is fed back as a follow-up call.

_RECORD_SLOT_TOOL = {
    "name": "record_slot",
    "description": "Record a confirmed slot value.",
    "input_schema": {
        "type": "object",
        "properties": {"slot": {"type": "string"}, "value": {"type": "string"}},
        "required": ["slot", "value"],
    },
}


def test_claude_tools_to_ollama_translates_input_schema_to_parameters():
    ollama_tools = llm_engine._claude_tools_to_ollama([_RECORD_SLOT_TOOL])

    assert ollama_tools == [
        {
            "type": "function",
            "function": {
                "name": "record_slot",
                "description": "Record a confirmed slot value.",
                "parameters": _RECORD_SLOT_TOOL["input_schema"],
            },
        }
    ]


def test_claude_history_to_ollama_passes_plain_turns_through():
    history = [{"role": "user", "content": "Haan main aa jaunga."}]
    assert llm_engine._claude_history_to_ollama(history) == history


def test_claude_history_to_ollama_translates_tool_use_and_tool_result_blocks():
    # The exact shape services/agents/conversation_agent.py's follow-up
    # round builds after a real tool call.
    history = [
        {"role": "user", "content": "Haan main aa jaunga."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_0", "name": "record_slot", "input": {"slot": "x", "value": "y"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_0", "content": '{"status": "recorded"}'}],
        },
    ]

    translated = llm_engine._claude_history_to_ollama(history)

    assert translated[0] == {"role": "user", "content": "Haan main aa jaunga."}
    assert translated[1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "record_slot", "arguments": {"slot": "x", "value": "y"}}}],
    }
    assert translated[2] == {"role": "tool", "content": '{"status": "recorded"}'}


@pytest.mark.asyncio
async def test_generate_conversation_turn_parses_a_tool_call_response(monkeypatch):
    fake_resp = _FakeResponse(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "record_slot", "arguments": {"slot": "x", "value": "y"}}}],
            },
            "done_reason": "stop",
            "prompt_eval_count": 186,
            "eval_count": 27,
        }
    )
    monkeypatch.setattr(llm_engine, "_client", lambda: _FakeAsyncClient(fake_resp))

    result = await llm_engine.generate_conversation_turn(
        "system prompt", [{"role": "user", "content": "haan"}], tools=[_RECORD_SLOT_TOOL],
    )

    assert result.text == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "record_slot"
    assert result.tool_calls[0].input == {"slot": "x", "value": "y"}
    assert result.tokens_in == 186
    assert result.tokens_out == 27
    # raw_content must be readable by this module's own translator on the
    # follow-up round - see test above for the shape it expects.
    assert result.raw_content == [
        {"type": "tool_use", "id": "local_call_0", "name": "record_slot", "input": {"slot": "x", "value": "y"}}
    ]


@pytest.mark.asyncio
async def test_generate_conversation_turn_parses_a_plain_text_response(monkeypatch):
    fake_resp = _FakeResponse({"message": {"role": "assistant", "content": "Dhanyavaad!"}, "done_reason": "stop"})
    monkeypatch.setattr(llm_engine, "_client", lambda: _FakeAsyncClient(fake_resp))

    result = await llm_engine.generate_conversation_turn(
        "system prompt", [{"role": "user", "content": "haan"}], tools=[],
    )

    assert result.text == "Dhanyavaad!"
    assert result.tool_calls == []
