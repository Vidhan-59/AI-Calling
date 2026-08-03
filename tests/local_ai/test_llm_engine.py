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
