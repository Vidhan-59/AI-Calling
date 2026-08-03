"""Local response generation via Ollama - a drop-in alternative to
services/claude_client.py.

Deliberately mirrors claude_client.py's function names and signatures
(classify_structured, generate_reply) rather than inventing a new shape, so
services/scenario_engine.py can pick between them with a single if-branch (see
SCENARIO_LLM_BACKEND in local_ai/config.py) instead of two divergent call
patterns.

Requires `ollama serve` running locally (default http://127.0.0.1:11434) and
the model already pulled: `ollama pull phi4-mini`.
"""

import json
from typing import AsyncIterator

import httpx

from local_ai import config


class OllamaUnavailableError(RuntimeError):
    """Ollama isn't running or the model isn't pulled. Distinct from a plain
    RuntimeError so callers (e.g. scenario_engine's fallback path) can catch
    this specifically rather than swallowing every possible failure."""


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=config.OLLAMA_BASE_URL, timeout=config.OLLAMA_TIMEOUT_SECONDS)


async def classify_structured(
    system_prompt: str,
    user_text: str,
    schema: dict,
    max_tokens: int = 400,
) -> dict:
    """Same contract as services/claude_client.classify_structured: given a
    system prompt describing the task/branches and the user's utterance,
    return a dict matching `schema`.

    Ollama's `format` field accepts a JSON schema directly (not just the
    string "json") on the versions this was built against (0.32.x) - that is
    what makes this a real apples-to-apples comparison with Claude's
    structured outputs, not a hand-rolled "hope the model returns valid JSON"
    prompt hack.
    """
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "format": schema,
        "stream": False,
        "options": {"num_predict": max_tokens},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    try:
        async with _client() as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
    except httpx.ConnectError as e:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Is `ollama serve` running?"
        ) from e
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise OllamaUnavailableError(
                f"Model '{config.OLLAMA_MODEL}' is not pulled. Run: ollama pull {config.OLLAMA_MODEL}"
            ) from e
        raise

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Small local models occasionally wrap JSON in prose despite the
        # format constraint - salvage the first {...} block rather than
        # failing the whole classification on a formatting slip.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(content[start : end + 1])
        raise


async def generate_reply(
    system_prompt: str,
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 1024,
    **_ignored,
) -> str:
    """Same contract as services/claude_client.generate_reply (non-streaming).
    `**_ignored` absorbs Claude-only kwargs (e.g. `effort`) so the two engines
    are interchangeable at the call site without an if-branch there too."""
    payload = {
        "model": model or config.OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": False,
        "options": {"num_predict": max_tokens},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    try:
        async with _client() as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
    except httpx.ConnectError as e:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Is `ollama serve` running?"
        ) from e
    return resp.json().get("message", {}).get("content", "").strip()


async def generate_reply_stream(
    system_prompt: str,
    messages: list[dict],
    model: str | None = None,
    max_tokens: int = 1024,
    **_ignored,
) -> AsyncIterator[str]:
    """Streaming counterpart, for parity with claude_client.generate_reply_stream."""
    payload = {
        "model": model or config.OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": True,
        "options": {"num_predict": max_tokens},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    try:
        async with _client() as client:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
    except httpx.ConnectError as e:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Is `ollama serve` running?"
        ) from e


_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        "score": {"type": "number"},
        "emotion": {"type": "string"},
        "escalate": {"type": "boolean"},
    },
    "required": ["sentiment", "score", "emotion", "escalate"],
    "additionalProperties": False,
}


async def classify_sentiment(text: str) -> dict:
    """Same contract as services/claude_client.classify_sentiment - lets
    voice_scenario.py's sentiment annotation stay off Claude too when
    SCENARIO_LLM_BACKEND=local, not just the branch classification."""
    try:
        return await classify_structured(
            system_prompt=(
                "You classify the sentiment/emotion of a single sentence spoken "
                "by a caller during a phone conversation. Be decisive and terse."
            ),
            user_text=text,
            schema=_SENTIMENT_SCHEMA,
            max_tokens=200,
        )
    except Exception:
        return {"sentiment": "neutral", "score": 0.0, "emotion": "neutral", "escalate": False}


async def health_check() -> dict:
    """Is Ollama up, and is the configured model actually pulled? Used by the
    CLI/README setup check rather than on the live-call path."""
    try:
        async with _client() as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
    except httpx.ConnectError:
        return {"ollama_running": False, "model_pulled": False}

    models = [m["name"] for m in resp.json().get("models", [])]
    pulled = any(m == config.OLLAMA_MODEL or m.startswith(config.OLLAMA_MODEL + ":") for m in models)
    return {"ollama_running": True, "model_pulled": pulled, "available_models": models}
