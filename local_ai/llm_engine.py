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
import logging
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from local_ai import config
from services.claude_client import ConversationTurnResult, ToolCallRequest

logger = logging.getLogger("local_ai.llm_engine")


class OllamaUnavailableError(RuntimeError):
    """Ollama isn't running or the model isn't pulled. Distinct from a plain
    RuntimeError so callers (e.g. scenario_engine's fallback path) can catch
    this specifically rather than swallowing every possible failure."""


_shared_client: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    """One persistent client for the process lifetime, not a fresh one per
    call. Latency fix (2026-09-30): the previous `async with
    httpx.AsyncClient(...) as client:` per call opened and tore down a new
    connection (pool init, socket connect) on every single Ollama request -
    real overhead on top of inference time, paid again on every turn even
    though every request goes to the same localhost server. Reusing one
    client keeps a warm connection instead."""
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.AsyncClient(
            base_url=config.OLLAMA_BASE_URL, timeout=config.OLLAMA_TIMEOUT_SECONDS
        )
    return _shared_client


async def aclose_client() -> None:
    """Close the shared client - call from an app shutdown hook. Not
    load-bearing for correctness (process exit reclaims the socket either
    way), just avoids an "unclosed client" resource warning."""
    global _shared_client
    if _shared_client is not None:
        await _shared_client.aclose()
        _shared_client = None


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
        # temperature 0: this is a classifier - the same utterance must get
        # the same branch every time, and sampling only adds variance (and
        # longer, rambling string fields).
        "options": {"num_predict": max_tokens, "temperature": 0},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    try:
        client = _client()
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
        return _parse_classification_json(content)
    except json.JSONDecodeError as first_error:
        # Observed live (2026-09-30), reproducibly across real calls, not a
        # one-off: Ollama's grammar-constrained "format" decoding is a strong
        # prior, not a guarantee - qwen2.5:7b-instruct sometimes emits a
        # truncated/invalid JSON string for real (messy, code-mixed) caller
        # speech even though the exact same call type works fine on clean
        # test input. One retry (a second, independent inference call) gives
        # a real chance of a different, valid completion; if the model
        # reliably breaks on THIS input, the retry will fail the same way,
        # which is exactly why this no longer raises after that - see below.
        logger.warning(
            "classify_structured: first attempt produced invalid JSON (%s), retrying once. raw=%r",
            first_error, content,
        )
        client = _client()
        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        try:
            return _parse_classification_json(content)
        except json.JSONDecodeError as second_error:
            # Both attempts produced invalid JSON. Raising here is what used
            # to happen - services/scenario_engine.py's caller then treated
            # the exception as if the caller had said nothing at all
            # (on_silence(): a generic "no response" line + hang up), which
            # is exactly wrong for "I heard you, I just couldn't parse the
            # classification" - it makes the app look like it's ignoring the
            # caller. Falling back to "unclear" (always a valid branch value
            # per scenario_engine._classification_schema's enum) instead
            # lets the flow's own re-ask/clarify branch handle it naturally.
            # Scoped to this caller's known schema shape (branch/confidence/
            # detected_language/rationale) - scenario_engine.py's `.get(...)`
            # calls already default missing keys, so a partial dict is safe
            # even if a future caller's schema shape differs.
            logger.error(
                "classify_structured: retry also produced invalid JSON (%s) - "
                "falling back to an 'unclear' classification rather than raising. raw=%r",
                second_error, content,
            )
            return {
                "branch": "unclear",
                "confidence": 0.0,
                "detected_language": "unknown",
                "rationale": "local_llm_returned_invalid_json_after_retry",
            }


async def prime_prompt(system_prompt: str, schema: dict) -> None:
    """Run a prompt through Ollama for its KV cache only (1 output token,
    result ignored), so a following request with the same prefix skips
    re-processing it. Deliberately not classify_structured: a 1-token
    output is never valid JSON, and that would trigger its retry path."""
    await _client().post("/api/chat", json={
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "ok"}],
        "format": schema,
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    })


async def prime_conversation_turn(system_prompt: str, messages: list[dict], *, tools: list[dict]) -> None:
    """KV-cache warm-up for the NEXT conversation turn (1 output token,
    result ignored): same system prompt, tools and history the next
    generate_conversation_turn[_stream] call will send, plus a placeholder
    caller message. When the real turn arrives only the caller's new words
    are left to process. Run while the bot is still speaking."""
    await _client().post("/api/chat", json={
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *_claude_history_to_ollama(messages),
            {"role": "user", "content": "."},
        ],
        "tools": _claude_tools_to_ollama(tools) if tools else [],
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    })


def _parse_classification_json(content: str) -> dict:
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
        client = _client()
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
    temperature: float | None = None,
    **_ignored,
) -> AsyncIterator[str]:
    """Streaming counterpart, for parity with claude_client.generate_reply_stream."""
    options = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature  # low = stick to the given facts (Studio calls)
    payload = {
        "model": model or config.OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": True,
        "options": options,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    try:
        client = _client()
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


def _claude_tools_to_ollama(tools: list[dict]) -> list[dict]:
    """services/agents/conversation_agent.py builds `tools` as Claude-shaped
    dicts ({"name","description","input_schema"} - see RECORD_SLOT_TOOL and
    services/tools/registry.py::to_claude_tool_schema). Ollama's /api/chat
    wants OpenAI-style {"type":"function","function":{"name","description",
    "parameters"}} - verified against a live qwen2.5:7b-instruct response
    (2026-09-24), see local_ai's own probe notes in git history for the
    request/response shapes this was built against."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", t["name"]),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _claude_history_to_ollama(messages: list[dict]) -> list[dict]:
    """Translates conversation_agent.py's message history into Ollama's
    shape. Most turns are plain {"role","content": str} and pass through
    unchanged. The one exception is the tool-resume round
    conversation_agent.run_turn builds after a real tool call:

        {"role": "assistant", "content": result.raw_content}   # Claude content blocks
        {"role": "user", "content": tool_result_blocks}        # Claude tool_result blocks

    `result.raw_content` here is whatever THIS module's own
    generate_conversation_turn returned last round (see its own raw_content
    construction below, deliberately built in the same Claude block shape
    so this translator can read its own prior output back) - text blocks
    become the assistant message's `content` string, tool_use blocks become
    its `tool_calls` list; tool_result blocks each become their own
    {"role":"tool","content":...} message, matching what the live probe
    against qwen2.5:7b-instruct confirmed Ollama expects for the follow-up
    round to produce a real spoken reply instead of calling the tool again.
    """
    out: list[dict] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue

        if role == "assistant":
            text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
            tool_calls = [
                {"function": {"name": b.get("name"), "arguments": b.get("input", {})}}
                for b in content
                if b.get("type") == "tool_use"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        else:
            for b in content:
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "content": str(b.get("content", ""))})
                else:
                    out.append({"role": role, "content": str(b.get("content", b))})
    return out


async def generate_conversation_turn(
    system_prompt: str,
    messages: list[dict],
    *,
    model: str | None = None,
    tools: list[dict],
    max_tokens: int = 150,
    temperature: float = 0.4,
) -> ConversationTurnResult:
    """Local (Ollama) equivalent of services.claude_client.generate_conversation_turn
    - same signature and same ConversationTurnResult/ToolCallRequest return
    types, so services/agents/conversation_agent.py's run_turn works
    unchanged regardless of which one it's given as `llm_call`. See
    _claude_history_to_ollama for the message-shape translation this relies
    on, and _claude_tools_to_ollama for the tool-schema translation.
    """
    payload = {
        "model": model or config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *_claude_history_to_ollama(messages),
        ],
        "tools": _claude_tools_to_ollama(tools) if tools else [],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    try:
        client = _client()
        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
    except httpx.ConnectError as e:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Is `ollama serve` running?"
        ) from e

    data = resp.json()
    message = data.get("message", {})
    text = (message.get("content") or "").strip()

    tool_calls: list[ToolCallRequest] = []
    raw_content: list[dict] = []
    if text:
        raw_content.append({"type": "text", "text": text})
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function", {})
        call_id = f"local_call_{i}"
        tool_calls.append(ToolCallRequest(id=call_id, name=fn.get("name", ""), input=fn.get("arguments", {}) or {}))
        raw_content.append({"type": "tool_use", "id": call_id, "name": fn.get("name", ""), "input": fn.get("arguments", {}) or {}})

    return ConversationTurnResult(
        text=text,
        tool_calls=tool_calls,
        stop_reason=data.get("done_reason"),
        raw_content=raw_content,
        tokens_in=data.get("prompt_eval_count", 0) or 0,
        tokens_out=data.get("eval_count", 0) or 0,
    )


async def generate_conversation_turn_stream(
    system_prompt: str,
    messages: list[dict],
    *,
    model: str | None = None,
    tools: list[dict],
    max_tokens: int = 150,
    temperature: float = 0.4,
    on_text: Callable[[str], Awaitable[None]],
) -> ConversationTurnResult:
    """Streaming generate_conversation_turn: every text delta is handed to
    `on_text` the moment Ollama produces it (so speech can start on the first
    few words), and the same ConversationTurnResult - tool calls included -
    is returned once the stream ends. Ollama streams content token by token
    even with tools attached; tool calls arrive whole, in their own chunk."""
    payload = {
        "model": model or config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            *_claude_history_to_ollama(messages),
        ],
        "tools": _claude_tools_to_ollama(tools) if tools else [],
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": temperature},
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
    }
    text_parts: list[str] = []
    raw_tool_calls: list[dict] = []
    final: dict = {}
    try:
        async with _client().stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                message = chunk.get("message", {})
                piece = message.get("content") or ""
                if piece:
                    text_parts.append(piece)
                    await on_text(piece)
                raw_tool_calls.extend(message.get("tool_calls") or [])
                if chunk.get("done"):
                    final = chunk
    except httpx.ConnectError as e:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {config.OLLAMA_BASE_URL}. Is `ollama serve` running?"
        ) from e

    text = "".join(text_parts).strip()
    tool_calls: list[ToolCallRequest] = []
    raw_content: list[dict] = []
    if text:
        raw_content.append({"type": "text", "text": text})
    for i, tc in enumerate(raw_tool_calls):
        fn = tc.get("function", {})
        call_id = f"local_call_{i}"
        tool_calls.append(ToolCallRequest(id=call_id, name=fn.get("name", ""), input=fn.get("arguments", {}) or {}))
        raw_content.append({"type": "tool_use", "id": call_id, "name": fn.get("name", ""), "input": fn.get("arguments", {}) or {}})

    return ConversationTurnResult(
        text=text,
        tool_calls=tool_calls,
        stop_reason=final.get("done_reason"),
        raw_content=raw_content,
        tokens_in=final.get("prompt_eval_count", 0) or 0,
        tokens_out=final.get("eval_count", 0) or 0,
    )


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
        client = _client()
        resp = await client.get("/api/tags")
        resp.raise_for_status()
    except httpx.ConnectError:
        return {"ollama_running": False, "model_pulled": False}

    models = [m["name"] for m in resp.json().get("models", [])]
    pulled = any(m == config.OLLAMA_MODEL or m.startswith(config.OLLAMA_MODEL + ":") for m in models)
    return {"ollama_running": True, "model_pulled": pulled, "available_models": models}
