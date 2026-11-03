from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

import anthropic

from config import ANTHROPIC_API_KEY, SENTIMENT_MODEL

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Lazily construct the Anthropic client so the server can still start
    (health check, agent CRUD, browsing the DB) before a key is configured -
    the clear error only surfaces when something actually tries to talk to
    Claude, e.g. /chat/test or an inbound call."""
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "Missing ANTHROPIC_API_KEY. Set it in your .env file, then restart the "
                "server. Get a key at https://console.anthropic.com/settings/keys"
            )
        _client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def get_client() -> anthropic.AsyncAnthropic:
    """Public accessor for the post-call agents (analyzer/qa/summarizer),
    which need direct client access for structured-output calls that don't
    fit generate_conversation_turn's tool-use shape."""
    return _get_client()

NO_TAG_LEAK_INSTRUCTION = (
    "\n\nDo not include internal or system XML tags in your response. "
    "Respond only with the words you want spoken aloud to the caller - no "
    "labels, no markdown, no stage directions."
)

_VALID_EFFORTS = {"low", "medium", "high"}


async def generate_reply_stream(
    system_prompt: str,
    messages: list[dict],
    model: str,
    effort: str = "medium",
    max_tokens: int = 1024,
) -> AsyncIterator[str]:
    """Stream a conversational reply token-by-token.

    Thinking is explicitly disabled: this is a real-time voice call, and
    waiting on extended thinking tokens before the callee hears anything
    would be a bad user experience. `effort` is capped to "high" because
    Claude Opus 5 only allows disabled thinking at effort <= high.
    """
    effort = effort if effort in _VALID_EFFORTS else "medium"

    async with _get_client().messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt + NO_TAG_LEAK_INSTRUCTION,
        thinking={"type": "disabled"},
        output_config={"effort": effort},
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def generate_reply(
    system_prompt: str,
    messages: list[dict],
    model: str,
    effort: str = "medium",
    max_tokens: int = 1024,
) -> str:
    """Non-streaming variant, used by the text-based test-chat endpoint."""
    chunks = []
    async for chunk in generate_reply_stream(system_prompt, messages, model, effort, max_tokens):
        chunks.append(chunk)
    return "".join(chunks)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ConversationTurnResult:
    text: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str | None = None
    raw_content: list[Any] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0


async def generate_conversation_turn(
    system_prompt: str,
    messages: list[dict],
    *,
    model: str,
    tools: list[dict],
    max_tokens: int = 150,
    temperature: float = 0.4,
) -> ConversationTurnResult:
    """One Conversation Agent turn, tool-use enabled. Non-streaming: the
    Conversation Agent needs to see whether a tool_use block came back
    before it can decide whether to resume (see
    services/agents/conversation_agent.py) - real sentence-level streaming
    to TTS happens downstream of the FINAL text this returns, via
    services/pipeline/chunker.py, not here.
    """
    response = await _get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt + NO_TAG_LEAK_INSTRUCTION,
        tools=tools or [],
        messages=messages,
    )
    text_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCallRequest(id=block.id, name=block.name, input=block.input))

    usage = getattr(response, "usage", None)
    return ConversationTurnResult(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        stop_reason=response.stop_reason,
        raw_content=[b.model_dump() for b in response.content],
        tokens_in=getattr(usage, "input_tokens", 0) or 0,
        tokens_out=getattr(usage, "output_tokens", 0) or 0,
    )


async def generate_conversation_turn_stream(
    system_prompt: str,
    messages: list[dict],
    *,
    model: str,
    tools: list[dict],
    max_tokens: int = 150,
    temperature: float = 0.4,
    on_text: Callable[[str], Awaitable[None]],
) -> ConversationTurnResult:
    """Streaming generate_conversation_turn: text deltas go to `on_text` as
    they arrive (live calls start speaking on the first few words), and the
    full result - tool_use blocks included - is returned at the end, so
    run_turn's tool-resume logic is unchanged."""
    async with _get_client().messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt + NO_TAG_LEAK_INSTRUCTION,
        tools=tools or [],
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            if text:
                await on_text(text)
        response = await stream.get_final_message()

    text_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCallRequest(id=block.id, name=block.name, input=block.input))

    usage = getattr(response, "usage", None)
    return ConversationTurnResult(
        text="".join(text_parts).strip(),
        tool_calls=tool_calls,
        stop_reason=response.stop_reason,
        raw_content=[b.model_dump() for b in response.content],
        tokens_in=getattr(usage, "input_tokens", 0) or 0,
        tokens_out=getattr(usage, "output_tokens", 0) or 0,
    )


async def generate_supervisor_guidance(
    transcript_summary: str,
    *,
    model: str,
    current_objective: str,
) -> str | None:
    """Async, off the critical path (see services/agents/supervisor_agent.py
    for the caller - it must never be awaited inline in the turn loop).
    Sonnet reasoning over the rolling transcript to catch what Haiku
    shouldn't be trusted to self-referee: multi-intent utterances,
    escalation calls, guardrail near-misses."""
    response = await _get_client().messages.create(
        model=model,
        max_tokens=200,
        system=(
            "You are a supervisor watching a live voice-agent phone call. You "
            "never speak to the caller directly. Given the rolling transcript "
            "and the agent's current objective, decide if the frontline agent "
            "needs guidance for its next turn. Respond with ONLY the guidance "
            "sentence if needed, or the exact word NONE if the call is on track."
        ),
        messages=[
            {
                "role": "user",
                "content": f"CURRENT OBJECTIVE: {current_objective}\n\nTRANSCRIPT:\n{transcript_summary}",
            }
        ],
    )
    for block in response.content:
        if block.type == "text":
            text = block.text.strip()
            return None if text.upper() == "NONE" else text
    return None


async def classify_structured(
    system_prompt: str,
    user_text: str,
    schema: dict,
    max_tokens: int = 400,
) -> dict:
    """Generic structured-output classification against a caller-supplied schema.

    Used by services/scenario_engine.py to map an operator utterance onto one
    scripted branch. Runs on the conversation model (Haiku by default) because
    it sits on the live-call critical path - a slow classification is heard by
    the operator as dead air. Structured outputs guarantee the response parses,
    so there is no regex/JSON-repair fallback to get wrong.
    """
    from config import settings

    response = await _get_client().messages.create(
        model=settings.conversation_model,
        max_tokens=max_tokens,
        system=system_prompt,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user_text}],
    )
    for block in response.content:
        if block.type == "text":
            import json

            return json.loads(block.text)
    return {}


_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative"],
        },
        "score": {
            "type": "number",
            "description": "Sentiment score from -1.0 (very negative) to 1.0 (very positive)",
        },
        "emotion": {
            "type": "string",
            "description": "Single word for the dominant emotion, e.g. happy, frustrated, confused, angry, calm",
        },
        "escalate": {
            "type": "boolean",
            "description": "True if the caller sounds angry/frustrated enough that a human agent should step in",
        },
    },
    "required": ["sentiment", "score", "emotion", "escalate"],
    "additionalProperties": False,
}


async def classify_sentiment(text: str) -> dict:
    """Classify the sentiment of a single caller utterance.

    Uses Haiku 4.5 - fast and cheap, which is all a short classification
    task like this needs, keeping the expensive conversational model free
    for actually talking to the caller.
    """
    response = await _get_client().messages.create(
        model=SENTIMENT_MODEL,
        max_tokens=200,
        system=(
            "You classify the sentiment/emotion of a single sentence spoken by a "
            "caller during a phone conversation. Be decisive and terse."
        ),
        output_config={"format": {"type": "json_schema", "schema": _SENTIMENT_SCHEMA}},
        messages=[{"role": "user", "content": text}],
    )
    for block in response.content:
        if block.type == "text":
            import json

            return json.loads(block.text)
    return {"sentiment": "neutral", "score": 0.0, "emotion": "neutral", "escalate": False}
