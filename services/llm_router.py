"""Resolves which LLM engine backs Track A's free-form replies and the
shared sentiment classifier (services/claude_client.py vs
local_ai/llm_engine.py), based on local_ai.config.LLM_BACKEND.

Read fresh on every call (not cached at import time) so flipping
LLM_BACKEND in .env takes effect on the next request, no restart - same
reasoning as services/scenario_engine.py's own backend switch.

The rebuild's conversation agent has its own, separate resolution
(services/agents/conversation_agent.py::_default_llm_call) since it needs
tool-calling support that Track A's plain reply/sentiment calls don't -
this module is only for routers/voice.py and routers/calls.py's /chat/test.
"""

from local_ai import config as local_ai_config
from services import claude_client


def engine():
    """Returns the claude_client module or the local_ai.llm_engine module -
    both expose the same generate_reply / generate_reply_stream /
    classify_sentiment function names and signatures (local_ai's absorb
    Claude-only kwargs like `effort` via **_ignored), so callers can use
    the result interchangeably."""
    if local_ai_config.LLM_BACKEND == "local":
        from local_ai import llm_engine

        return llm_engine
    return claude_client


def model_kwarg(agent_model: str) -> str | None:
    """Track A's per-agent `model` field names a Claude model
    (e.g. "claude-opus-5") - meaningless to Ollama, which serves exactly
    one model (local_ai.config.OLLAMA_MODEL). Pass the agent's own choice
    through only when actually talking to Claude; None lets the local
    engine fall back to its own configured default."""
    return agent_model if engine() is claude_client else None
