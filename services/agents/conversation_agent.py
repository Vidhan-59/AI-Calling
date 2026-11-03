"""The Conversation Agent - Claude Haiku, on the critical path, every turn.
Node-scoped prompt (services/flow/resolver.py), <=150 output tokens,
voice-formatted only. This module is the one place that ties together the
flow engine, the tool registry, and the output guardrail into a single
per-turn decision.

Slot-filling mechanic: the model has a synthetic `record_slot` tool
available on every turn (not part of the DB-backed tool registry - it never
touches Odoo or gets idempotency-logged) that it calls whenever it has just
confirmed one of the node's `collect` values. This keeps slot extraction and
spoken reply in the SAME model call instead of a second extraction pass,
which matters for the latency budget.

Real tool calls (lookup_roster, confirm_shift, ...) get ONE resume round:
call the tool(s), feed tool_result back, get final text. No deeper
recursion - a tool that needs a second tool call is a flow-design smell,
not something this loop should paper over.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from services import agent_registry
from services.claude_client import ConversationTurnResult, generate_conversation_turn
from services.flow.graph import FlowGraph
from services.flow.resolver import prompt_resolver
from services.flow.slots import SlotState
from services.guardrails.input import scan_input
from services.guardrails.output import GuardrailViolation, enforce as enforce_guardrails
import services.tools  # noqa: F401 - side effect: registers all @tool functions
from services.tools.registry import ToolTimeoutError, get_tool, invoke_tool, to_claude_tool_schema

logger = logging.getLogger("conversation_agent")

CALL_CONNECTED_SENTINEL = "[CALL_CONNECTED]"
"""Synthetic first "user" message run_turn() is called with to produce the
opening line from the flow graph's entry node objective (there's no fixed
greeting script to render, unlike Track B's scenario_engine.opening_line()).
Shared between routers/agent_voice.py (ConversationRelay path) and
services/voice_pipeline/ (raw Media Streams path) - both need the exact same
bootstrap turn so a call's history matches what was actually said regardless
of which live-call transport handled it."""

RECORD_SLOT_TOOL = {
    "name": "record_slot",
    "description": (
        "Record a single piece of information you have just confirmed from "
        "the caller. Call this as soon as you have clearly confirmed a value "
        "for anything listed under INFORMATION YOU STILL NEED - do not wait "
        "until the end of the call, and do not call it for anything you "
        "have not actually confirmed yet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slot": {"type": "string", "description": "The exact slot name from INFORMATION YOU STILL NEED."},
            "value": {
                "type": "string",
                "description": (
                    "The confirmed value as a short token: true / false for "
                    "yes-no slots, or a short lowercase word like confirm / "
                    "decline / uncertain for categorical slots, or the "
                    "captured text itself for free-text slots."
                ),
            },
        },
        "required": ["slot", "value"],
    },
}

LLMCall = Callable[..., Awaitable[ConversationTurnResult]]


@dataclass
class ToolCallLog:
    tool: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class TurnOutcome:
    text_to_speak: str
    slot_state: SlotState
    guardrail_violations: list[GuardrailViolation] = field(default_factory=list)
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    decision: str = "advance"
    # advance | advance_terminal | reprompt | handoff | transfer_to_human | end_call
    decision_detail: dict[str, Any] | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    input_injection_flagged: bool = False
    """True if the caller's latest utterance matched a prompt-injection
    pattern (services/guardrails/input.scan_input). Flagging only, never
    blocking - the flow engine already structurally limits what a turn can do
    (no tool exists that approves leave, transfers money, etc.), so the right
    response to a flagged utterance is visibility for QA/logging, not
    refusing to answer a caller who tripped a keyword heuristic."""
    spoken_while_streaming: bool = False
    """True when run_turn was given on_speech and has ALREADY handed
    text_to_speak to it piece by piece - the caller must not speak it again."""


SpeechCallback = Callable[[str], Awaitable[None]]

# Said while a tool call (roster lookup etc.) runs - see _SpeechGate.say_filler.
# Gender-neutral on purpose: the persona's voice may be either.
_TOOL_FILLER = {"hi": "जी, एक सेकंड।", "gu": "જી, એક સેકન્ડ.", "en": "Sure, one moment."}


class TurnAbandoned(Exception):
    """Raised by an on_speech callback to stop a streamed turn mid-generation
    (the caller barged in, or kept talking). It propagates out of run_turn
    and closes the LLM stream, so generation stops too. Slot fills and tool
    calls are only applied after a round's stream completes, so a turn
    abandoned in its first round leaves slot_state untouched."""


class _SpeechGate:
    """Streams one turn's reply to `on_speech` piece by piece (Layer 1),
    with the output guardrail applied BEFORE each piece is spoken - spoken
    audio can't be taken back, so every piece is checked together with
    everything already said this turn. On a BLOCK the rule's deflection is
    spoken instead and the rest of the turn is dropped, the streaming
    equivalent of enforce() replacing the whole utterance.

    `hold` buffers everything until the end (no streaming backend).

    `await_disclosure` (the opening turn) holds pieces back only until the
    AI/recording disclosure has been said, then streams the rest: a
    compliant opening starts playing a few words in, and one that never
    discloses gets the legal line prepended at the end, as enforce() would."""

    def __init__(self, on_speech: SpeechCallback, *, guardrail_ids, slot_state: SlotState, hold: bool,
                 await_disclosure: bool = False, piece_budget=None):
        from config import settings
        from services.pipeline.chunker import SpeechChunker

        self._on_speech = on_speech
        self._guardrail_ids = guardrail_ids
        self._slot_state = slot_state
        self.hold = hold
        self.awaiting_disclosure = await_disclosure
        self._held: list[str] = []
        self._chunker = SpeechChunker(
            first_chunk_words=settings.stream_first_chunk_words, min_chunk_words=settings.stream_min_chunk_words,
            budget_words=piece_budget,
        )
        self.spoken: list[str] = []
        self.violations: list[GuardrailViolation] = []
        self.blocked = False

    async def on_text(self, delta: str) -> None:
        if self.hold or self.blocked:
            return
        for piece in self._chunker.feed(delta):
            await self._offer(piece)

    async def end_round(self) -> None:
        if self.hold or self.blocked:
            return
        for piece in self._chunker.flush():
            await self._offer(piece)

    async def _offer(self, piece: str) -> None:
        if not self.awaiting_disclosure:
            await self._speak(piece)
            return
        from services.guardrails.output import _discloses_ai_and_recording

        self._held.append(piece)
        if _discloses_ai_and_recording(" ".join(self._held)):
            self.awaiting_disclosure = False
            held, self._held = self._held, []
            for p in held:
                await self._speak(p)

    async def finish_undisclosed(self, language: str) -> None:
        """End of an opening turn that never disclosed: the legal line first,
        then what the model said."""
        from services.guardrails.output import _DISCLOSURE_PREPEND

        self.awaiting_disclosure = False
        held, self._held = self._held, []
        for p in [_DISCLOSURE_PREPEND.get(language, _DISCLOSURE_PREPEND["hi"]).strip(), *held]:
            await self._speak(p)

    async def say_filler(self, language: str) -> None:
        """A tool call means a second LLM round before the real answer; say
        a short acknowledgement now so the caller isn't left in silence."""
        await self._speak(_TOOL_FILLER.get(language, _TOOL_FILLER["hi"]))

    async def speak_all(self, text: str) -> None:
        from services.pipeline.chunker import split_sentences

        for piece in split_sentences(text):
            self.spoken.append(piece)
            await self._on_speech(piece)

    async def _speak(self, piece: str) -> None:
        from services.guardrails.output import check_output_text
        from services.guardrails.rules import GUARDRAIL_CATALOG, Severity

        candidate = " ".join([*self.spoken, piece])
        blocking = [
            v for v in check_output_text(
                candidate, guardrail_ids=self._guardrail_ids,
                roster_available=self._slot_state.roster_available,
            )
            if v.severity == Severity.BLOCK
        ]
        if blocking:
            self.blocked = True
            top = blocking[0]
            self.violations.append(top)
            rule = GUARDRAIL_CATALOG.get(top.rule_id)
            deflection = top.deflection or (rule.deflection if rule else "Main is baare mein abhi baat nahi kar sakta.")
            self.spoken.append(deflection)
            await self._on_speech(deflection)
            return
        self.spoken.append(piece)
        await self._on_speech(piece)


def _coerce_slot_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return raw


def _default_llm_call() -> LLMCall:
    """Resolves which engine backs every conversation-agent turn, read fresh
    per call (not cached at import/def time) so flipping LLM_BACKEND in .env
    takes effect on the next call, no restart - same reasoning as
    services/scenario_engine.py's own backend switch. Only consulted when
    run_turn's caller doesn't pass an explicit llm_call (e.g. a test's
    FakeLLM), so nothing here affects existing test coverage."""
    from local_ai import config as local_ai_config

    if local_ai_config.LLM_BACKEND == "local":
        from local_ai.llm_engine import generate_conversation_turn as local_generate_conversation_turn

        return local_generate_conversation_turn
    return generate_conversation_turn


def _default_stream_llm_call():
    """Streaming counterpart of _default_llm_call, same LLM_BACKEND switch."""
    from local_ai import config as local_ai_config

    if local_ai_config.LLM_BACKEND == "local":
        from local_ai.llm_engine import generate_conversation_turn_stream as local_stream

        return local_stream
    from services.claude_client import generate_conversation_turn_stream

    return generate_conversation_turn_stream


def _turn_prompt(*, agent_id, flow_graph, slot_state, contact, language, extra_vars=None, supervisor_guidance=None):
    """(system prompt, tool schemas) for the turn at slot_state's current node."""
    config = agent_registry.get_agent_config(agent_id)
    if not slot_state.current_node_id:
        slot_state.current_node_id = flow_graph.entry
    node = flow_graph.node(slot_state.current_node_id)
    system_prompt = prompt_resolver.render_system_prompt(
        agent_id=agent_id,
        guardrail_ids=config.guardrails,
        node=node,
        slot_state=slot_state,
        contact=contact,
        language=language,
        extra_vars=extra_vars,
        supervisor_guidance=supervisor_guidance,
    )
    return system_prompt, [RECORD_SLOT_TOOL] + [to_claude_tool_schema(t) for t in node.tools]


async def prewarm_next_turn(*, agent_id, flow_graph, slot_state, contact, language, history) -> None:
    """Think ahead while the bot is still talking (Layer 1): the next turn's
    prompt only changes by the caller's not-yet-spoken words, so the local
    LLM processes everything else now. Measured: the first turn at a new
    flow node took ~490ms to its first words, ~200ms with the cache warm.
    Local backend only - Claude's own prompt caching is per request."""
    from local_ai import config as local_ai_config

    if local_ai_config.LLM_BACKEND != "local":
        return
    from local_ai.llm_engine import prime_conversation_turn

    system_prompt, tool_schemas = _turn_prompt(
        agent_id=agent_id, flow_graph=flow_graph, slot_state=slot_state, contact=contact, language=language,
    )
    await prime_conversation_turn(system_prompt, history, tools=tool_schemas)


async def run_turn(
    *,
    session,
    call_id: str,
    agent_id: str,
    flow_graph: FlowGraph,
    slot_state: SlotState,
    contact: dict[str, Any],
    language: str,
    history: list[dict[str, Any]],
    extra_vars: dict[str, Any] | None = None,
    supervisor_guidance: str | None = None,
    llm_call: LLMCall | None = None,
    on_speech: SpeechCallback | None = None,
    stream_llm_call=None,
    piece_budget=None,
) -> TurnOutcome:
    """`on_speech` turns on Layer 1 streaming: the reply is spoken piece by
    piece while the LLM is still generating (first piece after a few words),
    through the same guardrails, and the returned TurnOutcome says so via
    spoken_while_streaming. `stream_llm_call` overrides the streaming backend
    (tests); an explicit `llm_call` without one falls back to speaking the
    finished reply in sentence pieces."""
    if on_speech is not None and stream_llm_call is None and llm_call is None:
        stream_llm_call = _default_stream_llm_call()
    llm_call = llm_call or _default_llm_call()
    config = agent_registry.get_agent_config(agent_id)

    latest_user_text = next(
        (m.get("content") for m in reversed(history) if m.get("role") == "user"),
        None,
    )
    input_injection_flagged = bool(
        isinstance(latest_user_text, str)
        and latest_user_text
        and scan_input(latest_user_text).prompt_injection_flagged
    )
    if input_injection_flagged:
        logger.warning(
            "[%s] caller utterance matched a prompt-injection pattern: %r",
            call_id, latest_user_text,
        )

    system_prompt, tool_schemas = _turn_prompt(
        agent_id=agent_id, flow_graph=flow_graph, slot_state=slot_state, contact=contact,
        language=language, extra_vars=extra_vars, supervisor_guidance=supervisor_guidance,
    )
    # configs/agents/*.yaml's llm.conversation.model names a Claude model
    # (e.g. "claude-haiku-4-5") - meaningless to local_ai's Ollama engine,
    # which serves exactly one model (local_ai.config.OLLAMA_MODEL). Pass
    # the configured name through only when actually talking to Claude;
    # None lets a non-Claude llm_call (the local engine, or a test's
    # FakeLLM, which ignores it) fall back to its own default.
    from services.claude_client import generate_conversation_turn_stream as claude_stream

    talks_to_claude = (
        stream_llm_call is claude_stream if stream_llm_call is not None
        else llm_call is generate_conversation_turn
    )
    model_name = config.llm.conversation.model if talks_to_claude else None

    gate = None
    if on_speech is not None:
        gate = _SpeechGate(
            on_speech, guardrail_ids=config.guardrails, slot_state=slot_state,
            hold=stream_llm_call is None,
            await_disclosure=slot_state.turn_count == 0 and "disclose_ai_and_recording" in config.guardrails,
            piece_budget=piece_budget,
        )

    async def call_llm(messages: list[dict[str, Any]]):
        kwargs = dict(
            model=model_name,
            tools=tool_schemas,
            max_tokens=config.llm.conversation.max_tokens,
            temperature=config.llm.conversation.temperature,
        )
        if gate is not None and stream_llm_call is not None:
            out = await stream_llm_call(system_prompt, messages, on_text=gate.on_text, **kwargs)
            await gate.end_round()
            return out
        return await llm_call(system_prompt, messages, **kwargs)

    result = await call_llm(history)

    tool_call_logs: list[ToolCallLog] = []
    terminal_decision: dict[str, Any] | None = None
    final_text = result.text
    tokens_in, tokens_out = result.tokens_in, result.tokens_out

    slot_calls = [tc for tc in result.tool_calls if tc.name == "record_slot"]
    real_tool_calls = [tc for tc in result.tool_calls if tc.name != "record_slot"]
    for sc in slot_calls:
        slot_name = sc.input.get("slot")
        if slot_name:
            slot_state.fill(slot_name, _coerce_slot_value(str(sc.input.get("value", ""))))

    if real_tool_calls:
        logger.info("[%s] tool round: %s", call_id, [tc.name for tc in real_tool_calls])
        if gate is not None and not gate.hold and not gate.awaiting_disclosure and not gate.spoken:
            await gate.say_filler(language)
        tool_result_blocks = []
        for tc in real_tool_calls:
            try:
                tool_output = await invoke_tool(tc.name, call_id=call_id, session=session, args=tc.input)
                if tc.name == "lookup_roster":
                    slot_state.roster_available = True
                tool_call_logs.append(ToolCallLog(tool=tc.name, status="ok", result=tool_output))
                tool_result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": json.dumps(tool_output)}
                )
            except ToolTimeoutError:
                tool_call_logs.append(ToolCallLog(tool=tc.name, status="timeout"))
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "is_error": True,
                        "content": (
                            "Tool timed out. Tell the caller you'll confirm this by SMS "
                            "shortly and move on - do not guess the missing information."
                        ),
                    }
                )
            except Exception as e:  # noqa: BLE001 - surfaced to the model as a tool error, not raised
                tool_call_logs.append(ToolCallLog(tool=tc.name, status="error", error=str(e)))
                tool_result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "is_error": True, "content": f"Tool error: {e}"}
                )

            if get_tool(tc.name).terminal:
                terminal_decision = {"tool": tc.name, "input": tc.input}

        follow_up_messages = history + [
            {"role": "assistant", "content": result.raw_content},
            {"role": "user", "content": tool_result_blocks},
        ]
        follow_up = await call_llm(follow_up_messages)
        for sc in [tc for tc in follow_up.tool_calls if tc.name == "record_slot"]:
            slot_name = sc.input.get("slot")
            if slot_name:
                slot_state.fill(slot_name, _coerce_slot_value(str(sc.input.get("value", ""))))
        if gate is not None and not gate.hold:
            # Streamed: anything said before the tool call was already spoken,
            # so the whole turn's text is both rounds, not just the last one.
            final_text = " ".join(t for t in (final_text, follow_up.text) if t)
        else:
            final_text = follow_up.text or final_text
        tokens_in += follow_up.tokens_in
        tokens_out += follow_up.tokens_out

    safe_text, violations = enforce_guardrails(
        final_text,
        guardrail_ids=config.guardrails,
        roster_available=slot_state.roster_available,
        is_first_turn=slot_state.turn_count == 0,
        language=language,
    )
    if gate is not None:
        if gate.hold:
            await gate.speak_all(safe_text)
        else:
            if gate.awaiting_disclosure:
                await gate.finish_undisclosed(language)
            # What the caller actually heard - each piece already passed the
            # guardrail on its way out (see _SpeechGate).
            safe_text = " ".join(gate.spoken)
            seen = {v.rule_id for v in violations}
            violations += [v for v in gate.violations if v.rule_id not in seen]
    spoken_while_streaming = gate is not None

    slot_state.turn_count += 1

    if terminal_decision:
        return TurnOutcome(
            text_to_speak=safe_text,
            slot_state=slot_state,
            guardrail_violations=violations,
            tool_calls=tool_call_logs,
            decision=terminal_decision["tool"],
            decision_detail=terminal_decision["input"],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            input_injection_flagged=input_injection_flagged,
            spoken_while_streaming=spoken_while_streaming,
        )

    next_id = flow_graph.next_node_id(slot_state.current_node_id, slot_state.context())
    if next_id is None:
        slot_state.bump_reprompt(slot_state.current_node_id)
        decision = "reprompt"
        decision_detail = None
    elif next_id.startswith("HANDOFF:"):
        decision = "handoff"
        decision_detail = {"target_agent_id": next_id.split(":", 1)[1]}
    else:
        slot_state.current_node_id = next_id
        decision = "advance_terminal" if flow_graph.is_terminal(next_id) else "advance"
        decision_detail = {"next_node_id": next_id}

    return TurnOutcome(
        text_to_speak=safe_text,
        slot_state=slot_state,
        guardrail_violations=violations,
        tool_calls=tool_call_logs,
        decision=decision,
        decision_detail=decision_detail,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        input_injection_flagged=input_injection_flagged,
        spoken_while_streaming=spoken_while_streaming,
    )
