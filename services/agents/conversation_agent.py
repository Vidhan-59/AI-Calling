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
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from services import agent_registry
from services.claude_client import ConversationTurnResult, generate_conversation_turn
from services.flow.graph import FlowGraph
from services.flow.resolver import prompt_resolver
from services.flow.slots import SlotState
from services.guardrails.output import GuardrailViolation, enforce as enforce_guardrails
import services.tools  # noqa: F401 - side effect: registers all @tool functions
from services.tools.registry import ToolTimeoutError, get_tool, invoke_tool, to_claude_tool_schema

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


def _coerce_slot_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return raw


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
    llm_call: LLMCall = generate_conversation_turn,
) -> TurnOutcome:
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
    tool_schemas = [RECORD_SLOT_TOOL] + [to_claude_tool_schema(t) for t in node.tools]

    result = await llm_call(
        system_prompt,
        history,
        model=config.llm.conversation.model,
        tools=tool_schemas,
        max_tokens=config.llm.conversation.max_tokens,
        temperature=config.llm.conversation.temperature,
    )

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
        follow_up = await llm_call(
            system_prompt,
            follow_up_messages,
            model=config.llm.conversation.model,
            tools=tool_schemas,
            max_tokens=config.llm.conversation.max_tokens,
            temperature=config.llm.conversation.temperature,
        )
        for sc in [tc for tc in follow_up.tool_calls if tc.name == "record_slot"]:
            slot_name = sc.input.get("slot")
            if slot_name:
                slot_state.fill(slot_name, _coerce_slot_value(str(sc.input.get("value", ""))))
        final_text = follow_up.text or final_text
        tokens_in += follow_up.tokens_in
        tokens_out += follow_up.tokens_out

    safe_text, violations = enforce_guardrails(
        final_text,
        guardrail_ids=config.guardrails,
        roster_available=slot_state.roster_available,
    )

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
    )
