"""Terminal tools - calling one of these ends the Conversation Agent's
turn-taking loop for this agent. The actual mechanics (SIP REFER warm
transfer, closing an AgentSpan and opening the next one, updating Call
status) are performed by services/agents/handoff.py, which the turn manager
calls once it sees a terminal tool_call in the model's response. The tool
bodies here are intentionally thin: they validate/normalize the model's
stated intent and return it as a structured decision, they do not reach
into telephony or DB state themselves - that would split the transfer
mechanics across two places and both would need the same call_id/session
plumbing anyway, so handoff.py, which already has that context, is the one
right place for it.

Handoff rule enforced in the conversation agent, not here: max 2
transfer_to_agent handoffs per call, then transfer_to_human is the only
option left.
"""

from services.tools.registry import tool


@tool(terminal=True)
async def transfer_to_human(reason: str, target: str = "terminal_supervisor") -> dict:
    return {"decision": "transfer_to_human", "reason": reason, "target": target}


@tool(terminal=True)
async def transfer_to_agent(
    target_agent_id: str,
    reason: str,
    carry_slots: list[str],
    context_summary: str,
) -> dict:
    """Receiving agent gets carry_slots + context_summary ONLY - never the
    raw transcript. That's enforced by handoff.py only copying these two
    fields into the new AgentSpan's context, not the Turn history."""
    return {
        "decision": "transfer_to_agent",
        "target_agent_id": target_agent_id,
        "reason": reason,
        "carry_slots": carry_slots,
        "context_summary": context_summary,
    }


@tool(terminal=True)
async def end_call(outcome: str, notes: str) -> dict:
    return {"decision": "end_call", "outcome": outcome, "notes": notes}
