"""Per-call mutable state the flow engine reasons over: what's been filled,
how many times we've reprompted for what, and where we are in the graph.
Lives in Redis per call_id in the real pipeline (services/pipeline); kept as
a plain in-memory dataclass here so the flow engine has zero dependency on
where its state is persisted.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlotState:
    current_node_id: str = ""
    filled: dict[str, Any] = field(default_factory=dict)
    reprompt_counts: dict[str, int] = field(default_factory=dict)
    turn_count: int = 0
    handoff_count: int = 0
    roster_available: bool = False
    """Set True the first time lookup_roster succeeds this call. Passed to
    the output guardrail's hallucination_guard check - it can only trust a
    stated shift/date/time if this is True."""

    def context(self) -> dict[str, Any]:
        """Flattened view transitions evaluate against: filled slots plus
        the counters transitions commonly reference (reprompts, turn_count)."""
        ctx: dict[str, Any] = dict(self.filled)
        ctx["reprompts"] = self.reprompt_counts.get(self.current_node_id, 0)
        ctx["turn_count"] = self.turn_count
        return ctx

    def fill(self, slot: str, value: Any) -> None:
        self.filled[slot] = value

    def missing_slots(self, node) -> list[str]:
        return [s for s in node.collect if s not in self.filled]

    def bump_reprompt(self, node_id: str) -> int:
        self.reprompt_counts[node_id] = self.reprompt_counts.get(node_id, 0) + 1
        return self.reprompt_counts[node_id]

    def reprompt_count_for(self, node_id: str) -> int:
        return self.reprompt_counts.get(node_id, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_node_id": self.current_node_id,
            "filled": self.filled,
            "reprompt_counts": self.reprompt_counts,
            "turn_count": self.turn_count,
            "handoff_count": self.handoff_count,
            "roster_available": self.roster_available,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlotState":
        return cls(
            current_node_id=data.get("current_node_id", ""),
            filled=data.get("filled", {}),
            reprompt_counts=data.get("reprompt_counts", {}),
            turn_count=data.get("turn_count", 0),
            handoff_count=data.get("handoff_count", 0),
            roster_available=data.get("roster_available", False),
        )
