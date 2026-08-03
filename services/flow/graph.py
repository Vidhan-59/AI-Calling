"""The flow graph: a deterministic state machine of objectives. The LLM
handles natural language *within* a node; it never sees the graph itself,
only the current node's objective (see resolver.py) - that's both the
safety property (it can't wander onto an off-limits topic that isn't in its
context) and the latency property (the prompt stays small).

Transition conditions are a small, safe expression grammar over slot values
and counters - deliberately NOT `eval()` on arbitrary strings, since this
YAML is meant to be editable by whoever owns a scenario, not just engineers.
Supported forms: "always", "slot == value", "slot != value",
"counter >= N" (and <=, >, <). Values are bool/int/quoted-string literals.
"""

from dataclasses import dataclass, field
from typing import Any

import yaml

_COMPARATORS = ("==", "!=", ">=", "<=", ">", "<")


def _parse_literal(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        return raw


def _compare(left: Any, op: str, right: Any) -> bool:
    if left is None:
        # An unfilled slot satisfies nothing except "!= <something concrete>".
        if op == "!=":
            return right is not None
        if op == "==":
            return False
        left = 0
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    raise ValueError(f"Unsupported operator: {op}")


@dataclass
class Transition:
    when: str
    goto: str
    """`goto` is either a node id in this graph, or the sentinel
    "HANDOFF:<agent_id>" - the flow engine itself doesn't act on handoffs,
    it just surfaces the sentinel for the conversation loop to interpret."""

    def matches(self, context: dict[str, Any]) -> bool:
        expr = self.when.strip()
        if expr == "always":
            return True
        for op in _COMPARATORS:
            if op in expr:
                left_raw, right_raw = expr.split(op, 1)
                left_val = context.get(left_raw.strip())
                right_val = _parse_literal(right_raw)
                return _compare(left_val, op, right_val)
        raise ValueError(f"Unparseable transition condition: {self.when!r}")


@dataclass
class FlowNode:
    id: str
    objective: str
    collect: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    on_max_reprompts: str | None = None


@dataclass
class FlowGraph:
    agent_id: str
    entry: str
    nodes: dict[str, FlowNode]
    terminal_nodes: set[str]

    @classmethod
    def from_yaml(cls, agent_id: str, yaml_text: str) -> "FlowGraph":
        data = yaml.safe_load(yaml_text)
        nodes: dict[str, FlowNode] = {}
        for node_id, node_data in data["nodes"].items():
            transitions = [
                Transition(when=t["when"], goto=t["goto"]) for t in node_data.get("transitions", [])
            ]
            nodes[node_id] = FlowNode(
                id=node_id,
                objective=node_data["objective"].strip(),
                collect=node_data.get("collect", []) or [],
                tools=node_data.get("tools", []) or [],
                transitions=transitions,
                on_max_reprompts=node_data.get("on_max_reprompts"),
            )
        entry = data["entry"]
        if entry not in nodes:
            raise ValueError(f"Flow entry node {entry!r} is not defined in nodes")
        terminal_nodes = set(data.get("terminal_nodes", []))
        for t in terminal_nodes:
            if t not in nodes:
                raise ValueError(f"terminal_nodes references undefined node {t!r}")
        return cls(agent_id=agent_id, entry=entry, nodes=nodes, terminal_nodes=terminal_nodes)

    def node(self, node_id: str) -> FlowNode:
        return self.nodes[node_id]

    def is_terminal(self, node_id: str) -> bool:
        return node_id in self.terminal_nodes

    def next_node_id(self, current_node_id: str, context: dict[str, Any]) -> str | None:
        """First matching transition wins. Returns None if nothing matched,
        meaning: stay on this node and reprompt."""
        for transition in self.node(current_node_id).transitions:
            if transition.matches(context):
                return transition.goto
        return None
