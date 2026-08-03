import pytest

from services.flow.graph import FlowGraph
from services.flow.slots import SlotState

SAMPLE_FLOW_YAML = """
entry: greet
nodes:
  greet:
    objective: Greet and confirm identity.
    collect: [identity_verified]
    transitions:
      - when: "identity_verified == true"
        goto: state_shift
      - when: "identity_verified == false"
        goto: wrong_person
      - when: "reprompts >= 2"
        goto: schedule_callback_node
  state_shift:
    objective: State the shift and ask if they will attend.
    tools: [lookup_roster]
    collect: [attendance_intent]
    transitions:
      - when: "attendance_intent == \\"confirm\\""
        goto: confirm_and_close
      - when: "attendance_intent == \\"decline\\""
        goto: capture_reason
      - when: "intent == \\"wants_ot\\""
        goto: "HANDOFF:ot_request"
  capture_reason:
    objective: Capture the decline reason with empathy.
    collect: [decline_reason]
    transitions:
      - when: always
        goto: close_decline
  wrong_person:
    objective: Politely end the call.
    transitions: []
  schedule_callback_node:
    objective: Offer a callback later.
    transitions: []
  confirm_and_close:
    objective: Confirm and close.
    transitions: []
  close_decline:
    objective: Close after capturing decline.
    transitions: []
terminal_nodes: [wrong_person, schedule_callback_node, confirm_and_close, close_decline]
"""


@pytest.fixture
def graph() -> FlowGraph:
    return FlowGraph.from_yaml("shift_confirm", SAMPLE_FLOW_YAML)


def test_entry_and_terminal_nodes(graph: FlowGraph):
    assert graph.entry == "greet"
    assert graph.is_terminal("wrong_person")
    assert not graph.is_terminal("greet")


def test_transition_on_identity_confirmed(graph: FlowGraph):
    assert graph.next_node_id("greet", {"identity_verified": True}) == "state_shift"


def test_transition_on_identity_declined(graph: FlowGraph):
    assert graph.next_node_id("greet", {"identity_verified": False}) == "wrong_person"


def test_transition_falls_through_to_reprompt_gate(graph: FlowGraph):
    assert graph.next_node_id("greet", {"reprompts": 2}) == "schedule_callback_node"


def test_no_matching_transition_returns_none_for_reprompt(graph: FlowGraph):
    assert graph.next_node_id("greet", {}) is None


def test_string_literal_condition_with_quotes(graph: FlowGraph):
    assert graph.next_node_id("state_shift", {"attendance_intent": "confirm"}) == "confirm_and_close"
    assert graph.next_node_id("state_shift", {"attendance_intent": "decline"}) == "capture_reason"


def test_handoff_sentinel_is_surfaced_not_acted_on(graph: FlowGraph):
    next_id = graph.next_node_id("state_shift", {"intent": "wants_ot"})
    assert next_id == "HANDOFF:ot_request"


def test_always_transition(graph: FlowGraph):
    assert graph.next_node_id("capture_reason", {}) == "close_decline"


def test_entry_node_missing_raises():
    bad_yaml = "entry: nope\nnodes:\n  greet:\n    objective: x\n"
    with pytest.raises(ValueError):
        FlowGraph.from_yaml("bad", bad_yaml)


def test_terminal_node_referencing_undefined_node_raises():
    bad_yaml = (
        "entry: greet\nnodes:\n  greet:\n    objective: x\n"
        "terminal_nodes: [does_not_exist]\n"
    )
    with pytest.raises(ValueError):
        FlowGraph.from_yaml("bad", bad_yaml)


def test_slot_state_missing_slots_and_reprompt_tracking(graph: FlowGraph):
    state = SlotState(current_node_id="greet")
    node = graph.node("greet")
    assert state.missing_slots(node) == ["identity_verified"]
    state.fill("identity_verified", True)
    assert state.missing_slots(node) == []
    assert state.reprompt_count_for("greet") == 0
    state.bump_reprompt("greet")
    state.bump_reprompt("greet")
    assert state.reprompt_count_for("greet") == 2
    ctx = state.context()
    assert ctx["reprompts"] == 2
    assert ctx["identity_verified"] is True
