"""Assembles the exact system prompt sent to the Conversation Agent on every
turn, per 03_AGENT_RUNTIME_PROMPT_TEMPLATE.md:

  base_persona + voice_output_rules + language_policy + guardrails
  + agent persona + CURRENT NODE + SLOTS + SUPERVISOR GUIDANCE

Sections 1-5 are stable for the whole call (same agent, same language) and
are meant to sit before a prompt-cache breakpoint; sections 6-8 change every
turn. This module only builds the string - where the cache breakpoint goes
is a claude_client.py concern (system blocks vs one string), see
services/agents/conversation_agent.py.
"""

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from config import PROMPTS_DIR
from services.flow.graph import FlowNode
from services.flow.slots import SlotState

LANGUAGE_NAMES = {"hi": "Hindi", "gu": "Gujarati", "en": "English"}

_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


class PromptResolver:
    def __init__(self, jinja_env: Environment | None = None) -> None:
        self.env = jinja_env or _env

    def _render(self, template_name: str, context: dict) -> str:
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            return ""
        return template.render(**context).strip()

    def render_system_prompt(
        self,
        *,
        agent_id: str,
        guardrail_ids: list[str],
        node: FlowNode,
        slot_state: SlotState,
        contact: dict,
        language: str,
        extra_vars: dict | None = None,
        supervisor_guidance: str | None = None,
    ) -> str:
        shared_ctx = {
            "contact": contact,
            "language": language,
            "language_name": LANGUAGE_NAMES.get(language, language),
            "guardrails": guardrail_ids,
            **(extra_vars or {}),
        }

        sections = [
            self._render("base_persona.j2", shared_ctx),
            self._render("voice_output_rules.j2", shared_ctx),
            self._render("language_policy.j2", shared_ctx),
            self._render("guardrails.j2", shared_ctx),
            self._render(f"agents/{agent_id}/persona.j2", shared_ctx),
            self._render_node_block(
                node=node,
                slot_state=slot_state,
                supervisor_guidance=supervisor_guidance,
            ),
        ]
        return "\n\n".join(s for s in sections if s)

    def _render_node_block(
        self,
        *,
        node: FlowNode,
        slot_state: SlotState,
        supervisor_guidance: str | None,
    ) -> str:
        reprompt_count = slot_state.reprompt_count_for(node.id)
        context = {
            "node": node,
            "missing_slots": slot_state.missing_slots(node),
            "filled_slots": slot_state.filled,
            "reprompt_count": reprompt_count,
            "supervisor_guidance": supervisor_guidance,
        }
        return self._render("node_block.j2", context)


prompt_resolver = PromptResolver()
