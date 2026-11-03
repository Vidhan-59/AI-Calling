"""Config-driven agent registry - Sec 3.1 of 01_MASTER_ARCHITECTURE.md.

Acceptance test for this whole module: adding a sixth calling scenario must
require ZERO Python changes - one configs/agents/<id>.yaml, one
configs/flows/<id>.yaml, one prompts/agents/<id>/persona.j2. This file is
where that promise either holds or breaks, so config validation is strict
(extra="forbid": an unknown key in the YAML fails loudly instead of being
silently ignored) and nothing here should ever special-case an agent id.

configs/agents/*.yaml is the source of truth; Agent/AgentVersion/Prompt/
FlowGraph rows in the database are a published, auditable snapshot of it
(sync_configs_to_db, called from db.init_db()) so every Call can point at
the exact config that ran even after the YAML file is later edited.
"""

import hashlib
import os
from functools import lru_cache
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict

from config import CONFIGS_DIR, PROMPTS_DIR
from services.flow.graph import FlowGraph


class VoiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["omnivoice"] = "omnivoice"
    # "default" = no preference: Voice Studio's assignment, else the library
    # default. Any other value must be a voice id from Voice Studio.
    voice_id: str = "default"
    speed: float = 1.0


class ConversationLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    max_tokens: int = 150
    temperature: float = 0.4


class SupervisorLLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    enabled: bool = True


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation: ConversationLLMConfig
    supervisor: SupervisorLLMConfig


class LimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_duration_sec: int = 180
    max_turns: int = 24
    max_reprompts_per_slot: int = 2


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempts: int = 3
    backoff: list[str] = ["15m", "2h", "6h"]
    on_no_answer: str = "sms_fallback"


class CallingWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str = "07:00"
    end: str = "21:00"
    tz: str = "Asia/Kolkata"


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: int
    display_name: str
    languages: list[str]
    default_language: str
    voice: dict[str, VoiceConfig]
    llm: LLMConfig
    persona_file: str
    flow: str
    tools: list[str] = []
    guardrails: list[str] = []
    limits: LimitsConfig = LimitsConfig()
    retry: RetryConfig = RetryConfig()
    calling_window: CallingWindow = CallingWindow()
    success_slots: list[str] = []


def _agents_dir() -> str:
    return os.path.join(CONFIGS_DIR, "agents")


def _flows_dir() -> str:
    return os.path.join(CONFIGS_DIR, "flows")


def list_agent_ids() -> list[str]:
    if not os.path.isdir(_agents_dir()):
        return []
    return sorted(
        fname[:-5] for fname in os.listdir(_agents_dir()) if fname.endswith(".yaml")
    )


@lru_cache
def _raw_config_yaml(agent_id: str) -> str:
    path = os.path.join(_agents_dir(), f"{agent_id}.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@lru_cache
def get_agent_config(agent_id: str) -> AgentConfig:
    raw = _raw_config_yaml(agent_id)
    data = yaml.safe_load(raw)
    config = AgentConfig.model_validate(data)
    if config.id != agent_id:
        raise ValueError(
            f"configs/agents/{agent_id}.yaml declares id={config.id!r}, filename must match"
        )
    return config


@lru_cache
def get_flow_graph(agent_id: str) -> FlowGraph:
    config = get_agent_config(agent_id)
    flow_filename = os.path.basename(config.flow)
    path = os.path.join(_flows_dir(), flow_filename)
    with open(path, "r", encoding="utf-8") as f:
        return FlowGraph.from_yaml(agent_id, f.read())


def get_persona_template_path(agent_id: str) -> str:
    config = get_agent_config(agent_id)
    return os.path.join(PROMPTS_DIR, os.path.relpath(config.persona_file, "prompts"))


def clear_config_cache() -> None:
    """Call after editing YAML on disk during local dev / hot-reload."""
    _raw_config_yaml.cache_clear()
    get_agent_config.cache_clear()
    get_flow_graph.cache_clear()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def sync_configs_to_db() -> None:
    """Upsert every configs/agents/*.yaml into Agent/AgentVersion/Prompt/
    FlowGraph. Safe to call on every startup - a no-op if nothing changed
    (compares persona_hash/flow_hash before writing a new version row)."""
    import models
    from db import session_scope
    from sqlalchemy import select

    async with session_scope() as session:
        for agent_id in list_agent_ids():
            clear_config_cache()
            config = get_agent_config(agent_id)
            persona_path = get_persona_template_path(agent_id)
            persona_body = ""
            if os.path.exists(persona_path):
                with open(persona_path, "r", encoding="utf-8") as f:
                    persona_body = f.read()
            flow_yaml_path = os.path.join(_flows_dir(), os.path.basename(config.flow))
            with open(flow_yaml_path, "r", encoding="utf-8") as f:
                flow_yaml_text = f.read()

            persona_hash = _sha256(persona_body)
            flow_hash = _sha256(flow_yaml_text)

            agent_row = await session.get(models.Agent, agent_id)
            if agent_row is None:
                agent_row = models.Agent(
                    id=agent_id,
                    display_name=config.display_name,
                    current_version=config.version,
                    status="published",
                )
                session.add(agent_row)
            else:
                agent_row.display_name = config.display_name
                agent_row.current_version = config.version

            existing_version = await session.scalar(
                select(models.AgentVersion).where(
                    models.AgentVersion.agent_id == agent_id,
                    models.AgentVersion.version == config.version,
                )
            )
            if existing_version is not None and existing_version.persona_hash == persona_hash and (
                existing_version.flow_hash == flow_hash
            ):
                continue  # nothing changed for this version, skip the write

            raw_yaml = _raw_config_yaml(agent_id)
            session.add(
                models.AgentVersion(
                    agent_id=agent_id,
                    version=config.version,
                    config_yaml=raw_yaml,
                    persona_hash=persona_hash,
                    flow_hash=flow_hash,
                )
            )
            session.add(
                models.Prompt(
                    agent_id=agent_id,
                    version=config.version,
                    role="persona",
                    body=persona_body,
                    checksum=persona_hash,
                )
            )
            flow_graph = get_flow_graph(agent_id)
            session.add(
                models.FlowGraph(
                    agent_id=agent_id,
                    version=config.version,
                    definition_json={
                        "entry": flow_graph.entry,
                        "terminal_nodes": sorted(flow_graph.terminal_nodes),
                        "nodes": list(flow_graph.nodes.keys()),
                    },
                )
            )
