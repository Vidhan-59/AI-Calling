import json
import re
import secrets
from typing import Optional

from simple_db import get_cursor, now_iso


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "agent"


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["knowledge_base"] = json.loads(d["knowledge_base"])
    return d


def list_agents() -> list[dict]:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM agents ORDER BY created_at DESC")
        return [_row_to_dict(r) for r in cur.fetchall()]


def get_agent(agent_id: str) -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


def create_agent(
    name: str,
    system_prompt: str,
    welcome_greeting: str,
    voice: str,
    model: str,
    effort: str = "medium",
    knowledge_base: Optional[list[str]] = None,
    agent_id: Optional[str] = None,
) -> dict:
    if not agent_id:
        base_slug = _slugify(name)
        agent_id = base_slug
        with get_cursor() as cur:
            cur.execute("SELECT 1 FROM agents WHERE id = ?", (agent_id,))
            if cur.fetchone():
                agent_id = f"{base_slug}-{secrets.token_hex(3)}"

    timestamp = now_iso()
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO agents
                (id, name, system_prompt, welcome_greeting, voice, model, effort, knowledge_base, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                name,
                system_prompt,
                welcome_greeting,
                voice,
                model,
                effort,
                json.dumps(knowledge_base or []),
                timestamp,
                timestamp,
            ),
        )
    return get_agent(agent_id)


def update_agent(agent_id: str, **fields) -> Optional[dict]:
    existing = get_agent(agent_id)
    if not existing:
        return None

    allowed = {"name", "system_prompt", "welcome_greeting", "voice", "model", "effort", "knowledge_base"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return existing

    if "knowledge_base" in updates:
        updates["knowledge_base"] = json.dumps(updates["knowledge_base"])

    updates["updated_at"] = now_iso()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with get_cursor() as cur:
        cur.execute(
            f"UPDATE agents SET {set_clause} WHERE id = ?",
            (*updates.values(), agent_id),
        )
    return get_agent(agent_id)


def delete_agent(agent_id: str) -> bool:
    with get_cursor() as cur:
        cur.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        return cur.rowcount > 0


def build_system_prompt(agent: dict) -> str:
    """Compose the final system prompt: base instructions + knowledge base.

    This is the "training" mechanism - editing an agent's system_prompt or
    knowledge_base via the API immediately changes how it talks on the very
    next call, with no redeploy and no fine-tuning needed.
    """
    prompt = agent["system_prompt"]
    kb = agent.get("knowledge_base") or []
    if kb:
        facts = "\n".join(f"- {item}" for item in kb)
        prompt += (
            "\n\nUse the following reference information when relevant to the "
            "conversation. Do not mention that you were given this list; just "
            f"use it naturally:\n{facts}"
        )
    return prompt
