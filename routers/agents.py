from fastapi import APIRouter, HTTPException

from schemas import AgentCreateRequest, AgentUpdateRequest
from services import agent_manager

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("")
def list_agents():
    """List every configured agent (each can be used by a different phone number / call)."""
    return agent_manager.list_agents()


@router.post("")
def create_agent(payload: AgentCreateRequest):
    """Create a new agent. This is the 'training' mechanism: system_prompt + knowledge_base
    define its behavior entirely through the prompt, no model fine-tuning needed."""
    agent = agent_manager.create_agent(
        agent_id=payload.agent_id,
        name=payload.name,
        system_prompt=payload.system_prompt,
        welcome_greeting=payload.welcome_greeting,
        voice=payload.voice,
        model=payload.model,
        effort=payload.effort,
        knowledge_base=payload.knowledge_base,
    )
    return agent


@router.get("/{agent_id}")
def get_agent(agent_id: str):
    agent = agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.put("/{agent_id}")
def update_agent(agent_id: str, payload: AgentUpdateRequest):
    """Edit an agent's prompt/knowledge/voice/model live. Takes effect on the next call."""
    agent = agent_manager.update_agent(agent_id, **payload.model_dump(exclude_unset=True))
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.delete("/{agent_id}")
def delete_agent(agent_id: str):
    if agent_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default agent")
    if not agent_manager.delete_agent(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "deleted", "agent_id": agent_id}
