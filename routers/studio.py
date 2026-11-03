"""AI Agent Studio API: create and configure voice agents, train them with
scripts / Q&A / demo conversations / documents, and inspect what the audio
pipeline is doing.

  GET  /studio                                   the single-page UI (static/agent_studio.html)
  GET  /studio/meta                              enums, templates, voice-control capability map, use cases
  GET  /studio/stats                             dashboard numbers + recent activity
  GET/POST /studio/agents                        list / create
  GET/PUT/DELETE /studio/agents/{id}             read / update / delete
  POST /studio/agents/{id}/duplicate|status      copy / publish / activate
  GET  /studio/agents/{id}/prompt                the exact system prompt the agent runs with
  GET/POST /studio/agents/{id}/training          training items (script | qa | scenario | document)
  PUT/DELETE /studio/training/{item}
  POST /studio/agents/{id}/training/qa/import    CSV / JSON
  GET  /studio/agents/{id}/training/export       JSON, or CSV of the Q&A pairs
  POST /studio/agents/{id}/knowledge/document    PDF / DOCX / CSV / TXT upload
  POST /studio/agents/{id}/knowledge/url         fetch a public web page
  POST /studio/agents/{id}/knowledge/search      what retrieval would hand the LLM for a question
  GET  /studio/diagnostics/audio                 live-call audio continuity metrics
  GET  /studio/call-config                       can "Call now" work? (masked number, what is missing)
  POST /studio/agents/{id}/call                  dial STUDIO_CALL_NUMBER with the agent (Twilio)
  GET  /studio/calls/{call_id}                   call status + live transcript
  POST /studio/calls/{call_id}/end               hang up
  GET/POST /studio/voice, WS /studio/media       Twilio's webhook + media stream (no admin key: Twilio cannot send it)

Voices themselves (library, design, preview) are routers/voices.py - the Studio
reuses those endpoints rather than duplicating them. Everything except the
HTML page requires X-API-Key (services/auth.py).
"""

import csv
import io
import json
import logging
import os
import subprocess
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile, WebSocket
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator

from config import BASE_DIR
from services import agent_registry, scenario_engine
from services.auth import require_admin_key
from services.studio import calling, knowledge, store, templates, voice_caps
from services.studio.prompt import PERSONALITY_FIELDS, build_system_prompt, sections_loaded

logger = logging.getLogger("studio")
router = APIRouter(tags=["studio"])
guard = [Depends(require_admin_key)]

LANGUAGES = ("hi", "gu", "en")
_PAGE = os.path.join(BASE_DIR, "static", "agent_studio.html")
_PERSONALITY_KEYS = {k for k, _ in PERSONALITY_FIELDS} | {"greeting"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


# ------------------------------------------------------------------ models

class VoiceConfig(BaseModel):
    voice_id: str | None = Field(default=None, max_length=120)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    style: str | None = None
    tone: str | None = None

    @field_validator("style")
    @classmethod
    def _style(cls, v):
        if v and v not in voice_caps.STYLES:
            raise ValueError(f"style must be one of {voice_caps.STYLES}")
        return v or None

    @field_validator("tone")
    @classmethod
    def _tone(cls, v):
        if v and v not in voice_caps.TONES:
            raise ValueError(f"tone must be one of {voice_caps.TONES}")
        return v or None


class BasicInfo(BaseModel):
    purpose: str = Field(default="", max_length=500)
    objective: str = Field(default="", max_length=800)


class AgentIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)
    avatar: str | None = Field(default=None, max_length=150_000)  # an emoji, or a small data: URL image
    use_case_id: str | None = Field(default=None, max_length=80)
    language: str | None = None
    voice: VoiceConfig | None = None
    personality: dict[str, str] | None = None
    system_instructions: str | None = Field(default=None, max_length=12_000)
    basic: BasicInfo | None = None

    @field_validator("language")
    @classmethod
    def _lang(cls, v):
        if v is not None and v not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        return v

    @field_validator("avatar")
    @classmethod
    def _avatar(cls, v):
        if v and v.startswith("data:") and not v.startswith("data:image/"):
            raise ValueError("avatar image must be an image")
        return v

    @field_validator("personality")
    @classmethod
    def _personality(cls, v):
        if v is None:
            return v
        bad = set(v) - _PERSONALITY_KEYS
        if bad:
            raise ValueError(f"unknown personality field(s): {sorted(bad)}")
        for key, text in v.items():
            if len(text) > 4000:
                raise ValueError(f"{key} is longer than 4000 characters")
        return v


class StatusIn(BaseModel):
    status: str


class TrainingIn(BaseModel):
    training_type: str
    title: str = Field(default="", max_length=300)
    content: str = Field(default="", max_length=60_000)
    metadata: dict = Field(default_factory=dict)


class TrainingPatch(BaseModel):
    title: str | None = Field(default=None, max_length=300)
    content: str | None = Field(default=None, max_length=60_000)
    metadata: dict | None = None


class UrlIn(BaseModel):
    url: str = Field(min_length=8, max_length=2000)


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)


# ----------------------------------------------------------------- helpers

def _use_cases() -> list[dict]:
    """Things an agent can be attached to: the scripted ShiftEaze scenarios and
    the config-driven agents already in the app."""
    cases = [{"id": s["id"], "name": s["display_name"], "kind": "scenario"} for s in scenario_engine.list_scenarios()]
    known = {c["id"] for c in cases}
    cases += [{"id": a, "name": a.replace("_", " ").title(), "kind": "agent"} for a in agent_registry.list_agent_ids() if a not in known]
    return cases


def _agent_or_404(agent_id: str) -> dict:
    agent = store.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, "agent not found")
    return agent


def readiness(agent: dict) -> list[str]:
    """What is still missing before an agent can be published."""
    problems = []
    p = agent.get("personality") or {}
    if not (agent.get("system_instructions") or "").strip() and not (p.get("role") or "").strip():
        problems.append("Add system instructions or an agent role.")
    if not (agent.get("basic") or {}).get("objective", "").strip() and not (p.get("objective") or "").strip():
        problems.append("Add a conversation objective.")
    return problems


def _with_meta(agent: dict) -> dict:
    items = store.list_training(agent["id"])
    counts = {t: 0 for t in store.TRAINING_TYPES}
    for it in items:
        counts[it["training_type"]] = counts.get(it["training_type"], 0) + 1
    return {**agent, "training_counts": counts, "ready_to_publish": not readiness(agent), "missing": readiness(agent),
            "prompt_sections": sections_loaded(agent)}


def _scenario_content(metadata: dict, fallback: str) -> str:
    """Demo conversations are edited as turns; the stored text (what retrieval
    and the prompt see) is always derived from them so the two can't drift."""
    turns = metadata.get("turns")
    if not isinstance(turns, list):
        return fallback
    lines = []
    for t in turns:
        if not isinstance(t, dict) or not str(t.get("text", "")).strip():
            continue
        who = "Customer" if str(t.get("speaker", "")).lower().startswith("c") else "AI"
        lines.append(f"{who}: {str(t['text']).strip()}")
    return "\n".join(lines)


def _validate_training(training_type: str, title: str, content: str, metadata: dict) -> tuple[str, dict]:
    if training_type not in store.TRAINING_TYPES:
        raise HTTPException(422, f"training_type must be one of {store.TRAINING_TYPES}")
    if training_type == "qa":
        if not title.strip() or not content.strip():
            raise HTTPException(422, "a Q&A pair needs both a question and an answer")
    elif training_type == "scenario":
        content = _scenario_content(metadata, content)
        if not title.strip():
            raise HTTPException(422, "a demo conversation needs a scenario name")
        if "AI:" not in content or "Customer:" not in content:
            raise HTTPException(422, "a demo conversation needs at least one Customer turn and one AI turn")
    elif training_type == "script":
        if not content.strip():
            raise HTTPException(422, "the script is empty")
        kinds = [k for k, _ in templates.SCRIPT_KINDS]
        if metadata.get("kind") and metadata["kind"] not in kinds:
            raise HTTPException(422, f"script kind must be one of {kinds}")
    return content, metadata


# -------------------------------------------------------------------- pages

@router.get("/studio", include_in_schema=False)
async def studio_page():
    return FileResponse(_PAGE, media_type="text/html", headers={"Cache-Control": "no-cache"})


@router.get("/studio/meta", dependencies=guard)
async def meta():
    return {
        "languages": list(LANGUAGES),
        "statuses": list(store.STATUSES),
        "styles": list(voice_caps.STYLES),
        "tones": list(voice_caps.TONES),
        "design_options": voice_caps.DESIGN_OPTIONS,
        "capabilities": voice_caps.CAPABILITIES,
        "use_cases": _use_cases(),
        "personality_fields": [{"key": k, "label": label} for k, label in PERSONALITY_FIELDS],
        "script_kinds": [{"key": k, "label": label} for k, label in templates.SCRIPT_KINDS],
        "script_example": templates.SCRIPT_EXAMPLE,
        "instruction_templates": templates.INSTRUCTION_TEMPLATES,
        "document_types": [".pdf", ".docx", ".csv", ".txt", ".md"],
    }


@router.get("/studio/stats", dependencies=guard)
async def stats():
    s = store.stats()
    agents = store.list_agents()[:6]
    return {
        **s,
        "success_rate": round(100 * s["calls_successful"] / s["calls_total"]) if s["calls_total"] else None,
        "recent_agents": [{"id": a["id"], "name": a["name"], "status": a["status"], "updated_at": a["updated_at"]} for a in agents],
        "recent_calls": store.list_calls(limit=6),
    }


# ------------------------------------------------------------------ agents

@router.get("/studio/agents", dependencies=guard)
async def list_agents():
    return {"agents": [{**a, "ready_to_publish": not readiness(a)} for a in store.list_agents()]}


@router.post("/studio/agents", dependencies=guard, status_code=201)
async def create_agent(body: AgentIn):
    fields = body.model_dump(exclude_none=True)
    fields["status"] = "draft"
    return _with_meta(store.create_agent(**fields))


@router.get("/studio/agents/{agent_id}", dependencies=guard)
async def get_agent(agent_id: str):
    return _with_meta(_agent_or_404(agent_id))


@router.put("/studio/agents/{agent_id}", dependencies=guard)
async def update_agent(agent_id: str, body: AgentIn):
    _agent_or_404(agent_id)
    fields = body.model_dump(exclude_unset=True, exclude_none=True)
    return _with_meta(store.update_agent(agent_id, **fields))


@router.delete("/studio/agents/{agent_id}", dependencies=guard, status_code=204)
async def delete_agent(agent_id: str):
    if not store.delete_agent(agent_id):
        raise HTTPException(404, "agent not found")
    return Response(status_code=204)


@router.post("/studio/agents/{agent_id}/duplicate", dependencies=guard, status_code=201)
async def duplicate_agent(agent_id: str):
    copy = store.duplicate_agent(agent_id)
    if not copy:
        raise HTTPException(404, "agent not found")
    return _with_meta(copy)


@router.post("/studio/agents/{agent_id}/status", dependencies=guard)
async def set_status(agent_id: str, body: StatusIn):
    agent = _agent_or_404(agent_id)
    if body.status not in store.STATUSES:
        raise HTTPException(422, f"status must be one of {store.STATUSES}")
    if body.status != "draft":
        problems = readiness(agent)
        if body.status == "active" and not agent.get("use_case_id"):
            problems.append("Attach the agent to a use case to make it active.")
        if problems:
            raise HTTPException(422, " ".join(problems))
    return _with_meta(store.update_agent(agent_id, status=body.status))


@router.get("/studio/agents/{agent_id}/prompt", dependencies=guard)
async def agent_prompt(agent_id: str, query: str = Query(default="", max_length=500)):
    """The exact prompt a call would start from; with `query`, including the
    knowledge retrieved for that question - proof the training is applied."""
    agent = _agent_or_404(agent_id)
    retrieved = knowledge.retrieve(agent_id, query) if query.strip() else []
    return {
        "prompt": build_system_prompt(agent, retrieved),
        "sections_loaded": sections_loaded(agent),
        "retrieved": [{"source": r["source"], "label": r["label"], "score": r["score"]} for r in retrieved],
    }


# ---------------------------------------------------------------- training

@router.get("/studio/agents/{agent_id}/training", dependencies=guard)
async def list_training(agent_id: str, type: str | None = None):
    _agent_or_404(agent_id)
    if type and type not in store.TRAINING_TYPES:
        raise HTTPException(422, f"type must be one of {store.TRAINING_TYPES}")
    return {"items": store.list_training(agent_id, type)}


@router.post("/studio/agents/{agent_id}/training", dependencies=guard, status_code=201)
async def add_training(agent_id: str, body: TrainingIn):
    _agent_or_404(agent_id)
    content, metadata = _validate_training(body.training_type, body.title, body.content, body.metadata)
    return store.add_training(agent_id, body.training_type, body.title.strip(), content, metadata)


@router.put("/studio/training/{item_id}", dependencies=guard)
async def update_training(item_id: str, body: TrainingPatch):
    item = store.get_training(item_id)
    if not item:
        raise HTTPException(404, "training item not found")
    title = body.title if body.title is not None else item["title"]
    content = body.content if body.content is not None else item["content"]
    metadata = body.metadata if body.metadata is not None else item["metadata"]
    if item["training_type"] == "document":  # uploaded text: only the title is editable
        return store.update_training(item_id, title=title)
    content, metadata = _validate_training(item["training_type"], title, content, metadata)
    return store.update_training(item_id, title=title.strip(), content=content, metadata=metadata)


@router.delete("/studio/training/{item_id}", dependencies=guard, status_code=204)
async def delete_training(item_id: str):
    if not store.delete_training(item_id):
        raise HTTPException(404, "training item not found")
    return Response(status_code=204)


@router.post("/studio/agents/{agent_id}/training/qa/import", dependencies=guard)
async def import_qa(agent_id: str, file: UploadFile = File(...), mode: str = Form("append")):
    _agent_or_404(agent_id)
    if mode not in ("append", "replace"):
        raise HTTPException(422, "mode must be append or replace")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large")
    try:
        pairs = knowledge.parse_qa_import(file.filename or "qa.csv", data)
    except knowledge.IngestError as e:
        raise HTTPException(422, str(e)) from None
    items = [{"title": p["question"], "content": p["answer"], "metadata": {"category": p["category"]}} for p in pairs]
    if mode == "replace":
        store.replace_training(agent_id, "qa", items)
    else:
        for it in items:
            store.add_training(agent_id, "qa", it["title"], it["content"], it["metadata"])
    return {"imported": len(items), "mode": mode}


@router.get("/studio/agents/{agent_id}/training/export", dependencies=guard)
async def export_training(agent_id: str, format: str = Query("json", pattern="^(json|csv)$")):
    agent = _agent_or_404(agent_id)
    safe = "".join(c if c.isalnum() else "_" for c in agent["name"]) or "agent"
    if format == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["question", "answer", "category"])
        for it in store.list_training(agent_id, "qa"):
            w.writerow([it["title"], it["content"], it["metadata"].get("category", "")])
        return Response(buf.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{safe}_qa.csv"'})
    payload = {
        "agent": {k: agent[k] for k in ("name", "description", "language", "voice", "personality", "system_instructions", "basic")},
        "training": [{k: it[k] for k in ("training_type", "title", "content", "metadata")} for it in store.list_training(agent_id)],
    }
    return Response(json.dumps(payload, ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{safe}_training.json"'})


# --------------------------------------------------------------- knowledge

@router.post("/studio/agents/{agent_id}/knowledge/document", dependencies=guard, status_code=201)
async def upload_document(agent_id: str, file: UploadFile = File(...)):
    _agent_or_404(agent_id)
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large (8 MB max)")
    try:
        text = knowledge.extract_text(file.filename or "", data)
        return knowledge.add_document(agent_id, file.filename or "document", text,
                                      {"source": "upload", "filename": file.filename, "bytes": len(data)})
    except knowledge.IngestError as e:
        raise HTTPException(422, str(e)) from None


@router.post("/studio/agents/{agent_id}/knowledge/url", dependencies=guard, status_code=201)
async def add_url(agent_id: str, body: UrlIn):
    _agent_or_404(agent_id)
    try:
        text = await knowledge.fetch_url_text(body.url)
        return knowledge.add_document(agent_id, body.url, text, {"source": "url", "url": body.url})
    except knowledge.IngestError as e:
        raise HTTPException(422, str(e)) from None


@router.post("/studio/agents/{agent_id}/knowledge/search", dependencies=guard)
async def search_knowledge(agent_id: str, body: SearchIn):
    _agent_or_404(agent_id)
    return {"results": [{"source": r["source"], "label": r["label"], "score": r["score"], "text": r["text"]}
                        for r in knowledge.retrieve(agent_id, body.query)]}


# -------------------------------------------------------------- diagnostics

_gpu_cache: tuple[float, dict] = (0.0, {})


def _gpu_info() -> dict:
    """GPU clocks and throttle state, best effort. A laptop GPU on battery is
    power-capped to a fraction of its clocks, which makes OmniVoice 2-4x slower
    and is the usual reason speech breaks on an otherwise healthy setup."""
    global _gpu_cache
    now = time.monotonic()
    if now - _gpu_cache[0] < 5.0:
        return _gpu_cache[1]
    info: dict = {"available": False}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,power.draw,utilization.gpu,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
        ).stdout.strip().splitlines()[0].split(", ")
        reasons = int(out[5], 16)
        info = {
            "available": True, "name": out[0], "sm_clock_mhz": float(out[1]), "max_sm_clock_mhz": float(out[2]),
            "power_w": float(out[3]), "utilization_pct": float(out[4]),
            "power_capped": bool(reasons & 0x4), "thermal_throttled": bool(reasons & (0x20 | 0x40 | 0x8)),
        }
    except Exception:  # noqa: BLE001 - no NVIDIA GPU / nvidia-smi: just report unavailable
        pass
    _gpu_cache = (now, info)
    return info


@router.get("/studio/diagnostics/audio", dependencies=guard)
async def audio_diagnostics():
    from services.tts import client
    from services.tts.playout import DIAGNOSTICS

    try:
        tts = await client.health()
    except client.TTSUnavailable as e:
        tts = {"status": "unreachable", "error": str(e)}
    return {"playout": DIAGNOSTICS.summary(), "tts_service": tts, "gpu": _gpu_info()}


# -------------------------------------------------------------- phone calls

@router.get("/studio/call-config", dependencies=guard)
async def call_config():
    return calling.config_status()


@router.post("/studio/agents/{agent_id}/call", dependencies=guard)
async def call_now(agent_id: str):
    """Dial STUDIO_CALL_NUMBER with this agent, using its SAVED configuration."""
    agent = _agent_or_404(agent_id)
    problems = readiness(agent)
    if problems:
        raise HTTPException(422, " ".join(problems))
    try:
        return await calling.place_call(agent)
    except calling.CallSetupError as e:
        raise HTTPException(e.status, e.message) from None


@router.get("/studio/calls/{call_id}", dependencies=guard)
async def call_status(call_id: str):
    call = store.get_call(call_id)
    if not call:
        raise HTTPException(404, "call not found")
    call = calling.expire_unanswered(call)
    keys = ("id", "agent_id", "status", "started_at", "ended_at", "duration_secs", "outcome", "error")
    return {k: call[k] for k in keys} | {"messages": store.list_messages(call_id)}


@router.post("/studio/calls/{call_id}/end", dependencies=guard)
async def end_call(call_id: str):
    call = store.get_call(call_id)
    if not call:
        raise HTTPException(404, "call not found")
    if call["status"] not in calling.ACTIVE:
        return {"status": call["status"]}
    try:
        await calling.hang_up(call)
    except calling.CallSetupError as e:
        raise HTTPException(e.status, e.message) from None
    return {"status": "ending"}


@router.api_route("/studio/voice", methods=["GET", "POST"], include_in_schema=False)
async def studio_twiml(request: Request):
    """Twilio webhook: connect the answered call to the Studio media stream.
    Only serves a call that was just placed by /studio/agents/{id}/call."""
    from services.voice_pipeline.scenario_pipeline import build_stream_twiml

    call_id = request.query_params.get("call_id", "")
    call = store.get_call(call_id) if call_id else None
    if not call or call["status"] != "connecting":
        return HTMLResponse('<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>', media_type="application/xml")
    twiml = build_stream_twiml(f"wss://{request.url.hostname}/studio/media", {"callId": call_id})
    return HTMLResponse(content=twiml, media_type="application/xml")


@router.websocket("/studio/media")
async def studio_media(websocket: WebSocket):
    from services.voice_pipeline.studio_pipeline import run_studio_media_pipeline

    await run_studio_media_pipeline(websocket)
