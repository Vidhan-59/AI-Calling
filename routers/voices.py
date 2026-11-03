"""Voice Studio backend: create/manage OmniVoice voices and decide which voice
speaks for which scenario/agent. The voices themselves live in the TTS service
(tts_service/); this router is the authenticated front door the browser uses.

  GET  /voice-studio                  the single-page UI (static/voice_studio.html)
  GET  /voices                        library + assignments + assignable owners
  POST /voices/clone                  upload/recorded sample -> new voice
  POST /voices/design                 attributes -> new voice
  POST /voices/preview                speak arbitrary text in a voice (24 kHz WAV)
  PATCH/DELETE /voices/{id}           rename / make default / delete
  GET  /voices/{id}/audio             the voice's reference sample
  PUT  /voices/assignments            assign (or clear) a voice for owner+language
  GET  /voices/health                 TTS service status

Everything except the HTML page requires X-API-Key (services/auth.py): voice
cloning is a sensitive capability, and the page itself holds no data.
"""

import logging
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import BASE_DIR
from services import agent_registry, scenario_engine
from services.auth import require_admin_key
from services.tts import client, voice_registry

logger = logging.getLogger("voices")
router = APIRouter(tags=["voices"])
guard = [Depends(require_admin_key)]

LANGUAGES = ("hi", "gu", "en")
_PAGE = os.path.join(BASE_DIR, "static", "voice_studio.html")


async def _call(coro):
    """Translate TTS-client errors into HTTP errors."""
    try:
        return await coro
    except client.TTSRequestError as e:
        raise HTTPException(e.status, e.detail) from None
    except client.TTSUnavailable as e:
        raise HTTPException(503, str(e)) from None


@router.get("/voice-studio", include_in_schema=False)
async def voice_studio_page():
    return FileResponse(_PAGE, media_type="text/html", headers={"Cache-Control": "no-cache"})


@router.get("/voices/health", dependencies=guard)
async def tts_health():
    return await _call(client.health())


def _owners() -> list[dict]:
    owners = [{"id": sid, "kind": "scenario"} for sid in sorted(scenario_engine.load_scenarios())]
    owners += [{"id": aid, "kind": "agent"} for aid in agent_registry.list_agent_ids()]
    return owners


@router.get("/voices", dependencies=guard)
async def list_voices():
    data = await _call(client.list_voices())
    return {
        **data,
        "assignments": voice_registry.load_assignments(),
        "owners": _owners(),
        "languages": LANGUAGES,
    }


@router.post("/voices/clone", dependencies=guard, status_code=201)
async def clone_voice(
    name: str = Form(min_length=1, max_length=80),
    language: str = Form("hi"),
    description: str = Form("", max_length=300),
    ref_text: str = Form("", max_length=500),
    file: UploadFile = File(...),
):
    audio = await file.read()
    voice = await _call(client.clone_voice(
        name=name, language=language, description=description, ref_text=ref_text,
        filename=file.filename or "reference.wav", audio=audio, content_type=file.content_type or "audio/wav",
    ))
    voice_registry.invalidate()
    return voice


class DesignRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    language: str = "hi"
    instruct: str = Field(min_length=3, max_length=200)
    description: str = Field(default="", max_length=300)


@router.post("/voices/design", dependencies=guard, status_code=201)
async def design_voice(req: DesignRequest):
    voice = await _call(client.design_voice(req.model_dump()))
    voice_registry.invalidate()
    return voice


class PreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    language: str = "hi"
    voice_id: str | None = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@router.post("/voices/preview", dependencies=guard)
async def preview(req: PreviewRequest):
    wav = await _call(client.synthesize_preview_wav(req.text, language=req.language, voice_id=req.voice_id, speed=req.speed))
    return Response(wav, media_type="audio/wav")


class VoiceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=300)
    is_default: bool | None = None


@router.patch("/voices/{voice_id}", dependencies=guard)
async def update_voice(voice_id: str, patch: VoiceUpdate):
    voice = await _call(client.update_voice(voice_id, patch.model_dump(exclude_none=True)))
    voice_registry.invalidate()
    return voice


@router.delete("/voices/{voice_id}", dependencies=guard, status_code=204)
async def delete_voice(voice_id: str):
    await _call(client.delete_voice(voice_id))
    # Drop assignments pointing at it, so they don't silently fall through.
    for key, entry in list(voice_registry.load_assignments().items()):
        if entry["voice_id"] == voice_id:
            owner, _, language = key.rpartition(":")
            voice_registry.set_assignment(owner, language, None)
    voice_registry.invalidate()
    return Response(status_code=204)


@router.get("/voices/{voice_id}/audio", dependencies=guard)
async def voice_audio(voice_id: str):
    return Response(await _call(client.voice_audio(voice_id)), media_type="audio/wav")


class AssignRequest(BaseModel):
    owner: str = Field(min_length=1, max_length=80)  # scenario/agent id, or "*"
    language: str = "*"  # "hi" | "gu" | "en" | "*"
    voice_id: str | None = None  # None clears the assignment
    speed: float | None = Field(default=None, ge=0.5, le=2.0)


@router.put("/voices/assignments", dependencies=guard)
async def assign(req: AssignRequest):
    if req.language != "*" and req.language not in LANGUAGES:
        raise HTTPException(422, f"language must be one of {LANGUAGES} or '*'")
    if req.voice_id is not None:
        known = {v["id"] for v in (await _call(client.list_voices()))["voices"]}
        if req.voice_id not in known:
            raise HTTPException(404, f"unknown voice {req.voice_id!r}")
    return {"assignments": voice_registry.set_assignment(req.owner, req.language, req.voice_id, req.speed)}
