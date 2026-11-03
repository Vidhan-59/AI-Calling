"""OmniVoice TTS service (run: `uvicorn tts_service.app:app --port 8100`).

  POST /v1/synthesize               text + voice -> raw PCM16 8 kHz (telephony) or WAV 24 kHz (preview)
  GET  /v1/voices                   list the voice library
  POST /v1/voices/clone             multipart: reference audio (+ optional transcript) -> saved voice
  POST /v1/voices/design            attributes ("female, low pitch") -> saved voice
  PATCH/DELETE /v1/voices/{id}      rename / set default / delete
  GET  /v1/voices/{id}/audio        the voice's reference sample
  GET  /health                      readiness + queue depth (used by the main app's startup check)

Auth: when TTS_SERVICE_TOKEN is set every /v1 call needs `X-TTS-Token`.
"""

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from tts_service import audio as aud
from tts_service import settings, voices
from tts_service.engine import QueueFull, create_engine

logger = logging.getLogger("tts_service")
engine = create_engine()

# Sentence used to render a design voice's reference clip, per language.
DESIGN_SAMPLE_TEXT = {
    "hi": "नमस्ते, मैं आपकी सहायक बोल रही हूँ। आज आपकी शिफ्ट के बारे में एक छोटी सी जानकारी देनी है।",
    "gu": "નમસ્તે, હું તમારી સહાયક બોલી રહી છું. આજે તમારી શિફ્ટ વિશે એક નાની માહિતી આપવાની છે.",
    "en": "Hello, this is your assistant calling. I have a short update about your shift today.",
}
SUPPORTED_LANGUAGES = ("hi", "gu", "en")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.load()
    watcher = asyncio.create_task(engine.idle_watcher())
    yield
    watcher.cancel()


app = FastAPI(title="OmniVoice TTS service", lifespan=lifespan)


async def require_token(x_tts_token: str | None = Header(default=None)) -> None:
    if settings.TOKEN and not hmac.compare_digest(x_tts_token or "", settings.TOKEN):
        raise HTTPException(401, "missing or invalid X-TTS-Token")


auth = [Depends(require_token)]


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    language: str = "hi"
    voice_id: str | None = None  # None -> the library default
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    num_step: int | None = Field(default=None, ge=4, le=64)
    format: str = Field(default="pcm8k", pattern="^(pcm8k|wav24k)$")


def _resolve_voice(voice_id: str | None) -> voices.Voice:
    vid = voice_id or voices.default_id()
    if not vid:
        raise HTTPException(409, "no voice selected and no default voice set - create one in Voice Studio")
    try:
        return voices.get(vid)
    except voices.VoiceNotFound:
        raise HTTPException(404, f"unknown voice_id {vid!r}") from None


@app.get("/health")
async def health():
    return {
        "status": "ok" if engine.loaded else "loading",
        "engine": engine.name,
        "model": settings.MODEL,
        "device": settings.DEVICE,
        "queue_depth": engine.queue_depth,
        "on_gpu": engine.on_gpu,
        "idle_offload_secs": engine.idle_offload_secs,
        "last_synth_ms": engine.last_synth_ms,
        "voices": len(voices.list_voices()),
        "default_voice_id": voices.default_id(),
    }


@app.post("/v1/warm", dependencies=auth, status_code=202)
async def warm():
    """Fire-and-forget hint: a request is likely soon - make sure the model is
    on the GPU. Never fails the caller (it is only an optimisation)."""
    try:
        await engine.warm()
    except QueueFull:
        pass
    return {"on_gpu": engine.on_gpu}


@app.post("/v1/synthesize", dependencies=auth)
async def synthesize(req: SynthesizeRequest):
    if len(req.text) > settings.MAX_TEXT_CHARS:
        raise HTTPException(413, f"text longer than {settings.MAX_TEXT_CHARS} chars - send one sentence at a time")
    voice = _resolve_voice(req.voice_id)
    try:
        wav = await engine.synthesize(req.text, voice.id, req.language, req.speed, req.num_step)
    except QueueFull:
        raise HTTPException(503, "TTS queue full", headers={"Retry-After": "1"}) from None
    except Exception as e:  # noqa: BLE001
        logger.exception("synthesis failed (voice=%s)", voice.id)
        raise HTTPException(500, f"synthesis failed: {e}") from None
    wav = aud.trim_silence(wav, engine.sample_rate)
    wav = aud.fade_edges(aud.normalize_loudness(wav), engine.sample_rate)
    headers = {"X-Voice-Id": voice.id, "X-Voice-Version": str(voice.version)}
    if req.format == "wav24k":
        out = aud.to_wav_bytes(wav, engine.sample_rate)
        return Response(out, media_type="audio/wav", headers=headers)
    tel = aud.resample(wav, engine.sample_rate, settings.TELEPHONY_RATE)
    headers.update({"X-Sample-Rate": str(settings.TELEPHONY_RATE), "X-Duration": f"{aud.duration_secs(tel, settings.TELEPHONY_RATE):.3f}"})
    return Response(aud.to_pcm16(tel), media_type="audio/L16", headers=headers)


@app.get("/v1/voices", dependencies=auth)
async def list_voices():
    default = voices.default_id()
    return {"voices": [v.to_dict(default) for v in voices.list_voices()], "default_voice_id": default}


@app.get("/v1/voices/{voice_id}", dependencies=auth)
async def get_voice(voice_id: str):
    try:
        return voices.get(voice_id).to_dict(voices.default_id())
    except voices.VoiceNotFound:
        raise HTTPException(404, "voice not found") from None


@app.get("/v1/voices/{voice_id}/audio", dependencies=auth)
async def voice_audio(voice_id: str):
    try:
        voices.get(voice_id)
        return Response(voices.reference_path(voice_id).read_bytes(), media_type="audio/wav")
    except (voices.VoiceNotFound, FileNotFoundError):
        raise HTTPException(404, "voice not found") from None


async def _finish_voice(voice: voices.Voice, ref_wav: bytes, ref_text: str | None) -> dict:
    """Persist the reference, build the OmniVoice prompt, roll back on failure
    so a failed clone never leaves a voice that errors on every call."""
    voices.save(voice, reference_wav=ref_wav)
    try:
        voice.ref_text = await engine.make_prompt(voice.id, voices.reference_path(voice.id), ref_text)
        voices.save(voice)
    except QueueFull:
        voices.delete(voice.id)
        raise HTTPException(503, "TTS busy - try again in a moment") from None
    except Exception as e:  # noqa: BLE001
        voices.delete(voice.id)
        logger.exception("voice creation failed")
        raise HTTPException(500, f"could not build voice: {e}") from None
    if voices.default_id() is None:
        voices.set_default(voice.id)
    return voice.to_dict(voices.default_id())


def _check_language(language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(422, f"language must be one of {SUPPORTED_LANGUAGES}")
    return language


@app.post("/v1/voices/clone", dependencies=auth, status_code=201)
async def clone_voice(
    name: str = Form(min_length=1, max_length=80),
    language: str = Form("hi"),
    description: str = Form("", max_length=300),
    ref_text: str = Form(""),
    file: UploadFile = File(...),
):
    _check_language(language)
    try:
        samples, rate = aud.decode_upload(await file.read())
    except aud.AudioError as e:
        raise HTTPException(422, str(e)) from None
    samples = aud.trim_silence(aud.resample(samples, rate, settings.MODEL_RATE), settings.MODEL_RATE)
    secs = aud.duration_secs(samples, settings.MODEL_RATE)
    if secs < settings.MIN_REF_SECS:
        raise HTTPException(422, f"reference too short ({secs:.1f}s) - use 3-10 seconds of clear speech")
    if secs > settings.MAX_REF_SECS:
        raise HTTPException(422, f"reference too long ({secs:.0f}s) - OmniVoice works best with 3-10 seconds")
    voice = voices.Voice(id=voices.new_id(name), name=name.strip(), kind="clone", language=language, description=description)
    return await _finish_voice(voice, aud.to_wav_bytes(aud.normalize_peak(samples), settings.MODEL_RATE), ref_text.strip() or None)


class DesignRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    language: str = "hi"
    instruct: str = Field(min_length=3, max_length=200)  # e.g. "female, young adult, moderate pitch"
    description: str = Field(default="", max_length=300)
    sample_text: str | None = Field(default=None, max_length=300)


@app.post("/v1/voices/design", dependencies=auth, status_code=201)
async def design_voice(req: DesignRequest):
    _check_language(req.language)
    text = req.sample_text or DESIGN_SAMPLE_TEXT[req.language]
    try:
        sample = await engine.design_sample(text, req.instruct, req.language)
    except QueueFull:
        raise HTTPException(503, "TTS busy - try again in a moment") from None
    except ValueError as e:  # OmniVoice rejects unsupported attribute words
        raise HTTPException(422, str(e)) from None
    except Exception as e:  # noqa: BLE001
        logger.exception("voice design failed")
        raise HTTPException(500, f"voice design failed: {e}") from None
    sample = aud.normalize_peak(aud.trim_silence(sample, engine.sample_rate))
    voice = voices.Voice(
        id=voices.new_id(req.name), name=req.name.strip(), kind="design", language=req.language,
        description=req.description, instruct=req.instruct,
    )
    return await _finish_voice(voice, aud.to_wav_bytes(sample, engine.sample_rate), text)


class VoiceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=300)
    is_default: bool | None = None


@app.patch("/v1/voices/{voice_id}", dependencies=auth)
async def update_voice(voice_id: str, patch: VoiceUpdate):
    try:
        voice = voices.get(voice_id)
    except voices.VoiceNotFound:
        raise HTTPException(404, "voice not found") from None
    if patch.name is not None:
        voice.name = patch.name.strip()
    if patch.description is not None:
        voice.description = patch.description
    voices.save(voice)
    if patch.is_default:
        voices.set_default(voice_id)
    return voice.to_dict(voices.default_id())


@app.delete("/v1/voices/{voice_id}", dependencies=auth, status_code=204)
async def delete_voice(voice_id: str):
    try:
        voices.delete(voice_id)
    except voices.VoiceNotFound:
        raise HTTPException(404, "voice not found") from None
    engine.forget(voice_id)
    return Response(status_code=204)
