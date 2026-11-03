"""HTTP client for the OmniVoice TTS service (tts_service/). The service owns
the GPU, the model and the voice library; this app only sends it text.

Every live-call path synthesizes through `synthesize_pcm`, which returns raw
PCM16 mono at 8 kHz - the telephony rate - ready to hand to Pipecat/Twilio.
"""

import logging

import httpx

from config import settings

logger = logging.getLogger("tts.client")

SAMPLE_RATE = 8000

_client: httpx.AsyncClient | None = None


class TTSUnavailable(RuntimeError):
    """The TTS service is down, overloaded, or has no usable voice."""


class TTSRequestError(RuntimeError):
    """The TTS service rejected the request (bad voice id, bad audio...)."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        headers = {"X-TTS-Token": settings.tts_service_token} if settings.tts_service_token else {}
        _client = httpx.AsyncClient(
            base_url=settings.tts_service_url.rstrip("/"),
            headers=headers,
            timeout=settings.tts_request_timeout_seconds,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=8),
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _raise_for(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    if resp.status_code in (503, 409) or resp.status_code >= 500:
        raise TTSUnavailable(f"TTS service {resp.status_code}: {detail}")
    raise TTSRequestError(resp.status_code, str(detail))


async def _request(method: str, path: str, **kw) -> httpx.Response:
    try:
        resp = await _http().request(method, path, **kw)
    except httpx.HTTPError as e:
        raise TTSUnavailable(f"TTS service unreachable at {settings.tts_service_url}: {e!r}") from e
    _raise_for(resp)
    return resp


async def warm() -> None:
    """Best-effort: ask the TTS service to have the model on the GPU now.
    Never raises - a missed warm-up only costs the wake-up time later."""
    try:
        await _http().post("/v1/warm", timeout=3.0)
    except Exception:  # noqa: BLE001
        pass


async def health() -> dict:
    return (await _request("GET", "/health")).json()


async def synthesize_pcm(
    text: str, *, language: str, voice_id: str | None, speed: float = 1.0, num_step: int | None = None
) -> bytes:
    """PCM16 mono 8 kHz for `text`. One retry: a 503 is the service shedding
    load for a moment, and a dead-air line is worse than 200ms of waiting."""
    body = {
        "text": text, "language": language, "voice_id": voice_id, "speed": speed, "format": "pcm8k",
        "num_step": num_step or settings.tts_live_num_step,
    }
    last: Exception | None = None
    for attempt in (1, 2):
        try:
            resp = await _request("POST", "/v1/synthesize", json=body)
            rate = int(resp.headers.get("x-sample-rate", SAMPLE_RATE))
            if rate != SAMPLE_RATE:
                raise TTSUnavailable(f"TTS service returned {rate} Hz, expected {SAMPLE_RATE}")
            return resp.content
        except TTSUnavailable as e:
            last = e
            logger.warning("TTS attempt %d failed for %r: %s", attempt, text[:60], e)
    raise last  # type: ignore[misc]


async def synthesize_preview_wav(text: str, *, language: str, voice_id: str | None, speed: float = 1.0) -> bytes:
    """24 kHz WAV for Voice Studio's "try this voice" button."""
    body = {"text": text, "language": language, "voice_id": voice_id, "speed": speed, "format": "wav24k"}
    return (await _request("POST", "/v1/synthesize", json=body)).content


# --- voice library passthroughs (Voice Studio) ---

async def list_voices() -> dict:
    return (await _request("GET", "/v1/voices")).json()


async def voice_audio(voice_id: str) -> bytes:
    return (await _request("GET", f"/v1/voices/{voice_id}/audio")).content


async def clone_voice(*, name: str, language: str, description: str, ref_text: str, filename: str, audio: bytes, content_type: str) -> dict:
    resp = await _request(
        "POST", "/v1/voices/clone",
        data={"name": name, "language": language, "description": description, "ref_text": ref_text},
        files={"file": (filename, audio, content_type)},
        timeout=600.0,  # first clone without a transcript loads Whisper (minutes on CPU)
    )
    return resp.json()


async def design_voice(payload: dict) -> dict:
    return (await _request("POST", "/v1/voices/design", json=payload, timeout=120.0)).json()


async def update_voice(voice_id: str, payload: dict) -> dict:
    return (await _request("PATCH", f"/v1/voices/{voice_id}", json=payload)).json()


async def delete_voice(voice_id: str) -> None:
    await _request("DELETE", f"/v1/voices/{voice_id}")
