"""/voices router with the TTS service client faked."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import voices as voices_router
from services.tts import client, voice_registry


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_registry, "_PATH", str(tmp_path / "assign.json"))
    voice_registry._snapshot = None
    lib = {"voices": [{"id": "anita-aaa111", "name": "Anita", "version": 1, "kind": "clone", "language": "hi", "is_default": True}],
           "default_voice_id": "anita-aaa111"}
    calls = {}

    async def list_voices():
        return lib

    async def clone_voice(**kw):
        calls["clone"] = kw
        if kw["audio"] == b"bad":
            raise client.TTSRequestError(422, "reference too short")
        return {"id": "new-ccc333", "name": kw["name"]}

    async def delete_voice(vid):
        if vid == "gone":
            raise client.TTSRequestError(404, "voice not found")

    async def preview(text, **kw):
        return b"RIFFfake"

    async def down():
        raise client.TTSUnavailable("TTS service unreachable")

    monkeypatch.setattr(client, "list_voices", list_voices)
    monkeypatch.setattr(client, "clone_voice", clone_voice)
    monkeypatch.setattr(client, "delete_voice", delete_voice)
    monkeypatch.setattr(client, "synthesize_preview_wav", preview)
    monkeypatch.setattr(client, "health", down)
    app = FastAPI()
    app.include_router(voices_router.router)
    return TestClient(app), calls


def test_page_served(api):
    c, _ = api
    r = c.get("/voice-studio")
    assert r.status_code == 200 and "Voice Studio" in r.text


def test_list_includes_assignments_and_owners(api):
    c, _ = api
    body = c.get("/voices").json()
    assert body["voices"][0]["id"] == "anita-aaa111"
    assert {"hi", "gu", "en"} <= set(body["languages"])
    assert any(o["kind"] == "scenario" for o in body["owners"])


def test_clone_forwards_upload_and_maps_service_errors(api):
    c, calls = api
    ok = c.post("/voices/clone", data={"name": "New", "language": "hi"}, files={"file": ("r.wav", b"audio", "audio/wav")})
    assert ok.status_code == 201 and calls["clone"]["audio"] == b"audio"
    bad = c.post("/voices/clone", data={"name": "New"}, files={"file": ("r.wav", b"bad", "audio/wav")})
    assert bad.status_code == 422 and "too short" in bad.json()["detail"]


def test_assign_validates_voice_and_language(api):
    c, _ = api
    assert c.put("/voices/assignments", json={"owner": "shift_confirm", "language": "hi", "voice_id": "nope"}).status_code == 404
    assert c.put("/voices/assignments", json={"owner": "x", "language": "fr", "voice_id": None}).status_code == 422
    r = c.put("/voices/assignments", json={"owner": "shift_confirm", "language": "hi", "voice_id": "anita-aaa111"})
    assert r.json()["assignments"]["shift_confirm:hi"]["voice_id"] == "anita-aaa111"
    r = c.put("/voices/assignments", json={"owner": "shift_confirm", "language": "hi", "voice_id": None})
    assert r.json()["assignments"] == {}


def test_delete_clears_assignments_pointing_at_voice(api):
    c, _ = api
    voice_registry.set_assignment("shift_confirm", "hi", "anita-aaa111")
    assert c.delete("/voices/anita-aaa111").status_code == 204
    assert voice_registry.load_assignments() == {}
    assert c.delete("/voices/gone").status_code == 404


def test_service_down_is_503(api):
    c, _ = api
    assert c.get("/voices/health").status_code == 503


def test_preview_returns_wav(api):
    c, _ = api
    r = c.post("/voices/preview", json={"text": "hi", "language": "en"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
