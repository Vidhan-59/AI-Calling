"""TTS service API with the mock engine (no GPU). Run from the repo root with
the service venv:  tts_service/.venv/Scripts/python -m pytest tts_service/tests"""

import io
import os

import numpy as np
import pytest
import soundfile as sf

os.environ["TTS_ENGINE"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

from tts_service import settings  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "VOICES_DIR", tmp_path / "voices")
    monkeypatch.setattr(settings, "TOKEN", "")
    from tts_service import app as service

    with TestClient(service.app) as c:
        yield c


def wav_bytes(seconds=4.0, rate=16000, freq=220.0):
    t = np.arange(int(seconds * rate)) / rate
    buf = io.BytesIO()
    sf.write(buf, (0.4 * np.sin(2 * np.pi * freq * t)).astype("float32"), rate, format="WAV")
    return buf.getvalue()


def clone(client, name="Anita", **extra):
    return client.post("/v1/voices/clone", data={"name": name, "language": "hi", **extra},
                       files={"file": ("ref.wav", wav_bytes(), "audio/wav")})


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["engine"] == "mock" and h["voices"] == 0


def test_synthesize_without_any_voice_is_409(client):
    r = client.post("/v1/synthesize", json={"text": "hello"})
    assert r.status_code == 409


def test_clone_then_synthesize_telephony_pcm(client):
    r = clone(client, ref_text="namaste")
    assert r.status_code == 201
    voice = r.json()
    assert voice["kind"] == "clone" and voice["is_default"] is True  # first voice becomes default
    r = client.post("/v1/synthesize", json={"text": "नमस्ते आप कैसे हैं", "voice_id": voice["id"]})
    assert r.status_code == 200 and r.headers["x-sample-rate"] == "8000"
    pcm = np.frombuffer(r.content, dtype="<i2")
    assert abs(len(pcm) / 8000 - float(r.headers["x-duration"])) < 0.01
    assert np.abs(pcm).max() > 1000  # audible, normalised


def test_synthesize_preview_wav_is_24k(client):
    vid = clone(client).json()["id"]
    r = client.post("/v1/synthesize", json={"text": "hello", "voice_id": vid, "format": "wav24k"})
    data, rate = sf.read(io.BytesIO(r.content))
    assert rate == 24000 and len(data) > 0


def test_speed_changes_duration(client):
    vid = clone(client).json()["id"]
    slow = client.post("/v1/synthesize", json={"text": "x" * 40, "voice_id": vid, "speed": 0.7}).headers["x-duration"]
    fast = client.post("/v1/synthesize", json={"text": "x" * 40, "voice_id": vid, "speed": 1.5}).headers["x-duration"]
    assert float(slow) > float(fast)


def test_clone_rejects_bad_audio(client):
    short = client.post("/v1/voices/clone", data={"name": "S"}, files={"file": ("r.wav", wav_bytes(0.5), "audio/wav")})
    assert short.status_code == 422 and "short" in short.json()["detail"]
    junk = client.post("/v1/voices/clone", data={"name": "J"}, files={"file": ("r.wav", b"not audio", "audio/wav")})
    assert junk.status_code == 422
    long = client.post("/v1/voices/clone", data={"name": "L"}, files={"file": ("r.wav", wav_bytes(30), "audio/wav")})
    assert long.status_code == 422 and "long" in long.json()["detail"]
    assert client.get("/v1/voices").json()["voices"] == []  # nothing half-created


def test_design_voice_renders_reference_and_is_reusable(client):
    r = client.post("/v1/voices/design", json={"name": "Calm", "language": "en", "instruct": "female, low pitch"})
    assert r.status_code == 201 and r.json()["kind"] == "design"
    vid = r.json()["id"]
    ref = client.get(f"/v1/voices/{vid}/audio")
    assert sf.read(io.BytesIO(ref.content))[1] == 24000
    assert client.post("/v1/synthesize", json={"text": "hi", "voice_id": vid}).status_code == 200


def test_default_rename_delete(client):
    a = clone(client, "A").json()["id"]
    b = clone(client, "B").json()["id"]
    assert client.patch(f"/v1/voices/{b}", json={"is_default": True, "name": "Bee"}).json()["name"] == "Bee"
    assert client.get("/v1/voices").json()["default_voice_id"] == b
    assert client.delete(f"/v1/voices/{b}").status_code == 204
    assert client.get("/v1/voices").json()["default_voice_id"] is None
    assert client.get(f"/v1/voices/{a}").status_code == 200
    assert client.get(f"/v1/voices/{b}").status_code == 404


def test_path_traversal_ids_rejected(client):
    assert client.get("/v1/voices/..%2F..%2Fetc/audio").status_code == 404
    assert client.post("/v1/synthesize", json={"text": "x", "voice_id": "../x"}).status_code == 404


def test_text_length_limit(client):
    clone(client)
    assert client.post("/v1/synthesize", json={"text": "a" * (settings.MAX_TEXT_CHARS + 1)}).status_code == 413


def test_token_auth(client, monkeypatch):
    monkeypatch.setattr(settings, "TOKEN", "s3cret")
    assert client.get("/v1/voices").status_code == 401
    assert client.get("/v1/voices", headers={"X-TTS-Token": "wrong"}).status_code == 401
    assert client.get("/v1/voices", headers={"X-TTS-Token": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for probes
