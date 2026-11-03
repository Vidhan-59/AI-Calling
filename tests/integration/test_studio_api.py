"""Agent Studio API (routers/studio.py) against an in-memory copy of the
legacy DB. Voices/TTS are not touched - the studio router only stores config."""

import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import simple_db
from routers import studio
from services.studio import store


@pytest.fixture
def client(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    monkeypatch.setattr(simple_db, "_connection", conn)
    monkeypatch.setattr(store, "_ready", False)
    app = FastAPI()
    app.include_router(studio.router)
    return TestClient(app)


def _agent(client, **over):
    body = {"name": "Support Bot", "language": "hi", "system_instructions": "You are a support assistant.",
            "basic": {"objective": "Resolve the question."}, **over}
    r = client.post("/studio/agents", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_create_read_update_agent_round_trips_every_field(client):
    a = _agent(client, description="d", avatar="🎧", use_case_id="shift_confirm",
               voice={"voice_id": "calm-female-1", "speed": 1.2, "style": "calm", "tone": "warm"},
               personality={"role": "Support", "greeting": "Namaste!"})
    got = client.get(f"/studio/agents/{a['id']}").json()
    assert got["voice"] == {"voice_id": "calm-female-1", "speed": 1.2, "style": "calm", "tone": "warm"}
    assert got["personality"]["greeting"] == "Namaste!" and got["status"] == "draft"
    upd = client.put(f"/studio/agents/{a['id']}", json={"name": "Renamed", "voice": {"speed": 0.8}}).json()
    assert upd["name"] == "Renamed" and upd["voice"]["speed"] == 0.8
    assert upd["system_instructions"] == "You are a support assistant."  # untouched fields survive


@pytest.mark.parametrize("body", [
    {"name": ""}, {"name": "x", "language": "fr"}, {"name": "x", "voice": {"speed": 3.0}},
    {"name": "x", "voice": {"style": "sarcastic"}}, {"name": "x", "personality": {"bogus": "y"}},
    {"name": "x", "avatar": "data:text/html;base64,AAAA"},
])
def test_invalid_agent_input_is_rejected(client, body):
    assert client.post("/studio/agents", json=body).status_code == 422


def test_publishing_needs_instructions_and_objective(client):
    empty = client.post("/studio/agents", json={"name": "Blank"}).json()
    r = client.post(f"/studio/agents/{empty['id']}/status", json={"status": "published"})
    assert r.status_code == 422 and "instructions" in r.json()["detail"]
    ready = _agent(client)
    assert client.post(f"/studio/agents/{ready['id']}/status", json={"status": "published"}).json()["status"] == "published"


def test_active_requires_a_use_case(client):
    a = _agent(client)
    r = client.post(f"/studio/agents/{a['id']}/status", json={"status": "active"})
    assert r.status_code == 422 and "use case" in r.json()["detail"]
    client.put(f"/studio/agents/{a['id']}", json={"use_case_id": "shift_confirm"})
    assert client.post(f"/studio/agents/{a['id']}/status", json={"status": "active"}).json()["status"] == "active"


def test_duplicate_copies_training_and_resets_to_draft(client):
    a = _agent(client)
    client.post(f"/studio/agents/{a['id']}/training", json={"training_type": "qa", "title": "Hours?", "content": "9 to 6."})
    client.post(f"/studio/agents/{a['id']}/status", json={"status": "published"})
    copy = client.post(f"/studio/agents/{a['id']}/duplicate").json()
    assert copy["name"].endswith("(copy)") and copy["status"] == "draft" and copy["training_counts"]["qa"] == 1


def test_delete_agent_removes_training(client):
    a = _agent(client)
    client.post(f"/studio/agents/{a['id']}/training", json={"training_type": "qa", "title": "q", "content": "a"})
    assert client.delete(f"/studio/agents/{a['id']}").status_code == 204
    assert client.get(f"/studio/agents/{a['id']}").status_code == 404
    assert store.list_training(a["id"]) == []


def test_qa_crud_and_validation(client):
    a = _agent(client)
    assert client.post(f"/studio/agents/{a['id']}/training", json={"training_type": "qa", "title": "Q only"}).status_code == 422
    item = client.post(f"/studio/agents/{a['id']}/training", json={
        "training_type": "qa", "title": "Working hours?", "content": "Mon-Sat 9-6", "metadata": {"category": "hours"}}).json()
    upd = client.put(f"/studio/training/{item['id']}", json={"content": "Mon-Fri 9-6"}).json()
    assert upd["content"] == "Mon-Fri 9-6" and upd["metadata"]["category"] == "hours"
    assert len(client.get(f"/studio/agents/{a['id']}/training?type=qa").json()["items"]) == 1
    assert client.delete(f"/studio/training/{item['id']}").status_code == 204
    assert client.get(f"/studio/agents/{a['id']}/training").json()["items"] == []


def test_demo_conversation_text_is_derived_from_turns(client):
    a = _agent(client)
    turns = [{"speaker": "customer", "text": "I want to book."}, {"speaker": "ai", "text": "Which date?"},
             {"speaker": "customer", "text": ""}]  # blank turns are dropped
    item = client.post(f"/studio/agents/{a['id']}/training", json={
        "training_type": "scenario", "title": "Booking", "metadata": {"turns": turns}}).json()
    assert item["content"] == "Customer: I want to book.\nAI: Which date?"
    bad = client.post(f"/studio/agents/{a['id']}/training", json={
        "training_type": "scenario", "title": "One-sided", "metadata": {"turns": turns[:1]}})
    assert bad.status_code == 422


def test_script_kind_is_validated(client):
    a = _agent(client)
    ok = client.post(f"/studio/agents/{a['id']}/training", json={
        "training_type": "script", "title": "Open", "content": "Hello!", "metadata": {"kind": "opening"}})
    assert ok.status_code == 201
    bad = client.post(f"/studio/agents/{a['id']}/training", json={
        "training_type": "script", "title": "x", "content": "y", "metadata": {"kind": "nonsense"}})
    assert bad.status_code == 422


def test_qa_import_csv_json_and_export_round_trip(client):
    a = _agent(client)
    csv_data = "question,answer,category\nWorking hours?,Mon-Sat 9-6,hours\nContact?,Call support,contact\n"
    r = client.post(f"/studio/agents/{a['id']}/training/qa/import",
                    files={"file": ("qa.csv", csv_data.encode(), "text/csv")}, data={"mode": "append"})
    assert r.json()["imported"] == 2
    js = json.dumps([{"question": "Refund?", "answer": "5 days", "category": "billing"}])
    r = client.post(f"/studio/agents/{a['id']}/training/qa/import",
                    files={"file": ("qa.json", js.encode(), "application/json")}, data={"mode": "replace"})
    assert r.json()["imported"] == 1
    items = client.get(f"/studio/agents/{a['id']}/training?type=qa").json()["items"]
    assert [i["title"] for i in items] == ["Refund?"]  # replace dropped the CSV pairs
    exported = client.get(f"/studio/agents/{a['id']}/training/export?format=csv")
    assert "Refund?,5 days,billing" in exported.text
    full = client.get(f"/studio/agents/{a['id']}/training/export").json()
    assert full["training"][0]["title"] == "Refund?" and full["agent"]["name"] == "Support Bot"
    bad = client.post(f"/studio/agents/{a['id']}/training/qa/import", files={"file": ("bad.csv", b"x,y\n1,2\n", "text/csv")})
    assert bad.status_code == 422


def test_document_upload_feeds_retrieval_and_prompt(client):
    a = _agent(client)
    doc = b"Our service centre is open Monday to Saturday.\n\nThe annual maintenance plan costs 4999 rupees."
    r = client.post(f"/studio/agents/{a['id']}/knowledge/document", files={"file": ("faq.txt", doc, "text/plain")})
    assert r.status_code == 201 and r.json()["metadata"]["chunks"] >= 1
    found = client.post(f"/studio/agents/{a['id']}/knowledge/search", json={"query": "how much is the maintenance plan"}).json()["results"]
    assert found and "4999" in found[0]["text"]
    prompt = client.get(f"/studio/agents/{a['id']}/prompt", params={"query": "maintenance plan price"}).json()
    assert "4999" in prompt["prompt"] and prompt["retrieved"][0]["source"] == "document"
    miss = client.get(f"/studio/agents/{a['id']}/prompt", params={"query": "zzzz qqqq"}).json()
    assert miss["retrieved"] == [] and "do not invent facts" in miss["prompt"]


def test_unsupported_document_type_is_a_clean_error(client):
    a = _agent(client)
    r = client.post(f"/studio/agents/{a['id']}/knowledge/document", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 422


def test_url_ingest_refuses_private_hosts(client):
    a = _agent(client)
    for url in ("http://127.0.0.1:8000/", "http://localhost/x", "file:///etc/passwd"):
        assert client.post(f"/studio/agents/{a['id']}/knowledge/url", json={"url": url}).status_code == 422


def test_saved_configuration_appears_in_the_prompt(client):
    a = _agent(client, personality={"role": "Support lead", "escalation": "Offer a human if upset."},
               voice={"style": "empathetic", "tone": "reassuring"})
    prompt = client.get(f"/studio/agents/{a['id']}/prompt").json()
    text = prompt["prompt"]
    assert "You are a support assistant." in text and "Support lead" in text
    assert "Offer a human if upset." in text and "Speaking style: empathetic." in text
    assert "Emotional tone: reassuring." in text
    assert "System instructions" in prompt["sections_loaded"]


def test_meta_states_what_voice_controls_really_do(client):
    meta = client.get("/studio/meta").json()
    status = {c["key"]: c["status"] for c in meta["capabilities"]}
    assert status["speed"] == "supported" and status["pitch"] == "creation_only"
    assert status["stability"] == status["expressiveness"] == "unsupported"
    assert any(u["id"] == "shift_confirm" for u in meta["use_cases"])


def test_stats_count_agents_by_status(client):
    a = _agent(client)
    _agent(client, name="Second")
    client.post(f"/studio/agents/{a['id']}/status", json={"status": "published"})
    s = client.get("/studio/stats").json()
    assert s["agents_total"] == 2 and s["agents_published"] == 1 and s["agents_draft"] == 1


def test_audio_diagnostics_shape(client, monkeypatch):
    from services.tts import client as tts_client

    async def down():
        raise tts_client.TTSUnavailable("down")

    monkeypatch.setattr(tts_client, "health", down)
    d = client.get("/studio/diagnostics/audio").json()
    assert {"playout", "tts_service", "gpu"} <= set(d) and d["tts_service"]["status"] == "unreachable"
    assert "underruns" in d["playout"]
