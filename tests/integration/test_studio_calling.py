"""Agent Studio "Call now": placing the call (Twilio mocked - nothing is ever
dialled), the TwiML/stream hand-off, and what the call processor does with
the agent's instructions and training. In-memory DB, no Twilio, no GPU."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection

import simple_db
from routers import studio
from services.studio import calling, knowledge, store
from services.voice_pipeline.studio_processor import StudioCallProcessor

NUMBER = "+919876543210"


@pytest.fixture
def db(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    monkeypatch.setattr(simple_db, "_connection", conn)
    monkeypatch.setattr(store, "_ready", False)
    return conn


class FakeTwilioCalls:
    def __init__(self, fail=None):
        self.created, self.updated, self.fail = [], [], fail

    def create(self, **kw):
        if self.fail:
            raise RuntimeError(self.fail)
        self.created.append(kw)
        return type("Call", (), {"sid": "CAtest123"})()

    def __call__(self, sid):
        outer = self
        return type("One", (), {"update": lambda self, **kw: outer.updated.append((sid, kw))})()


@pytest.fixture
def phone(db, monkeypatch):
    """A fully configured environment with a fake Twilio."""
    fake = FakeTwilioCalls()
    monkeypatch.setattr(calling, "_read_env_value", lambda key: None)
    monkeypatch.setenv(calling.ENV_NAME, NUMBER)
    monkeypatch.setattr(calling, "get_public_url", lambda: "https://abc.ngrok.app")
    monkeypatch.setattr(calling, "_check_tunnel", lambda url: None)
    monkeypatch.setattr(calling, "_twilio_client", lambda: type("C", (), {"calls": fake})())
    for name, value in (("twilio_account_sid", "ACx"), ("twilio_auth_token", "tok"), ("twilio_phone_number", "+12542725145")):
        monkeypatch.setattr(calling.settings, name, value)
    app = FastAPI()
    app.include_router(studio.router)
    client = TestClient(app)
    client.fake = fake
    return client


def _agent(client, **over):
    body = {"name": "Support", "language": "en", "system_instructions": "You are a support assistant for ABC Services.",
            "basic": {"objective": "Resolve questions."}, **over}
    return client.post("/studio/agents", json=body).json()


# ------------------------------------------------------------- placing the call

def test_config_masks_the_number_and_reports_what_is_missing(phone, monkeypatch):
    cfg = phone.get("/studio/call-config").json()
    assert cfg["ready"] and cfg["number_masked"] == "+91•••••••210" and NUMBER not in json.dumps(cfg)
    monkeypatch.delenv(calling.ENV_NAME)
    cfg = phone.get("/studio/call-config").json()
    assert not cfg["ready"] and "STUDIO_CALL_NUMBER" in " ".join(cfg["problems"])
    monkeypatch.setenv(calling.ENV_NAME, "98765 not a number")
    assert "not a valid phone number" in " ".join(phone.get("/studio/call-config").json()["problems"])


def test_call_now_dials_the_configured_number_with_the_agent(phone):
    a = _agent(phone, voice={"voice_id": "v1", "speed": 1.1})
    r = phone.post(f"/studio/agents/{a['id']}/call")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["call_sid"] == "CAtest123" and body["to"] == "+91•••••••210" and NUMBER not in json.dumps(body)
    sent = phone.fake.created[0]
    assert sent["to"] == NUMBER and sent["from_"] == "+12542725145"
    assert sent["url"] == f"https://abc.ngrok.app/studio/voice?call_id={body['call_id']}"
    call = store.get_call(body["call_id"])
    assert call["status"] == "connecting" and call["agent_id"] == a["id"]
    assert call["agent_snapshot"]["voice"]["voice_id"] == "v1" and call["metrics"]["twilio_call_sid"] == "CAtest123"


def test_an_unfinished_agent_cannot_call(phone):
    blank = phone.post("/studio/agents", json={"name": "Blank"}).json()
    r = phone.post(f"/studio/agents/{blank['id']}/call")
    assert r.status_code == 422 and phone.fake.created == []


def test_missing_configuration_blocks_the_call_with_the_reason(phone, monkeypatch):
    a = _agent(phone)
    monkeypatch.delenv(calling.ENV_NAME)
    r = phone.post(f"/studio/agents/{a['id']}/call")
    assert r.status_code == 409 and "STUDIO_CALL_NUMBER" in r.json()["detail"] and phone.fake.created == []


def test_dead_tunnel_is_reported_and_nothing_is_dialled(phone, monkeypatch):
    a = _agent(phone)
    monkeypatch.setattr(calling, "_check_tunnel", lambda url: "is unreachable (ConnectError)")
    r = phone.post(f"/studio/agents/{a['id']}/call")
    assert r.status_code == 502 and "NGROK_URL" in r.json()["detail"]
    assert phone.fake.created == [] and store.list_calls() == []


def test_twilio_failure_marks_the_call_failed(phone):
    a = _agent(phone)
    phone.fake.fail = "The number is unverified"
    r = phone.post(f"/studio/agents/{a['id']}/call")
    assert r.status_code == 502 and "unverified" in r.json()["detail"]
    [call] = store.list_calls()
    assert call["status"] == "failed" and "unverified" in call["error"]


def test_double_click_does_not_place_two_calls(phone):
    a = _agent(phone)
    assert phone.post(f"/studio/agents/{a['id']}/call").status_code == 200
    r = phone.post(f"/studio/agents/{a['id']}/call")
    assert r.status_code == 409 and len(phone.fake.created) == 1


# ------------------------------------------------- Twilio hand-off + status

def test_twiml_connects_only_a_call_that_was_just_placed(phone):
    a = _agent(phone)
    call_id = phone.post(f"/studio/agents/{a['id']}/call").json()["call_id"]
    ok = phone.post(f"/studio/voice?call_id={call_id}")
    assert "<Stream" in ok.text and "/studio/media" in ok.text and f'value="{call_id}"' in ok.text
    for bad in ("nope", ""):
        assert "<Hangup/>" in phone.post(f"/studio/voice?call_id={bad}").text
    store.update_call(call_id, status="in_progress")
    assert "<Hangup/>" in phone.post(f"/studio/voice?call_id={call_id}").text  # a stream can't be replayed


def test_media_socket_refuses_an_unknown_call_id(phone):
    with TestClient(phone.app) as c:
        with c.websocket_connect("/studio/media") as ws:
            ws.send_text(json.dumps({"event": "connected"}))
            ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1",
                                                                  "customParameters": {"callId": "call_forged"}}}))
            with pytest.raises(Exception):
                ws.receive_text()  # server closed it


def test_status_returns_the_live_transcript_and_expires_unanswered_calls(phone):
    a = _agent(phone)
    call_id = phone.post(f"/studio/agents/{a['id']}/call").json()["call_id"]
    store.add_message(call_id, "agent", "Hello!")
    store.add_message(call_id, "customer", "Hi.")
    s = phone.get(f"/studio/calls/{call_id}").json()
    assert s["status"] == "connecting" and [m["speaker"] for m in s["messages"]] == ["agent", "customer"]
    old = (datetime.now(timezone.utc) - timedelta(seconds=calling.NO_ANSWER_AFTER_SECS + 5)).isoformat()
    simple_db._connection.execute("UPDATE studio_calls SET started_at = ? WHERE id = ?", (old, call_id))
    s = phone.get(f"/studio/calls/{call_id}").json()
    assert s["status"] == "failed" and s["outcome"] == "no_answer" and "not answered" in s["error"]


def test_hang_up_ends_the_twilio_call(phone):
    a = _agent(phone)
    call_id = phone.post(f"/studio/agents/{a['id']}/call").json()["call_id"]
    assert phone.post(f"/studio/calls/{call_id}/end").json()["status"] == "ending"
    assert phone.fake.updated == [("CAtest123", {"status": "completed"})]


# ------------------------------------------------------ the call processor

class Recorder:
    def __init__(self):
        self.frames = []

    async def __call__(self, frame, direction=None):
        self.frames.append(frame)

    @property
    def spoken(self):
        return [f.text for f in self.frames if isinstance(f, TTSSpeakFrame)]


def _processor(db, *, reply, agent_over=None):
    fields = dict(name="Alex", language="en", system_instructions="You are Alex, a support assistant for ABC Services.",
                  personality={"greeting": "Hello! Thank you for calling ABC Services."},
                  voice={"style": "friendly"}, basic={"objective": "Resolve questions."})
    fields["personality"] = {**fields["personality"], **(agent_over or {}).get("personality", {})}
    agent = store.create_agent(**fields)
    store.add_training(agent["id"], "qa", "What are your working hours?", "We are open Monday to Saturday, 9 AM to 6 PM.", {})
    call_id = store.create_call(agent=agent, kind="call", language="en")
    seen = {}

    async def stream(prompt, messages):
        seen["prompt"], seen["messages"] = prompt, [dict(m) for m in messages]
        for word in reply:
            yield word

    proc = StudioCallProcessor(agent=agent, call_id=call_id, stream_reply=stream, max_call_secs=0)
    rec = Recorder()
    proc.push_frame = rec
    return proc, rec, call_id, seen


async def test_greeting_is_the_agents_own_opening_and_is_recorded(db):
    proc, rec, call_id, _ = _processor(db, reply=[])
    await proc._greet()
    assert rec.spoken == ["Hello!", "Thank you for calling ABC Services."]  # the saved greeting, spoken sentence by sentence
    assert [(m["speaker"], m["message"]) for m in store.list_messages(call_id)] == [("agent", "Hello! Thank you for calling ABC Services.")]


async def test_reply_uses_saved_instructions_and_retrieved_training(db):
    proc, rec, call_id, seen = _processor(db, reply=["We are open ", "Monday to Saturday, ", "from nine to six."])
    await proc._greet()
    await proc._on_transcript("Please tell me the working hours on weekdays")
    assert "You are Alex, a support assistant for ABC Services." in seen["prompt"]
    assert "Speaking style: friendly." in seen["prompt"]
    assert "Monday to Saturday, 9 AM to 6 PM" in seen["prompt"]  # the Q&A pair was retrieved into the prompt
    assert seen["messages"][-1] == {"role": "user", "content": "Please tell me the working hours on weekdays"}
    assert " ".join(rec.spoken[2:]) == "We are open Monday to Saturday, from nine to six."
    messages = [(m["speaker"], m["message"]) for m in store.list_messages(call_id)]
    assert messages[1] == ("customer", "Please tell me the working hours on weekdays") and messages[2][0] == "agent"
    cats = {t["category"] for t in store.list_traces(call_id)}
    assert {"speech_recognition", "knowledge", "prompt", "llm", "response"} <= cats


async def test_unanswerable_question_gets_no_invented_context(db):
    proc, _, _, seen = _processor(db, reply=["Sorry, I do not have that."])
    await proc._on_transcript("what is the capital of mars")
    assert "No reference material matched" in seen["prompt"]


async def test_barge_in_stops_the_rest_of_the_reply(db):
    proc, rec, call_id, _ = _processor(db, reply=["First part here, ", "second part here, ", "third part here, ", "and the last."])
    orig = proc._speak_piece

    async def speak_then_interrupt(piece, epoch):
        await orig(piece, epoch)
        if len(rec.spoken) == 1:
            await proc.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    proc._speak_piece = speak_then_interrupt
    await proc._on_transcript("hello there")
    assert len(rec.spoken) == 1  # nothing after the interruption was sent to TTS


async def test_a_failing_llm_falls_back_to_the_agents_fallback_line(db):
    async def boom(prompt, messages):
        raise RuntimeError("model down")
        yield  # pragma: no cover

    proc, rec, call_id, _ = _processor(db, reply=[], agent_over={"personality": {"fallback": "Sorry, could you repeat that?"}})
    proc._stream_reply = boom
    await proc._on_transcript("anything")
    assert rec.spoken[-1] == "Sorry, could you repeat that?"
    assert any(t["status"] == "error" for t in store.list_traces(call_id))


async def test_finish_records_duration_and_outcome(db):
    proc, _, call_id, _ = _processor(db, reply=["Okay."])
    proc.finish()  # nobody spoke
    assert store.get_call(call_id)["outcome"] == "no_conversation"
    proc2, _, call2, _ = _processor(db, reply=["Okay."])
    await proc2._on_transcript("hi")
    proc2.finish()
    c = store.get_call(call2)
    assert c["status"] == "completed" and c["outcome"] == "completed" and c["duration_secs"] is not None
    assert any(t["category"] == "session" and t["name"] == "Call ended" for t in store.list_traces(call2))


async def test_a_question_matching_a_qa_pair_is_answered_with_the_stored_answer(db):
    proc, rec, call_id, seen = _processor(db, reply=["this should never be spoken"])
    await proc._on_transcript("What are your working hours")
    assert " ".join(rec.spoken) == "We are open Monday to Saturday, 9 AM to 6 PM."
    assert "prompt" not in seen  # the LLM was not even called
    assert any(t["data"].get("verbatim") for t in store.list_traces(call_id) if t["category"] == "response")


async def test_a_loosely_related_question_still_goes_through_the_llm(db):
    proc, rec, call_id, seen = _processor(db, reply=["We open in the morning."])
    await proc._on_transcript("Please tell me the working hours on weekdays")
    assert "prompt" in seen and "Monday to Saturday, 9 AM to 6 PM" in seen["prompt"]  # retrieved, but the LLM answers
    assert "We open in the morning." in " ".join(rec.spoken)


@pytest.mark.parametrize("raw,expected", [
    ("+919876543210", "+919876543210"),
    ("9876543210", "+919876543210"),          # bare Indian mobile
    ("09876543210", "+919876543210"),
    ("91 98765-43210", "+919876543210"),
    ("(+91) 98765 43210", "+919876543210"),
    ("+16175551234", "+16175551234"),          # other countries must already be international
    ("16175551234", None), ("12345", None), ("5876543210", None), ("", None), ("abc", None),
])
def test_phone_number_normalization(raw, expected):
    assert calling.normalize(raw) == expected
