"""Call details + post-call analysis on scenario_calls (services/call_analysis.py),
against an in-memory copy of the legacy DB and a fake Qwen."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import simple_db
from local_ai import llm_engine as local_llm_engine
from services import call_analysis, conversation_store, scenario_store

ANALYSIS = {
    "sentiment": "neutral", "sentiment_score": 0.1, "emotion": "apologetic",
    "reason_category": "illness", "reason_english": "has a fever",
    "summary": "Worker cannot come tomorrow because of a fever.",
    "follow_up_required": False, "follow_up_note": "",
}


@pytest.fixture
def db(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    monkeypatch.setattr(simple_db, "_connection", conn)
    simple_db.init_db()
    monkeypatch.setattr(call_analysis, "set_call_analysis", lambda *a, **k: None)  # no log files
    return conn


def _call(sid="CA1", reason="बुखार है", user_turns=("नहीं आ पाऊँगा", "बुखार है")):
    scenario_store.create(sid, "shift_confirm", "hi", "+910000000000", {"employee_name": "Krish"})
    conversation_store.create_call(sid, "shift_confirm", "", "+910000000000", "outbound")
    conversation_store.add_message(sid, "assistant", "नमस्ते, क्या आप कल आएंगे?")
    for text in user_turns:
        conversation_store.add_message(sid, "user", text)
    scenario_store.record_outcome(sid, outcome="absent_with_reason", matched_branch="no",
                                  collected={"absence_reason": reason} if reason else {})


def _fake_llm(monkeypatch, result):
    async def fake(*, system_prompt, user_text, schema, max_tokens=400):
        fake.user_text = user_text
        return dict(result)

    monkeypatch.setattr(local_llm_engine, "classify_structured", fake)
    return fake


def test_migration_adds_columns_and_keeps_old_rows(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE scenario_calls (call_sid TEXT PRIMARY KEY, scenario_id TEXT NOT NULL,
        language TEXT NOT NULL, to_number TEXT, employee_name TEXT, rgid TEXT,
        facts_json TEXT NOT NULL DEFAULT '{}', attempt INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'in_progress', outcome TEXT, matched_branch TEXT,
        collected_json TEXT NOT NULL DEFAULT '{}', detected_language TEXT,
        started_at TEXT NOT NULL, ended_at TEXT)""")
    conn.execute("INSERT INTO scenario_calls (call_sid, scenario_id, language, started_at, outcome) "
                 "VALUES ('CAold', 'shift_confirm', 'hi', '2026-09-30T00:00:00+00:00', 'confirmed')")
    monkeypatch.setattr(simple_db, "_connection", conn)

    simple_db.init_db()

    cols = {r[1] for r in conn.execute("PRAGMA table_info(scenario_calls)")}
    assert {"absence_reason", "sentiment", "reason_category", "summary", "follow_up_required",
            "transcript_text", "analysis_status"} <= cols
    assert scenario_store.get("CAold")["outcome"] == "confirmed"


async def test_details_and_analysis_land_on_the_row(db, monkeypatch):
    _call()
    fake = _fake_llm(monkeypatch, ANALYSIS)

    assert call_analysis.store_details("CA1", avg_turn_latency_ms=1300) is True
    assert scenario_store.get("CA1")["analysis_status"] == "pending"
    await call_analysis.analyze_call("CA1")

    row = scenario_store.get("CA1")
    assert row["absence_reason"] == "बुखार है"
    assert row["turn_count"] == 2 and row["avg_turn_latency_ms"] == 1300
    assert row["transcript_text"].splitlines() == ["Bot: नमस्ते, क्या आप कल आएंगे?", "Caller: नहीं आ पाऊँगा", "Caller: बुखार है"]
    assert row["sentiment"] == "neutral" and row["emotion"] == "apologetic"
    assert row["reason_category"] == "illness" and row["reason_english"] == "has a fever"
    assert row["follow_up_required"] is False and row["follow_up_note"] is None
    assert row["analysis_status"] == "done" and row["analyzed_at"]
    assert "Caller: बुखार है" in fake.user_text  # the model saw the transcript


async def test_reanalysis_keeps_the_measured_latency(db, monkeypatch):
    _call()
    _fake_llm(monkeypatch, ANALYSIS)
    call_analysis.store_details("CA1", avg_turn_latency_ms=1300)
    call_analysis.store_details("CA1")  # POST /scenario-calls/{sid}/analyze path
    assert scenario_store.get("CA1")["avg_turn_latency_ms"] == 1300


async def test_invalid_model_output_marks_failed_but_keeps_details(db, monkeypatch):
    _call()
    # llm_engine's invalid-JSON fallback is classification-shaped, not an analysis
    _fake_llm(monkeypatch, {"branch": "unclear", "confidence": 0.0})
    call_analysis.store_details("CA1")

    assert await call_analysis.analyze_call("CA1") is None

    row = scenario_store.get("CA1")
    assert row["analysis_status"] == "failed" and row["sentiment"] is None
    assert row["absence_reason"] == "बुखार है"


async def test_silent_call_is_skipped_without_calling_the_model(db, monkeypatch):
    _call(reason=None, user_turns=())

    async def must_not_run(**_):
        raise AssertionError("LLM called for a call where nobody spoke")

    monkeypatch.setattr(local_llm_engine, "classify_structured", must_not_run)

    assert call_analysis.store_details("CA1") is False
    assert scenario_store.get("CA1")["analysis_status"] == "skipped"


async def test_collected_reason_is_never_reported_as_no_reason(db, monkeypatch):
    _call(reason="अरे मेरे फैमिली में बहुत चल रहा है")
    _fake_llm(monkeypatch, {**ANALYSIS, "reason_category": "none", "sentiment_score": 7})
    await call_analysis.analyze_call("CA1")
    row = scenario_store.get("CA1")
    assert row["reason_category"] == "other"
    assert row["sentiment_score"] == 1.0  # clamped


def test_sweep_closes_stale_rows_only(db):
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    for sid in ("pending:abc", "CAstuck", "CAlive"):
        scenario_store.create(sid, "shift_confirm", "hi", "+91", {})
    with simple_db.get_cursor() as cur:
        cur.execute("UPDATE scenario_calls SET started_at = ? WHERE call_sid != 'CAlive'", (old,))

    assert scenario_store.sweep_stale() == 2

    assert scenario_store.get("pending:abc")["status"] == "not_connected"
    assert scenario_store.get("pending:abc")["outcome"] == "not_connected"
    assert scenario_store.get("CAstuck")["status"] == "abandoned"
    assert scenario_store.get("CAlive")["status"] == "in_progress"
