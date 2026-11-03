"""Persistence for ShiftEaze scenario calls: the facts we called with, the
branch the AI matched, and the resulting outcome.

The per-turn transcript itself still goes to services/conversation_store.py
(shared with the generic agent), so `GET /scenario-calls/{sid}` can join the
two: scenario outcome here, verbatim turns + sentiment there.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from simple_db import get_cursor, now_iso


def create(
    call_sid: str,
    scenario_id: str,
    language: str,
    to_number: str,
    facts: dict[str, Any],
    attempt: int = 1,
    rgid: Optional[str] = None,
) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT OR REPLACE INTO scenario_calls
                (call_sid, scenario_id, language, to_number, employee_name, rgid,
                 facts_json, attempt, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'in_progress', ?)
            """,
            (
                call_sid,
                scenario_id,
                language,
                to_number,
                (facts or {}).get("employee_name"),
                rgid,
                json.dumps(facts or {}, ensure_ascii=False),
                attempt,
                now_iso(),
            ),
        )


def record_outcome(
    call_sid: str,
    outcome: str,
    matched_branch: Optional[str] = None,
    collected: Optional[dict] = None,
    detected_language: Optional[str] = None,
    status: str = "completed",
) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE scenario_calls
               SET outcome = ?, matched_branch = ?, collected_json = ?,
                   detected_language = ?, status = ?, ended_at = ?
             WHERE call_sid = ?
            """,
            (
                outcome,
                matched_branch,
                json.dumps(collected or {}, ensure_ascii=False),
                detected_language,
                status,
                now_iso(),
                call_sid,
            ),
        )


def record_call_details(
    call_sid: str,
    *,
    absence_reason: Optional[str],
    duration_secs: Optional[float],
    turn_count: int,
    avg_turn_latency_ms: Optional[int],
    transcript_text: str,
    analysis_status: str,
) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE scenario_calls
               SET absence_reason = ?, duration_secs = ?, turn_count = ?,
                   avg_turn_latency_ms = COALESCE(?, avg_turn_latency_ms), transcript_text = ?,
                   analysis_status = ?
             WHERE call_sid = ?
            """,
            (absence_reason, duration_secs, turn_count, avg_turn_latency_ms, transcript_text,
             analysis_status, call_sid),
        )


def record_analysis(call_sid: str, analysis: Optional[dict], status: str) -> None:
    a = analysis or {}
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE scenario_calls
               SET sentiment = ?, sentiment_score = ?, emotion = ?, reason_category = ?,
                   reason_english = ?, summary = ?, follow_up_required = ?, follow_up_note = ?,
                   analysis_status = ?, analyzed_at = ?
             WHERE call_sid = ?
            """,
            (
                a.get("sentiment"), a.get("sentiment_score"), a.get("emotion"), a.get("reason_category"),
                a.get("reason_english"), a.get("summary"),
                None if "follow_up_required" not in a else int(bool(a["follow_up_required"])),
                a.get("follow_up_note"), status, now_iso(), call_sid,
            ),
        )


def set_placed_call_sid(row_id: str, twilio_call_sid: str) -> None:
    with get_cursor() as cur:
        cur.execute("UPDATE scenario_calls SET placed_call_sid = ? WHERE call_sid = ?", (twilio_call_sid, row_id))


def delete(call_sid: str) -> None:
    with get_cursor() as cur:
        cur.execute("DELETE FROM scenario_calls WHERE call_sid = ?", (call_sid,))


def sweep_stale(older_than_minutes: int = 15) -> int:
    """Close rows that will never finish on their own:
    - "pending:*" placement rows the phone never connected (no answer, busy,
      rejected) - the connected call gets its own row and deletes this one;
    - connected rows left "in_progress" by a server crash/restart mid-call.
    A call lasts at most a few minutes, so 15 minutes is unambiguous."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)).isoformat()
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE scenario_calls
               SET status = CASE WHEN call_sid LIKE 'pending:%' THEN 'not_connected' ELSE 'abandoned' END,
                   outcome = COALESCE(outcome, CASE WHEN call_sid LIKE 'pending:%' THEN 'not_connected' ELSE 'unknown' END),
                   analysis_status = COALESCE(analysis_status, 'skipped'),
                   ended_at = COALESCE(ended_at, ?)
             WHERE status = 'in_progress' AND started_at < ?
            """,
            (now_iso(), cutoff),
        )
        return cur.rowcount


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["facts"] = json.loads(d.pop("facts_json") or "{}")
    d["collected"] = json.loads(d.pop("collected_json") or "{}")
    if d.get("follow_up_required") is not None:
        d["follow_up_required"] = bool(d["follow_up_required"])
    return d


def get(call_sid: str) -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM scenario_calls WHERE call_sid = ?", (call_sid,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


def list_calls(scenario_id: Optional[str] = None, limit: int = 100) -> list[dict]:
    sweep_stale()
    with get_cursor() as cur:
        if scenario_id:
            cur.execute(
                "SELECT * FROM scenario_calls WHERE scenario_id = ? ORDER BY started_at DESC LIMIT ?",
                (scenario_id, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM scenario_calls ORDER BY started_at DESC LIMIT ?", (limit,)
            )
        return [_row_to_dict(r) for r in cur.fetchall()]
