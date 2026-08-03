"""Persistence for ShiftEaze scenario calls: the facts we called with, the
branch the AI matched, and the resulting outcome.

The per-turn transcript itself still goes to services/conversation_store.py
(shared with the generic agent), so `GET /scenario-calls/{sid}` can join the
two: scenario outcome here, verbatim turns + sentiment there.
"""

import json
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


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["facts"] = json.loads(d.pop("facts_json") or "{}")
    d["collected"] = json.loads(d.pop("collected_json") or "{}")
    return d


def get(call_sid: str) -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM scenario_calls WHERE call_sid = ?", (call_sid,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None


def list_calls(scenario_id: Optional[str] = None, limit: int = 100) -> list[dict]:
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
