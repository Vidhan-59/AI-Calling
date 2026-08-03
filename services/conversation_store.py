from typing import Optional

from simple_db import get_cursor, now_iso


def create_call(call_sid: str, agent_id: str, from_number: str, to_number: str, direction: str) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT OR IGNORE INTO calls (call_sid, agent_id, from_number, to_number, direction, status, started_at)
            VALUES (?, ?, ?, ?, ?, 'in_progress', ?)
            """,
            (call_sid, agent_id, from_number, to_number, direction, now_iso()),
        )


def end_call(call_sid: str, status: str = "completed") -> None:
    with get_cursor() as cur:
        cur.execute(
            "UPDATE calls SET status = ?, ended_at = ? WHERE call_sid = ?",
            (status, now_iso(), call_sid),
        )


def get_call(call_sid: str) -> Optional[dict]:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM calls WHERE call_sid = ?", (call_sid,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_calls(limit: int = 100) -> list[dict]:
    with get_cursor() as cur:
        cur.execute("SELECT * FROM calls ORDER BY started_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]


def add_message(
    call_sid: str,
    role: str,
    content: str,
    sentiment_label: Optional[str] = None,
    sentiment_score: Optional[float] = None,
    emotion: Optional[str] = None,
    escalate: bool = False,
) -> int:
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages (call_sid, role, content, sentiment_label, sentiment_score, emotion, escalate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (call_sid, role, content, sentiment_label, sentiment_score, emotion, int(escalate), now_iso()),
        )
        return cur.lastrowid


def get_messages(call_sid: str) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM messages WHERE call_sid = ? ORDER BY id ASC",
            (call_sid,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_call_summary(call_sid: str) -> Optional[dict]:
    call = get_call(call_sid)
    if not call:
        return None
    messages = get_messages(call_sid)

    scores = [m["sentiment_score"] for m in messages if m["sentiment_score"] is not None]
    escalate_flagged = any(m["escalate"] for m in messages)
    avg_sentiment = sum(scores) / len(scores) if scores else None

    return {
        "call": call,
        "transcript": messages,
        "message_count": len(messages),
        "average_sentiment_score": avg_sentiment,
        "escalation_recommended": escalate_flagged,
    }
