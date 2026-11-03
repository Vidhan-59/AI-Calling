"""Persistence for Studio agents, training material, calls, messages and
trace events (tables created on first use in the existing simple_db file)."""

import json
import secrets
import threading
from typing import Any, Optional

from simple_db import get_cursor, now_iso

STATUSES = ("draft", "published", "active")
TRAINING_TYPES = ("script", "qa", "scenario", "document")

_ready = False
_ready_lock = threading.Lock()

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS studio_agents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        avatar TEXT NOT NULL DEFAULT '',
        use_case_id TEXT,
        language TEXT NOT NULL DEFAULT 'hi',
        status TEXT NOT NULL DEFAULT 'draft',
        voice TEXT NOT NULL DEFAULT '{}',
        personality TEXT NOT NULL DEFAULT '{}',
        system_instructions TEXT NOT NULL DEFAULT '',
        basic TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS studio_training (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        training_type TEXT NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL DEFAULT '',
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_studio_training_agent ON studio_training(agent_id)",
    """CREATE TABLE IF NOT EXISTS studio_calls (
        id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        use_case_id TEXT,
        kind TEXT NOT NULL DEFAULT 'call',
        language TEXT NOT NULL DEFAULT 'hi',
        status TEXT NOT NULL DEFAULT 'connecting',
        started_at TEXT NOT NULL,
        ended_at TEXT,
        duration_secs REAL,
        outcome TEXT,
        error TEXT,
        agent_snapshot TEXT NOT NULL DEFAULT '{}',
        metrics TEXT NOT NULL DEFAULT '{}'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_studio_calls_agent ON studio_calls(agent_id, started_at)",
    """CREATE TABLE IF NOT EXISTS studio_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_id TEXT NOT NULL,
        speaker TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_studio_messages_call ON studio_messages(call_id, id)",
    """CREATE TABLE IF NOT EXISTS studio_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_id TEXT NOT NULL,
        category TEXT NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'ok',
        detail TEXT NOT NULL DEFAULT '',
        duration_ms REAL,
        data TEXT NOT NULL DEFAULT '{}',
        timestamp TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_studio_traces_call ON studio_traces(call_id, id)",
]


def ensure_tables() -> None:
    global _ready
    if _ready:
        return
    with _ready_lock:
        if _ready:
            return
        with get_cursor() as cur:
            for stmt in _SCHEMA:
                cur.execute(stmt)
        _ready = True


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(5)}"


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------- agents

def _agent(row) -> dict:
    d = dict(row)
    for key in ("voice", "personality", "basic"):
        d[key] = json.loads(d[key] or "{}")
    return d


def list_agents() -> list[dict]:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute(
            """SELECT a.*, (SELECT COUNT(*) FROM studio_calls c WHERE c.agent_id = a.id AND c.kind = 'call') AS call_count
               FROM studio_agents a ORDER BY a.updated_at DESC"""
        )
        return [_agent(r) for r in cur.fetchall()]


def get_agent(agent_id: str) -> Optional[dict]:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("SELECT * FROM studio_agents WHERE id = ?", (agent_id,))
        row = cur.fetchone()
        return _agent(row) if row else None


def agent_for_use_case(use_case_id: str) -> Optional[dict]:
    """The agent bound to a use case: an active/published one wins over a draft."""
    ensure_tables()
    with get_cursor() as cur:
        cur.execute(
            """SELECT * FROM studio_agents WHERE use_case_id = ?
               ORDER BY CASE status WHEN 'active' THEN 0 WHEN 'published' THEN 1 ELSE 2 END, updated_at DESC LIMIT 1""",
            (use_case_id,),
        )
        row = cur.fetchone()
        return _agent(row) if row else None


_AGENT_COLUMNS = ("name", "description", "avatar", "use_case_id", "language", "status", "voice", "personality",
                  "system_instructions", "basic")
_JSON_COLUMNS = ("voice", "personality", "basic")


def create_agent(**fields) -> dict:
    ensure_tables()
    agent_id = _new_id("agt")
    ts = now_iso()
    values = {c: fields.get(c) for c in _AGENT_COLUMNS if fields.get(c) is not None}
    values.setdefault("name", "Untitled agent")
    for c in _JSON_COLUMNS:
        values[c] = _j(values.get(c) or {})
    cols = ["id", *values, "created_at", "updated_at"]
    with get_cursor() as cur:
        cur.execute(
            f"INSERT INTO studio_agents ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            (agent_id, *values.values(), ts, ts),
        )
    return get_agent(agent_id)


def update_agent(agent_id: str, **fields) -> Optional[dict]:
    if not get_agent(agent_id):
        return None
    updates = {c: v for c, v in fields.items() if c in _AGENT_COLUMNS and v is not None}
    for c in _JSON_COLUMNS:
        if c in updates:
            updates[c] = _j(updates[c])
    if updates:
        updates["updated_at"] = now_iso()
        with get_cursor() as cur:
            cur.execute(
                f"UPDATE studio_agents SET {', '.join(f'{k} = ?' for k in updates)} WHERE id = ?",
                (*updates.values(), agent_id),
            )
    return get_agent(agent_id)


def clear_use_case(agent_id: str) -> None:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("UPDATE studio_agents SET use_case_id = NULL, updated_at = ? WHERE id = ?", (now_iso(), agent_id))


def delete_agent(agent_id: str) -> bool:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("DELETE FROM studio_training WHERE agent_id = ?", (agent_id,))
        cur.execute("DELETE FROM studio_agents WHERE id = ?", (agent_id,))
        return cur.rowcount > 0


def duplicate_agent(agent_id: str) -> Optional[dict]:
    src = get_agent(agent_id)
    if not src:
        return None
    fields = {c: src[c] for c in _AGENT_COLUMNS if c != "use_case_id"}
    copy = create_agent(**{**fields, "name": f"{src['name']} (copy)", "status": "draft"})
    for item in list_training(agent_id):
        add_training(copy["id"], item["training_type"], item["title"], item["content"], item["metadata"])
    return copy


# -------------------------------------------------------------- training

def _training(row) -> dict:
    d = dict(row)
    d["metadata"] = json.loads(d["metadata"] or "{}")
    return d


def list_training(agent_id: str, training_type: str | None = None) -> list[dict]:
    ensure_tables()
    with get_cursor() as cur:
        if training_type:
            cur.execute("SELECT * FROM studio_training WHERE agent_id = ? AND training_type = ? ORDER BY created_at, id",
                        (agent_id, training_type))
        else:
            cur.execute("SELECT * FROM studio_training WHERE agent_id = ? ORDER BY created_at, id", (agent_id,))
        return [_training(r) for r in cur.fetchall()]


def add_training(agent_id: str, training_type: str, title: str, content: str, metadata: dict | None = None) -> dict:
    ensure_tables()
    item_id = _new_id("trn")
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO studio_training (id, agent_id, training_type, title, content, metadata, created_at) VALUES (?,?,?,?,?,?,?)",
            (item_id, agent_id, training_type, title, content, _j(metadata or {}), now_iso()),
        )
    return get_training(item_id)


def get_training(item_id: str) -> Optional[dict]:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("SELECT * FROM studio_training WHERE id = ?", (item_id,))
        row = cur.fetchone()
        return _training(row) if row else None


def update_training(item_id: str, *, title=None, content=None, metadata=None) -> Optional[dict]:
    if not get_training(item_id):
        return None
    sets, vals = [], []
    for col, val in (("title", title), ("content", content)):
        if val is not None:
            sets.append(f"{col} = ?")
            vals.append(val)
    if metadata is not None:
        sets.append("metadata = ?")
        vals.append(_j(metadata))
    if sets:
        with get_cursor() as cur:
            cur.execute(f"UPDATE studio_training SET {', '.join(sets)} WHERE id = ?", (*vals, item_id))
    return get_training(item_id)


def delete_training(item_id: str) -> bool:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("DELETE FROM studio_training WHERE id = ?", (item_id,))
        return cur.rowcount > 0


def replace_training(agent_id: str, training_type: str, items: list[dict]) -> int:
    """Bulk import: replace every item of one type for the agent."""
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("DELETE FROM studio_training WHERE agent_id = ? AND training_type = ?", (agent_id, training_type))
    for it in items:
        add_training(agent_id, training_type, it.get("title", ""), it.get("content", ""), it.get("metadata") or {})
    return len(items)


# ----------------------------------------------------------------- calls

def create_call(*, agent: dict, kind: str, language: str, use_case_id: str | None = None) -> str:
    ensure_tables()
    call_id = _new_id("call")
    snapshot = {k: agent[k] for k in ("id", "name", "voice", "personality", "system_instructions", "language")}
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO studio_calls (id, agent_id, use_case_id, kind, language, status, started_at, agent_snapshot)
               VALUES (?,?,?,?,?,?,?,?)""",
            (call_id, agent["id"], use_case_id or agent.get("use_case_id"), kind, language, "connecting", now_iso(), _j(snapshot)),
        )
    return call_id


def update_call(call_id: str, **fields) -> None:
    ensure_tables()
    allowed = {"status", "ended_at", "duration_secs", "outcome", "error", "metrics"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if "metrics" in updates:
        updates["metrics"] = _j(updates["metrics"])
    if not updates:
        return
    with get_cursor() as cur:
        cur.execute(f"UPDATE studio_calls SET {', '.join(f'{k} = ?' for k in updates)} WHERE id = ?", (*updates.values(), call_id))


def _call(row) -> dict:
    d = dict(row)
    d["agent_snapshot"] = json.loads(d["agent_snapshot"] or "{}")
    d["metrics"] = json.loads(d["metrics"] or "{}")
    return d


def get_call(call_id: str) -> Optional[dict]:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("SELECT * FROM studio_calls WHERE id = ?", (call_id,))
        row = cur.fetchone()
        return _call(row) if row else None


def list_calls(*, agent_id=None, use_case_id=None, status=None, date_from=None, date_to=None, limit=100) -> list[dict]:
    ensure_tables()
    where, vals = ["1=1"], []
    for col, val in (("agent_id", agent_id), ("use_case_id", use_case_id), ("status", status)):
        if val:
            where.append(f"c.{col} = ?")
            vals.append(val)
    if date_from:
        where.append("c.started_at >= ?")
        vals.append(date_from)
    if date_to:
        where.append("c.started_at <= ?")
        vals.append(date_to + "T23:59:59.999999+00:00" if len(date_to) == 10 else date_to)
    with get_cursor() as cur:
        cur.execute(
            f"""SELECT c.*, a.name AS agent_name,
                       (SELECT COUNT(*) FROM studio_messages m WHERE m.call_id = c.id) AS message_count
                FROM studio_calls c LEFT JOIN studio_agents a ON a.id = c.agent_id
                WHERE {' AND '.join(where)} ORDER BY c.started_at DESC LIMIT ?""",
            (*vals, max(1, min(limit, 500))),
        )
        return [_call(r) | {"agent_name": r["agent_name"], "message_count": r["message_count"]} for r in cur.fetchall()]


def stats() -> dict:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("SELECT status, COUNT(*) n FROM studio_agents GROUP BY status")
        by_status = {r["status"]: r["n"] for r in cur.fetchall()}
        cur.execute(
            """SELECT COUNT(*) total, SUM(CASE WHEN outcome = 'completed' THEN 1 ELSE 0 END) ok,
                      AVG(duration_secs) avg_dur FROM studio_calls WHERE kind = 'call' AND status != 'connecting'"""
        )
        c = cur.fetchone()
    return {
        "agents_total": sum(by_status.values()),
        "agents_active": by_status.get("active", 0),
        "agents_published": by_status.get("published", 0),
        "agents_draft": by_status.get("draft", 0),
        "calls_total": c["total"] or 0,
        "calls_successful": c["ok"] or 0,
        "avg_duration_secs": round(c["avg_dur"] or 0, 1),
    }


# ------------------------------------------------- messages & trace rows

def add_message(call_id: str, speaker: str, message: str, metadata: dict | None = None) -> dict:
    ensure_tables()
    ts = now_iso()
    with get_cursor() as cur:
        cur.execute("INSERT INTO studio_messages (call_id, speaker, message, timestamp, metadata) VALUES (?,?,?,?,?)",
                    (call_id, speaker, message, ts, _j(metadata or {})))
        mid = cur.lastrowid
    return {"id": mid, "call_id": call_id, "speaker": speaker, "message": message, "timestamp": ts, "metadata": metadata or {}}


def list_messages(call_id: str) -> list[dict]:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("SELECT * FROM studio_messages WHERE call_id = ? ORDER BY id", (call_id,))
        return [dict(r) | {"metadata": json.loads(r["metadata"] or "{}")} for r in cur.fetchall()]


def add_trace(call_id: str, category: str, name: str, status: str, detail: str,
              duration_ms: float | None, data: dict | None) -> dict:
    ensure_tables()
    ts = now_iso()
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO studio_traces (call_id, category, name, status, detail, duration_ms, data, timestamp) VALUES (?,?,?,?,?,?,?,?)",
            (call_id, category, name, status, detail, duration_ms, _j(data or {}), ts),
        )
        tid = cur.lastrowid
    return {"id": tid, "call_id": call_id, "category": category, "name": name, "status": status, "detail": detail,
            "duration_ms": duration_ms, "data": data or {}, "timestamp": ts}


def list_traces(call_id: str, after_id: int = 0) -> list[dict]:
    ensure_tables()
    with get_cursor() as cur:
        cur.execute("SELECT * FROM studio_traces WHERE call_id = ? AND id > ? ORDER BY id", (call_id, after_id))
        return [dict(r) | {"data": json.loads(r["data"] or "{}")} for r in cur.fetchall()]
