"""Isolated SQLite store for the simple flat agent-CRUD / calls / messages
tables used by agent_manager.py, conversation_store.py, and routers/*.py.

Why this file exists instead of reusing db.py: db.py was replaced (by a
separate, in-progress effort in this same repo) with an async SQLAlchemy
engine targeting data/app.db for the full ShiftEaze data model (Agent,
AgentVersion, Contact, Call, Turn, ...). That work is still in progress and
owned elsewhere - this module intentionally does NOT touch db.py, config.py,
or models.py, and writes to its own separate file (data/legacy_app.db) so
there is zero risk of colliding with that schema. If/when the ShiftEaze
rebuild reaches the point of replacing the simple /agents and /chat/test
demo path, this whole file can just be deleted.
"""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "legacy_app.db")

_lock = threading.Lock()
_connection = sqlite3.connect(DB_PATH, check_same_thread=False)
_connection.row_factory = sqlite3.Row


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_cursor():
    """Serialize all DB access through a single connection + lock.

    Call volume here is low (one write per conversational turn), so a
    single shared connection is simpler and safer than a pool.
    """
    with _lock:
        cursor = _connection.cursor()
        try:
            yield cursor
            _connection.commit()
        except Exception:
            _connection.rollback()
            raise
        finally:
            cursor.close()


def init_db() -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                welcome_greeting TEXT NOT NULL,
                voice TEXT NOT NULL,
                model TEXT NOT NULL,
                effort TEXT NOT NULL DEFAULT 'medium',
                knowledge_base TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_sid TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                from_number TEXT,
                to_number TEXT,
                direction TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'in_progress',
                started_at TEXT NOT NULL,
                ended_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_sid TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sentiment_label TEXT,
                sentiment_score REAL,
                emotion TEXT,
                escalate INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_call_sid ON messages(call_sid)"
        )
        # ShiftEaze scenario calls: one row per placed scenario call, carrying
        # the facts it was given and the classified outcome. Separate from the
        # generic `calls` table because the outcome vocabulary here is defined
        # by the scenario JSON (confirmed / ot_declined / disputed / ...) and is
        # what Odoo ultimately consumes.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scenario_calls (
                call_sid TEXT PRIMARY KEY,
                scenario_id TEXT NOT NULL,
                language TEXT NOT NULL,
                to_number TEXT,
                employee_name TEXT,
                rgid TEXT,
                facts_json TEXT NOT NULL DEFAULT '{}',
                attempt INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'in_progress',
                outcome TEXT,
                matched_branch TEXT,
                collected_json TEXT NOT NULL DEFAULT '{}',
                detected_language TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenario_calls_scenario ON scenario_calls(scenario_id)"
        )

    _seed_default_agent()


def _load_seed_prompt() -> str:
    from config import BASE_DIR

    path = os.path.join(BASE_DIR, "prompts", "system_prompt.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return (
            "You are a helpful, friendly AI voice assistant speaking with a caller "
            "over the phone. Keep responses concise (1-3 sentences) since this is a "
            "voice call, not a chat window."
        )


def _seed_default_agent() -> None:
    from config import DEFAULT_CLAUDE_MODEL

    with get_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM agents")
        if cur.fetchone()["c"] > 0:
            return

        default_prompt = _load_seed_prompt()
        timestamp = now_iso()
        cur.execute(
            """
            INSERT INTO agents
                (id, name, system_prompt, welcome_greeting, voice, model, effort, knowledge_base, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "default",
                "Default Assistant",
                default_prompt,
                "Hello! Thanks for calling. How can I help you today?",
                "Google.en-US-Neural2-C",
                DEFAULT_CLAUDE_MODEL,
                "medium",
                json.dumps([]),
                timestamp,
                timestamp,
            ),
        )
