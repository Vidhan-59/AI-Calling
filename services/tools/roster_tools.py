"""Read-only roster lookup. Hits a pre-warmed Redis snapshot, never a live
Odoo ORM query - the architecture is explicit that a cold Odoo call inside a
live conversation is not acceptable within the tool timeout budget. The
snapshot is populated by workers/roster_sync.py (a periodic job pulling from
Odoo, not implemented in this build) or, for local dev/tests, by
seed_mock_roster below.
"""

import json
from dataclasses import dataclass

from db import get_redis
from services.tools.registry import tool


@dataclass
class RosterInfo:
    rgid: str
    name: str
    date: str
    shift_code: str  # D | N | O
    terminal: str
    reporting_time: str  # "HH:MM"
    roster_line_id: str


def _roster_key(rgid: str, date: str) -> str:
    return f"roster:{rgid}:{date}"


@tool(idempotent=True, timeout_ms=1200)
async def lookup_roster(rgid: str, date: str) -> dict:
    """Returns the roster snapshot for one operator on one date. Raises if
    the snapshot is missing - callers must not fabricate a shift when this
    fails; the hallucination_guard guardrail depends on that."""
    redis = get_redis()
    raw = await redis.get(_roster_key(rgid, date))
    if raw is None:
        raise LookupError(f"No roster snapshot cached for rgid={rgid} date={date}")
    return json.loads(raw)


async def seed_mock_roster(entries: list[RosterInfo], ttl_seconds: int = 24 * 3600) -> None:
    """Local dev / eval-harness helper - writes fixture roster rows into
    Redis in the exact shape lookup_roster expects, so the rest of the
    stack can be exercised without a real Odoo sync job."""
    redis = get_redis()
    for entry in entries:
        await redis.set(
            _roster_key(entry.rgid, entry.date),
            json.dumps(entry.__dict__),
            ex=ttl_seconds,
        )
