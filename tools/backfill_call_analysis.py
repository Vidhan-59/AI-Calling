"""Fill the call-detail and post-call-analysis columns of scenario_calls
(data/legacy_app.db) for calls made before they existed.

    .\\venv\\Scripts\\python.exe tools\\backfill_call_analysis.py          # rows not analysed yet
    .\\venv\\Scripts\\python.exe tools\\backfill_call_analysis.py --all    # re-analyse every call

Needs `ollama serve` running (the analysis is Qwen). Safe to run while the
server is up; calls are analysed one at a time.
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from simple_db import get_cursor, init_db  # noqa: E402
from services import call_analysis, scenario_store  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="re-analyse calls that already have an analysis")
    args = parser.parse_args()

    init_db()  # adds the new columns if the server hasn't been restarted yet
    swept = scenario_store.sweep_stale()
    with get_cursor() as cur:
        where = "status != 'in_progress' AND call_sid NOT LIKE 'pending:%'"
        if not args.all:
            where += " AND (analysis_status IS NULL OR analysis_status IN ('pending', 'failed'))"
        cur.execute(f"SELECT call_sid FROM scenario_calls WHERE {where} ORDER BY started_at")
        sids = [r[0] for r in cur.fetchall()]
    print(f"closed {swept} stale row(s); {len(sids)} call(s) to process")

    done = skipped = failed = 0
    for i, sid in enumerate(sids, 1):
        if not call_analysis.store_details(sid):
            skipped += 1
            print(f"[{i}/{len(sids)}] {sid}: caller said nothing - details saved, no analysis")
            continue
        result = await call_analysis.analyze_call(sid)
        if result:
            done += 1
            print(f"[{i}/{len(sids)}] {sid}: {result['sentiment']}, {result['reason_category']} - {result['summary']}")
        else:
            failed += 1
            print(f"[{i}/{len(sids)}] {sid}: analysis FAILED (is ollama running?)")
    print(f"done: {done} analysed, {skipped} silent, {failed} failed")


if __name__ == "__main__":
    asyncio.run(main())
