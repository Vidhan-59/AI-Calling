"""routers/analytics.py - the read/reporting surface for models.Call/Turn/
CallOutcome. Calls the route functions directly against an in-memory
session (same pattern as tests/unit/test_dialer.py) rather than through
FastAPI's dependency injection - the routes take `session` as a plain
keyword argument, so this exercises the same query logic without needing a
TestClient.
"""

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import models
from db import Base
from routers.analytics import analytics_summary, get_call_detail, list_calls


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _seed_call(session, *, agent_id="shift_confirm", status="completed", duration_sec=60, outcome=None):
    call = models.Call(agent_id=agent_id, status=status, duration_sec=duration_sec)
    session.add(call)
    await session.flush()
    if outcome:
        session.add(models.CallOutcome(call_id=call.id, slots_json={}, primary_outcome=outcome, extracted_by_model="test"))
    await session.flush()
    return call


async def _seed_turn(session, call, *, seq, role, text, sentiment_label=None, latency_ms=None):
    session.add(
        models.Turn(
            call_id=call.id, seq=seq, role=role, text=text,
            sentiment_label=sentiment_label, latency_total_ms=latency_ms,
        )
    )
    await session.flush()


async def test_list_calls_returns_joined_outcome(session):
    call = await _seed_call(session, outcome="shift_confirmed")
    await session.commit()

    results = await list_calls(session=session)

    assert len(results) == 1
    assert results[0]["id"] == call.id
    assert results[0]["outcome"]["primary_outcome"] == "shift_confirmed"


async def test_list_calls_filters_by_agent_id(session):
    await _seed_call(session, agent_id="shift_confirm")
    await _seed_call(session, agent_id="punch_in_remind")
    await session.commit()

    results = await list_calls(agent_id="punch_in_remind", session=session)

    assert len(results) == 1
    assert results[0]["agent_id"] == "punch_in_remind"


async def test_get_call_detail_includes_turns_in_order(session):
    call = await _seed_call(session)
    await _seed_turn(session, call, seq=1, role="assistant", text="Namaste")
    await _seed_turn(session, call, seq=2, role="user", text="Haan", sentiment_label="positive")
    await session.commit()

    detail = await get_call_detail(call.id, session=session)

    assert [t["seq"] for t in detail["turns"]] == [1, 2]
    assert detail["turns"][1]["sentiment_label"] == "positive"


async def test_get_call_detail_404s_on_unknown_call(session):
    with pytest.raises(HTTPException) as exc_info:
        await get_call_detail("does-not-exist", session=session)
    assert exc_info.value.status_code == 404


async def test_analytics_summary_aggregates_outcomes_and_sentiment(session):
    call1 = await _seed_call(session, status="completed", duration_sec=100, outcome="shift_confirmed")
    call2 = await _seed_call(session, status="completed", duration_sec=200, outcome="shift_confirmed")
    call3 = await _seed_call(session, status="disconnected", duration_sec=None, outcome="call_dropped")
    await _seed_turn(session, call1, seq=1, role="user", text="haan", sentiment_label="positive", latency_ms=500)
    await _seed_turn(session, call2, seq=1, role="user", text="nahi", sentiment_label="negative", latency_ms=700)
    await session.commit()

    summary = await analytics_summary(session=session)

    assert summary["total_calls"] == 3
    assert summary["completed_calls"] == 2
    assert summary["completion_rate"] == pytest.approx(2 / 3)
    assert summary["avg_duration_sec"] == pytest.approx(150.0)
    assert summary["outcome_breakdown"] == {"shift_confirmed": 2, "call_dropped": 1}
    assert summary["sentiment_breakdown"] == {"positive": 1, "negative": 1}
    assert summary["avg_turn_latency_ms"] == pytest.approx(600.0)


async def test_analytics_summary_handles_no_calls(session):
    summary = await analytics_summary(session=session)

    assert summary["total_calls"] == 0
    assert summary["completion_rate"] is None
    assert summary["outcome_breakdown"] == {}
    assert summary["sentiment_breakdown"] == {}
