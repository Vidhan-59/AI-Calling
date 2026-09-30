"""services/dispatch/dialer.py - the queue-backed outbound dialer.

Master-prompt requirement: "unit tests for the compliance gate and dialer
logic at minimum." This file covers the dialer; tests/unit/test_admission.py
covers the compliance gate it calls.
"""

import fakeredis
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import models
from db import Base
from services.dispatch import dialer
from services.telephony.mock import MockTelephonyProvider


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr("services.dispatch.admission.get_redis", lambda: redis)
    monkeypatch.setattr("services.dispatch.dialer.get_redis", lambda: redis)
    return redis


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _make_campaign(session, **overrides) -> models.Campaign:
    defaults = dict(agent_id="shift_confirm", agent_version=1, name="c1", scenario="shift_confirm")
    defaults.update(overrides)
    c = models.Campaign(**defaults)
    session.add(c)
    await session.flush()
    return c


_contact_counter = 0


async def _make_contact(session, **overrides) -> models.Contact:
    global _contact_counter
    _contact_counter += 1
    defaults = dict(
        rgid=f"RG-{_contact_counter}", name="Amit", phone_e164="+919999999999", consent_status="granted",
    )
    defaults.update(overrides)
    c = models.Contact(**defaults)
    session.add(c)
    await session.flush()
    return c


async def _make_task(session, campaign, contact, **overrides) -> models.CallTask:
    defaults = dict(
        campaign_id=campaign.id, contact_id=contact.id, state="pending",
        idempotency_key=f"{campaign.id}:{contact.id}:1",
    )
    defaults.update(overrides)
    t = models.CallTask(**defaults)
    session.add(t)
    await session.flush()
    return t


class _FailingProvider(MockTelephonyProvider):
    async def originate_call(self, **kwargs):
        raise RuntimeError("simulated carrier failure")


def _webhook_url_for(task_id: str) -> str:
    return f"https://example.test/agent-voice/twiml?task_id={task_id}"


async def test_originates_eligible_task_and_creates_a_call_row(session):
    campaign = await _make_campaign(session)
    contact = await _make_contact(session)
    task = await _make_task(session, campaign, contact)

    result = await dialer.run_dispatch_cycle(
        session, answer_webhook_url_for=_webhook_url_for, provider=MockTelephonyProvider(),
    )
    await session.commit()

    assert result["originated"] == 1
    assert result["skipped"] == 0
    await session.refresh(task)
    assert task.state == "in_call"

    call = (
        await session.scalars(select(models.Call).where(models.Call.call_task_id == task.id))
    ).first()
    assert call is not None
    assert call.provider == "mock"
    assert await dialer.active_call_count() == 1


async def test_skips_dnc_contact_and_marks_task_abandoned(session):
    campaign = await _make_campaign(session)
    contact = await _make_contact(session, dnc=True)
    task = await _make_task(session, campaign, contact)

    result = await dialer.run_dispatch_cycle(
        session, answer_webhook_url_for=_webhook_url_for, provider=MockTelephonyProvider(),
    )

    assert result["originated"] == 0
    assert result["skipped"] == 1
    assert result["skip_reasons"] == {"dnc": 1}
    await session.refresh(task)
    assert task.state == "abandoned"
    assert await dialer.active_call_count() == 0


async def test_respects_the_concurrency_ceiling(session, _fake_redis):
    await _fake_redis.set(dialer.ACTIVE_CALLS_REDIS_KEY, 5)

    campaign = await _make_campaign(session)
    contact = await _make_contact(session)
    task = await _make_task(session, campaign, contact)

    result = await dialer.run_dispatch_cycle(
        session, answer_webhook_url_for=_webhook_url_for,
        provider=MockTelephonyProvider(), max_concurrent=5,
    )

    assert result["capacity"] == 0
    assert result["originated"] == 0
    await session.refresh(task)
    assert task.state == "pending"  # never even picked up


async def test_kill_switch_leaves_task_pending_for_retry_not_abandoned(session, _fake_redis):
    await _fake_redis.set("ai_calling:kill_switch", "1")

    campaign = await _make_campaign(session)
    contact = await _make_contact(session)
    task = await _make_task(session, campaign, contact)

    result = await dialer.run_dispatch_cycle(
        session, answer_webhook_url_for=_webhook_url_for, provider=MockTelephonyProvider(),
    )

    assert result["originated"] == 0
    await session.refresh(task)
    assert task.state == "pending"  # transient - must not need a human to un-abandon it
    assert task.attempt_no == 0  # a kill-switch skip doesn't consume a retry attempt


async def test_origination_failure_schedules_a_retry(session):
    campaign = await _make_campaign(session)
    contact = await _make_contact(session)
    task = await _make_task(session, campaign, contact)

    result = await dialer.run_dispatch_cycle(
        session, answer_webhook_url_for=_webhook_url_for, provider=_FailingProvider(),
    )

    assert result["skipped"] == 1
    assert result["skip_reasons"] == {"origination_error": 1}
    await session.refresh(task)
    assert task.state == "pending"
    assert task.attempt_no == 1
    assert task.next_attempt_at is not None


async def test_exhausted_retries_marks_task_failed(session):
    campaign = await _make_campaign(session)  # shift_confirm.yaml: retry.attempts == 3
    contact = await _make_contact(session)
    task = await _make_task(session, campaign, contact, attempt_no=3)

    await dialer.run_dispatch_cycle(
        session, answer_webhook_url_for=_webhook_url_for, provider=_FailingProvider(),
    )
    await session.refresh(task)
    assert task.state == "failed"


async def test_batch_size_and_priority_ordering(session):
    campaign = await _make_campaign(session)
    tasks = []
    for i in range(3):
        contact = await _make_contact(session)
        tasks.append(await _make_task(session, campaign, contact, priority=i))

    result = await dialer.run_dispatch_cycle(
        session, answer_webhook_url_for=_webhook_url_for,
        provider=MockTelephonyProvider(), batch_size=2, max_concurrent=10,
    )
    assert result["picked"] == 2
    assert result["originated"] == 2
    # Highest priority (2) and next (1) go first; priority 0 stays pending.
    await session.refresh(tasks[0])
    assert tasks[0].state == "pending"


async def test_call_status_completed_marks_task_completed_and_frees_concurrency_slot(session, _fake_redis):
    await _fake_redis.set(dialer.ACTIVE_CALLS_REDIS_KEY, 1)
    campaign = await _make_campaign(session)
    contact = await _make_contact(session)
    task = await _make_task(session, campaign, contact, state="in_call")
    call = models.Call(call_task_id=task.id, provider_call_sid="CA-123", provider="mock")
    session.add(call)
    await session.flush()

    status = await dialer.handle_call_status_callback(
        session, provider_call_sid="CA-123", provider_status="completed",
    )

    assert status == "completed"
    await session.refresh(task)
    assert task.state == "completed"
    assert await dialer.active_call_count() == 0


async def test_call_status_no_answer_schedules_a_retry(session):
    campaign = await _make_campaign(session)
    contact = await _make_contact(session)
    task = await _make_task(session, campaign, contact, state="in_call")
    call = models.Call(call_task_id=task.id, provider_call_sid="CA-456", provider="mock")
    session.add(call)
    await session.flush()

    status = await dialer.handle_call_status_callback(
        session, provider_call_sid="CA-456", provider_status="no-answer",
    )

    assert status == "no-answer"
    await session.refresh(task)
    assert task.state == "pending"
    assert task.next_attempt_at is not None


async def test_call_status_for_unknown_sid_is_a_safe_noop(session):
    result = await dialer.handle_call_status_callback(
        session, provider_call_sid="CA-does-not-exist", provider_status="completed",
    )
    assert result is None


async def test_reconcile_active_call_count_recomputes_from_in_call_tasks(session, _fake_redis):
    campaign = await _make_campaign(session)
    contact1 = await _make_contact(session)
    contact2 = await _make_contact(session)
    contact3 = await _make_contact(session)
    await _make_task(session, campaign, contact1, state="in_call")
    await _make_task(session, campaign, contact2, state="in_call")
    await _make_task(session, campaign, contact3, state="pending")
    await session.commit()

    # Simulate a stale counter left over from an ungraceful shutdown - a
    # fresh reconcile must overwrite it, not trust it.
    await _fake_redis.set(dialer.ACTIVE_CALLS_REDIS_KEY, 99)

    count = await dialer.reconcile_active_call_count(session)

    assert count == 2
    assert await dialer.active_call_count() == 2


async def test_reconcile_active_call_count_is_zero_with_no_in_call_tasks(session, _fake_redis):
    count = await dialer.reconcile_active_call_count(session)
    assert count == 0
    assert await dialer.active_call_count() == 0
