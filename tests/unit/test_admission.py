"""services/dispatch/admission.py - the pre-dial compliance gate.

Master-prompt requirement: "unit tests for the compliance gate and dialer
logic at minimum." This file covers the gate; tests/unit/test_dialer.py
covers the dialer that calls it.
"""

from datetime import datetime, timezone

import fakeredis
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import models
from db import Base
from services.dispatch import admission


@pytest.fixture(autouse=True)
def _fake_redis(monkeypatch):
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr("services.dispatch.admission.get_redis", lambda: redis)
    return redis


def _contact(**overrides) -> models.Contact:
    defaults = dict(
        rgid="RG-1", name="Amit Patel", phone_e164="+919999999999",
        dnc=False, consent_status="granted",
    )
    defaults.update(overrides)
    return models.Contact(**defaults)


def _campaign(**overrides) -> models.Campaign:
    defaults = dict(
        agent_id="shift_confirm", agent_version=1, name="c1", scenario="shift_confirm",
        calling_window_start="07:00", calling_window_end="21:00", calling_window_tz="Asia/Kolkata",
    )
    defaults.update(overrides)
    return models.Campaign(**defaults)


async def test_allows_consented_contact_within_calling_window():
    # 10:00 UTC = 15:30 IST, inside 07:00-21:00 Asia/Kolkata.
    now = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
    result = await admission.can_dial(_contact(), _campaign(), now=now)
    assert result.allowed is True
    assert result.reason_code is None


async def test_refuses_dnc_contact():
    result = await admission.can_dial(_contact(dnc=True))
    assert result.allowed is False
    assert result.reason_code == "dnc"


@pytest.mark.parametrize("consent_status", ["unknown", "revoked"])
async def test_refuses_contact_without_granted_consent(consent_status):
    result = await admission.can_dial(_contact(consent_status=consent_status))
    assert result.allowed is False
    assert result.reason_code == "consent"


async def test_refuses_outside_calling_window():
    # 20:30 UTC on the 19th = 02:00 IST on the 20th - well outside 07:00-21:00.
    now = datetime(2026, 8, 19, 20, 30, tzinfo=timezone.utc)
    result = await admission.can_dial(_contact(), _campaign(), now=now)
    assert result.allowed is False
    assert result.reason_code == "calling_window"


async def test_wraparound_calling_window_is_evaluated_correctly():
    campaign = _campaign(calling_window_start="22:00", calling_window_end="06:00", calling_window_tz="UTC")
    within = datetime(2026, 8, 20, 23, 0, tzinfo=timezone.utc)
    outside = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    assert (await admission.can_dial(_contact(), campaign, now=within)).allowed is True
    assert (await admission.can_dial(_contact(), campaign, now=outside)).allowed is False


async def test_no_campaign_skips_the_calling_window_check():
    # Would fail the window check if a campaign were passed - omitting the
    # campaign (the ad-hoc/non-campaign call path) must not enforce a window
    # that was never configured for this call.
    now = datetime(2026, 8, 19, 20, 30, tzinfo=timezone.utc)
    result = await admission.can_dial(_contact(), None, now=now)
    assert result.allowed is True


async def test_kill_switch_blocks_everything_regardless_of_consent(_fake_redis):
    await _fake_redis.set("ai_calling:kill_switch", "1")
    result = await admission.can_dial(_contact(), _campaign())
    assert result.allowed is False
    assert result.reason_code == "kill_switch"


async def test_kill_switch_takes_priority_over_other_reasons(_fake_redis):
    """If the kill switch is engaged, that's the reason reported - not "DNC"
    or anything else - because the operational response (someone needs to
    find out why it's engaged) is different from a per-contact skip."""
    await _fake_redis.set("ai_calling:kill_switch", "1")
    result = await admission.can_dial(_contact(dnc=True))
    assert result.reason_code == "kill_switch"


async def test_record_opt_out_sets_dnc_and_revokes_consent_together():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        contact = _contact()
        session.add(contact)
        await session.flush()

        await admission.record_opt_out(session, contact, source="call")
        await session.commit()

        assert contact.dnc is True
        assert contact.consent_status == "revoked"

        from sqlalchemy import select

        rows = (
            await session.scalars(
                select(models.ConsentRecord).where(models.ConsentRecord.contact_id == contact.id)
            )
        ).all()
        assert len(rows) == 1
        assert rows[0].event == "revoked"
        assert rows[0].source == "call"

    await engine.dispose()


async def test_opted_out_contact_is_refused_by_can_dial():
    """The point of record_opt_out: the very next admission check for this
    contact must refuse it, with no separate propagation step."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async with factory() as session:
        contact = _contact()
        session.add(contact)
        await session.flush()
        await admission.record_opt_out(session, contact, source="call")
        await session.commit()

        result = await admission.can_dial(contact)
        assert result.allowed is False
        assert result.reason_code in ("dnc", "consent")

    await engine.dispose()
