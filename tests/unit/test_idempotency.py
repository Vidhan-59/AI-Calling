import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import models  # noqa: F401 - registers ORM tables on Base.metadata
from db import Base
from services.tools.registry import (
    ToolTimeoutError,
    canonical_idempotency_key,
    get_tool,
    invoke_tool,
    tool,
)

CALL_COUNT = {"n": 0}


@tool(idempotent=True)
async def _fake_write_tool(rgid: str, amount: int) -> dict:
    CALL_COUNT["n"] += 1
    return {"rgid": rgid, "amount": amount, "invocation_number": CALL_COUNT["n"]}


@tool(idempotent=False)
async def _fake_non_idempotent_tool(rgid: str) -> dict:
    CALL_COUNT["n"] += 1
    return {"rgid": rgid, "invocation_number": CALL_COUNT["n"]}


@tool(timeout_ms=50)
async def _fake_slow_tool() -> dict:
    import asyncio

    await asyncio.sleep(1)
    return {"should": "never get here"}


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.fixture(autouse=True)
def _reset_call_count():
    CALL_COUNT["n"] = 0
    yield


async def test_idempotent_tool_only_executes_once_for_same_key(session):
    args = {"rgid": "RG-1", "amount": 100}
    result1 = await invoke_tool("_fake_write_tool", call_id="call-1", session=session, args=args)
    result2 = await invoke_tool("_fake_write_tool", call_id="call-1", session=session, args=args)
    await session.commit()

    assert CALL_COUNT["n"] == 1
    assert result1 == result2
    assert result1["invocation_number"] == 1


async def test_idempotent_tool_executes_again_for_different_call_id(session):
    args = {"rgid": "RG-1", "amount": 100}
    await invoke_tool("_fake_write_tool", call_id="call-1", session=session, args=args)
    await invoke_tool("_fake_write_tool", call_id="call-2", session=session, args=args)
    await session.commit()

    assert CALL_COUNT["n"] == 2


async def test_idempotent_tool_executes_again_for_different_args(session):
    await invoke_tool(
        "_fake_write_tool", call_id="call-1", session=session, args={"rgid": "RG-1", "amount": 100}
    )
    await invoke_tool(
        "_fake_write_tool", call_id="call-1", session=session, args={"rgid": "RG-1", "amount": 200}
    )
    await session.commit()

    assert CALL_COUNT["n"] == 2


async def test_non_idempotent_tool_executes_every_time(session):
    args = {"rgid": "RG-1"}
    await invoke_tool("_fake_non_idempotent_tool", call_id="call-1", session=session, args=args)
    await invoke_tool("_fake_non_idempotent_tool", call_id="call-1", session=session, args=args)
    await session.commit()

    assert CALL_COUNT["n"] == 2


async def test_canonical_idempotency_key_is_order_independent():
    key_a = canonical_idempotency_key("call-1", "tool_x", {"a": 1, "b": 2})
    key_b = canonical_idempotency_key("call-1", "tool_x", {"b": 2, "a": 1})
    assert key_a == key_b


async def test_canonical_idempotency_key_differs_per_call():
    key_a = canonical_idempotency_key("call-1", "tool_x", {"a": 1})
    key_b = canonical_idempotency_key("call-2", "tool_x", {"a": 1})
    assert key_a != key_b


async def test_tool_timeout_raises_and_is_logged(session):
    with pytest.raises(ToolTimeoutError):
        await invoke_tool("_fake_slow_tool", call_id="call-1", session=session, args={})
    await session.commit()

    invocation = get_tool("_fake_slow_tool")
    assert invocation.timeout_ms == 50


async def test_tool_result_is_logged_to_tool_invocation_table(session):
    from sqlalchemy import select

    await invoke_tool(
        "_fake_write_tool", call_id="call-1", session=session, args={"rgid": "RG-1", "amount": 5}
    )
    await session.commit()

    rows = (await session.scalars(select(models.ToolInvocation))).all()
    assert len(rows) == 1
    assert rows[0].status == "ok"
    assert rows[0].tool_name == "_fake_write_tool"
    assert rows[0].result_json["amount"] == 5
