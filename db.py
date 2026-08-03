"""Async SQLAlchemy engine/session + Redis client.

Replaces the old raw-sqlite3 `get_cursor()` helper. Swapping SQLite for
Postgres in production is exactly one env var (DATABASE_URL) - no code here
is database-specific.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

import redis.asyncio as redis_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config import settings


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_redis: redis_asyncio.Redis | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url = settings.resolved_database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_async_engine(url, echo=False, connect_args=connect_args)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Use as `async with session_scope() as session:` for a single
    transactional unit of work outside of a FastAPI request (workers,
    scripts, tests)."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: `session: AsyncSession = Depends(get_session)`."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create tables if they don't exist yet.

    Convenience for local dev / tests so `docker run postgres` + `uvicorn
    main:app` works with zero extra steps. Real deployments should run
    `alembic upgrade head` instead and treat this as a no-op safety net
    (CREATE TABLE IF NOT EXISTS semantics via SQLAlchemy metadata).
    """
    import models  # noqa: F401  (registers all ORM classes on Base.metadata)

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from services import agent_registry

    await agent_registry.sync_configs_to_db()


def get_redis() -> redis_asyncio.Redis:
    global _redis
    if _redis is None:
        _redis = redis_asyncio.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def close_connections() -> None:
    global _engine, _redis
    if _engine is not None:
        await _engine.dispose()
        _engine = None
    if _redis is not None:
        await _redis.aclose()
        _redis = None
