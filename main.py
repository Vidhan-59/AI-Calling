import asyncio
import logging
import sys

from fastapi import FastAPI

from config import settings
from simple_db import init_db as init_legacy_db
from db import init_db as init_rebuild_db
from routers import agent_voice, agents, calls, campaigns, scenarios, voice, voice_scenario

# ---------------------------------------------------------------------------
# Logging - structured, consistent across all modules
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Quiet down noisy libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

logger = logging.getLogger("main")

app = FastAPI(title="AI Calling Agent", description="Claude-powered voice AI calling agent")

init_legacy_db()
logger.info("Legacy DB initialized (data/legacy_app.db - agents/calls/messages)")

# Generic free-form agent (any prompt, any use case)
app.include_router(agents.router)
app.include_router(calls.router)
app.include_router(voice.router)

# ShiftEaze scripted scenario calling (5 cases x English/Hindi/Gujarati)
app.include_router(scenarios.router)
app.include_router(voice_scenario.router)

# Rebuild: config-driven flow-graph agent (contacts/campaigns/dialer + the
# live call entrypoint that drives services/agents/conversation_agent.py)
app.include_router(campaigns.router)
app.include_router(agent_voice.router)

logger.info("All routers registered")

_dialer_task: asyncio.Task | None = None


async def _dialer_loop() -> None:
    """Polls for eligible CallTask rows and originates them, forever, every
    `settings.dialer_poll_interval_seconds`. Only runs at all if
    dialer_enabled=True (default False - see config.py) - this loop places
    real calls, so it must be an explicit opt-in per environment, not
    something that starts just because the server booted."""
    from public_url import get_public_url
    from db import session_scope
    from services.dispatch.dialer import run_dispatch_cycle

    logger.warning("Background dialer loop starting - this WILL place real outbound calls.")
    while True:
        try:
            public_url = get_public_url()
            if not public_url:
                logger.warning("dialer loop: NGROK_URL is not configured, skipping this cycle")
            else:
                async with session_scope() as session:
                    result = await run_dispatch_cycle(
                        session,
                        answer_webhook_url_for=lambda tid: f"{public_url}/agent-voice/twiml?task_id={tid}",
                        status_callback_url_for=lambda tid: f"{public_url}/dialer/call-status",
                    )
                if result.get("originated") or result.get("skipped"):
                    logger.info("dialer cycle: %s", result)
        except Exception:  # noqa: BLE001 - one bad cycle must not kill the loop
            logger.exception("dialer cycle failed")
        await asyncio.sleep(settings.dialer_poll_interval_seconds)


@app.on_event("startup")
async def _start_dialer_loop() -> None:
    global _dialer_task
    if settings.dialer_enabled:
        _dialer_task = asyncio.create_task(_dialer_loop())
    else:
        logger.info("Dialer loop not started (dialer_enabled=False) - use POST /dialer/run-once to test manually.")


@app.on_event("shutdown")
async def _stop_dialer_loop() -> None:
    if _dialer_task is not None:
        _dialer_task.cancel()


@app.on_event("startup")
async def _init_rebuild_db() -> None:
    """Creates models.py's SQLAlchemy tables (Agent/AgentVersion/Prompt/
    FlowGraph/Contact/Campaign/CallTask/Call/...) and syncs configs/agents/*.yaml
    into them via agent_registry.sync_configs_to_db().

    Previously nothing called this - only init_legacy_db() (above) ran, so
    every table in models.py was created only inside test fixtures, never by
    running `uvicorn main:app`. See AUDIT.md, "One correctness note worth a
    second look before Phase 3". This alone does not make the rebuild's
    conversation_agent/flow-graph path reachable from a live call - it only
    means the tables it needs actually exist and stay in sync with the YAML
    configs, which the dispatcher/router work in a later phase will need.
    """
    await init_rebuild_db()
    logger.info("Rebuild DB initialized (models.py tables + agent config sync)")


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "AI Voice Assistant is running!"}
