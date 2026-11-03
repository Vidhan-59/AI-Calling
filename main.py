import asyncio
import logging
import sys

from fastapi import FastAPI

# Call logs contain Hindi/Gujarati text. Under run-all.ps1, stdout is a file
# opened in the Windows default encoding (cp1252), which can't hold Devanagari
# or Gujarati: every such log line failed with UnicodeEncodeError and was
# silently dropped - live calls looked like they stalled after the opening
# when they had actually completed.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config import settings
from simple_db import init_db as init_legacy_db
from db import init_db as init_rebuild_db
from routers import agent_voice, agents, analytics, calls, campaigns, scenarios, voice, voice_scenario, studio, voices, webrtc

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

# Fail loud at boot, not quietly at request time (services/auth.py still logs
# a warning and allows requests through when ADMIN_API_KEY is unset - but only
# in development, since this check stops the server from starting anywhere
# else). "Forgot to set it" must not be able to silently ship an
# unauthenticated routers/campaigns.py + routers/analytics.py to production.
if settings.environment != "development" and not settings.admin_api_key:
    raise RuntimeError(
        f"ADMIN_API_KEY must be set when ENVIRONMENT={settings.environment!r} - "
        "routers/campaigns.py (contacts/campaigns/kill-switch/dialer admin) and "
        "routers/analytics.py (call transcripts/outcomes) would otherwise run "
        "with no authentication. Set ADMIN_API_KEY in .env, or set "
        "ENVIRONMENT=development if this really is a dev/test deployment."
    )

app = FastAPI(title="AI Calling Agent", description="Claude-powered voice AI calling agent")

init_legacy_db()
logger.info("Legacy DB initialized (data/legacy_app.db - agents/calls/messages)")
from services import scenario_store as _scenario_store  # noqa: E402

_swept = _scenario_store.sweep_stale()
if _swept:
    logger.info("Closed %d stale scenario call row(s) (never connected / interrupted by a restart)", _swept)

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
app.include_router(analytics.router)

# Voice Studio: create/clone OmniVoice voices and assign them to scenarios/agents
app.include_router(voices.router)

# AI Agent Studio: user-configurable voice agents, training material, audio diagnostics
app.include_router(studio.router)

# Browser calls over WebRTC - same pipelines as phone calls, different transport
app.include_router(webrtc.router)

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


@app.on_event("shutdown")
async def _stop_dialer_loop() -> None:
    if _dialer_task is not None:
        _dialer_task.cancel()


@app.on_event("shutdown")
async def _close_local_llm_client() -> None:
    from local_ai.llm_engine import aclose_client
    from services.voice_pipeline import tts_cache

    await aclose_client()
    await tts_cache.aclose()
    from services.voice_pipeline import webrtc_pipeline

    await webrtc_pipeline.close_all()
    if not settings.use_sarvam_stt and settings.local_stt_isolation == "process":
        from services.voice_pipeline.stt_worker import worker

        await asyncio.to_thread(worker.close)


async def _warm_up_voice_stack() -> None:
    """Load Qwen into VRAM and fill the TTS cache before the first call,
    so the first caller of the day doesn't pay a ~10s model load or TTS
    synthesis as dead air (the OmniVoice service loads its own model at its
    own startup). Best-effort: failures are logged, not fatal."""
    from local_ai import config as local_ai_config

    if local_ai_config.LLM_BACKEND == "local":
        try:
            from local_ai.llm_engine import prime_prompt

            await prime_prompt(
                "Reply with the JSON.",
                {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
            )
            logger.info("Local LLM warm (%s resident in GPU memory)", local_ai_config.OLLAMA_MODEL)
        except Exception as e:  # noqa: BLE001
            logger.warning("Local LLM warm-up failed (is `ollama serve` running?): %s", e)

    if settings.scenario_voice_transport == "media" and not settings.use_sarvam_stt:
        from services.voice_pipeline.local_stt import preload

        try:
            langs = [l.strip() for l in settings.local_stt_preload.split(",") if l.strip()]
            specs = [(settings.local_stt_model_for(l), settings.local_stt_compute_type) for l in langs]
            await asyncio.to_thread(preload, specs, settings.local_stt_isolation)
            logger.info("Local STT warm for %s (%s): %s", langs, settings.local_stt_isolation, [s for s, _ in specs])
        except Exception as e:  # noqa: BLE001
            logger.warning("Local STT preload failed - calls will load it on first use: %s", e)

    if settings.scenario_voice_transport == "media":
        from services.tts import client as tts_client
        from services.voice_pipeline.scenario_pipeline import prewarm_static_lines

        try:
            await tts_client.health()
            await prewarm_static_lines()
        except Exception as e:  # noqa: BLE001
            logger.warning("TTS prewarm failed: %s", e)


@app.on_event("startup")
async def _start_voice_warmup() -> None:
    asyncio.create_task(_warm_up_voice_stack())


# Registration order still matters for this one: _init_rebuild_db must run
# before _reconcile_and_start_dialer, since it creates the tables
# reconcile_active_call_count reads. (The reconcile-then-start pair below used
# to be two separate startup hooks, which meant a future edit could silently
# reorder them and reintroduce the concurrency-counter leak this closes -
# merged into one function so that specific pair can no longer be separated.)
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

    sync_configs_to_db() (inside init_rebuild_db) already calls
    agent_registry.get_agent_config()/get_flow_graph() for every
    configs/agents/*.yaml, which raise ValueError on a bad config - so a
    broken agent config fails the server at boot with a clear traceback
    instead of surfacing on the first live call to it. Not caught here on
    purpose: a broken agent config should stop the server from starting,
    the same way a broken migration would.
    """
    await init_rebuild_db()
    logger.info("Rebuild DB initialized (models.py tables + agent config sync)")


@app.on_event("startup")
async def _reconcile_and_start_dialer() -> None:
    """Recomputes the active-call concurrency counter from the database
    instead of trusting whatever Redis held across a restart - closes the
    counter leak documented in PHASE2_ARCHITECTURE_PROPOSAL.md's "known
    limitations" (an ungraceful shutdown skips the decrement in
    mark_call_ended, permanently occupying a concurrency slot until this
    runs) - then starts the dialer loop, in that order, in the same
    function, so no future edit can silently separate them and reintroduce
    the leak (they used to be two independent @app.on_event hooks).

    Redis is optional for local dev (Track A/B calling and most of the API
    don't touch it at all) - a missing/unreachable Redis must not prevent
    the whole server from booting just to serve those. Reconciliation is
    only meaningful once Redis and the dialer are actually in play, so a
    connection failure here is logged and skipped, not fatal - the dialer
    loop still starts (if enabled) on an unreconciled counter in that case,
    same behavior as before this change."""
    from db import session_scope
    from services.dispatch.dialer import reconcile_active_call_count

    try:
        async with session_scope() as session:
            count = await reconcile_active_call_count(session)
        logger.info("Reconciled active_call_count from DB: %d in_call task(s)", count)
    except Exception as e:  # noqa: BLE001 - Redis being down must not block server startup
        logger.warning(
            "Could not reconcile active_call_count (Redis unreachable?) - skipping. "
            "This only matters if the dialer is in use. Error: %s", e,
        )

    global _dialer_task
    if settings.dialer_enabled:
        _dialer_task = asyncio.create_task(_dialer_loop())
    else:
        logger.info("Dialer loop not started (dialer_enabled=False) - use POST /dialer/run-once to test manually.")


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "AI Voice Assistant is running!"}
