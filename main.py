import logging
import sys

from fastapi import FastAPI

from config import settings
from simple_db import init_db
from routers import agents, calls, scenarios, voice, voice_scenario

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

init_db()
logger.info("Database initialized")

# Generic free-form agent (any prompt, any use case)
app.include_router(agents.router)
app.include_router(calls.router)
app.include_router(voice.router)

# ShiftEaze scripted scenario calling (5 cases x English/Hindi/Gujarati)
app.include_router(scenarios.router)
app.include_router(voice_scenario.router)

logger.info("All routers registered")


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "AI Voice Assistant is running!"}
