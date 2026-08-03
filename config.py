"""Central settings object. Every other module imports `settings` from here
instead of reading os.environ directly, so the whole app has exactly one
place that knows how to interpret configuration."""

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Settings below reads .env for its own pydantic fields only - that does NOT
# populate os.environ. Anything reading a plain os.getenv() (local_ai/config.py's
# SCENARIO_LLM_BACKEND, LOCAL_LLM_MODEL, etc.) needs .env loaded into the real
# process environment too. config.py is the first thing main.py imports, so
# this is the one place that guarantees it happens before anything else runs -
# a module-level import elsewhere doing this was fragile (see git history:
# removing one incidental import silently broke SCENARIO_LLM_BACKEND=local).
load_dotenv()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Anthropic ---
    anthropic_api_key: str | None = None
    conversation_model: str = "claude-haiku-4-5"
    supervisor_model: str = "claude-sonnet-5"
    sentiment_model: str = "claude-haiku-4-5"
    default_claude_model: str = "claude-opus-5"

    # --- Database ---
    # Empty => aiosqlite file under ./data/app.db. This is the one line you
    # change to point at Postgres in any environment beyond a laptop demo;
    # nothing else in the codebase is SQLite- or Postgres-specific.
    database_url: str = ""

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = os.path.join(DATA_DIR, "app.db").replace("\\", "/")
        return f"sqlite+aiosqlite:///{db_path}"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Telephony ---
    telephony_provider: Literal["mock", "twilio", "exotel", "plivo"] = "mock"
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_phone_number: str | None = None
    exotel_api_key: str | None = None
    exotel_api_token: str | None = None
    exotel_sid: str | None = None
    exotel_virtual_number: str | None = None
    plivo_auth_id: str | None = None
    plivo_auth_token: str | None = None
    plivo_phone_number: str | None = None

    # --- STT ---
    stt_provider: Literal["mock", "sarvam", "deepgram"] = "mock"
    sarvam_api_key: str | None = None
    deepgram_api_key: str | None = None

    # --- TTS ---
    tts_provider: Literal["mock", "sarvam", "elevenlabs", "azure"] = "mock"
    elevenlabs_api_key: str | None = None
    azure_speech_key: str | None = None
    azure_speech_region: str | None = None

    # --- Odoo ---
    odoo_webhook_url: str | None = None
    odoo_webhook_hmac_secret: str = "change_me_in_production"

    # --- SMS ---
    sms_provider: Literal["mock", "twilio"] = "mock"

    # --- Server ---
    ngrok_url: str | None = None
    port: int = 8000
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # --- Safety ---
    global_kill_switch_redis_key: str = "ai_calling:kill_switch"
    max_concurrent_calls: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# --- Backwards-compatible module-level names -------------------------------
# The pre-existing codebase (services/claude_client.py, routers/calls.py, ...)
# imports these directly. Keep them as thin aliases so nothing downstream
# breaks while it's migrated to `from config import settings` incrementally.
ANTHROPIC_API_KEY = settings.anthropic_api_key
DEFAULT_CLAUDE_MODEL = settings.default_claude_model
SENTIMENT_MODEL = settings.sentiment_model
TWILIO_ACCOUNT_SID = settings.twilio_account_sid
TWILIO_AUTH_TOKEN = settings.twilio_auth_token
TWILIO_PHONE_NUMBER = settings.twilio_phone_number
NGROK_URL = settings.ngrok_url
PORT = settings.port
DB_PATH = os.path.join(DATA_DIR, "app.db")
