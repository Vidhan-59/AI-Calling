"""Central settings object. Every other module imports `settings` from here
instead of reading os.environ directly, so the whole app has exactly one
place that knows how to interpret configuration."""

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import field_validator
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
    sarvam_api_key: str | None = None

    # --- Voice pipeline (services/voice_pipeline/) ---
    # Scripts use feminine first-person Hindi/Gujarati verb forms ("bol rahi
    # hoon", "samajh gayi"), so the voice assigned in Voice Studio should be
    # female - a male voice saying "rahi hoon" sounds wrong to any Hindi
    # speaker.
    # saaras:v3 measured 0.31-0.42s and exact on Hindi test phrases; local
    # faster-whisper (small, CPU) was 2-5s and garbled - see LOCAL_AI_README.md.
    sarvam_stt_model: str = "saaras:v3"
    # STT switch. True = Sarvam cloud STT (saaras:v3, needs SARVAM_API_KEY);
    # False = open-source faster-whisper on this machine's GPU
    # (services/voice_pipeline/local_stt.py). TTS is always OmniVoice. The
    # local model was picked by tools/stt_benchmark.py on this hardware -
    # re-run it on the deployment server, where a bigger GPU may make a
    # larger model the winner.
    use_sarvam_stt: bool = True

    @property
    def stt_engine(self) -> str:
        return "sarvam" if self.use_sarvam_stt else "local"
    # Per call language - picked by tools/stt_benchmark.py on this laptop
    # (RTX 5060, 8GB shared with Qwen). Hindi: IIT Madras Whisper-Indic
    # Hindi medium + domain prompt = 3.1% CER, 94% intent, ~700ms. Gujarati:
    # Whisper-Indic Gujarati medium = 14% CER, 85% intent. Generic Whisper
    # large-v3 was both slower (1.8s) and worse on Gujarati; turbo was
    # unusable for Gujarati (84% CER). On a server with a bigger GPU, the
    # Hindi large-v2 fine-tune is the more accurate choice - re-run the
    # benchmark there. Local paths come from tools/convert_stt_models.py.
    local_stt_model_hi: str = os.path.join(os.path.expanduser("~"), ".cache", "ai-calling", "stt_models", "whisper-hindi-medium")
    local_stt_model_gu: str = os.path.join(os.path.expanduser("~"), ".cache", "ai-calling", "stt_models", "whisper-gujarati-medium")
    # Any other language (e.g. English, where stock Whisper is strong).
    local_stt_model: str = "Systran/faster-whisper-medium"
    # Languages whose model is loaded at server startup; others load on first use.
    local_stt_preload: str = "hi,gu"

    def local_stt_model_for(self, language: str) -> str:
        return {"hi": self.local_stt_model_hi, "gu": self.local_stt_model_gu}.get(language, self.local_stt_model)
    # float16, not int8_float16: int8 GPU kernels fail on Blackwell GPUs (RTX 50xx)
    # with CUBLAS_STATUS_NOT_SUPPORTED - found by tools/stt_benchmark.py.
    local_stt_compute_type: str = "float16"
    local_stt_beam_size: int = 1
    # "process": decode in a dedicated worker process (services/voice_pipeline/
    # stt_worker.py) - in-process, the server's event loop starves Whisper of
    # the GIL and decodes ran ~2x slower during live calls. "thread": in-process.
    local_stt_isolation: Literal["process", "thread"] = "process"
    # GPU residency. The 8GB laptop GPU cannot hold Qwen (4.6GB) + OmniVoice
    # (2.1GB) + even one Whisper-medium (2.0GB) at once; the overflow is paged
    # to system RAM by Windows and decodes stall for tens of seconds (seen as
    # "local STT failed: TimeoutError"). So only `local_stt_gpu_models` Whisper
    # models stay on the GPU (the call's language); the rest are parked in CPU
    # RAM and come back in ~0.3s (measured). Idle models are parked after
    # `local_stt_idle_offload_secs` so Qwen/OmniVoice get the VRAM between
    # calls. 0 disables idle parking. Set gpu_models=2 on a >=16GB GPU.
    # If a local decode fails or times out, retry that one utterance on Sarvam
    # (needs SARVAM_API_KEY) instead of the caller hearing silence. Costs only
    # on failures; set false for strictly zero cloud spend.
    local_stt_fallback_sarvam: bool = True
    local_stt_gpu_models: int = 1
    local_stt_idle_offload_secs: float = 30.0
    # "media": scripted scenario calls go through Twilio Media Streams + our
    # own STT/OmniVoice pipeline (services/voice_pipeline/scenario_pipeline.py).
    # "relay": the older Twilio ConversationRelay path with Google STT/TTS -
    # kept as a one-line rollback.
    scenario_voice_transport: Literal["media", "relay"] = "media"
    # After a "no", ALWAYS ask why - even when the model thinks the caller
    # already gave a reason (it often guessed wrong and the question was
    # silently skipped). A reason they volunteered is kept as a fallback in
    # case the answer to the question is empty. false = old skip behaviour.
    scenario_always_ask_reason: bool = True

    @field_validator("scenario_voice_transport", mode="before")
    @classmethod
    def _legacy_transport_name(cls, v):
        # Existing .env files say "sarvam" (the old name of the media path).
        return "media" if v == "sarvam" else v


    # --- Low-latency audio front end (services/voice_pipeline/audio_frontend.py) ---
    # Layer 2: neural noise suppression -> Silero VAD -> barge-in controller.
    # RNNoise runs on the CPU before VAD/STT, so street/factory noise neither
    # trips barge-in nor reaches Whisper. Needs pyrnnoise (pipecat-ai[rnnoise]);
    # without it the filter passes audio through unchanged and logs once.
    noise_suppression: bool = True
    # Speech must stay above vad_confidence (and vad_min_volume) for this long
    # before it counts as the caller talking - 150ms: long enough that a
    # cough, a door or a "hmm" from the background doesn't cut the bot off,
    # short enough that a real interruption stops it almost at once.
    vad_start_secs: float = 0.15
    vad_confidence: float = 0.7
    vad_min_volume: float = 0.6
    # Speculative STT: start decoding the caller's words this long into a
    # pause, instead of after the full end-of-turn window (VAD stop_secs).
    # If the caller turns out to be done, the transcript is (nearly) ready the
    # moment VAD confirms it; if they resume, the guess is thrown away.
    # Trade-off: a guess started on a dip between words and then discarded
    # still occupies the one GPU worker, and the real decode queues behind it
    # (seen once: transcript 811ms late instead of ~140ms). 0.2s didn't stop
    # those dips but made every turn's guess later; 0.12s measured best
    # overall (tools/latency_smoke_test.py, 11 of 12 turns under 1s).
    speculative_stt: bool = True
    speculate_after_secs: float = 0.12

    # --- Layer 1: token streaming (agent calls, services/voice_pipeline/conversation_processor.py) ---
    # The LLM reply is streamed and spoken in pieces: the first piece goes to
    # TTS after this many words, so audio starts while the LLM is still
    # writing the rest of the sentence.
    stream_llm_replies: bool = True
    stream_first_chunk_words: int = 3
    # Later pieces are merged to at least this many words (a sentence end
    # releases at half): fewer piece boundaries = fewer audible gaps.
    stream_min_chunk_words: int = 6

    # --- TTS: OmniVoice service (tts_service/, separate process + venv) ---
    tts_service_url: str = "http://127.0.0.1:8100"
    tts_service_token: str | None = None
    tts_request_timeout_seconds: float = 20.0
    # OmniVoice diffusion steps. Live (free-form agent replies) trades quality
    # for latency: measured on an RTX 5060 shared with Qwen, 16 steps ~2.4s for
    # a 5.5s line vs 5.1s at 32. Scripted lines are rendered once into the
    # cache (prefetched while the phone rings), so they get the best quality.
    tts_live_num_step: int = 16
    # A short first piece (the few words the processor releases early) gets
    # fewer steps still: it is the one the caller waits for, and it is short
    # enough that the quality difference is hard to hear. Measured on the
    # 8GB laptop GPU: 8 steps ~0.9s, 12 ~1.1s, 16 ~1.5s for a short phrase.
    tts_short_piece_num_step: int = 10
    tts_short_piece_words: int = 5
    # Free-form agent calls: if the reply hasn't started this many ms after the
    # caller finished (Qwen + OmniVoice on a shared 8GB GPU take ~1.5-2.5s to
    # render the first words), a tiny pre-rendered acknowledgement ("जी।") plays
    # at once so the caller is never left in silence. 0 = off.
    late_filler_ms: int = 700

    # --- Agent Studio phone calls (services/studio/calling.py) ---
    # Longest reply (tokens) a Studio agent speaks per turn, and the hard cap on a call's length.
    studio_reply_max_tokens: int = 220
    studio_max_call_secs: int = 600
    tts_cached_num_step: int = 32

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

    # --- Dialer ---
    # Off by default on purpose: starting a background dispatch loop that
    # dials real numbers just because the server booted is exactly how a
    # number gets blacklisted by an operator who never asked to be called.
    # An operator has to opt in explicitly per environment.
    dialer_enabled: bool = False
    dialer_poll_interval_seconds: int = 15

    # --- Admin auth ---
    # Guards routers/campaigns.py (contacts/campaigns/kill-switch/dialer
    # admin) and routers/analytics.py (call transcripts/outcomes) - see
    # services/auth.py. None by default so local dev/tests aren't blocked;
    # unset means those endpoints run with NO authentication, which
    # services/auth.py logs loudly on first use.
    admin_api_key: str | None = None


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
