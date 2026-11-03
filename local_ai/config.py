"""Config for the local pipeline - every model is swappable from here, or by
setting the matching env var. Nothing in stt_engine.py / llm_engine.py /
tts_engine.py / pipeline.py hardcodes a model name.

Defaults are picked for this project's actual hardware (see LOCAL_AI_README.md
for the measurement this was based on): 8-core/16-thread CPU, ~15GB RAM, no
CUDA GPU. Every default favors speed on that hardware over maximum accuracy;
bump WHISPER_MODEL_SIZE / OLLAMA_MODEL to a larger variant if you have more
RAM or a GPU and want to trade latency for quality.
"""

import os

from dotenv import load_dotenv

# Nothing else in this codebase loads .env into the process environment -
# config.py's pydantic Settings reads .env for its own model fields only, it
# never touches os.environ. Every constant below is a plain os.getenv(), so
# without this call .env edits to LOCAL_LLM_MODEL / SCENARIO_LLM_BACKEND /
# etc. would silently do nothing. load_dotenv() never overrides a variable
# that's already set in the real process environment.
load_dotenv()

# --- STT (faster-whisper) --------------------------------------------------
# Model sizes, smallest to largest: tiny, base, small, medium, large-v3.
# "small" is the speed/accuracy sweet spot on CPU for the code-mixed
# Hindi/Gujarati/English speech this project deals with. Gujarati accuracy in
# particular improves noticeably at "medium" - worth trying if small mislabels
# too much, at roughly 2-3x the inference time.
WHISPER_MODEL_SIZE = os.getenv("LOCAL_WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("LOCAL_WHISPER_COMPUTE_TYPE", "int8")

# --- LLM (Ollama) ------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# UPDATE (2026-09-30, GPU machine - RTX 5060 Laptop, 8GB VRAM): the numbers
# below this line were measured on a CPU-only box and no longer describe this
# machine. Ollama 100%-GPU-offloads qwen2.5:7b-instruct here; re-measured
# through the REAL run_turn() code path (flow graph + tools + guardrails,
# not just a raw chat ping): 0.9-1.7s/turn, correct tool-calling behavior.
# Also benchmarked qwen3:8b as a candidate upgrade the same way: 3.2-3.8s/turn
# steady-state (2-4x slower - likely its default "thinking mode" generating
# hidden reasoning tokens before the answer) AND it returned an EMPTY reply
# on every non-tool-calling turn in this app's Ollama tool-calling contract -
# a real compatibility defect, not just a speed tradeoff. Verdict: stayed on
# qwen2.5:7b-instruct as the default. Revisit qwen3 only after disabling its
# thinking mode and re-verifying output is non-empty - don't just flip the
# model name.
#
# Original CPU-only measurement (kept for history, no longer this machine's
# reality): classifying the same 3 Hindi test utterances via scenario_engine
# on a Ryzen 7 5800HS, 8c/16t, CPU-only, no GPU:
#   phi4-mini (MIT, 2.5GB):           23-33s/turn, 1/3 correct (missed the
#     negation trap and the supervisor-escalation intent)
#   qwen2.5:7b-instruct (Apache-2.0, 4.7GB): 33-60s/turn, 3/3 correct
OLLAMA_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("LOCAL_LLM_TIMEOUT", "150"))
# Ollama unloads a model from RAM after this much idle time, then pays a
# one-time reload cost (~60-90s for a 4.7GB model on this hardware, on top of
# the usual per-turn inference time) on the next request - this is exactly
# what turned one call into an httpx.ReadTimeout past the 90s client timeout
# above. "30m" keeps the model resident through a normal test/demo session
# instead of unloading after Ollama's 5-minute default. Set to "-1" to never
# unload for the length of a demo.
# "-1m" = never unload (verified: `ollama ps` shows UNTIL "Forever"). The
# local model is now on the live-call critical path, and a cold reload is
# ~9.6s measured on this machine - the first call after 30 idle minutes paid
# it as dead air. On a machine dedicated to calling, keep it resident.
OLLAMA_KEEP_ALIVE = os.getenv("LOCAL_LLM_KEEP_ALIVE", "-1m")

# --- TTS (Piper) --------------------------------------------------------------
# See LOCAL_AI_README.md "Known limitation" section before relying on this for
# Hindi, and do not rely on it at all for Gujarati - no usable open-source
# Gujarati TTS voice exists at the time of writing.
PIPER_VOICES_DIR = os.getenv(
    "LOCAL_PIPER_VOICES_DIR",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "voices"),
)
PIPER_VOICE_EN = os.getenv("LOCAL_PIPER_VOICE_EN", "en_US-lessac-medium")
PIPER_VOICE_HI = os.getenv("LOCAL_PIPER_VOICE_HI", "hi_IN-pratham-medium")
# No default Gujarati voice - intentional, see caveat above.
PIPER_VOICE_GU = os.getenv("LOCAL_PIPER_VOICE_GU", "")

# --- Wiring into services/scenario_engine.py ---------------------------------
# "claude" (default, unchanged behavior) or "local" (routes classification
# through llm_engine.py / Ollama instead of Claude). See scenario_engine.py's
# classify_utterance() for the branch point. Kept as its own name for
# backward compat with existing docs/deployments that already set it - LLM_BACKEND
# below is the newer, broader switch and falls back to this value.
SCENARIO_LLM_BACKEND = os.getenv("SCENARIO_LLM_BACKEND", "claude")

# --- Global LLM backend switch ------------------------------------------------
# Unlike SCENARIO_LLM_BACKEND above (Track B's branch classifier only),
# LLM_BACKEND governs every LLM call in the app: Track A's free-form replies
# + sentiment (routers/voice.py, routers/calls.py's /chat/test), and the
# rebuild's flow-graph conversation agent (services/agents/conversation_agent.py).
# Falls back to SCENARIO_LLM_BACKEND so setting that one alone (as existing
# docs describe) still flips everything, not just Track B - set LLM_BACKEND
# explicitly only if you want the two to diverge (e.g. Track B on Claude,
# everything else local).
LLM_BACKEND = os.getenv("LLM_BACKEND", SCENARIO_LLM_BACKEND)
