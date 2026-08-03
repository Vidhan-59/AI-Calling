"""Fully local, offline-capable voice AI pipeline: STT (faster-whisper) ->
response generation (Ollama) -> TTS (Piper).

Entirely additive to the rest of this repo - nothing here is imported by
default, and nothing outside local_ai/ is required to import this package.
The one opt-in touch point is services/scenario_engine.py, which can route
classification through local_ai.llm_engine instead of Claude when
SCENARIO_LLM_BACKEND=local (see local_ai/config.py). Default behavior is
unchanged.

See LOCAL_AI_README.md for setup, hardware requirements, and how to swap
any stage's model via config.
"""
