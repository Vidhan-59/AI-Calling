"""File-in/file-out test runner for the local pipeline.

Two response modes:
  --scenario shift_confirm   route through the real ShiftEaze scenario logic
                             (services/scenario_engine.py), using the local
                             LLM for classification instead of Claude
  (no --scenario)            plain chat: Ollama replies directly, for testing
                             STT/LLM/TTS in isolation from scenario JSON

Usage:
    python -m local_ai.cli --audio caller.wav --out reply.wav
    python -m local_ai.cli --audio caller.wav --out reply.wav --scenario shift_confirm --language hi
    python -m local_ai.cli --check    # verify Ollama + the model are ready
"""

import argparse
import asyncio
import json
import sys

from local_ai import config
from local_ai.pipeline import LocalVoicePipeline, OllamaUnavailableError, VoiceNotAvailableError


def _plain_chat_respond(text: str, language: str) -> str:
    from local_ai import llm_engine

    system = (
        "You are a helpful voice assistant. Keep replies to one or two short "
        "sentences - this is spoken aloud, not read as text."
    )
    return asyncio.run(
        llm_engine.generate_reply(system, [{"role": "user", "content": text}])
    )


def _scenario_respond(scenario_id: str, facts: dict, attempt: int):
    """Returns a respond(text, language) closure backed by the real scenario
    engine, forced onto the local LLM backend for this one call regardless of
    the SCENARIO_LLM_BACKEND env var - the CLI is explicitly for testing the
    local backend, so it should not silently fall back to Claude."""
    import os

    os.environ["SCENARIO_LLM_BACKEND"] = "local"
    from services import scenario_engine

    session_holder: dict = {}

    def respond(text: str, language: str) -> str:
        if "session" not in session_holder:
            session_holder["session"] = scenario_engine.ScenarioSession(
                scenario_id=scenario_id, language=language, facts=facts, attempt=attempt
            )
        session = session_holder["session"]
        result = asyncio.run(session.handle_utterance(text))
        return result.reply

    return respond


def cmd_check() -> int:
    from local_ai import llm_engine

    health = asyncio.run(llm_engine.health_check())
    print(json.dumps(health, indent=2))
    if not health["ollama_running"]:
        print(f"\nOllama is not running. Start it with: ollama serve", file=sys.stderr)
        return 1
    if not health["model_pulled"]:
        print(f"\nModel not pulled. Run: ollama pull {config.OLLAMA_MODEL}", file=sys.stderr)
        return 1
    print("\nOllama backend ready.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", help="Input WAV file (caller's speech)")
    parser.add_argument("--out", help="Output WAV file (agent's reply)")
    parser.add_argument("--language", default=None, help="Language hint: en, hi, gu (default: auto-detect)")
    parser.add_argument("--scenario", default=None, help="ShiftEaze scenario id, e.g. shift_confirm")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--facts", default="{}", help='JSON facts dict, e.g. \'{"employee_name":"Rajesh","terminal":"CT3"}\''
    )
    parser.add_argument("--check", action="store_true", help="Verify Ollama + model are ready, then exit")
    args = parser.parse_args()

    if args.check:
        return cmd_check()

    if not args.audio or not args.out:
        parser.error("--audio and --out are required unless --check is given")

    if args.scenario:
        respond = _scenario_respond(args.scenario, json.loads(args.facts), args.attempt)
    else:
        respond = _plain_chat_respond

    pipeline = LocalVoicePipeline()
    try:
        result = pipeline.run_file(args.audio, args.out, respond, language_hint=args.language)
    except OllamaUnavailableError as e:
        print(f"error: {e}", file=sys.stderr)
        print("run `python -m local_ai.cli --check` to diagnose", file=sys.stderr)
        return 1
    except VoiceNotAvailableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(
        {
            "transcript": result.transcript,
            "detected_language": result.detected_language,
            "language_probability": round(result.language_probability, 3),
            "reply_text": result.reply_text,
            "output_audio_path": result.output_audio_path,
            "timings_ms": result.timings_ms,
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
