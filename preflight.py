"""Pre-flight checks run by start.ps1 before booting the server.

Stdlib only, on purpose: this runs *before* dependencies are guaranteed to be
installed, so it must not import anything from requirements.txt.

Emits one `LEVEL|message` line per check on stdout for start.ps1 to colourise:

    OK|venv found
    WARN|Twilio not configured
    FAIL|ANTHROPIC_API_KEY is not set
    INSTALL|fastapi,uvicorn        <- start.ps1 pip-installs these, then re-runs

Exit code is 1 if any FAIL was emitted, 0 otherwise.
"""

import importlib.util
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
_failed = False
_lines: list[str] = []


def ok(msg: str) -> None:
    _lines.append(f"OK|{msg}")


def warn(msg: str) -> None:
    _lines.append(f"WARN|{msg}")


def fail(msg: str) -> None:
    global _failed
    _failed = True
    _lines.append(f"FAIL|{msg}")


def note(msg: str) -> None:
    _lines.append(f"NOTE|{msg}")


# ---------------------------------------------------------------------------
# 1. Dependencies this app actually imports at boot
# ---------------------------------------------------------------------------
# requirements.txt is shared with a larger in-progress rebuild whose heavier
# deps (pipecat, asyncpg, redis, ...) are not needed to run this server, so we
# check only what main.py's import graph really touches.
REQUIRED = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "anthropic": "anthropic",
    "twilio": "twilio",
    "pydantic_settings": "pydantic-settings",
    "dotenv": "python-dotenv",
    "multipart": "python-multipart",
}


def check_dependencies() -> None:
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        _lines.append("INSTALL|" + ",".join(missing))
    else:
        ok(f"all {len(REQUIRED)} required packages present")


# ---------------------------------------------------------------------------
# 2. .env configuration
# ---------------------------------------------------------------------------


def _read_env() -> dict[str, str]:
    path = os.path.join(BASE_DIR, ".env")
    values: dict[str, str] = {}
    if not os.path.isfile(path):
        return values
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _placeholder(value: str) -> bool:
    """True if the value is still an .env.example stub rather than a real one."""
    if not value:
        return True
    lowered = value.lower()
    return any(
        stub in lowered
        for stub in ("your_", "your-", "xxxx", "change_me", "here")
    )


def _llm_backend(env: dict[str, str]) -> str:
    # Mirrors local_ai/config.py's LLM_BACKEND resolution (LLM_BACKEND,
    # falling back to SCENARIO_LLM_BACKEND, default "claude") without
    # importing that module - preflight must stay stdlib-only. Kept in sync
    # by hand; if you change the fallback chain there, change it here too.
    return env.get("LLM_BACKEND") or env.get("SCENARIO_LLM_BACKEND") or "claude"


def check_ollama(env: dict[str, str]) -> None:
    """Only called when LLM_BACKEND=local - checks Ollama is reachable and
    the configured model is pulled, using stdlib urllib (no httpx
    dependency needed just for this check, same reasoning as check_env's
    ANTHROPIC_API_KEY check running before dependencies are confirmed
    installed)."""
    import urllib.error
    import urllib.request

    base_url = (env.get("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    model = env.get("LOCAL_LLM_MODEL") or "qwen2.5:7b-instruct"

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        warn(f"LLM_BACKEND=local but Ollama is unreachable at {base_url} ({e}) - run `ollama serve`")
        return
    except Exception as e:  # noqa: BLE001 - a malformed response must not crash preflight
        warn(f"LLM_BACKEND=local but Ollama's response at {base_url} could not be parsed: {e}")
        return

    names = [m.get("name", "") for m in data.get("models", [])]
    pulled = any(n == model or n.startswith(model + ":") for n in names)
    if pulled:
        ok(f"Ollama reachable at {base_url}, model '{model}' pulled")
    else:
        warn(f"Ollama reachable at {base_url}, but '{model}' is not pulled - run: ollama pull {model}")


def check_env() -> None:
    if not os.path.isfile(os.path.join(BASE_DIR, ".env")):
        if os.path.isfile(os.path.join(BASE_DIR, ".env.example")):
            fail(".env does not exist - copy it from .env.example and set ANTHROPIC_API_KEY")
        else:
            fail("No .env and no .env.example to copy from")
        return

    env = _read_env()
    backend = _llm_backend(env)

    if backend == "local":
        note("LLM_BACKEND=local - Claude/ANTHROPIC_API_KEY is not required")
        check_ollama(env)
    else:
        key = env.get("ANTHROPIC_API_KEY", "")
        if _placeholder(key):
            fail("ANTHROPIC_API_KEY is not set in .env "
                 "(get one at https://console.anthropic.com/settings/keys, "
                 "or set LLM_BACKEND=local to use Ollama instead)")
        else:
            ok(f"ANTHROPIC_API_KEY set (...{key[-6:]})")

    twilio_ready = all(
        not _placeholder(env.get(k, ""))
        for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER")
    )
    tunnel = env.get("NGROK_URL", "")
    tunnel_ready = not _placeholder(tunnel)

    if twilio_ready:
        ok("Twilio credentials set")
    else:
        warn("Twilio not configured - text testing works, real calls will fail")

    if tunnel_ready:
        ok(f"NGROK_URL = {tunnel}")
    else:
        warn("NGROK_URL is not a real public URL - real calls will fail")

    if not (twilio_ready and tunnel_ready):
        note("That is fine for now: /scenarios/test-turn and /scenarios/test-batch")
        note("exercise the full conversation logic with no phone or tunnel.")


# ---------------------------------------------------------------------------
# 3. Scenario knowledge bases
# ---------------------------------------------------------------------------


def check_scenarios() -> None:
    directory = os.path.join(BASE_DIR, "scenarios")
    if not os.path.isdir(directory):
        fail("scenarios/ folder not found - the five ShiftEaze scripts are missing")
        return

    loaded = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
            langs = "/".join(sorted(data["languages"]))
            loaded.append(f"{data['id']}({langs})")
        except Exception as e:
            fail(f"{filename} is invalid: {e}")

    if loaded:
        ok(f"{len(loaded)} scenarios: " + "  ".join(loaded))
    elif not _failed:
        fail("scenarios/ contains no .json files")


def main() -> int:
    check_dependencies()
    check_env()
    check_scenarios()
    print("\n".join(_lines))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
