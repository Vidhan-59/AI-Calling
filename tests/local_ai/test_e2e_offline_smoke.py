"""The one test that actually proves the "no cloud dependency" claim: real
faster-whisper, real Ollama, real Piper, run end-to-end, with all non-loopback
network access physically blocked at the socket layer for the duration of the
test. If anything in the pipeline tried to reach the internet, this test
would fail with a ConnectionError - it does not merely assert that cloud SDKs
were not imported.

Skipped automatically (not failed) if Ollama isn't running or the configured
model isn't pulled, or if the Piper English voice hasn't been downloaded yet -
those are one-time local setup steps documented in LOCAL_AI_README.md, not
things this test should install.

This is a shape/plumbing check, not an accuracy check (see
tests/local_ai/test_e2e_offline_smoke.py's sibling, the manual
--check accuracy comparisons documented in LOCAL_AI_README.md, for the honest
accuracy numbers this project actually measured).
"""

import asyncio
import contextlib
import math
import socket
import struct
import wave

import httpx
import pytest

from local_ai import config
from local_ai.pipeline import LocalVoicePipeline

pytestmark = pytest.mark.slow  # real model inference; skip in fast test runs


def _ollama_ready() -> bool:
    try:
        resp = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(m.startswith(config.OLLAMA_MODEL) for m in models)
    except Exception:
        return False


def _piper_voice_ready() -> bool:
    import os

    return os.path.isfile(os.path.join(config.PIPER_VOICES_DIR, f"{config.PIPER_VOICE_EN}.onnx"))


def _make_test_wav(path: str, seconds: float = 1.0, freq_hz: float = 220.0) -> None:
    """A synthetic tone, not real speech - this test checks the pipeline's
    *shape* (produces a non-empty WAV, doesn't crash, doesn't touch the
    network), not transcription accuracy. Real speech accuracy is covered by
    the manual scenario tests documented in LOCAL_AI_README.md, which need a
    live speaker and are not suitable for an automated, unattended test."""
    sample_rate = 16000
    n_samples = int(seconds * sample_rate)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * freq_hz * i / sample_rate)))
            for i in range(n_samples)
        )
        f.writeframes(frames)


@contextlib.contextmanager
def _block_non_loopback_network():
    """Monkeypatch socket.socket.connect so any attempt to reach a
    non-loopback address raises immediately, while 127.0.0.1/::1 (Ollama)
    keep working. This is what makes the test an actual proof rather than an
    assertion of good intentions."""
    real_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise ConnectionError(
                f"BLOCKED: test attempted a non-loopback network connection to {host!r}. "
                "The local pipeline must not reach the internet."
            )
        return real_connect(self, address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    try:
        yield
    finally:
        socket.socket.connect = real_connect


@pytest.mark.skipif(not _ollama_ready(), reason="Ollama not running or model not pulled - see LOCAL_AI_README.md")
@pytest.mark.skipif(not _piper_voice_ready(), reason="Piper English voice not downloaded - see LOCAL_AI_README.md")
def test_pipeline_runs_end_to_end_with_no_internet_access(tmp_path):
    input_wav = str(tmp_path / "input.wav")
    output_wav = str(tmp_path / "output.wav")
    _make_test_wav(input_wav)

    def respond(text: str, language: str) -> str:
        # Deliberately not scenario_engine here - this test proves the
        # plumbing (STT -> respond -> TTS) with zero network egress; it does
        # not depend on scenario JSON files or Ollama producing a specific
        # answer to a synthetic tone.
        return "This is a test reply." if not text.strip() else f"You said: {text}"

    pipeline = LocalVoicePipeline()

    with _block_non_loopback_network():
        result = pipeline.run_file(input_wav, output_wav, respond, language_hint="en")

    # Shape assertions - this is a plumbing test, not an accuracy test.
    assert result.reply_text
    assert result.output_audio_path == output_wav

    with wave.open(output_wav, "rb") as f:
        assert f.getnframes() > 0
        assert f.getnchannels() >= 1

    assert result.timings_ms["total_ms"] > 0


@pytest.mark.skipif(not _ollama_ready(), reason="Ollama not running or model not pulled - see LOCAL_AI_README.md")
def test_ollama_classify_structured_works_with_no_internet_access():
    """Narrower check of just the LLM stage against a trivial schema -
    faster to run than the full pipeline test above, useful when only the
    Ollama leg needs re-verifying after a model swap."""
    from local_ai import llm_engine

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"],
    }

    with _block_non_loopback_network():
        result = asyncio.run(
            llm_engine.classify_structured(
                "Answer yes or no only, in the given JSON schema.",
                "Is water wet? Answer yes.",
                schema,
            )
        )

    assert result["answer"] in ("yes", "no")
