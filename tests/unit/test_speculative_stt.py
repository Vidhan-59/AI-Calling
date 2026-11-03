"""Speculative decoding in LocalWhisperSTT (services/voice_pipeline/
local_stt.py): a decode started at the caller's pause is used at end-of-turn;
one invalidated by the caller resuming is not. No model is loaded -
isolation="process" skips the in-process load and _decode is replaced."""

import asyncio

from services.voice_pipeline.local_stt import LocalWhisperSTT

SECOND = b"\x10\x00" * 16000  # 1s of 16kHz PCM16


def _stt() -> tuple[LocalWhisperSTT, list[int]]:
    stt = LocalWhisperSTT(source="unused", compute_type="float16", language="hi", isolation="process", sample_rate=16000)
    decodes: list[int] = []

    async def fake_decode(audio: bytes):
        decodes.append(len(audio))
        await asyncio.sleep(0.05)
        return "हाँ जी", 50.0

    async def no_op(*_a, **_k):
        return None

    stt._decode = fake_decode
    stt.create_task = lambda coro, name=None: asyncio.create_task(coro)
    stt.start_processing_metrics = no_op
    stt.stop_processing_metrics = no_op
    stt._user_id = ""
    return stt, decodes


async def _run(stt: LocalWhisperSTT, audio: bytes) -> list:
    return [f async for f in stt.run_stt(audio)]


async def test_guess_taken_at_the_pause_is_used_at_end_of_turn():
    stt, decodes = _stt()
    stt._audio_buffer = bytearray(SECOND)
    stt._speculate()
    frames = await _run(stt, SECOND + b"\x00\x00" * 8000)  # + the silent tail VAD waited through
    assert decodes == [len(SECOND)]  # one decode only - the speculative one
    assert frames[0].text == "हाँ जी"
    assert stt.speculative_hits == 1


async def test_guess_is_dropped_when_the_caller_resumes():
    stt, decodes = _stt()
    stt._audio_buffer = bytearray(SECOND)
    stt._speculate()
    stt._discard_speculation()  # UserSpeechResumedFrame
    full_turn = SECOND * 2
    frames = await _run(stt, full_turn)
    assert len(full_turn) in decodes  # the real turn was decoded in full
    assert frames[0].text == "हाँ जी"
    assert stt.speculative_hits == 0 and stt.speculative_misses == 1


async def test_only_one_speculative_decode_in_flight():
    stt, decodes = _stt()
    stt._audio_buffer = bytearray(SECOND)
    stt._speculate()
    stt._speculate()
    await asyncio.sleep(0.1)
    assert len(decodes) == 1


async def test_too_little_audio_is_not_speculated_on():
    stt, decodes = _stt()
    stt._audio_buffer = bytearray(b"\x00\x00" * 800)
    stt._speculate()
    assert stt._spec is None
