"""services/voice_pipeline/local_stt.py with a fake Whisper model - no GPU
or model download needed."""

from types import SimpleNamespace


def seg(text, no_speech_prob=0.05, avg_logprob=-0.3, compression_ratio=1.2):
    return SimpleNamespace(text=text, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob,
                           compression_ratio=compression_ratio)

import numpy as np

from pipecat.frames.frames import TranscriptionFrame
from services.voice_pipeline import local_stt


class FakeWhisper:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return iter(self.segments), None


def _stt(monkeypatch, segments, **kwargs):
    fake = FakeWhisper(segments)
    monkeypatch.setitem(local_stt._models, ("fake", "float16", "cuda"), fake)
    stt = local_stt.LocalWhisperSTT(source="fake", compute_type="float16", language="hi", **kwargs)
    return stt, fake


async def _collect(stt, audio: bytes):
    return [f async for f in stt.run_stt(audio)]


def _speech(seconds: float = 1.0) -> bytes:
    return (np.ones(int(16000 * seconds), dtype=np.int16) * 1000).tobytes()


async def test_transcribes_and_emits_frame(monkeypatch):
    stt, fake = _stt(monkeypatch, [seg(" नहीं आ पाऊँगा")])
    monkeypatch.setattr(stt, "start_processing_metrics", _noop)
    monkeypatch.setattr(stt, "stop_processing_metrics", _noop)

    frames = await _collect(stt, _speech())

    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "नहीं आ पाऊँगा"
    call = fake.calls[0]
    assert call["language"] == "hi" and call["beam_size"] == 1
    assert call["vad_filter"] is False and call["condition_on_previous_text"] is False
    assert call["initial_prompt"] == local_stt.DOMAIN_PROMPTS["hi"]


async def test_drops_segments_whisper_flags_as_non_speech(monkeypatch):
    """Whisper hallucinates text for noise ("धन्यवाद"); segments it scores as
    probably-not-speech must not reach the call logic."""
    stt, _ = _stt(monkeypatch, [seg(" धन्यवाद", no_speech_prob=0.9)])
    monkeypatch.setattr(stt, "start_processing_metrics", _noop)
    monkeypatch.setattr(stt, "stop_processing_metrics", _noop)

    frames = await _collect(stt, _speech())

    assert frames == [None]


async def test_ignores_clicks_shorter_than_a_tenth_of_a_second(monkeypatch):
    stt, fake = _stt(monkeypatch, [seg("x")])

    frames = await _collect(stt, _speech(0.05))

    assert frames == [None]
    assert fake.calls == []


def test_model_is_loaded_once_and_shared(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(local_stt._models, ("shared", "int8", "cuda"), sentinel)
    a = local_stt.LocalWhisperSTT(source="shared", compute_type="int8", language="gu")
    b = local_stt.LocalWhisperSTT(source="shared", compute_type="int8", language="hi")
    assert a._model is sentinel and b._model is sentinel


async def _noop(*_a, **_k):
    return None


def test_gujarati_written_in_devanagari_is_mapped_back_to_gujarati_script():
    """Whisper writes Gujarati speech in Devanagari ("मारी तबियत सारी नती") -
    found by tools/stt_benchmark.py. Without mapping back, the Gujarati
    keyword lists never match and every Gujarati answer goes to the LLM."""
    assert local_stt.devanagari_to_gujarati("मारी तबियत सारी नथी") == "મારી તબિયત સારી નથી"
    assert local_stt.devanagari_to_gujarati("ना, हुं नथी आवतो।") == "ના, હું નથી આવતો।"  # danda kept
    assert local_stt.devanagari_to_gujarati("ok, T2 १२") == "ok, T2 ૧૨"  # Latin untouched, digits mapped


async def test_gujarati_call_output_is_transliterated(monkeypatch):
    fake = FakeWhisper([seg(" ना, हुं नथी आवतो")])
    monkeypatch.setitem(local_stt._models, ("fake-gu", "float16", "cuda"), fake)
    stt = local_stt.LocalWhisperSTT(source="fake-gu", compute_type="float16", language="gu")
    monkeypatch.setattr(stt, "start_processing_metrics", _noop)
    monkeypatch.setattr(stt, "stop_processing_metrics", _noop)

    frames = await _collect(stt, _speech())

    assert frames[0].text == "ના, હું નથી આવતો"


async def test_drops_low_confidence_hallucinations(monkeypatch):
    """Benchmark finding: on noisy audio Whisper invented a whole news
    sentence. Low average log-probability is its own signal for that."""
    stt, _ = _stt(monkeypatch, [seg(" उन्होंने बताया कि प्रदेश में टीके लगाये जा चुके हैं", avg_logprob=-1.6)])
    monkeypatch.setattr(stt, "start_processing_metrics", _noop)
    monkeypatch.setattr(stt, "stop_processing_metrics", _noop)

    assert await _collect(stt, _speech()) == [None]


async def test_process_isolation_decodes_in_the_worker_not_in_process(monkeypatch):
    """Production mode: the server process loads no model and hands raw PCM
    to the STT worker (in-process decodes were ~2x slower under call load)."""
    from concurrent.futures import Future

    from services.voice_pipeline import stt_worker

    sent = {}

    def fake_transcribe(source, compute_type, pcm16, **options):
        sent.update(source=source, pcm16=pcm16, options=options)
        fut = Future()
        fut.set_result(("हाँ जी", 512.0))
        return fut

    monkeypatch.setattr(stt_worker.worker, "transcribe", fake_transcribe)
    stt = local_stt.LocalWhisperSTT(source="not-loaded-here", compute_type="float16", language="hi",
                                    isolation="process")
    monkeypatch.setattr(stt, "start_processing_metrics", _noop)
    monkeypatch.setattr(stt, "stop_processing_metrics", _noop)
    audio = _speech()

    frames = await _collect(stt, audio)

    assert stt._model is None
    assert frames[0].text == "हाँ जी"
    assert sent["source"] == "not-loaded-here" and sent["pcm16"] == audio
    assert sent["options"]["prompt"] == local_stt.DOMAIN_PROMPTS["hi"]


def test_worker_process_round_trip_restart_and_respawn():
    """Real spawned process, no model: errors come back as exceptions, a
    restart fails nothing silently, and the next request respawns it."""
    import pytest

    from services.voice_pipeline.stt_worker import STTWorker

    w = STTWorker()
    try:
        with pytest.raises(RuntimeError, match="unknown op"):
            w._submit("bogus").result(timeout=60)
        first = w._proc
        w.restart()
        assert not first.is_alive()
        with pytest.raises(RuntimeError, match="unknown op"):
            w._submit("bogus").result(timeout=60)
        assert w._proc is not first and w._proc.is_alive()
    finally:
        w.close()


async def test_local_failure_falls_back_to_sarvam(monkeypatch):
    """A wedged/crashed local decode must not leave the caller in silence."""
    from config import settings

    stt, _ = _stt(monkeypatch, [seg(" x")])
    monkeypatch.setattr(stt, "start_processing_metrics", _noop)
    monkeypatch.setattr(stt, "stop_processing_metrics", _noop)

    async def boom(audio):
        raise TimeoutError()

    async def sarvam(audio, language):
        return "हाँ जी आ जाऊँगा"

    monkeypatch.setattr(stt, "_decode", boom)
    monkeypatch.setattr(local_stt, "sarvam_fallback", sarvam)
    monkeypatch.setattr(settings, "sarvam_api_key", "k")
    monkeypatch.setattr(settings, "local_stt_fallback_sarvam", True)
    frames = await _collect(stt, _speech())
    assert isinstance(frames[0], TranscriptionFrame) and frames[0].text == "हाँ जी आ जाऊँगा"

    monkeypatch.setattr(settings, "local_stt_fallback_sarvam", False)  # strict zero-spend mode
    frames = await _collect(stt, _speech())
    from pipecat.frames.frames import ErrorFrame

    assert isinstance(frames[0], ErrorFrame)
