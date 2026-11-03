"""Voice assignment resolution and the TTS audio cache - the TTS service
client is faked, so no GPU/service is needed."""

import pytest

from services.tts import client, voice_registry
from services.voice_pipeline import tts_cache

VOICES = {
    "voices": [
        {"id": "anita-aaa111", "name": "Anita", "version": 1},
        {"id": "ravi-bbb222", "name": "Ravi", "version": 3},
    ],
    "default_voice_id": "anita-aaa111",
}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_registry, "_PATH", str(tmp_path / "assignments.json"))
    monkeypatch.setattr(tts_cache, "_CACHE_DIR", str(tmp_path / "cache"))
    tts_cache._memory.clear()
    voice_registry._snapshot = None
    state = {"voices": dict(VOICES), "synth_calls": []}

    async def list_voices():
        return state["voices"]

    async def synth(text, *, language, voice_id, speed=1.0, num_step=None):
        state["synth_calls"].append((text, voice_id, speed))
        return b"\x01\x00" * 800

    monkeypatch.setattr(client, "list_voices", list_voices)
    monkeypatch.setattr(client, "synthesize_pcm", synth)
    return state


async def test_default_voice_when_nothing_assigned():
    v = await voice_registry.resolve("shift_confirm", "hi")
    assert v.voice_id == "anita-aaa111"


async def test_assignment_precedence_specific_beats_wildcard():
    voice_registry.set_assignment("*", "*", "ravi-bbb222")
    assert (await voice_registry.resolve("shift_confirm", "hi")).voice_id == "ravi-bbb222"
    voice_registry.set_assignment("*", "hi", "anita-aaa111")
    assert (await voice_registry.resolve("shift_confirm", "hi")).voice_id == "anita-aaa111"
    voice_registry.set_assignment("shift_confirm", "hi", "ravi-bbb222", speed=0.9)
    v = await voice_registry.resolve("shift_confirm", "hi")
    assert (v.voice_id, v.speed, v.version) == ("ravi-bbb222", 0.9, 3)
    # other language/owner unaffected
    assert (await voice_registry.resolve("shift_confirm", "gu")).voice_id == "ravi-bbb222"  # falls to "*:*"


async def test_assignment_to_deleted_voice_falls_through():
    voice_registry.set_assignment("shift_confirm", "hi", "ghost-000000")
    assert (await voice_registry.resolve("shift_confirm", "hi")).voice_id == "anita-aaa111"


async def test_agent_yaml_voice_is_fallback_before_default():
    v = await voice_registry.resolve("a", "hi", fallback_voice_id="ravi-bbb222", fallback_speed=0.95)
    assert (v.voice_id, v.speed) == ("ravi-bbb222", 0.95)
    v = await voice_registry.resolve("a", "hi", fallback_voice_id="default", fallback_speed=0.95)
    assert (v.voice_id, v.speed) == ("anita-aaa111", 0.95)


async def test_no_voices_raises(isolated):
    isolated["voices"] = {"voices": [], "default_voice_id": None}
    with pytest.raises(client.TTSUnavailable):
        await voice_registry.resolve("x", "hi")


async def test_cache_synthesizes_once_per_line(isolated):
    a = await tts_cache.get_audio("namaste", "hi", "shift_confirm")
    b = await tts_cache.get_audio("namaste", "hi", "shift_confirm")
    assert a == b and len(isolated["synth_calls"]) == 1
    tts_cache._memory.clear()  # new process: served from disk, still no synth
    await tts_cache.get_audio("namaste", "hi", "shift_confirm")
    assert len(isolated["synth_calls"]) == 1


async def test_cache_invalidated_by_voice_version_and_voice_change(isolated):
    await tts_cache.get_audio("namaste", "hi", "s")
    isolated["voices"] = {
        "voices": [{"id": "anita-aaa111", "name": "Anita", "version": 2}, VOICES["voices"][1]],
        "default_voice_id": "anita-aaa111",
    }
    voice_registry.invalidate()
    await tts_cache.get_audio("namaste", "hi", "s")  # re-recorded voice -> new audio
    voice_registry.set_assignment("s", "hi", "ravi-bbb222")
    await tts_cache.get_audio("namaste", "hi", "s")  # different voice -> new audio
    assert [c[1] for c in isolated["synth_calls"]] == ["anita-aaa111", "anita-aaa111", "ravi-bbb222"]


async def test_concurrent_requests_share_one_synthesis(isolated):
    import asyncio

    await asyncio.gather(*(tts_cache.get_audio("same line", "en", "s") for _ in range(5)))
    assert len(isolated["synth_calls"]) == 1


async def test_memory_cache_is_bounded_but_disk_still_serves_evicted_lines(isolated, monkeypatch):
    """Live agent replies are free text - the in-memory cache must not grow forever."""
    monkeypatch.setattr(tts_cache, "_MEMORY_MAX", 3)
    for i in range(6):
        await tts_cache.get_audio(f"line {i}", "en", "s")
    assert len(tts_cache._memory) == 3
    calls = len(isolated["synth_calls"])
    await tts_cache.get_audio("line 0", "en", "s")  # evicted from memory, still on disk
    assert len(isolated["synth_calls"]) == calls


async def test_live_piece_uses_fewer_steps_when_short(monkeypatch):
    """The short first piece of a reply is the one the caller waits for; a
    long piece behind a deep playback buffer can afford the full live quality."""
    from services.tts.pipecat_service import OmniVoiceTTSService

    seen = []

    async def fake_get_audio_ex(text, language, owner="*", *, num_step=None, **kw):
        seen.append((text, num_step))
        return b"\x00\x00" * 80000, False  # 10s of audio: a deep buffer for the next piece

    monkeypatch.setattr(tts_cache, "get_audio_ex", fake_get_audio_ex)
    svc = OmniVoiceTTSService(owner="a", language="hi")
    for text in ("जी बिल्कुल समझ गया", "आपकी शिफ्ट कल सुबह आठ बजे टर्मिनल तीन पर है और आपको समय से पहुँचना है।"):
        async for _ in svc.run_tts(text, "ctx"):
            pass
    from config import settings

    assert seen[0][1] == settings.tts_short_piece_num_step
    assert seen[1][1] == settings.tts_live_num_step


async def test_live_piece_downshifts_quality_when_buffer_is_thin(monkeypatch):
    """Continuity over quality: a long piece right behind a nearly-empty
    buffer is rendered at the cheapest live setting so it lands in time."""
    from services.tts.pipecat_service import OmniVoiceTTSService

    seen = []

    async def fake_get_audio_ex(text, language, owner="*", *, num_step=None, **kw):
        seen.append(num_step)
        return b"\x00\x00" * 800, False  # 0.1s of audio

    monkeypatch.setattr(tts_cache, "get_audio_ex", fake_get_audio_ex)
    svc = OmniVoiceTTSService(owner="a", language="hi")
    for text in ("जी बिल्कुल समझ गया", "आपकी शिफ्ट कल सुबह आठ बजे टर्मिनल तीन पर है और आपको समय से पहुँचना है।"):
        async for _ in svc.run_tts(text, "ctx"):
            pass
    from services.tts.playout import STEP_LADDER

    assert seen[1] == STEP_LADDER[-1]


async def test_warm_never_raises_when_service_is_down(monkeypatch):
    async def boom(*a, **k):
        raise client.TTSUnavailable("down")

    monkeypatch.setattr(client, "_http", boom)
    await client.warm()  # best-effort: must swallow the error
