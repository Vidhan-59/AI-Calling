"""Unit tests for local_ai/tts_engine.py with the Piper voice mocked - no
real voice model download or audio synthesis required to run these."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from local_ai.tts_engine import PiperTTS, VoiceNotAvailableError


def test_synthesize_writes_wav_via_voice(tmp_path):
    engine = PiperTTS(voices_dir=str(tmp_path))
    output_path = str(tmp_path / "out.wav")

    def fake_synthesize_wav(text, wav_file, *args, **kwargs):
        # Real Piper configures the wave.Wave_write header before writing
        # frames; replicate the minimum needed for `wave` to close cleanly.
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)

    fake_voice = MagicMock()
    fake_voice.synthesize_wav.side_effect = fake_synthesize_wav

    with patch.object(PiperTTS, "_get_voice", return_value=fake_voice):
        result = engine.synthesize("hello", output_path, language="en")

    assert result == output_path
    fake_voice.synthesize_wav.assert_called_once()
    call_args = fake_voice.synthesize_wav.call_args
    assert call_args[0][0] == "hello"


def test_synthesize_raises_for_unconfigured_language(tmp_path):
    engine = PiperTTS(voices_dir=str(tmp_path))

    with pytest.raises(VoiceNotAvailableError, match="No Piper voice configured"):
        engine.synthesize("hello", str(tmp_path / "out.wav"), language="xx")


def test_gujarati_raises_not_a_fallback_to_another_language(tmp_path):
    """Deliberately no silent fallback to English/Hindi phonemes for Gujarati
    text - that would speak the wrong language over real words, worse than
    failing loudly. See local_ai/config.py PIPER_VOICE_GU."""
    engine = PiperTTS(voices_dir=str(tmp_path))

    with pytest.raises(VoiceNotAvailableError):
        engine.synthesize("કેમ છો", str(tmp_path / "out.wav"), language="gu")


def test_get_voice_passes_a_path_object_to_download_voice(tmp_path):
    """Regression test: download_voice() requires a pathlib.Path for its
    download_dir argument, not a plain str - passing a str fails inside the
    library with a confusing 'unsupported operand type(s) for /' TypeError.
    This was a real bug caught while wiring this up; guard against it
    silently coming back."""
    engine = PiperTTS(voices_dir=str(tmp_path))
    # PiperTTS._voices is a class-level cache shared across every instance
    # (by design - loading a voice is expensive, see tts_engine.py). That
    # means it can carry a real, already-loaded voice over from a test in
    # another file within the same pytest session (e.g. the real smoke test
    # in test_e2e_offline_smoke.py, which sorts before this file). Clear it
    # so this test genuinely exercises the download path instead of hitting
    # a stale cache and silently passing for the wrong reason.
    PiperTTS._voices.clear()

    with patch("piper.download_voices.download_voice") as fake_download, \
         patch("piper.PiperVoice.load", return_value=MagicMock()):
        engine._get_voice("en")

        assert fake_download.called
        download_dir_arg = fake_download.call_args[0][1]
        assert isinstance(download_dir_arg, Path), (
            f"download_voice was called with a {type(download_dir_arg).__name__}, "
            "not a pathlib.Path - this is the exact bug that was fixed"
        )
