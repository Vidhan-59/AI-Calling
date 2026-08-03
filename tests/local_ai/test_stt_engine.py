"""Unit tests for local_ai/stt_engine.py with the whisper model mocked -
these must not require downloading or loading a real model."""

from unittest.mock import MagicMock, patch

from local_ai.stt_engine import FasterWhisperSTT, TranscriptionResult


def _fake_segment(text, start=0.0, end=1.0):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    return seg


def test_transcribe_joins_segments_and_reports_language():
    engine = FasterWhisperSTT()
    fake_info = MagicMock(language="hi", language_probability=0.97)
    fake_segments = [_fake_segment("नमस्ते "), _fake_segment("कैसे हैं")]

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segments), fake_info)

    with patch.object(FasterWhisperSTT, "_get_model", return_value=fake_model):
        result = engine.transcribe("fake_audio.wav")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "नमस्ते कैसे हैं"
    assert result.detected_language == "hi"
    assert result.language_probability == 0.97
    assert len(result.segments) == 2
    assert result.segments[0]["text"] == "नमस्ते "


def test_transcribe_passes_language_hint_through():
    engine = FasterWhisperSTT()
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([]), MagicMock(language="en", language_probability=1.0))

    with patch.object(FasterWhisperSTT, "_get_model", return_value=fake_model):
        engine.transcribe("fake_audio.wav", language="en")

    _, kwargs = fake_model.transcribe.call_args
    assert kwargs["language"] == "en"


def test_transcribe_empty_audio_returns_empty_text():
    engine = FasterWhisperSTT()
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter([]), MagicMock(language="en", language_probability=0.5))

    with patch.object(FasterWhisperSTT, "_get_model", return_value=fake_model):
        result = engine.transcribe("silence.wav")

    assert result.text == ""
    assert result.segments == []
