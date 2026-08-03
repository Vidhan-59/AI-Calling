"""Unit tests for local_ai/text_norm.py - pure functions, no model needed."""

from local_ai import text_norm


def test_strip_markdown_removes_bold_italic_code():
    assert text_norm.strip_markdown("**bold** and *italic* and `code`") == "bold and italic and code"


def test_strip_markdown_removes_headings_and_bullets():
    text = "# Heading\n- item one\n- item two"
    result = text_norm.strip_markdown(text)
    assert "#" not in result
    assert result.count("\n") == 2
    assert "item one" in result and "item two" in result


def test_strip_markdown_removes_links_keeps_text():
    assert text_norm.strip_markdown("see [the docs](https://example.com)") == "see the docs"


def test_expand_times_english():
    assert text_norm.expand_times("meet at 14:00", "en") == "meet at 2 PM"
    assert text_norm.expand_times("call at 09:30", "en") == "call at 9:30 AM"
    assert text_norm.expand_times("midnight is 00:00", "en") == "midnight is 12 AM"


def test_expand_times_noop_for_non_english():
    # Hindi/Gujarati times are supplied pre-spoken by the caller (see
    # scenario JSON `time` facts) - this function must not touch them.
    text = "समय सुबह 8 बजे है"
    assert text_norm.expand_times(text, "hi") == text


def test_split_sentences_handles_devanagari_terminator():
    text = "आपकी ड्यूटी है। क्या आप आएँगे?"
    assert text_norm.split_sentences(text) == ["आपकी ड्यूटी है।", "क्या आप आएँगे?"]


def test_split_sentences_empty_input():
    assert text_norm.split_sentences("") == []
    assert text_norm.split_sentences("   ") == []


def test_normalize_for_speech_full_pass():
    result = text_norm.normalize_for_speech("**Meeting** at 14:00  \n\n please come", "en")
    assert result == "Meeting at 2 PM please come"
