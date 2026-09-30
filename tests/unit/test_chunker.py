from services.pipeline.chunker import split_sentences


def test_empty_and_blank_text_yields_no_chunks():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_single_sentence_with_no_terminal_punctuation_is_one_chunk():
    assert split_sentences("Namaste kaise ho") == ["Namaste kaise ho"]


def test_splits_on_period_exclamation_and_question_mark():
    text = "Namaste! Kya aap Amit hain? Haan, main sun raha hoon."
    assert split_sentences(text) == [
        "Namaste!",
        "Kya aap Amit hain?",
        "Haan, main sun raha hoon.",
    ]


def test_splits_on_hindi_danda_and_double_danda():
    text = "नमस्ते। क्या आप अमित हैं॥ हाँ।"
    assert split_sentences(text) == ["नमस्ते।", "क्या आप अमित हैं॥", "हाँ।"]


def test_does_not_split_mid_sentence_without_trailing_whitespace():
    # No space after the period (e.g. a decimal or abbreviation glued to the
    # next word) - the heuristic only splits on punctuation followed by
    # whitespace, so this stays one chunk rather than mis-splitting.
    assert split_sentences("Shift 2.30pm confirm karo") == ["Shift 2.30pm confirm karo"]


def test_strips_surrounding_whitespace_on_each_chunk():
    assert split_sentences("  Hello there.   Bye now.  ") == ["Hello there.", "Bye now."]
