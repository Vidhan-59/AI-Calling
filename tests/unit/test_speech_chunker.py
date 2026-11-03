from services.pipeline.chunker import SpeechChunker


def _stream(chunker: SpeechChunker, tokens: list[str]) -> list[str]:
    out: list[str] = []
    for t in tokens:
        out += chunker.feed(t)
    return out + chunker.flush()


def test_first_piece_released_after_n_words_while_llm_still_writing():
    c = SpeechChunker(first_chunk_words=4)
    assert c.feed("Main aap") == []
    assert c.feed("ki kaise mad") == []  # "mad..." is half a word - not yet 4 whole ones
    assert c.feed("ad ") == ["Main aapki kaise madad"]


def test_later_pieces_wait_for_a_clause_boundary():
    c = SpeechChunker(first_chunk_words=4)
    pieces = _stream(c, ["Main ", "aapki ", "kaise ", "madad ", "kar ", "sakti ", "hoon? ", "Aapka ", "order ", "kya ", "hai"])
    assert pieces == ["Main aapki kaise madad", "kar sakti hoon?", "Aapka order kya hai"]


def test_clause_end_before_n_words_releases_early():
    c = SpeechChunker(first_chunk_words=4)
    assert c.feed("Namaste! ") == ["Namaste!"]


def test_hindi_danda_and_comma_are_cut_points():
    c = SpeechChunker(first_chunk_words=4)
    pieces = _stream(c, ["जी, ", "समझ ", "गई। ", "कल ", "आप ", "आएँगे?"])
    assert pieces == ["जी,", "समझ गई।", "कल आप आएँगे?"]


def test_token_glued_text_is_reassembled_exactly():
    text = "Theek hai, main aapki shift note kar leti hoon. Dhanyavaad!"
    c = SpeechChunker(first_chunk_words=4)
    pieces = _stream(c, [text[i:i + 3] for i in range(0, len(text), 3)])
    assert " ".join(pieces) == text


def test_runaway_clause_is_cut_at_a_space():
    c = SpeechChunker(first_chunk_words=1, max_chunk_chars=20)
    pieces = _stream(c, ["a " * 30])
    assert all(len(p) <= 20 for p in pieces)
    assert " ".join(pieces).split() == ["a"] * 30


def test_flush_on_empty_buffer_yields_nothing():
    assert SpeechChunker().flush() == []


def test_later_pieces_are_merged_up_to_min_words():
    text = "Ji bilkul, main samajh gayi, aapki shift kal subah hai, aur terminal CT3 par. Dhanyavaad!"
    tokens = [text[i:i + 3] for i in range(0, len(text), 3)]
    plain = _stream(SpeechChunker(first_chunk_words=3), tokens)
    merged = _stream(SpeechChunker(first_chunk_words=3, min_chunk_words=6), tokens)
    assert " ".join(merged) == text
    # the first piece is still released at the first clause (latency), but
    # the short clauses after it are merged instead of one piece per comma
    assert merged[0] == "Ji bilkul,"
    assert merged[1] == "main samajh gayi, aapki shift kal subah hai,"
    assert len(merged) < len(plain)


def test_short_sentence_is_released_at_half_min_words_not_held():
    pieces = _stream(
        SpeechChunker(first_chunk_words=3, min_chunk_words=6),
        ["Namaste ji aap kaise hain ", "aaj. ", "Kya ", "aap ", "kal ", "aa ", "sakenge?"],
    )
    assert pieces == ["Namaste ji aap", "kaise hain aaj.", "Kya aap kal aa sakenge?"]
