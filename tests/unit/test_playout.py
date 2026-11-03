"""PlayoutPlanner (services/tts/playout.py): quality adapts so the next piece
lands before the buffer runs dry; underruns are counted, not hidden."""

from services.tts.playout import Diagnostics, PlayoutPlanner


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _planner():
    clock = Clock()
    return PlayoutPlanner(clock=clock, diagnostics=Diagnostics()), clock


def _piece(p, clock, text, *, steps, synth, audio, cached=False):
    p.begin_piece(text)
    clock.t += synth
    return p.piece_ready(owner="a", text=text, steps=steps, synth_secs=synth, audio_secs=audio, cached=cached)


def test_first_piece_takes_default_quality():
    p, _ = _planner()
    p.begin_piece("जी, समझ गई")
    assert p.choose_steps("जी, समझ गई", default_steps=16) == 16


def test_thin_buffer_forces_cheaper_steps():
    p, clock = _planner()
    _piece(p, clock, "एक दो तीन", steps=10, synth=0.8, audio=1.0)  # 1.0s buffered, then 0.0s spent
    p.begin_piece("यह एक लंबा वाक्य है जो काफी देर तक चलेगा")  # ~3.8s of speech est.
    # 9 words ~3.5s audio: 16 steps (~3.3s) can't land inside a 1.0s buffer
    assert p.choose_steps("यह एक लंबा वाक्य है जो काफी देर तक चलेगा", default_steps=16) == 10


def test_deep_buffer_allows_best_quality():
    p, clock = _planner()
    _piece(p, clock, "एक दो तीन चार", steps=10, synth=0.5, audio=8.0)
    p.begin_piece("ठीक है")
    assert p.choose_steps("ठीक है", default_steps=16) == 16


def test_underrun_is_counted_with_its_gap():
    p, clock = _planner()
    _piece(p, clock, "पहला टुकड़ा", steps=10, synth=0.8, audio=1.0)
    clock.t += 1.0 - 0.0  # first piece finished playing...
    gap = _piece(p, clock, "दूसरा टुकड़ा", steps=10, synth=0.4, audio=1.0)
    assert 0.35 < gap < 0.45  # ...and the next one arrived 0.4s after it ended
    assert p._diag.underruns == 1
    assert p._diag.summary()["max_gap_ms"] > 300


def test_seamless_pieces_report_no_gap():
    p, clock = _planner()
    _piece(p, clock, "पहला टुकड़ा यहाँ है", steps=10, synth=0.5, audio=3.0)
    gap = _piece(p, clock, "दूसरा टुकड़ा", steps=10, synth=0.6, audio=2.0)
    assert gap == 0.0 and p._diag.underruns == 0


def test_long_silence_starts_a_new_reply_not_an_underrun():
    p, clock = _planner()
    _piece(p, clock, "पहला", steps=10, synth=0.5, audio=1.0)
    clock.t += 30.0  # the caller talked for a while
    p.begin_piece("नया जवाब")
    assert p.choose_steps("नया जवाब", default_steps=16) == 16
    _piece_gap = p.piece_ready(owner="a", text="नया जवाब", steps=16, synth_secs=0.5, audio_secs=1.0, cached=False)
    assert _piece_gap == 0.0 and p._diag.underruns == 0 and p._diag.replies == 2


def test_cost_model_follows_measurements():
    p, clock = _planner()
    for audio in (1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.8):  # slow contended GPU: ~1.5s + 1.0x audio
        _piece(p, clock, "एक दो तीन", steps=16, synth=1.5 + audio, audio=audio)
    fixed, slope = p.model()[16]
    assert 1.2 < fixed < 1.8 and 0.8 < slope < 1.2  # seeded (0.7, 0.7)


def test_budget_grows_as_the_buffer_builds():
    p, clock = _planner()
    assert p.next_budget_words() is None  # nothing queued: the next piece starts a reply
    p.release("एक दो तीन")
    p.begin_piece("एक दो तीन")
    clock.t += 0.8
    p.piece_ready(owner="a", text="एक दो तीन", steps=10, synth_secs=0.8, audio_secs=1.2, cached=False)
    p.release("चार पाँच छह सात आठ")
    second = p.next_budget_words()
    p.begin_piece("चार पाँच छह सात आठ")
    clock.t += 0.9
    p.piece_ready(owner="a", text="चार पाँच छह सात आठ", steps=10, synth_secs=0.9, audio_secs=2.0, cached=False)
    third = p.next_budget_words()
    assert second is not None and third is not None
    assert third > second >= 2  # more buffered audio -> the next piece may be longer


def test_unavoidable_gap_gets_one_long_piece_not_many_small_ones():
    p, clock = _planner()
    p.release("एक")  # its synthesis alone eats the whole (short) buffer
    assert p.next_budget_words() == 12


def test_interruption_resets_buffer_and_is_counted():
    p, clock = _planner()
    _piece(p, clock, "एक दो तीन", steps=10, synth=0.5, audio=5.0)
    p.reset(interrupted=True)
    assert p.buffered_secs() == 0.0 and p._diag.interruptions == 1
