import pytest

from services.verbalize import (
    verbalize,
    verbalize_date,
    verbalize_digits,
    verbalize_number,
    verbalize_shift_code,
    verbalize_time,
)


def test_verbalize_number_hindi():
    assert verbalize_number(2, "hi") == "do"
    assert verbalize_number(21, "hi") == "ikkees"
    assert verbalize_number(59, "hi") == "unsath"


def test_verbalize_number_english_is_numeral():
    assert verbalize_number(14, "en") == "14"


def test_verbalize_number_out_of_range_raises():
    with pytest.raises(ValueError):
        verbalize_number(100, "hi")


def test_verbalize_digits_reads_each_digit():
    assert verbalize_digits("4217", "hi") == "chaar do ek saat"
    assert verbalize_digits("07", "en") == "0 7"


def test_verbalize_time_on_the_hour_never_says_fourteen_hundred():
    result = verbalize_time("14:00", "hi")
    assert result == "do baje dopahar"
    assert "14" not in result
    assert "fourteen hundred" not in result.lower()


def test_verbalize_time_quarter_past():
    assert verbalize_time("06:15", "hi") == "sava chhah baje subah"


def test_verbalize_time_half_past_irregular_forms():
    assert verbalize_time("01:30", "hi") == "dedh baje raat"
    assert verbalize_time("02:30", "hi") == "dhai baje raat"


def test_verbalize_time_half_past_regular():
    assert verbalize_time("06:30", "hi") == "saadhe chhah baje subah"


def test_verbalize_time_quarter_to_rolls_to_next_hour():
    assert verbalize_time("05:45", "hi") == "paune chhah baje subah"


def test_verbalize_time_english_uses_am_pm():
    assert verbalize_time("14:00", "en") == "2 PM"
    assert verbalize_time("00:00", "en") == "12 AM"


def test_verbalize_time_gujarati_uses_vaage():
    result = verbalize_time("14:00", "gu")
    assert "vaage" in result
    assert "bapore" in result


def test_verbalize_date_keeps_month_in_english():
    assert verbalize_date("2026-08-03", "hi") == "teen August"
    assert verbalize_date("2026-08-03", "gu") == "tran August"
    assert verbalize_date("2026-08-03", "en") == "August 3rd"


def test_verbalize_shift_code_never_spells_letters():
    assert verbalize_shift_code("D", "hi") == "din ki shift"
    assert verbalize_shift_code("N", "en") == "night shift"
    assert verbalize_shift_code("O", "gu") == "raja no divas"


def test_verbalize_shift_code_unknown_raises():
    with pytest.raises(ValueError):
        verbalize_shift_code("X", "en")


def test_verbalize_dispatcher_matches_direct_calls():
    assert verbalize(2, "number", "hi") == verbalize_number(2, "hi")
    assert verbalize("14:00", "time", "hi") == verbalize_time("14:00", "hi")
    assert verbalize("D", "shift_code", "en") == verbalize_shift_code("D", "en")
