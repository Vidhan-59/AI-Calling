"""Numbers, times, dates and shift codes -> the way a person actually says
them out loud, per language. 14:00 must become "do baje dopahar", never
"fourteen hundred" - this single function is called out in the architecture
doc as a disproportionate source of "the agent sounds broken" complaints, so
it gets its own module and its own unit tests rather than being inlined into
prompt templates.

IMPORTANT: the Hindi number/time-of-day vocabulary below is standard and
reasonably safe to trust. The Gujarati tables are a best-effort scaffold -
Gujarati number words (especially compounds in the 20s-50s) have enough
regional/dialectal variation that this MUST be checked against real Mundra
operator recordings or a native Gujarati speaker before it's used in a live
pilot. Do not treat evals/personas passing against this module as proof the
Gujarati output actually sounds right to a Gujarati speaker.
"""

from datetime import date as date_type, datetime, time as time_type
from typing import Literal

Language = Literal["hi", "gu", "en"]
VerbalizeType = Literal["number", "digits", "time", "date", "shift_code"]

_MONTHS_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_NUMBER_WORDS_HI: dict[int, str] = {
    0: "shunya", 1: "ek", 2: "do", 3: "teen", 4: "chaar", 5: "paanch",
    6: "chhah", 7: "saat", 8: "aath", 9: "nau", 10: "dus",
    11: "gyarah", 12: "baarah", 13: "terah", 14: "chaudah", 15: "pandrah",
    16: "solah", 17: "satrah", 18: "atharah", 19: "unnees", 20: "bees",
    21: "ikkees", 22: "baees", 23: "teis", 24: "chaubees", 25: "pachchees",
    26: "chhabbees", 27: "sattaais", 28: "atthaais", 29: "unatees", 30: "tees",
    31: "ikatees", 32: "battees", 33: "taintees", 34: "chauntees", 35: "paintees",
    36: "chhattees", 37: "saintees", 38: "adtees", 39: "untaalees", 40: "chaalees",
    41: "iktaalees", 42: "bayaalees", 43: "taintaalees", 44: "chauvaalees", 45: "paintaalees",
    46: "chhiyaalees", 47: "saintaalees", 48: "adtaalees", 49: "uncchaas", 50: "pachaas",
    51: "ikyaavan", 52: "baavan", 53: "tirpan", 54: "chauvan", 55: "pachpan",
    56: "chhappan", 57: "sattaavan", 58: "atthaavan", 59: "unsath",
}

# Best-effort - see module docstring warning.
_NUMBER_WORDS_GU: dict[int, str] = {
    0: "shunya", 1: "ek", 2: "be", 3: "tran", 4: "chaar", 5: "paanch",
    6: "chha", 7: "saat", 8: "aath", 9: "nav", 10: "dus",
    11: "agiyar", 12: "baar", 13: "ter", 14: "chaud", 15: "pandar",
    16: "sol", 17: "sattar", 18: "adhar", 19: "ognis", 20: "vees",
    21: "ekvis", 22: "baavis", 23: "trevis", 24: "chovis", 25: "pachvis",
    26: "chhavis", 27: "sataavis", 28: "adyaavis", 29: "ognatris", 30: "tris",
    31: "ektris", 32: "battris", 33: "tettris", 34: "chotris", 35: "paintris",
    36: "chhatris", 37: "sadatris", 38: "adatris", 39: "ognachalis", 40: "chalis",
    41: "ektalis", 42: "betalis", 43: "tetalis", 44: "chotalis", 45: "paistalis",
    46: "chhetalis", 47: "sadtalis", 48: "adtalis", 49: "oganpachas", 50: "pachas",
    51: "ekaavan", 52: "baavan", 53: "trepan", 54: "chopan", 55: "pachpan",
    56: "chhappan", 57: "sattavan", 58: "adyaavan", 59: "ognasath",
}

_PERIOD_OF_DAY = {
    "hi": [(0, 5, "raat"), (5, 12, "subah"), (12, 17, "dopahar"), (17, 20, "shaam"), (20, 24, "raat")],
    "gu": [(0, 5, "raate"), (5, 12, "savare"), (12, 17, "bapore"), (17, 20, "sanje"), (20, 24, "raate")],
}

_SHIFT_CODE_WORDS = {
    "hi": {"D": "din ki shift", "N": "raat ki shift", "O": "chhutti ka din"},
    "gu": {"D": "divas ni shift", "N": "raatri ni shift", "O": "raja no divas"},
    "en": {"D": "day shift", "N": "night shift", "O": "day off"},
}

# Irregular half/quarter-hour forms (Hindi). Gujarati commonly borrows the
# same sava/saade/paune structure in spoken Mundra-region usage; treat with
# the same caution as the rest of the Gujarati table.
_QUARTER_PAST = {"hi": "sava", "gu": "sava"}
_QUARTER_TO = {"hi": "paune", "gu": "paune"}
_HALF_PAST_IRREGULAR = {
    "hi": {1: "dedh", 2: "dhai"},
    "gu": {1: "dedh", 2: "adhi"},
}
_HALF_PAST_REGULAR = {"hi": "saadhe", "gu": "saade"}


def _number_words(language: Language) -> dict[int, str]:
    return _NUMBER_WORDS_HI if language == "hi" else _NUMBER_WORDS_GU


def verbalize_number(n: int, language: Language) -> str:
    """Spoken form of a whole number 0-59 (the only range this domain needs:
    minutes, day-of-month, small counts). English returns the numeral - a
    voice model reads plain digits correctly and idiomatically already."""
    if language == "en":
        return str(n)
    words = _number_words(language)
    if n in words:
        return words[n]
    if n < 0 or n > 59:
        raise ValueError(f"verbalize_number only supports 0-59, got {n}")
    return words[n]


def verbalize_digits(value: str, language: Language) -> str:
    """Spoken digit-by-digit, e.g. RGID last-4 "4217" -> "chaar do ek saat".
    Use this - never verbalize_number - anywhere a number must NOT be read
    as a magnitude (RGID digits, OTP-style confirmations, phone numbers)."""
    digit_names_hi = ["shunya", "ek", "do", "teen", "chaar", "paanch", "chhah", "saat", "aath", "nau"]
    digit_names_gu = ["shunya", "ek", "be", "tran", "chaar", "paanch", "chha", "saat", "aath", "nav"]
    if language == "en":
        return " ".join(value)
    names = digit_names_hi if language == "hi" else digit_names_gu
    return " ".join(names[int(ch)] for ch in value if ch.isdigit())


def _period_of_day(hour: int, language: Language) -> str:
    buckets = _PERIOD_OF_DAY[language]
    for start, end, label in buckets:
        if start <= hour < end:
            return label
    return buckets[-1][2]


def verbalize_time(value: str | time_type | datetime, language: Language) -> str:
    """14:00 -> "do baje dopahar" (hi) / "be vaage bapore" (gu) / "2 PM" (en).
    Handles sava/saadhe/paune (quarter-past/half-past/quarter-to) for
    non-zero minutes instead of bolting "X minute" onto everything, since
    that's how these times are actually said."""
    if isinstance(value, str):
        hour, minute = (int(p) for p in value.split(":")[:2])
    elif isinstance(value, datetime):
        hour, minute = value.hour, value.minute
    elif isinstance(value, time_type):
        hour, minute = value.hour, value.minute
    else:
        raise TypeError(f"Unsupported time value type: {type(value)}")

    hour12 = hour % 12 or 12

    if language == "en":
        suffix = "AM" if hour < 12 else "PM"
        return f"{hour12}:{minute:02d} {suffix}" if minute else f"{hour12} {suffix}"

    period = _period_of_day(hour, language)
    baje_word = "vaage" if language == "gu" else "baje"

    if minute == 0:
        return f"{verbalize_number(hour12, language)} {baje_word} {period}"
    if minute == 15:
        return f"{_QUARTER_PAST[language]} {verbalize_number(hour12, language)} {baje_word} {period}"
    if minute == 30:
        irregular = _HALF_PAST_IRREGULAR[language].get(hour12)
        if irregular:
            return f"{irregular} {baje_word} {period}"
        return f"{_HALF_PAST_REGULAR[language]} {verbalize_number(hour12, language)} {baje_word} {period}"
    if minute == 45:
        next_hour12 = (hour12 % 12) + 1
        return f"{_QUARTER_TO[language]} {verbalize_number(next_hour12, language)} {baje_word} {period}"

    return (
        f"{verbalize_number(hour12, language)} {baje_word} {verbalize_number(minute, language)} "
        f"minute {period}"
    )


def verbalize_date(value: str | date_type | datetime, language: Language) -> str:
    """3 Aug 2026 -> "teen August" (hi) / "tran August" (gu) / "August 3rd" (en).
    Month names are kept in English on purpose - that's how they're actually
    said in Hindi/Gujarati speech at Mundra, per the language policy's
    fixed-English-terms list."""
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date_type):
        parsed = datetime(value.year, value.month, value.day)
    else:
        raise TypeError(f"Unsupported date value type: {type(value)}")

    month_name = _MONTHS_EN[parsed.month - 1]
    if language == "en":
        day = parsed.day
        suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{month_name} {day}{suffix}"
    return f"{verbalize_number(parsed.day, language)} {month_name}"


def verbalize_shift_code(code: str, language: Language) -> str:
    """D/N/O spoken as words, never spelled as letters."""
    code = code.strip().upper()
    if code not in _SHIFT_CODE_WORDS["en"]:
        raise ValueError(f"Unknown shift code: {code!r} (expected D, N, or O)")
    return _SHIFT_CODE_WORDS[language][code]


def verbalize(value, value_type: VerbalizeType, language: Language) -> str:
    """Single dispatcher entry point - this is what prompt-assembly code
    should import, so callers don't need to know the per-type function names."""
    if value_type == "number":
        return verbalize_number(int(value), language)
    if value_type == "digits":
        return verbalize_digits(str(value), language)
    if value_type == "time":
        return verbalize_time(value, language)
    if value_type == "date":
        return verbalize_date(value, language)
    if value_type == "shift_code":
        return verbalize_shift_code(str(value), language)
    raise ValueError(f"Unknown verbalize type: {value_type}")
