"""Scenario classification: keyword fast path, the local-model reason guard,
and the lean (no-rationale) local schema. See services/scenario_engine.py."""

import pytest

from local_ai import llm_engine as local_llm_engine
from services import scenario_engine
from services.scenario_engine import fast_classify, is_real_reason

SHIFT_BRANCHES = ["yes", "no", "repeat", "dispute", "already_confirmed", "supervisor", "wrong_number"]
FACTS = {"employee_name": "Amit", "date": "kal", "time": "8 baje", "shift_name": "Day", "terminal": "CT3"}


@pytest.mark.parametrize(
    "utterance",
    ["हाँ", "हाँ जी", "जी हाँ", "हां जी, आ जाऊँगा।", "आ रहा हूँ", "ठीक है", "जी", "yes", "ok", "I will come", "haan ji aa jaunga"],
)
def test_fast_path_yes(utterance):
    assert fast_classify(utterance, SHIFT_BRANCHES) == "yes"


@pytest.mark.parametrize(
    "utterance",
    ["नहीं", "जी नहीं", "नहीं, मैं कल नहीं आ पाऊँगा।", "नहीं आऊंगा", "नहीं आ सकता भाई", "no", "nahi aa paunga", "I can't come"],
)
def test_fast_path_no(utterance):
    assert fast_classify(utterance, SHIFT_BRANCHES) == "no"


@pytest.mark.parametrize(
    "utterance",
    [
        "हाँ नहीं आ पाऊंगा यार, तबीयत खराब है",  # negation trap - both polarities
        "नहीं, बुखार है",  # a reason - the LLM should extract it
        "कौन सा टर्मिनल?",  # a question
        "यह मेरी ड्यूटी नहीं है",  # a dispute
        "",
    ],
)
def test_fast_path_defers_ambiguous_to_llm(utterance):
    assert fast_classify(utterance, SHIFT_BRANCHES) is None


def test_fast_path_reminder_acknowledgement():
    reminder = ["acknowledged", "already_punched", "technical_issue", "supervisor", "wrong_number"]
    assert fast_classify("ठीक है", reminder) == "acknowledged"
    assert fast_classify("हाँ कर लूँगा", reminder) is None  # unknown verb -> LLM
    assert fast_classify("नहीं", reminder) is None  # negative needs the LLM to pick a branch


def test_fast_path_skipped_without_yes_no_or_ack_branches():
    assert fast_classify("हाँ", ["still_working", "already_punched"]) is None


def test_every_scenario_has_natural_lines_and_no_ivr_phrasing():
    """No "please answer yes or no", every language has an unclear
    re-prompt and a silence check-in, and no English supervisor-name
    fallback leaks into a Hindi/Gujarati sentence."""
    for scenario_id in ["shift_confirm", "absence_priority", "ot_request", "punch_in_remind", "punch_out_remind"]:
        for lang in ["hi", "en", "gu"]:
            cfg = scenario_engine.get_language_config(scenario_id, lang)
            lines = [cfg["opening"], cfg["attempt_2_opening"], cfg["unclear_reprompt"],
                     cfg["silence_checkin"], cfg["silence_reply"]]
            for b in cfg["branches"].values():
                lines += [b["reply"], b.get("after_collect_reply", "")]
            text = " ".join(lines)
            assert "उत्तर दें" not in text and "answer yes or no" not in text.lower(), (scenario_id, lang)
            assert "{supervisor_name}" not in text, (scenario_id, lang)
            assert all(line.strip() for line in lines[:5]), (scenario_id, lang)


def test_attendance_scenarios_ask_for_reason_on_no():
    for scenario_id in ["shift_confirm", "absence_priority"]:
        for lang in ["hi", "en", "gu"]:
            no = scenario_engine.get_language_config(scenario_id, lang)["branches"]["no"]
            assert no.get("collect") == "absence_reason", (scenario_id, lang)
            assert no.get("after_collect_reply"), (scenario_id, lang)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("कल नहीं आ पाऊँगा।", False),  # observed live from qwen2.5 as a "reason"
        ("नहीं आ सकता", False),
        ("बुखार है", True),
        ("मेरी तबीयत खराब है", True),
        ("wife is in hospital", True),
        ("", False),
    ],
)
def test_is_real_reason(text, expected):
    assert is_real_reason(text) is expected


@pytest.fixture
def local_backend(monkeypatch):
    monkeypatch.setenv("SCENARIO_LLM_BACKEND", "local")


async def test_fast_path_skips_the_llm_entirely(local_backend, monkeypatch):
    async def must_not_be_called(**_):
        raise AssertionError("LLM called for a plain yes/no")

    monkeypatch.setattr(local_llm_engine, "classify_structured", must_not_be_called)

    result = await scenario_engine.classify_utterance("shift_confirm", "hi", "नहीं आ पाऊँगा", FACTS)

    assert result.branch == "no"
    assert result.source == "fast"
    assert result.volunteered_reason is None


async def test_restated_refusal_is_not_treated_as_a_reason(local_backend, monkeypatch):
    """Live failure: the model flagged a reason for a plain refusal, which
    skipped the reason question. A refusal-only utterance never counts."""

    async def fake(*, system_prompt, user_text, schema, max_tokens=400):
        return {"branch": "no", "gave_reason": True}

    monkeypatch.setattr(local_llm_engine, "classify_structured", fake)

    result = await scenario_engine.classify_utterance(
        "shift_confirm", "hi", "देखिए सर, मैं कल नहीं आ पाऊँगा यार", FACTS
    )

    assert result.branch == "no"
    assert result.source == "llm"
    assert result.volunteered_reason is None


async def test_gave_reason_saves_the_callers_own_words(local_backend, monkeypatch):
    async def fake(*, system_prompt, user_text, schema, max_tokens=400):
        return {"branch": "no", "gave_reason": True}

    monkeypatch.setattr(local_llm_engine, "classify_structured", fake)
    utterance = "नहीं आ पाऊँगा, मेरी पत्नी अस्पताल में है"

    result = await scenario_engine.classify_utterance("shift_confirm", "hi", utterance, FACTS)

    assert result.volunteered_reason == utterance


async def test_gujarati_refusal_takes_the_fast_path_and_asks_reason(local_backend, monkeypatch):
    """Live failure on a Gujarati call: the LLM invented a "reason" from the
    opening's date phrase. The plain refusal now never reaches the LLM."""

    async def must_not_be_called(**_):
        raise AssertionError("LLM called for a plain Gujarati no")

    monkeypatch.setattr(local_llm_engine, "classify_structured", must_not_be_called)
    facts = {"employee_name": "અમિત", "date": "કાલે", "time": "સવારે આઠ", "shift_name": "ડે", "terminal": "CT3"}

    result = await scenario_engine.classify_utterance("shift_confirm", "gu", "ના, હું કાલે નહીં આવી શકું", facts)

    assert result.branch == "no"
    assert result.volunteered_reason is None


async def test_local_backend_uses_lean_schema_without_rationale(local_backend, monkeypatch):
    seen = {}

    async def fake(*, system_prompt, user_text, schema, max_tokens=400):
        seen["schema"], seen["max_tokens"] = schema, max_tokens
        return {"branch": "repeat"}

    monkeypatch.setattr(local_llm_engine, "classify_structured", fake)

    result = await scenario_engine.classify_utterance("shift_confirm", "hi", "कौन सा टर्मिनल बोला?", FACTS)

    assert "rationale" not in seen["schema"]["properties"]
    assert "detected_language" not in seen["schema"]["properties"]
    assert "volunteered_reason" not in seen["schema"]["properties"]
    assert seen["max_tokens"] <= 40
    assert result.branch == "repeat"
    assert result.detected_language == "hi"


@pytest.mark.parametrize(
    "utterance",
    ["ના, હું નથી આવતો.", "ના, હું કાલે નહીં આવી શકું", "नहीं, मैं नहीं आता", "I can't make it"],
)
def test_fast_path_no_live_call_variants(utterance):
    """"ના, હું નથી આવતો." is from a live Gujarati call where the unknown verb
    sent it to the LLM and the refusal itself was saved as the reason."""
    assert fast_classify(utterance, SHIFT_BRANCHES) == "no"
    assert not is_real_reason(utterance)


def test_scripts_use_feminine_first_person_to_match_the_female_voice():
    """The default Sarvam speaker (config.py) is female; a masculine
    first-person verb in a Hindi/Gujarati line would be spoken wrong."""
    masculine = ["बोल रहा हूँ", "कर लूँगा", "समझ गया।", "समझ गया,", "देता हूँ", "लेता हूँ", "चाहता हूँ",
                 "लिख रहा हूँ", "दे पाऊँगा", "સમજી ગયો", "રહ્યો છું"]
    for scenario_id in ["shift_confirm", "absence_priority", "ot_request", "punch_in_remind", "punch_out_remind"]:
        for lang in ["hi", "gu"]:
            cfg = scenario_engine.get_language_config(scenario_id, lang)
            lines = [cfg["opening"], cfg["attempt_2_opening"], cfg["unclear_reprompt"], cfg["silence_reply"]]
            for b in cfg["branches"].values():
                lines += [b["reply"], b.get("after_collect_reply", "")]
            for form in masculine:
                assert not any(form in line for line in lines), (scenario_id, lang, form)


@pytest.mark.parametrize("utterance", ["नहीं नहीं मैं आ जाऊंगा", "नहीं नहीं, मैं जाऊंगा"])
def test_doubled_refusal_then_coming_is_a_yes_correction(utterance):
    """Benchmark finding: local STT drops the comma, and Qwen then read this
    correction as a refusal on every voice."""
    assert fast_classify(utterance, SHIFT_BRANCHES) == "yes"


def test_doubled_refusal_that_stays_negative_is_still_no():
    assert fast_classify("नहीं नहीं मैं नहीं आ पाऊंगा", SHIFT_BRANCHES) == "no"


@pytest.mark.parametrize("utterance", ["नहीं मैं आ जाऊंगा", "नहीं, कल आ जाऊँगा", "ના, હું આવીશ"])
def test_single_refusal_word_then_coming_is_a_yes_correction(utterance):
    """Simulator finding: local STT heard "नहीं नहीं, मैं आ जाऊँगा" as "नहीं
    मैं आ जाऊंगा", and Qwen called it a refusal."""
    assert fast_classify(utterance, SHIFT_BRANCHES) == "yes"


@pytest.mark.parametrize(
    "utterance",
    ["नहीं आऊंगा", "नहीं मैं आ पाऊंगा", "नहीं मैं नहीं आ पाऊंगा", "नहीं मैं कल ही आ पाऊंगा", "ના, હું નથી આવતો"],
)
def test_refusals_near_the_correction_rule_stay_no(utterance):
    assert fast_classify(utterance, SHIFT_BRANCHES) == "no"


def test_misheard_negation_particle_does_not_count_as_a_reason():
    """Simulator finding: local STT heard "कल नहीं आ पाऊँगा" as "कल ही आ
    पाऊंगा"; the stray "ही" counted as reason content and the bot closed the
    call without asking why."""
    utterance = "नहीं मैं कल ही आ पाऊंगा"
    assert fast_classify(utterance, SHIFT_BRANCHES) == "no"
    assert not is_real_reason(utterance)


@pytest.mark.parametrize("utterance", ["હું", "मैं", "हूँ", "कल"])
def test_filler_only_fragment_is_unclear_not_a_guess(utterance):
    """Benchmark finding: a noisy fragment like "હું" went to Qwen, which
    guessed a confident "no". Nothing in it answers the question."""
    assert fast_classify(utterance, SHIFT_BRANCHES) == "unclear"


async def test_restated_refusal_while_collecting_gets_one_reask_not_recorded_as_reason(local_backend):
    """Simulator finding: "ના" alone was answered with "why?", then the rest
    of the refusal ("હું કાલે નહીં આવી શકું") was saved as the reason."""
    facts = {"employee_name": "અમિત", "date": "કાલે", "time": "સવારે આઠ", "shift_name": "ડે", "terminal": "CT3"}
    s = scenario_engine.ScenarioSession(scenario_id="shift_confirm", language="gu", facts=facts)
    s.opening_line()

    first = await s.handle_utterance("ના")
    assert first.branch == "no" and s.collecting == "absence_reason"

    reask = await s.handle_utterance("હું કાલે નહીં આવી શકું")
    assert reask.branch == "reason_reprompt" and not reask.end_call
    assert reask.reply == s.lang_cfg["reason_reprompt"]
    assert "absence_reason" not in s.collected

    done = await s.handle_utterance("મારી તબિયત સારી નથી")
    assert done.end_call and done.outcome == "absent_with_reason"
    assert s.collected["absence_reason"] == "મારી તબિયત સારી નથી"


async def test_second_reasonless_answer_is_recorded_so_the_call_cannot_loop(local_backend):
    s = scenario_engine.ScenarioSession(scenario_id="shift_confirm", language="hi", facts=FACTS)
    s.opening_line()
    await s.handle_utterance("नहीं")
    await s.handle_utterance("नहीं आ पाऊँगा")
    done = await s.handle_utterance("बस नहीं आ पाऊँगा")
    assert done.end_call
    assert s.collected["absence_reason"] == "बस नहीं आ पाऊँगा"


# --- the reason question after a "no" is compulsory -------------------------

from services import scenario_engine as _se  # noqa: E402


def _fake_classify(branch, volunteered=None):
    async def classify(*a, **k):
        return _se.Classification(branch=branch, confidence=1.0, detected_language="hi",
                                  rationale="", volunteered_reason=volunteered, source="llm")
    return classify


FACTS = {"employee_name": "Amit", "date": "2 अगस्त", "time": "सुबह 8", "shift_name": "Day",
         "terminal": "CT3", "supervisor_name": "Suresh"}


async def test_no_always_asks_for_the_reason_even_if_model_claims_one_was_given(monkeypatch):
    """Live failure: the model flagged a reason for "nahi aa paunga yaar", the
    question was skipped and the call closed with a fake reason."""
    monkeypatch.setattr(_se, "classify_utterance", _fake_classify("no", volunteered="नहीं आ पाऊँगा यार"))
    s = _se.ScenarioSession("shift_confirm", "hi", dict(FACTS))
    r = await s.handle_utterance("नहीं आ पाऊँगा यार")
    assert r.end_call is False and r.outcome == "awaiting_absence_reason"
    assert s.collecting == "absence_reason"


async def test_reason_given_after_the_question_is_recorded_and_call_ends(monkeypatch):
    monkeypatch.setattr(_se, "classify_utterance", _fake_classify("no"))
    s = _se.ScenarioSession("shift_confirm", "hi", dict(FACTS))
    await s.handle_utterance("नहीं")
    r = await s.handle_utterance("मेरी तबीयत खराब है")
    assert r.end_call is True and r.outcome == "absent_with_reason"
    assert r.collected["absence_reason"] == "मेरी तबीयत खराब है"


async def test_answer_without_a_reason_is_asked_again_up_to_the_limit(monkeypatch):
    monkeypatch.setattr(_se, "classify_utterance", _fake_classify("no"))
    s = _se.ScenarioSession("shift_confirm", "hi", dict(FACTS))
    await s.handle_utterance("नहीं")
    for _ in range(_se.MAX_REASON_REASKS):
        r = await s.handle_utterance("कल नहीं आ पाऊँगा")  # restates the refusal, no reason
        assert r.end_call is False and r.branch == "reason_reprompt"
    r = await s.handle_utterance("कल नहीं आ पाऊँगा")  # limit reached: never loops forever
    assert r.end_call is True


async def test_volunteered_reason_is_the_fallback_when_the_answer_adds_nothing(monkeypatch):
    monkeypatch.setattr(_se, "classify_utterance", _fake_classify("no", volunteered="मेरी तबीयत खराब है"))
    s = _se.ScenarioSession("shift_confirm", "hi", dict(FACTS))
    r = await s.handle_utterance("नहीं, मेरी तबीयत खराब है")
    assert r.end_call is False  # still asked
    for _ in range(_se.MAX_REASON_REASKS):
        await s.handle_utterance("हाँ")
    r = await s.handle_utterance("हाँ")
    assert r.end_call is True and r.collected["absence_reason"] == "मेरी तबीयत खराब है"


@pytest.mark.parametrize(
    "utterance",
    [
        "ના, નહીં આવી શકું.",  # live failure: recorded as CONFIRMED, reason never asked
        "ના નહીં આવી શકું",
        "ના, હું કાલે નહીં આવી શકું",
        "નહીં નહીં, આવી શકું નહીં",
        "नहीं नहीं, नहीं आ पाऊँगा",
        "नहीं, नहीं आ सकता",
        "no no, I can't come",
    ],
)
def test_double_refusal_is_a_no_not_a_correction(utterance):
    assert fast_classify(utterance, SHIFT_BRANCHES) == "no"


@pytest.mark.parametrize(
    "utterance",
    ["नहीं नहीं, मैं आ जाऊँगा", "नहीं मैं आ जाऊंगा", "નહીં નહીં, હું આવીશ", "no no, I'll come"],
)
def test_real_corrections_still_mean_yes(utterance):
    assert fast_classify(utterance, SHIFT_BRANCHES) == "yes"
