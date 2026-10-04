import pytest

from cricketfm import taxonomy
from cricketfm.evaluation.shot_metrics import INVALID
from cricketfm.models.vlm import parsing, prompts

TAX = taxonomy.load("v1")


def test_v1_taxonomy():
    assert TAX.classes == ("cover_drive", "straight_drive", "pull", "cut", "flick", "sweep", "defensive", "other")
    assert TAX.to_class("hook") == "other" and TAX.to_class("cover_drive") == "cover_drive"
    assert TAX.to_class("unclear") is None
    assert TAX.bat_plane["pull"] == TAX.bat_plane["cut"] == "horizontal" and "other" not in TAX.bat_plane
    with pytest.raises(ValueError):
        TAX.to_class("helicopter_shot")


def test_prompt_lists_every_class_once_and_is_stable():
    text = prompts.build(TAX, "clip")
    for cls in TAX.classes:
        assert text.count(f"- {cls}:") == 1
    assert prompts.prompt_id(text) == prompts.prompt_id(prompts.build(TAX, "clip"))
    with pytest.raises(ValueError):
        prompts.build(TAX, "audio")


@pytest.mark.parametrize("text, label, how", [
    ('{"shot": "pull", "confidence": 0.8}', "pull", "json"),
    ('```json\n{"shot": "Cover Drive", "confidence": 0.6}\n```', "cover_drive", "json"),
    ('{"shot": "forward defence", "confidence": 0.7}', "defensive", "json_alias"),
    ('{"shot": "hook", "confidence": 0.7}', "other", "json_alias"),
    ('{"note": "thinking"} then {"shot": "sweep"}', "sweep", "json"),
])
def test_strict_reading(text, label, how):
    p = parsing.parse(text, TAX)
    assert (p.label, p.lenient_label, p.how) == (label, label, how)


def test_prose_counts_only_in_the_lenient_reading():
    p = parsing.parse("The batter plays a pull shot off the back foot.", TAX)
    assert (p.label, p.lenient_label, p.how) == (INVALID, "pull", "prose")
    p = parsing.parse('{"shot": "lofted cover drive"}', TAX)
    assert (p.label, p.lenient_label) == (INVALID, "cover_drive")


@pytest.mark.parametrize("text", [
    "It could be a pull or a cut.",                                 # two classes
    "Other than that, the clip is cut off before the stroke.",      # names "other" and "cut"
    "",
])
def test_invalid(text):
    p = parsing.parse(text, TAX)
    assert (p.label, p.lenient_label, p.how) == (INVALID, INVALID, "invalid")


def test_prose_ignores_one_word_synonyms_but_not_class_names():
    assert parsing.parse("He shapes to hook, then plays a cut.", TAX).lenient_label == "cut"


def test_confidence_reading():
    assert parsing.parse('{"shot": "cut", "confidence": 87}', TAX).confidence == pytest.approx(0.87)
    assert parsing.parse('{"shot": "cut", "confidence": "high"}', TAX).confidence is None
    assert parsing.parse('{"shot": "cut"}', TAX).confidence is None
