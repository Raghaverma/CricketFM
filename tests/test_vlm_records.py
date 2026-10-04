import json
from pathlib import Path

import pytest

from cricketfm import taxonomy
from cricketfm.models.vlm import prompts, records

CLIP = {"clip_id": "c1", "start_frame": 0, "end_frame": 150, "contact_frame": 90, "label": "hook", "class": "other"}


def test_frames3_offsets_are_in_seconds_and_clamped():
    assert records.frames_needed(CLIP, "frames3", 60) == [78, 90, 102]       # +-0.2 s at 60 fps
    assert records.frames_needed(CLIP, "frames3", 30) == [84, 90, 96]
    edge = dict(CLIP, contact_frame=145)
    assert records.frames_needed(edge, "frames3", 60) == [133, 145, 149]
    assert records.frames_needed(CLIP, "frame", 60) == [90] and records.frames_needed(CLIP, "clip", 60) == []
    with pytest.raises(ValueError):
        records.frames_needed(CLIP, "audio", 60)


def test_record_schema_is_shared_by_evaluation_and_training():
    prompt = prompts.build(taxonomy.load(), "frames3")
    ev = records.record(CLIP, "frames3", 60, prompt, Path("m"))
    tr = records.record(CLIP, "frames3", 60, prompt, Path("m"), train=True)
    assert tr["messages"][0] == ev["messages"][0] and len(ev["messages"]) == 1
    assert [c["image"] for c in ev["messages"][0]["content"][:3]] == ["m/c1_000078.jpg", "m/c1_000090.jpg", "m/c1_000102.jpg"]
    assert json.loads(tr["messages"][1]["content"]) == {"shot": "other"}
    assert ev["prompt_id"] == prompts.prompt_id(prompt)
    with pytest.raises(ValueError):
        records.record(dict(CLIP, label="unclear", **{"class": None}), "frame", 60, prompt, Path("m"), train=True)
