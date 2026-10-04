"""Shot clips: the ShotFM dataset's records, one JSON object per line (data/shots/clips.jsonl,
private). docs/ANNOTATION.md says how each field is judged.

  clip_id        unique id
  video_id       source video (AutoClipping job id)
  fingerprint    the source's content fingerprint (duplicate uploads share it)
  player_id      the batter's pseudonymous id, from the job metadata
  start_frame    first frame of the clip, as a decoded index (data/video.py), inclusive
  end_frame      last frame + 1
  contact_frame  the frame where the ball reaches the batter: bat contact, or the ball passing the
                 bat on a leave or a miss. The single-frame and three-frame conditions are cut
                 around it.
  label          fine label (configs/shots.yaml), or an excluded label such as "unclear"
  batting_hand   right | left
  view           front_on | side_on | rear | oblique
  flags          clip conditions seen at labeling, possibly none: a list from CLIP_FLAGS. With the
                 view, they are the base rates of the failure analysis (docs/PROTOCOL.md §4.4), so
                 they are tagged on every clip, before any model has seen it.
  borderline     optional, true when the label was a close call between two strokes
  annotator      who labeled it, for agreement checks
"""
from __future__ import annotations

import json
from pathlib import Path

from cricketfm.taxonomy import Taxonomy

VIEWS = ("front_on", "side_on", "rear", "oblique")
HANDS = ("right", "left")
CLIP_FLAGS = ("occlusion", "poor_framing", "motion_blur", "bat_not_visible")
FIELDS = ("clip_id", "video_id", "fingerprint", "player_id", "start_frame", "end_frame", "contact_frame",
          "label", "batting_hand", "view", "flags", "annotator")


def problems(rec: dict, taxonomy: Taxonomy) -> list[str]:
    out = [f"missing {f}" for f in FIELDS if rec.get(f) in (None, "")]
    if out:
        return out
    if not (0 <= rec["start_frame"] <= rec["contact_frame"] < rec["end_frame"]):
        out.append(f"frames must satisfy 0 <= start <= contact < end, got "
                   f"{rec['start_frame']}, {rec['contact_frame']}, {rec['end_frame']}")
    if rec["label"] not in taxonomy.fine_to_class and rec["label"] not in taxonomy.excluded:
        out.append(f"label {rec['label']!r} is not in the taxonomy")
    if rec["batting_hand"] not in HANDS:
        out.append(f"batting_hand {rec['batting_hand']!r} not in {HANDS}")
    if rec["view"] not in VIEWS:
        out.append(f"view {rec['view']!r} not in {VIEWS}")
    if not isinstance(rec["flags"], list) or set(rec["flags"]) - set(CLIP_FLAGS):
        out.append(f"flags {rec['flags']!r} must be a list from {CLIP_FLAGS}")
    if not isinstance(rec.get("borderline", False), bool):
        out.append(f"borderline {rec['borderline']!r} must be true or false")
    return out


def load(path: Path, taxonomy: Taxonomy) -> list[dict]:
    """All records, validated; raises listing every bad line. Each record gains `class` (its
    evaluated class under `taxonomy`, None for an excluded label)."""
    records, errors, seen = [], [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        errs = problems(rec, taxonomy)
        if rec.get("clip_id") in seen:
            errs.append(f"duplicate clip_id {rec['clip_id']!r}")
        seen.add(rec.get("clip_id"))
        if errs:
            errors.append(f"line {n}: {'; '.join(errs)}")
            continue
        rec["class"] = taxonomy.to_class(rec["label"])
        records.append(rec)
    if errors:
        raise ValueError(f"{path}: {len(errors)} invalid records\n" + "\n".join(errors[:20]))
    return records
