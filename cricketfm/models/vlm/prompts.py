"""The zero-shot shot-classification prompt, generated from the taxonomy (configs/shots.yaml).

The prompt is part of the frozen protocol (docs/PROTOCOL.md §4.2): it is used as generated, with
no prompt search on the pilot set, which is too small to tune and test on. Every result file
records prompt_id() so a test number can be traced to the exact text that produced it. The same
prompt is used before and after LoRA, so the adaptation is the only thing that changes between
those two rows.

Input conditions (the temporal ablation, levels L1-L2):
  frame   - the contact frame alone
  frames3 - contact - dt, contact, contact + dt, in time order (primary condition for L3-L5)
  clip    - the whole clip as video
"""
from __future__ import annotations

import hashlib

from cricketfm.taxonomy import Taxonomy

INPUTS = {
    "frame": "one frame of cricket footage, taken as the ball reaches the batter",
    "frames3": "three frames of cricket footage in time order: just before, as, and just after the ball "
               "reaches the batter",
    "clip": "a short video clip of cricket footage",
}


def build(taxonomy: Taxonomy, condition: str) -> str:
    if condition not in INPUTS:
        raise ValueError(f"unknown input condition {condition!r}; expected one of {sorted(INPUTS)}")
    lines = [
        f"You are shown {INPUTS[condition]}. The batter on strike faces one delivery. They may be "
        "right- or left-handed; off side and leg side are relative to the batter, not the camera.",
        "",
        "Classify the stroke the batter attempts, whether or not they hit the ball, as exactly one of "
        "these labels:",
    ]
    lines += [f"- {cls}: {taxonomy.descriptions[cls]}" for cls in taxonomy.classes]
    lines += [
        "",
        'Answer with JSON only, no other text: {"shot": "<one label from the list>", "confidence": '
        "<your probability, from 0 to 1, that the label is correct>}",
    ]
    return "\n".join(lines)


def prompt_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
