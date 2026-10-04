"""ShotFM clips -> the conversational records a VLM is prompted and fine-tuned with.

One schema for zero-shot evaluation and LoRA training, so the two differ only in the assistant
turn, which training files have and evaluation files don't:

    {"clip_id": "...", "condition": "frames3", "prompt_id": "...",
     "messages": [{"role": "user", "content": [<media items>, {"type": "text", "text": <prompt>}]},
                  {"role": "assistant", "content": "{\"shot\": \"cover_drive\"}"}]}

Media items use the transformers chat-template form ({"type": "image", "image": path},
{"type": "video", "video": path}). The training answer is the JSON the prompt asks for, minus the
confidence: there is no true confidence to teach. After LoRA a model stops reporting one, so
calibration is measured for zero-shot only.

The frames3 condition takes contact - 0.2 s, contact and contact + 0.2 s, converted with the
source's average frame rate. The plan's +-2 frames would be +-33 ms at 60 fps: three
near-identical images.
"""
from __future__ import annotations

import json
from pathlib import Path

from cricketfm.models.vlm.prompts import INPUTS, prompt_id

FRAMES3_OFFSET_S = 0.2


def frames_needed(clip: dict, condition: str, fps: float) -> list[int]:
    """Decoded frame indices the condition shows, clamped to the clip; [] for the video condition."""
    c, lo, hi = clip["contact_frame"], clip["start_frame"], clip["end_frame"] - 1
    if condition == "frame":
        return [c]
    if condition == "frames3":
        d = max(1, round(FRAMES3_OFFSET_S * fps))
        return [min(max(c + k * d, lo), hi) for k in (-1, 0, 1)]
    if condition == "clip":
        return []
    raise ValueError(f"unknown input condition {condition!r}; expected one of {sorted(INPUTS)}")


def media(clip: dict, condition: str, fps: float, media_dir: Path) -> list[dict]:
    """Media items for one clip: frames as <clip_id>_<frame>.jpg, the clip as <clip_id>.mp4."""
    if condition == "clip":
        return [{"type": "video", "video": str(Path(media_dir) / f"{clip['clip_id']}.mp4")}]
    return [{"type": "image", "image": str(Path(media_dir) / f"{clip['clip_id']}_{i:06d}.jpg")}
            for i in frames_needed(clip, condition, fps)]


def record(clip: dict, condition: str, fps: float, prompt: str, media_dir: Path, train: bool = False) -> dict:
    messages = [{"role": "user", "content": media(clip, condition, fps, media_dir) + [{"type": "text", "text": prompt}]}]
    if train:
        if clip.get("class") is None:
            raise ValueError(f"{clip['clip_id']}: no evaluated class (excluded label {clip.get('label')!r})")
        messages.append({"role": "assistant", "content": json.dumps({"shot": clip["class"]})})
    return {"clip_id": clip["clip_id"], "condition": condition, "prompt_id": prompt_id(prompt), "messages": messages}
