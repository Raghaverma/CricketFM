"""COCO-17: the one skeleton every pose model is scored in.

Pose models don't share an output format. ViTPose++'s COCO expert emits COCO-17 directly;
Sapiens' Goliath models emit 308 keypoints in their own order, with the wrists among the hand
points, after the hips and feet. So each model declares its native keypoint *names*, and
`Mapping.by_name` selects the COCO-17 subset by name, refusing on a missing or duplicated name.
A mapping by index would pair one model's hips with another's wrists and still produce
plausible-looking numbers.

Mapping by name fixes the format, not the definition: two training sets can annotate "left_hip"
in different places. evaluation/pose_metrics.keypoint_bias() measures that gap per keypoint
(docs/PROTOCOL.md, "Keypoint conventions").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
K = len(COCO17)
INDEX = {name: i for i, name in enumerate(COCO17)}

# pycocotools' per-keypoint OKS falloffs.
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

# Disjoint body regions for the per-region breakdowns: every keypoint is in exactly one.
GROUPS = {
    "head": ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
    "torso": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "arms": ("left_elbow", "right_elbow", "left_wrist", "right_wrist"),
    "legs": ("left_knee", "right_knee", "left_ankle", "right_ankle"),
}

# Joint angles a coach reads off a batting pose, as (end, vertex, end): elbow and knee flexion,
# and the shoulder angle between the trunk and the upper arm.
ANGLES = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
}

# Limb edges for drawing (0-based COCO-17 indices).
EDGES = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
         (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

def _mirror_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


# Index of each keypoint's mirror image (nose maps to itself).
FLIP = np.array([INDEX[_mirror_name(n)] for n in COCO17], dtype=np.int64)


def indices(names: Sequence[str]) -> np.ndarray:
    return np.array([INDEX[n] for n in names], dtype=np.int64)


@dataclass(frozen=True)
class Mapping:
    """Native skeleton -> COCO-17, built by keypoint name."""

    source: str          # the native skeleton's name, e.g. "goliath-308"
    index: np.ndarray    # (17,) native index of each COCO-17 keypoint

    @classmethod
    def by_name(cls, source: str, names: Sequence[str], aliases: dict[str, str] | None = None) -> "Mapping":
        """`aliases` maps a COCO-17 name to the native name where they differ."""
        pos: dict[str, int] = {}
        for i, name in enumerate(names):
            if name in pos:
                raise ValueError(f"{source}: keypoint name {name!r} appears twice (indices {pos[name]} and {i})")
            pos[name] = i
        aliases = aliases or {}
        wanted = [aliases.get(n, n) for n in COCO17]
        missing = [w for w in wanted if w not in pos]
        if missing:
            raise ValueError(f"{source}: no keypoint named {missing}; refusing to guess a mapping")
        return cls(source, np.array([pos[w] for w in wanted], dtype=np.int64))

    @classmethod
    def identity(cls) -> "Mapping":
        return cls("coco-17", np.arange(K))

    def __call__(self, keypoints: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(..., K_native, 2), (..., K_native) -> (..., 17, 2), (..., 17)."""
        return keypoints[..., self.index, :], scores[..., self.index]


def mirror(keypoints: np.ndarray, width: float) -> np.ndarray:
    """Reflect COCO-17 keypoints (..., 17, >=2) about the vertical centre line of a `width`-px
    frame (x -> width - 1 - x, mmpose's flip) and swap left/right labels, so the result is a
    valid COCO-17 pose of the mirrored image. Used to put left-handed batters in a
    right-hander's frame for the pose classifier."""
    out = keypoints[..., FLIP, :].copy()
    out[..., 0] = width - 1 - out[..., 0]
    return out
