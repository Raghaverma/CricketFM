"""PoseFM ground truth: a COCO keypoint file with CricketFM's metadata on every image.

    python -m cricketfm.data.pose_gt merge <tool export.json> [--sampled data/pose/to_annotate.json]

Plain COCO (images, annotations, categories), so any COCO tool reads it. Each image also
carries the fields data/sampling.py writes: video_id, fingerprint, player_id, frame_index
(decoded index), role (striker | bowler), the per-video fields filled in data/videos.csv (view,
hand, lighting) and the sampling stratum (cell, motion_bin). Annotation tools drop those custom
fields on export (CVAT does), so `merge` copies them back from the sampling file by file name,
validates the result and writes data/pose/annotations.json. Each annotation is one person: 17
keypoints with COCO visibility (0 not labeled, 1 labeled but occluded, 2 visible), the person's
box, and the annotator's phase and hand tags (docs/ANNOTATION.md), which CVAT exports as the shape
attributes `attributes.phase` and `attributes.hand`. A missing area defaults to box area * 0.53 (pose_metrics.AREA_FROM_BOX).
"""
from __future__ import annotations

import argparse
import json
from pathlib import PurePath
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cricketfm import skeleton
from cricketfm.evaluation.pose_metrics import AREA_FROM_BOX

IMAGE_FIELDS = ("video_id", "fingerprint", "player_id", "frame_index", "role")


@dataclass
class PoseGT:
    coco: dict                 # normalized COCO dict (pycocotools reads it as is)
    ann_ids: np.ndarray        # (N,) persons with at least one labeled joint, in file order
    image_ids: np.ndarray      # (N,)
    file_names: list[str]
    gt: np.ndarray             # (N, 17, 3)
    boxes: np.ndarray          # (N, 4) xywh
    area: np.ndarray           # (N,)
    meta: list[dict]           # (N,) the person's image metadata

    def column(self, field: str) -> np.ndarray:
        return np.array([m.get(field) for m in self.meta])


def _attribute(ann: dict, name: str) -> str | None:
    return (ann.get("attributes") or {}).get(name) or ann.get(name) or None


def load(path: Path, require_metadata: bool = True) -> PoseGT:
    coco = json.loads(Path(path).read_text())
    images = {im["id"]: im for im in coco["images"]}
    errors = []
    if require_metadata:
        for im in coco["images"]:
            missing = [f for f in IMAGE_FIELDS if im.get(f) in (None, "")]
            if missing:
                errors.append(f"image {im['id']}: missing {missing}")
    keep = []
    for a in coco["annotations"]:
        kps = np.asarray(a.get("keypoints", []), np.float64)
        if kps.size != 3 * skeleton.K:
            errors.append(f"annotation {a['id']}: {kps.size} keypoint values, expected {3 * skeleton.K}")
            continue
        kps = kps.reshape(skeleton.K, 3)
        if not np.isin(kps[:, 2], (0, 1, 2)).all():
            errors.append(f"annotation {a['id']}: visibility flags must be 0, 1 or 2")
        labeled = int((kps[:, 2] > 0).sum())
        if "num_keypoints" in a and a["num_keypoints"] != labeled:
            errors.append(f"annotation {a['id']}: num_keypoints {a['num_keypoints']} != {labeled} labeled")
        a["num_keypoints"] = labeled
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            errors.append(f"annotation {a['id']}: empty box {a['bbox']}")
        a.setdefault("area", w * h * AREA_FROM_BOX)
        a.setdefault("iscrowd", 0)
        a.setdefault("category_id", 1)
        if a["image_id"] not in images:
            errors.append(f"annotation {a['id']}: unknown image {a['image_id']}")
        elif labeled:
            keep.append((a, kps))
    if errors:
        raise ValueError(f"{path}: {len(errors)} problems\n" + "\n".join(errors[:20]))
    return PoseGT(
        coco=coco,
        ann_ids=np.array([a["id"] for a, _ in keep]),
        image_ids=np.array([a["image_id"] for a, _ in keep]),
        file_names=[images[a["image_id"]]["file_name"] for a, _ in keep],
        gt=np.stack([k for _, k in keep]) if keep else np.zeros((0, skeleton.K, 3)),
        boxes=np.array([a["bbox"] for a, _ in keep], np.float64).reshape(-1, 4),
        area=np.array([a["area"] for a, _ in keep], np.float64),
        meta=[{**{k: v for k, v in images[a["image_id"]].items() if k not in ("id", "file_name")},
               "phase": _attribute(a, "phase"), "person_hand": _attribute(a, "hand")} for a, _ in keep],
    )


def merge(export: dict, sampled: dict) -> dict:
    """Copy CricketFM's image metadata from the sampling file into a tool's COCO export, matching
    images by file name (tools may prefix directories), and keep only the export's people."""
    by_name = {PurePath(im["file_name"]).name: im for im in sampled["images"]}
    missing = [im["file_name"] for im in export["images"] if PurePath(im["file_name"]).name not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} exported images not in the sampling file, e.g. {missing[:3]}")
    images = []
    for im in export["images"]:
        meta = by_name[PurePath(im["file_name"]).name]
        images.append({**meta, "id": im["id"], "file_name": meta["file_name"]})
    return {"images": images, "annotations": export["annotations"], "categories": sampled["categories"]}


def main() -> None:
    from cricketfm.paths import DATA, rel

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("export", type=Path)
    m.add_argument("--sampled", type=Path, default=DATA / "pose" / "to_annotate.json")
    m.add_argument("--out", type=Path, default=DATA / "pose" / "annotations.json")
    args = p.parse_args()
    merged = merge(json.loads(args.export.read_text()), json.loads(args.sampled.read_text()))
    args.out.write_text(json.dumps(merged, indent=1))
    gt = load(args.out)
    roles = {r: int((gt.column("role") == r).sum()) for r in sorted(set(gt.column("role")))}
    tagged = sum(m["phase"] is not None for m in gt.meta)
    print(f"[pose_gt] {len(gt.ann_ids)} labeled people on {len(merged['images'])} images {roles}, "
          f"{tagged} with a phase tag -> {rel(args.out)}")


if __name__ == "__main__":
    main()
