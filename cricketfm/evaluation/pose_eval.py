"""PoseFM: pose models on the cricket set (docs/PROTOCOL.md §3.3, §4.1).

    python -m cricketfm.evaluation.pose_eval --models vitpose-plus-l sapiens-0.3b-coco sapiens-0.3b-goliath

Every labeled person is posed from its ground-truth box by each model (models/pose: own
preprocessing, COCO-17 out), so the comparison is between pose models, not detectors. Reported
per model: every metric of pose_metrics.evaluate(), with intervals that resample source videos
(by content fingerprint); the same metrics per stratum (role, camera view, handedness, lighting,
motion bin, phase tag, crop height as the distance stratum, share of occluded joints); and the
paired difference of each model from the first one listed, on the same resampled videos.

Inputs: data/pose/annotations.json (data/pose_gt.py) and the frames beside it in images/.
Outputs: results/pose_eval/<annotations stem>.json (aggregates, committed),
results/raw/pose_eval/<model>.npz (keypoints, gitignored).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cricketfm.benchmark import env, gpu_guard
from cricketfm.data import pose_gt
from cricketfm.evaluation import pose_metrics, stats
from cricketfm.models.pose import base
from cricketfm.paths import DATA, RAW, RESULTS, rel

MIN_STRATUM = 20                       # people; smaller strata are listed but not scored
BOX_HEIGHT_EDGES = (128, 256, 512)     # px; the TensorRT repo found pose behaviour changes below 64-128 px
BOX_HEIGHT_LABELS = ("<128", "128-256", "256-512", ">=512")


def strata(gt: pose_gt.PoseGT) -> dict[str, np.ndarray]:
    """Stratum label per person, by field."""
    out = {field: gt.column(field).astype(str) for field in ("role", "view", "lighting", "motion_bin", "phase")}
    # the person's own hand when tagged; the video's otherwise (which may be "mixed" or "unknown")
    out["hand"] = np.array([str(m.get("person_hand") or m.get("hand")) for m in gt.meta])
    out["box_height_px"] = np.array(BOX_HEIGHT_LABELS)[np.digitize(gt.boxes[:, 3], BOX_HEIGHT_EDGES)]
    occluded = (gt.gt[..., 2] == 1).sum(1) / np.maximum((gt.gt[..., 2] > 0).sum(1), 1)
    out["occluded_share"] = np.where(occluded == 0, "none", np.where(occluded <= 0.25, "up to 25%", "over 25%"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="the first is the reference for paired differences")
    p.add_argument("--gt", type=Path, default=DATA / "pose" / "annotations.json")
    p.add_argument("--precision", default="fp16", choices=sorted(base.DTYPES))
    p.add_argument("--n-boot", type=int, default=stats.N_BOOT)
    p.add_argument("--out", type=Path, default=None, help="default results/pose_eval/<annotations stem>.json")
    args = p.parse_args()

    gt = pose_gt.load(args.gt)
    paths = [args.gt.parent / "images" / name for name in gt.file_names]
    groups = gt.column("fingerprint")
    by_stratum = strata(gt)
    tables, models = {}, {}
    gpu_guard.wait_for_no_prod()
    for name in args.models:
        t0 = time.perf_counter()
        model = base.load(name, precision=args.precision)
        pred, scores = base.pose_people(model, paths, gt.boxes)
        models[name] = {**model.describe(), "wall_seconds": round(time.perf_counter() - t0, 1)}
        del model
        raw = RAW / "pose_eval" / f"{name}.npz"
        raw.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw, ann_ids=gt.ann_ids, keypoints=pred, scores=scores)
        tables[name] = (pred, scores)

    report = {"set": {"annotations": rel(args.gt), "persons": int(len(gt.ann_ids)),
                      "videos": int(len(set(groups))), "players": int(len(set(gt.column("player_id"))))},
              "protocol": {"boxes": "ground truth", "flip_test": False, "intervals": f"{args.n_boot} resamples of videos",
                           "min_stratum_persons": MIN_STRATUM},
              "models": models, "results": {}, "paired_vs_first": {}}
    per_person = {}
    for name, (pred, scores) in tables.items():
        res = {"overall": pose_metrics.evaluate(pred, scores, gt.gt, gt.boxes, gt.area, groups, args.n_boot)}
        for field, labels in by_stratum.items():
            res[field] = {}
            for value in sorted(set(labels)):
                m = labels == value
                res[field][value] = ({"persons": int(m.sum()), "metrics": pose_metrics.evaluate(
                    pred[m], scores[m], gt.gt[m], gt.boxes[m], gt.area[m], groups[m], args.n_boot // 4)["metrics"]}
                    if m.sum() >= MIN_STRATUM else {"persons": int(m.sum()), "metrics": None})
        report["results"][name] = res
        per_person[name] = pose_metrics.per_person(pred, scores, gt.gt, gt.boxes, gt.area)
    ref = args.models[0]
    for name in args.models[1:]:
        report["paired_vs_first"][name] = pose_metrics.compare(per_person[ref], per_person[name], groups, args.n_boot)
    report["environment"] = env.fingerprint()

    out = args.out or RESULTS / "pose_eval" / f"{args.gt.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    for name in args.models:
        m = report["results"][name]["overall"]["metrics"]
        print(f"[pose_eval] {name:24s} AP {m['AP']['value']:.4f} {m['AP']['ci']}  PCK@0.05 "
              f"{m['pck_bbox@0.05']['value']:.4f}  occluded {m['pck_occluded']['value']:.4f}")
    print(f"[pose_eval] wrote {rel(out)}")


if __name__ == "__main__":
    main()
