"""The gate every pose adapter passes before its cricket numbers count: COCO val2017 keypoints.

    python -m cricketfm.evaluation.coco_check --model vitpose-plus-l [--images 500]

A wrong crop, a transposed decode or a scrambled keypoint mapping still produces plausible
looking keypoints. On public data with public reference numbers it shows up instead as a
collapsed AP, or as one keypoint far below the others. Checks:
  - reproduction: ViTPose++-L on the full set must land within 0.005 AP of the TensorRT repo's
    Stage 10 (0.809), which used the same processing;
  - mapping: every keypoint's PCK@0.1 (bbox) must be at least 0.5, which a mis-mapped joint
    can't reach;
  - convention: the per-keypoint bias table, where a Goliath-head Sapiens shows how far its
    keypoint definitions sit from COCO's, measured on public data before the cricket set.

Protocol, as the TensorRT repo's Stage 10: every non-crowd person with at least one labeled
keypoint, ground-truth boxes, no flip test, pycocotools keypoint AP over the evaluated images,
person score = mean joint score over joints > 0.2. Intervals resample images (people in one
image are not independent).

Outputs: results/coco_check/<model>.json (committed; COCO is public),
results/raw/coco_check/<model>.npz (keypoints, gitignored).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from cricketfm import skeleton
from cricketfm.benchmark import env, gpu_guard
from cricketfm.evaluation import pose_metrics
from cricketfm.models.pose import base
from cricketfm.paths import COCO_DIR, RAW, RESULTS, rel

TRT_STAGE10_AP = {"vitpose-plus-l": 0.809}   # TensorRT repo, results/evaluation/coco_pose.json, PyTorch FP16
AP_TOLERANCE = 0.005
MIN_KEYPOINT_PCK = 0.5
BATCH = 16


def load_instances(ann_path, n_images: int | None):
    gt = json.loads(ann_path.read_text())
    images = {im["id"]: im for im in gt["images"]}
    inst = sorted((a for a in gt["annotations"] if a["num_keypoints"] > 0 and not a["iscrowd"]),
                  key=lambda a: (a["image_id"], a["id"]))
    if n_images is not None:
        keep = sorted({a["image_id"] for a in inst})[:n_images]
        inst = [a for a in inst if a["image_id"] in set(keep)]
    return gt, images, inst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help=f"one of {sorted(base.registry())}")
    p.add_argument("--images", type=int, default=None, help="first N images only (smoke runs; no reproduction check)")
    p.add_argument("--precision", default="fp16", choices=sorted(base.DTYPES))
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    ann_path = COCO_DIR / "annotations" / "person_keypoints_val2017.json"
    gt, images, inst = load_instances(ann_path, args.images)
    gpu_guard.wait_for_no_prod()
    t0 = time.perf_counter()
    model = base.load(args.model, precision=args.precision)
    pred, scores = base.pose_people(model, [COCO_DIR / "val2017" / images[a["image_id"]]["file_name"] for a in inst],
                                    np.array([a["bbox"] for a in inst], np.float32), BATCH)
    seconds = time.perf_counter() - t0

    gt_kps = np.array([a["keypoints"] for a in inst], np.float64).reshape(-1, skeleton.K, 3)
    boxes = np.array([a["bbox"] for a in inst], np.float64)
    area = np.array([a["area"] for a in inst], np.float64)
    image_ids = np.array([a["image_id"] for a in inst])
    coco = pose_metrics.coco_ap(gt, pose_metrics.detections(image_ids, pred, scores), sorted(set(image_ids.tolist())))
    report = pose_metrics.evaluate(pred, scores, gt_kps, boxes, area, groups=image_ids, n_boot=args.n_boot)
    hits = pose_metrics.pck_hits(pred, gt_kps, pose_metrics.box_size(boxes), 0.1)
    pck01 = {name: float(np.nanmean(hits[:, j])) for j, name in enumerate(skeleton.COCO17)}

    checks = {"every_keypoint_pck@0.1_at_least": MIN_KEYPOINT_PCK,
              "keypoints_below": sorted(n for n, v in pck01.items() if v < MIN_KEYPOINT_PCK)}
    passed = not checks["keypoints_below"]
    if args.model in TRT_STAGE10_AP and args.images is None:
        ref = TRT_STAGE10_AP[args.model]
        checks["reproduction"] = {"reference_ap": ref, "ap": coco["AP"], "tolerance": AP_TOLERANCE}
        passed &= abs(coco["AP"] - ref) <= AP_TOLERANCE
    checks["passed"] = passed

    out = {"model": model.describe(), "set": {"annotations": rel(ann_path), "images": len(set(image_ids.tolist())),
                                              "persons": len(inst), "limited_to_first_images": args.images},
           "protocol": {"boxes": "ground truth", "flip_test": False, "person_score": "mean joint score over joints > 0.2",
                        "ap": "pycocotools keypoints, all area ranges, evaluated images only",
                        "intervals": f"{args.n_boot} resamples of images"},
           "coco_ap": coco, "pck@0.1_bbox_by_keypoint": pck01, "checks": checks, **report,
           "wall_seconds": round(seconds, 1), "environment": env.fingerprint()}
    name = args.model if args.images is None else f"{args.model}_first{args.images}"
    path = RESULTS / "coco_check" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    raw = RAW / "coco_check" / f"{name}.npz"
    raw.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw, ann_ids=[a["id"] for a in inst], keypoints=pred, scores=scores)
    print(f"[coco_check] {args.model}: AP {coco['AP']:.4f} (AP50 {coco['AP50']:.3f}), "
          f"mean OKS {report['metrics']['mean_oks']['value']:.4f}, {len(inst)} persons, {seconds:.0f}s; "
          f"lowest keypoint PCK@0.1 {min(pck01.values()):.3f} -> {'PASS' if passed else 'FAIL'} ({rel(path)})")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
