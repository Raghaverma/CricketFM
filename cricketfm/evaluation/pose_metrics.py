"""Pose accuracy against human labels, scored in COCO-17 (docs/PROTOCOL.md §4.1, PoseFM).

Protocol: top-down with ground-truth person boxes, so every prediction belongs to exactly one
annotation and nothing needs matching. Under that protocol COCO's keypoint AP has a closed form,
matched_ap(), which equals pycocotools' COCOeval whenever each image has one annotated person
(the cricket set annotates the striker only), and is fast enough to bootstrap.
tests/test_pose_metrics.py checks the two agree. coco_ap() runs pycocotools itself, for sets with
several people per image (COCO val2017).

Arrays, for N persons:
  pred    (N, 17, 2)  predicted keypoints, frame px
  scores  (N, 17)     predicted keypoint confidences
  gt      (N, 17, 3)  labels x, y, v; COCO visibility: 0 not labeled, 1 labeled but occluded,
                      2 labeled and visible
  boxes   (N, 4)      ground-truth person boxes, xywh
  area    (N,)        OKS scale: COCO's segment area where it exists, else box area * 0.53

Metrics:
  - OKS per person (pycocotools' formula), and AP / AP50 / AP75 over it. A person's score for
    the AP ranking is the mean joint score over joints scoring > 0.2 (mmpose's top-down
    rescoring, as in the TensorRT repo's Stage 10).
  - PCK@a: a labeled joint is correct within a * max(box w, h) of its label (mmpose's bbox
    normalization), or within a * torso size (the mean of the two shoulder-to-opposite-hip
    diagonals), which doesn't shrink when a sweep crouches the box.
  - Breakdowns of PCK@0.05 (bbox): per keypoint, per body region (skeleton.GROUPS), and visible
    (v=2) vs occluded (v=1) joints.
  - Joint-angle error in degrees, in the image plane (skeleton.ANGLES), where all three joints
    are labeled.
  - Keypoint bias: each keypoint's mean signed offset from its label, in units of box size. A
    model trained on a different keypoint definition shows a consistent offset (a convention
    gap) rather than scatter. Vertical offsets (e.g. hip height) show up reliably on upright
    batters; lateral ones partly cancel across facing directions.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np

from cricketfm import skeleton
from cricketfm.evaluation import stats

AREA_FROM_BOX = 0.53     # person segment area / box area: mmpose's fallback when an annotation has no area
KPT_SCORE_THR = 0.2
OKS_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
PCK = {"bbox@0.05": ("bbox", 0.05), "bbox@0.1": ("bbox", 0.1), "torso@0.2": ("torso", 0.2)}
PRIMARY_PCK = "bbox@0.05"


def _nanmean(x: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(x)) if np.size(x) else float("nan")


# ---- per-person quantities ----------------------------------------------------------------------

def oks(pred: np.ndarray, gt: np.ndarray, area: np.ndarray, joints: np.ndarray | None = None) -> np.ndarray:
    """COCO OKS of each prediction against its own label (pycocotools' formula); with `joints`,
    over that subset only. NaN for a person with no labeled joint in the subset."""
    var = (skeleton.SIGMAS * 2) ** 2
    d2 = ((pred - gt[..., :2]) ** 2).sum(-1)
    e = d2 / var / (area[:, None] + np.spacing(1)) / 2
    vis = gt[..., 2] > 0
    if joints is not None:
        vis = vis & np.isin(np.arange(skeleton.K), joints)[None]
    n = vis.sum(1)
    return np.where(n > 0, (np.exp(-e) * vis).sum(1) / np.maximum(n, 1), np.nan)


def box_size(boxes: np.ndarray) -> np.ndarray:
    return np.maximum(boxes[:, 2], boxes[:, 3]).astype(np.float64)


def torso_size(gt: np.ndarray) -> np.ndarray:
    """Mean of the two shoulder-to-opposite-hip diagonals, from the labels; NaN unless all four
    torso joints are labeled."""
    ls, rs, lh, rh = (skeleton.INDEX[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    d1 = np.linalg.norm(gt[:, ls, :2] - gt[:, rh, :2], axis=-1)
    d2 = np.linalg.norm(gt[:, rs, :2] - gt[:, lh, :2], axis=-1)
    ok = (gt[:, [ls, rs, lh, rh], 2] > 0).all(1)
    return np.where(ok, (d1 + d2) / 2, np.nan)


def pck_hits(pred: np.ndarray, gt: np.ndarray, norm: np.ndarray, alpha: float) -> np.ndarray:
    """(N, 17): 1.0 where a labeled joint is closer than alpha * norm to its label (strictly, as
    mmpose), 0.0 where it isn't, NaN where the joint is unlabeled or the person has no norm."""
    d = np.linalg.norm(pred - gt[..., :2], axis=-1)
    with np.errstate(invalid="ignore"):
        hit = (d < alpha * norm[:, None]).astype(np.float64)
    labeled = (gt[..., 2] > 0) & np.isfinite(norm)[:, None]
    return np.where(labeled, hit, np.nan)


def person_scores(scores: np.ndarray, thr: float = KPT_SCORE_THR) -> np.ndarray:
    m = scores > thr
    s = np.where(m, scores, 0).sum(1) / np.maximum(m.sum(1), 1)
    return np.where(m.any(1), s, 0.0)


def _angle(xy: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Interior angle at b in degrees, as atan2(|u x v|, u . v): unlike arccos of a cosine, exact
    near 180 degrees, where a batter's straight arms and legs put many joints."""
    u, v = xy[:, a] - xy[:, b], xy[:, c] - xy[:, b]
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return np.degrees(np.arctan2(np.abs(cross), (u * v).sum(-1)))


def angle_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N, len(skeleton.ANGLES)) absolute error of each joint angle in degrees; NaN unless all
    three of its joints are labeled."""
    out = np.full((len(pred), len(skeleton.ANGLES)), np.nan)
    for j, names in enumerate(skeleton.ANGLES.values()):
        ia, ib, ic = skeleton.indices(names)
        ok = (gt[:, [ia, ib, ic], 2] > 0).all(1)
        if ok.any():
            out[ok, j] = np.abs(_angle(pred[ok], ia, ib, ic) - _angle(gt[ok, :, :2], ia, ib, ic))
    return out


def keypoint_bias(pred: np.ndarray, gt: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per keypoint over its labeled instances: mean signed (dx, dy) offset from the label (17, 2)
    and mean absolute distance (17,), both in units of box size. NaN for a never-labeled keypoint."""
    d = (pred - gt[..., :2]) / box_size(boxes)[:, None, None]
    lab = gt[..., 2] > 0
    n = lab.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = np.where(lab[..., None], d, 0).sum(0) / n[:, None]
        err = np.where(lab, np.linalg.norm(d, axis=-1), 0).sum(0) / n
    return np.where(n[:, None] > 0, bias, np.nan), np.where(n > 0, err, np.nan)


# ---- aggregates ---------------------------------------------------------------------------------

def matched_ap(oks_values: np.ndarray, person_score: np.ndarray) -> dict[str, float]:
    """COCO keypoint AP / AP50 / AP75 when each prediction is paired with its own annotation.
    The steps of COCOeval.accumulate(): rank every prediction by score (stable sort); one is a
    true positive at threshold t when its OKS >= t; precision made non-increasing from the right;
    read at 101 recall points, 0 past the highest recall reached; averaged over points, then t."""
    n = len(oks_values)
    if n == 0:
        return {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}
    o = oks_values[np.argsort(-person_score, kind="mergesort")]
    aps = np.empty(len(OKS_THRESHOLDS))
    for i, t in enumerate(OKS_THRESHOLDS):
        hit = o >= t
        tp = np.cumsum(hit)
        fp = np.cumsum(~hit)
        recall = tp / n
        precision = np.maximum.accumulate((tp / (tp + fp + np.spacing(1)))[::-1])[::-1]
        at = np.searchsorted(recall, RECALL_POINTS, side="left")
        aps[i] = np.where(at < n, precision[np.minimum(at, n - 1)], 0.0).mean()
    return {"AP": float(aps.mean()), "AP50": float(aps[0]), "AP75": float(aps[5])}


def per_person(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, boxes: np.ndarray,
               area: np.ndarray) -> dict[str, np.ndarray]:
    """The per-person quantities every aggregate is a function of (what the bootstrap resamples).
    Persons without a single labeled joint must be removed beforehand, as COCOeval ignores them."""
    if not (gt[..., 2] > 0).any(1).all():
        raise ValueError("every person needs at least one labeled joint (COCOeval ignores the others)")
    norms = {"bbox": box_size(boxes), "torso": torso_size(gt)}
    t = {"oks": oks(pred, gt, area), "score": person_scores(scores), "visibility": gt[..., 2].astype(np.int64),
         "angle_err": angle_errors(pred, gt)}
    for key, (norm, alpha) in PCK.items():
        t[key] = pck_hits(pred, gt, norms[norm], alpha)
    return t


def aggregate(t: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict[str, float]:
    """Scalar metrics over persons idx (all if None; repeats allowed, for the bootstrap)."""
    if idx is not None:
        t = {k: v[idx] for k, v in t.items()}
    out = matched_ap(t["oks"], t["score"])
    out["mean_oks"] = _nanmean(t["oks"])
    for key in PCK:
        out[f"pck_{key}"] = _nanmean(t[key])
    hits = t[PRIMARY_PCK]
    for region, names in skeleton.GROUPS.items():
        out[f"pck_{region}"] = _nanmean(hits[:, skeleton.indices(names)])
    out["pck_visible"] = _nanmean(hits[t["visibility"] == 2])
    out["pck_occluded"] = _nanmean(hits[t["visibility"] == 1])
    out["angle_mae_deg"] = _nanmean(t["angle_err"])
    return out


def evaluate(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, boxes: np.ndarray,
             area: np.ndarray, groups=None, n_boot: int = stats.N_BOOT, seed: int = 0) -> dict:
    """Every PoseFM metric for one model on one set: point estimates, cluster-bootstrap
    intervals (groups = clip or video id per person; None = persons independent), and the
    per-keypoint tables."""
    t = per_person(pred, scores, gt, boxes, area)
    hits = t[PRIMARY_PCK]
    bias, err = keypoint_bias(pred, gt, boxes)
    vis = t["visibility"]
    report = {
        "persons": int(len(gt)),
        "joints_labeled": int((vis > 0).sum()),
        "joints_occluded": int((vis == 1).sum()),
        "metrics": stats.bootstrap(lambda i: aggregate(t, i), len(gt), groups, n_boot, seed),
        "per_keypoint": {name: {"pck": _nanmean(hits[:, j]), "labeled": int((vis[:, j] > 0).sum()),
                                "bias_box_units": bias[j].round(5).tolist(), "error_box_units": float(err[j])}
                         for j, name in enumerate(skeleton.COCO17)},
        "angle_mae_deg": {name: _nanmean(t["angle_err"][:, j]) for j, name in enumerate(skeleton.ANGLES)},
    }
    if groups is not None:
        report["clusters"] = stats.describe_clusters(groups)
    return report


def compare(ref: dict[str, np.ndarray], cand: dict[str, np.ndarray], groups=None,
            n_boot: int = stats.N_BOOT, seed: int = 0) -> dict:
    """Paired differences (cand - ref) of every aggregate, two models on the same persons, given
    the per_person() tables of each."""
    return stats.paired(lambda i: aggregate(ref, i), lambda i: aggregate(cand, i), len(ref["oks"]),
                        groups, n_boot, seed)


# ---- pycocotools ----------------------------------------------------------------------------------

def detections(image_ids, pred: np.ndarray, scores: np.ndarray, box_scores: np.ndarray | None = None) -> list[dict]:
    """COCO results-format keypoint detections, scored by person_scores()."""
    ps = person_scores(scores) * (1.0 if box_scores is None else box_scores)
    return [{"image_id": int(i), "category_id": 1, "score": float(p),
             "keypoints": np.concatenate([k, s[:, None]], 1).ravel().round(3).tolist()}
            for i, k, s, p in zip(image_ids, pred, scores, ps)]


def coco_ap(gt: dict, dets: list[dict], img_ids=None) -> dict[str, float]:
    """pycocotools' keypoint AP/AR on a COCO-format ground truth dict."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        cd = cg.loadRes(dets)
        ev = COCOeval(cg, cd, "keypoints")
        ev.params.imgIds = sorted(img_ids if img_ids is not None else {d["image_id"] for d in dets})
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    names = ("AP", "AP50", "AP75", "AP_M", "AP_L", "AR", "AR50", "AR75", "AR_M", "AR_L")
    return {k: float(v) for k, v in zip(names, ev.stats)}
