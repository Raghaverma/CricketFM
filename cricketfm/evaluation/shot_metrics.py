"""Shot-classification metrics (ShotFM), identical for every system: VLM zero-shot, VLM + LoRA,
the pose + temporal classifier, and the VLM with a pose overlay.

Every test clip counts. A response that can't be mapped to a class (models/vlm/parsing.py) is
INVALID: scored as wrong and reported as its own rate, never dropped, so a model can't raise its
accuracy by failing to answer hard clips.

Accuracy is reported beside balanced accuracy and macro-F1 because the class distribution is
whatever the footage contains (drives and defence dominate nets sessions), and a majority-class
guesser can post a respectable accuracy.

Pooled out-of-fold metrics are reported beside per_fold() ones: with about two players per test
fold the folds differ a lot, and that spread is part of the result. Chance (Q2) is two
baselines: majority_predictions(), and permutation_test(), which shuffles a system's own answers
within each player (and, as a second number, across all clips).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from cricketfm.evaluation import stats

INVALID = "__invalid__"
ECE_BINS = 10


def encode(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Labels -> class indices; an invalid prediction -> len(labels). An unknown true label is an
    error (the test set and the taxonomy disagree), an unknown predicted label is INVALID."""
    index = {name: i for i, name in enumerate(labels)}
    unknown = sorted({y for y in y_true if y not in index})
    if unknown:
        raise ValueError(f"true labels not in the taxonomy: {unknown}")
    t = np.array([index[y] for y in y_true], dtype=np.int64)
    p = np.array([index.get(y, len(labels)) for y in y_pred], dtype=np.int64)
    return t, p


def confusion(t: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    """(C, C + 1) counts: rows true class, columns predicted class, last column INVALID."""
    return np.bincount(t * (n_classes + 1) + p, minlength=n_classes * (n_classes + 1)).reshape(n_classes, n_classes + 1)


def _per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = cm.shape[0]
    tp = np.diag(cm[:, :c]).astype(np.float64)
    support = cm.sum(1).astype(np.float64)
    predicted = cm[:, :c].sum(0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    f1 = np.where(support > 0, f1, np.nan)
    return precision, recall, f1, support


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error over equal-width confidence bins (NaN confidences skipped)."""
    ok = np.isfinite(confidence)
    conf, corr = confidence[ok], correct[ok].astype(np.float64)
    if len(conf) == 0:
        return float("nan")
    b = np.minimum((conf * bins).astype(np.int64), bins - 1)
    total = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            total += m.sum() / len(conf) * abs(corr[m].mean() - conf[m].mean())
    return float(total)


def aggregate(t: np.ndarray, p: np.ndarray, n_classes: int, confidence: np.ndarray | None = None,
              idx: np.ndarray | None = None) -> dict[str, float]:
    """Scalar metrics over clips idx (all if None; repeats allowed, for the bootstrap). Classes
    absent from the (resampled) truth are left out of the macro averages."""
    if idx is not None:
        t, p = t[idx], p[idx]
        confidence = None if confidence is None else confidence[idx]
    cm = confusion(t, p, n_classes)
    _, recall, f1, _ = _per_class(cm)
    correct = t == p
    out = {"accuracy": float(correct.mean()), "balanced_accuracy": float(np.nanmean(recall)),
           "macro_f1": float(np.nanmean(f1)), "invalid_rate": float((p == n_classes).mean())}
    if confidence is not None:
        out["ece"] = ece(confidence, correct)
    return out


def report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str],
           confidence: Sequence[float] | None = None, groups=None, n_boot: int = stats.N_BOOT,
           seed: int = 0) -> dict:
    """Every ShotFM metric for one system on one set: point estimates with cluster-bootstrap
    intervals (groups = the batter's player_id per clip), per-class precision/recall/F1 and the
    confusion matrix (columns: labels + INVALID)."""
    t, p = encode(y_true, y_pred, labels)
    c = len(labels)
    conf = None if confidence is None else np.array([np.nan if v is None else v for v in confidence], np.float64)
    precision, recall, f1, support = _per_class(confusion(t, p, c))
    out = {
        "clips": int(len(t)),
        "labels": list(labels),
        "metrics": stats.bootstrap(lambda i: aggregate(t, p, c, conf, i), len(t), groups, n_boot, seed),
        "per_class": {name: {"precision": float(precision[k]), "recall": float(recall[k]),
                             "f1": float(f1[k]), "support": int(support[k])} for k, name in enumerate(labels)},
        "confusion": {"columns": list(labels) + [INVALID], "rows": confusion(t, p, c).tolist()},
    }
    if groups is not None:
        out["clusters"] = stats.describe_clusters(groups)
    return out


def compare(y_true: Sequence[str], pred_ref: Sequence[str], pred_cand: Sequence[str], labels: Sequence[str],
            groups=None, n_boot: int = stats.N_BOOT, seed: int = 0) -> dict:
    """Paired differences (cand - ref) of every aggregate, two systems on the same clips."""
    t, p_ref = encode(y_true, pred_ref, labels)
    _, p_cand = encode(y_true, pred_cand, labels)
    c = len(labels)
    return stats.paired(lambda i: aggregate(t, p_ref, c, idx=i), lambda i: aggregate(t, p_cand, c, idx=i),
                        len(t), groups, n_boot, seed)


def per_fold(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], folds: Sequence,
             players: Sequence[str] | None = None) -> dict:
    """Point metrics on each test fold, and their mean, SD, min and max across folds."""
    t, p = encode(y_true, y_pred, labels)
    folds = np.asarray(folds)
    rows = {}
    for f in sorted(set(folds.tolist()), key=str):
        m = folds == f
        rows[str(f)] = {"clips": int(m.sum()), **aggregate(t[m], p[m], len(labels))}
        if players is not None:
            rows[str(f)]["players"] = len(set(np.asarray(players)[m].tolist()))
    keys = [k for k in next(iter(rows.values())) if k not in ("clips", "players")] if rows else []
    across = {}
    for k in keys:
        v = np.array([r[k] for r in rows.values()], np.float64)
        across[k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"),
                     "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
    return {"folds": rows, "across_folds": across}


def majority_predictions(y_true: Sequence[str], folds: Sequence) -> list[str]:
    """The majority-class baseline under the same folds: every test clip gets the most frequent
    class among the other folds' clips (ties to the alphabetically first)."""
    y, folds = np.asarray(y_true), np.asarray(folds)
    out = np.empty(len(y), dtype=object)
    for f in set(folds.tolist()):
        train = y[folds != f]
        classes, counts = np.unique(train, return_counts=True)
        out[folds == f] = classes[np.argmax(counts)]
    return out.tolist()


def permutation_test(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], metric: str = "macro_f1",
                     groups=None, n_perm: int = 10_000, seed: int = 0) -> dict:
    """Is a system better than chance? Its predictions are shuffled n_perm times, which keeps how
    often it names each class but cuts any link to the clip's stroke. With groups (the player per
    clip), predictions are shuffled only among a player's own clips: the null then keeps whatever
    the answers share with the batter and camera setup, and a system beats it only by telling one
    stroke from another within a batter. Returns the observed metric, the null's mean and 95th
    percentile, and the one-sided p-value (with the +1 correction, so never 0)."""
    t, p = encode(y_true, y_pred, labels)
    c = len(labels)
    observed = aggregate(t, p, c)[metric]
    rng = np.random.default_rng(seed)
    blocks = [np.arange(len(t))] if groups is None else [
        np.flatnonzero(np.asarray(groups) == g) for g in np.unique(np.asarray(groups))]
    null = np.empty(n_perm)
    for b in range(n_perm):
        shuffled = p.copy()
        for idx in blocks:
            shuffled[idx] = p[rng.permutation(idx)]
        null[b] = aggregate(t, shuffled, c)[metric]
    return {"metric": metric, "observed": float(observed), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p_value": float((1 + (null >= observed).sum()) / (1 + n_perm)),
            "permutations": n_perm, "shuffled_within": "all clips" if groups is None else f"{len(blocks)} groups"}
