"""Uncertainty for small, clustered test sets.

Frames from one clip are not independent samples, and a cricket test set is a few hundred
frames from a few dozen clips: a frame-level bootstrap would report intervals far narrower than
the data supports. So intervals resample *clusters* (clips or videos) with replacement, and every
instance of a drawn cluster comes along. Differences between two systems are paired: each
resample scores both systems on the same clusters, as the TensorRT repo's Stage 10 did per
person.

A statistic is a function of an index array (the resampled instances, with repeats) returning a
dict of floats, so one resample scores many metrics at once.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

N_BOOT = 2000
Stat = Callable[[np.ndarray], dict[str, float]]


class _Sampler:
    def __init__(self, n: int, groups=None):
        self.n = n
        if groups is None:
            self.members = None
            self.units = n
        else:
            groups = np.asarray(groups)
            if len(groups) != n:
                raise ValueError(f"groups has {len(groups)} entries for {n} instances")
            _, inv = np.unique(groups, return_inverse=True)
            order = np.argsort(inv, kind="stable")
            bounds = np.cumsum(np.bincount(inv))[:-1]
            self.members = np.split(order, bounds)
            self.units = len(self.members)

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        picks = rng.integers(0, self.units, self.units)
        if self.members is None:
            return picks
        return np.concatenate([self.members[p] for p in picks])


def _interval(draws: np.ndarray, level: float) -> list[float]:
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return [float("nan"), float("nan")]
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(finite, [tail, 100 - tail])
    return [float(lo), float(hi)]


def bootstrap(stat: Stat, n: int, groups=None, n_boot: int = N_BOOT, seed: int = 0,
              level: float = 0.95) -> dict[str, dict]:
    """{metric: {"value": stat on all instances, "ci": [lo, hi]}} with a percentile interval from
    n_boot cluster resamples. A metric undefined in some resamples (e.g. no occluded joint drawn)
    is summarized over the resamples where it is defined, and the count of the others reported."""
    sampler = _Sampler(n, groups)
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n))
    draws = {k: np.empty(n_boot) for k in point}
    for b in range(n_boot):
        r = stat(sampler.draw(rng))
        for k in point:
            draws[k][b] = r.get(k, np.nan)
    out = {}
    for k, v in point.items():
        out[k] = {"value": float(v), "ci": _interval(draws[k], level)}
        undefined = int((~np.isfinite(draws[k])).sum())
        if undefined:
            out[k]["resamples_undefined"] = undefined
    return out


def paired(ref: Stat, cand: Stat, n: int, groups=None, n_boot: int = N_BOOT, seed: int = 0,
           level: float = 0.95) -> dict[str, dict]:
    """Paired comparison of two systems scored on the same n instances: for every metric,
    {"ref", "cand", "diff" (cand - ref), "ci": interval of the difference}. Both systems see the
    same resampled clusters in every draw."""
    sampler = _Sampler(n, groups)
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    point_ref, point_cand = ref(all_idx), cand(all_idx)
    keys = [k for k in point_ref if k in point_cand]
    draws = {k: np.empty(n_boot) for k in keys}
    for b in range(n_boot):
        idx = sampler.draw(rng)
        r, c = ref(idx), cand(idx)
        for k in keys:
            draws[k][b] = c.get(k, np.nan) - r.get(k, np.nan)
    return {k: {"ref": float(point_ref[k]), "cand": float(point_cand[k]),
                "diff": float(point_cand[k] - point_ref[k]), "ci": _interval(draws[k], level)}
            for k in keys}


def describe_clusters(groups) -> dict:
    """Cluster count and sizes, recorded beside every interval so a reader can judge it."""
    _, counts = np.unique(np.asarray(groups), return_counts=True)
    return {"clusters": int(len(counts)), "instances": int(counts.sum()),
            "cluster_size_min": int(counts.min()), "cluster_size_median": float(np.median(counts)),
            "cluster_size_max": int(counts.max())}
