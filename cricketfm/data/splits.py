"""Group-disjoint splits: no player and no video on both sides of a split.

Two leaks this guards against, both present in the footage on this box (results/data_inventory.json):
  - player identity: 10 batters account for 62 batting videos, one of them for 20, so a split by
    video would test mostly on people the model trained on;
  - duplicate uploads: 33 of 157 job videos re-upload footage under a new job id. They share a
    content fingerprint (data/video.content_fingerprint).
Records that share any group key (by default player and fingerprint) are joined into one
component by union-find, and components are assigned whole.

Assignment is greedy, largest component first: each goes to the bucket that keeps bucket sizes
closest to their targets and each bucket's class mix closest to the overall mix. With few
players a fixed train/val/test split leaves a test set of two or three people, so ShotFM uses
assign_folds() (grouped K-fold over players): every record is tested exactly once, by a model
that never saw its player.

per_class_subsets() gives the data-efficiency curve's training subsets, k clips per class
(docs/PROTOCOL.md §4.2): nested, so a larger budget only adds clips. budget_ladder() fixes the
pilot's values of k from label counts alone. The VLM + LoRA and the pose classifier train on the
same subsets, so their curves differ only in the learner.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

GROUP_KEYS = ("player_id", "fingerprint")


def components(records: Sequence[dict], keys: Sequence[str] = GROUP_KEYS) -> np.ndarray:
    """Component id per record: records sharing a value of any key end up together."""
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, rec in enumerate(records):
        for key in keys:
            value = rec.get(key)
            if value in (None, ""):
                raise ValueError(f"record {i} has no {key!r}; every record needs every group key")
            j = seen.setdefault((key, str(value)), i)
            parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(len(records))])
    return np.unique(roots, return_inverse=True)[1]


def _assign(comp: np.ndarray, labels: Sequence[str] | None, targets: np.ndarray, seed: int) -> np.ndarray:
    n_comp = comp.max() + 1
    sizes = np.bincount(comp, minlength=n_comp)
    classes = sorted(set(labels)) if labels is not None else ["_"]
    lab = np.array([classes.index(x) for x in labels]) if labels is not None else np.zeros(len(comp), int)
    comp_counts = np.zeros((n_comp, len(classes)))
    np.add.at(comp_counts, (comp, lab), 1)
    total = len(comp)
    want_mix = comp_counts.sum(0) / total                  # overall class fractions
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n_comp), -sizes))       # largest first, seeded ties
    bucket_counts = np.zeros((len(targets), len(classes)))
    bucket_of = np.empty(n_comp, int)
    for c in order:
        best, best_cost = 0, np.inf
        for b in range(len(targets)):
            trial = bucket_counts.copy()
            trial[b] += comp_counts[c]
            size_cost = ((trial.sum(1) / total - targets) ** 2).sum()
            mix_cost = ((trial / total - targets[:, None] * want_mix[None]) ** 2).sum()
            cost = size_cost + mix_cost
            if cost < best_cost - 1e-12:
                best, best_cost = b, cost
        bucket_counts[best] += comp_counts[c]
        bucket_of[c] = best
    return bucket_of[comp]


def assign_splits(records: Sequence[dict], fractions: dict[str, float], label_key: str | None = "label",
                  keys: Sequence[str] = GROUP_KEYS, seed: int = 0) -> list[str]:
    """Split name per record, e.g. fractions={"train": .6, "val": .2, "test": .2}."""
    names = list(fractions)
    targets = np.array([fractions[n] for n in names], float)
    targets /= targets.sum()
    labels = [r[label_key] for r in records] if label_key else None
    bucket = _assign(components(records, keys), labels, targets, seed)
    return [names[b] for b in bucket]


def assign_folds(records: Sequence[dict], k: int, label_key: str | None = "label",
                 keys: Sequence[str] = GROUP_KEYS, seed: int = 0) -> list[int]:
    """Fold index (0..k-1) per record for grouped K-fold."""
    comp = components(records, keys)
    if comp.max() + 1 < k:
        raise ValueError(f"only {comp.max() + 1} independent groups for {k} folds")
    labels = [r[label_key] for r in records] if label_key else None
    return _assign(comp, labels, np.full(k, 1.0 / k), seed).tolist()


def check_disjoint(records: Sequence[dict], assignment: Sequence, keys: Sequence[str] = GROUP_KEYS) -> None:
    """Raise if any group-key value appears under two different assignments."""
    for key in keys:
        where: dict[str, set] = {}
        for rec, a in zip(records, assignment):
            where.setdefault(str(rec[key]), set()).add(a)
        leaks = {v: sorted(map(str, s)) for v, s in where.items() if len(s) > 1}
        if leaks:
            raise AssertionError(f"{key} values in several splits: {dict(list(leaks.items())[:5])}")


def describe(records: Sequence[dict], assignment: Sequence, label_key: str | None = "label") -> dict:
    """Per bucket: records, players, videos and class counts, for the split report."""
    out = {}
    for a in sorted(set(assignment), key=str):
        rs = [r for r, x in zip(records, assignment) if x == a]
        entry = {"records": len(rs), "players": len({r["player_id"] for r in rs}),
                 "videos": len({r["fingerprint"] for r in rs})}
        if label_key:
            entry["classes"] = dict(sorted(Counter(r[label_key] for r in rs).items()))
        out[str(a)] = entry
    return out


def per_class_subsets(ids: Sequence[str], labels: Sequence[str], budgets: Sequence[int],
                      seed: int = 0) -> dict[int, list[str]]:
    """Training subsets of up to k clips of every class, for each budget k. Each class's clips are
    shuffled once and a budget takes the first k, so a larger budget only adds clips; a class with
    fewer than k clips contributes all it has (the subset records what each class actually got)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    order = {}
    for cls in sorted(set(labels)):
        members = np.flatnonzero(labels == cls)
        order[cls] = members[rng.permutation(len(members))]
    return {k: [ids[i] for cls in sorted(order) for i in order[cls][:k]] for k in sorted(budgets)}


def budget_ladder(train_labels_per_fold: Sequence[Sequence[str]], classes: Sequence[str], start: int = 2) -> list[int]:
    """The pilot's budgets in clips per class: powers of two from `start` up to the largest k that
    at least half of `classes` reach in every training fold. Uses label counts only, so it is fixed
    before any model is trained. Empty when the data don't reach `start`; the curve then has only
    zero-shot and the whole training fold."""
    counts = [Counter(fold) for fold in train_labels_per_fold]
    reach = sorted((min(c[cls] for c in counts) for cls in classes), reverse=True)
    k_max = reach[math.ceil(len(classes) / 2) - 1]
    ladder, k = [], start
    while k <= k_max:
        ladder.append(k)
        k *= 2
    return ladder
