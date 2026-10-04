from collections import Counter

import numpy as np
import pytest

from cricketfm.data import splits


def _records(seed: int = 0, players: int = 12, per_player: int = 15):
    rng = np.random.default_rng(seed)
    recs = []
    for p in range(players):
        for v in range(3):
            for c in range(per_player // 3):
                recs.append({"clip_id": f"p{p}v{v}c{c}", "player_id": f"P{p}", "fingerprint": f"F{p}-{v}",
                             "label": rng.choice(["cover_drive", "pull", "defensive", "other"])})
    return recs


def test_components_join_transitively_through_shared_videos():
    recs = [{"player_id": "A", "fingerprint": "f1"}, {"player_id": "B", "fingerprint": "f1"},   # same video, 2 players
            {"player_id": "B", "fingerprint": "f2"}, {"player_id": "C", "fingerprint": "f3"}]
    comp = splits.components(recs)
    assert comp[0] == comp[1] == comp[2] and comp[3] != comp[0]


def test_components_need_every_key():
    with pytest.raises(ValueError):
        splits.components([{"player_id": "A", "fingerprint": ""}])


def test_assign_splits_is_group_disjoint_and_near_target():
    recs = _records()
    a = splits.assign_splits(recs, {"train": 0.6, "val": 0.2, "test": 0.2})
    splits.check_disjoint(recs, a)
    share = Counter(a)
    assert abs(share["train"] / len(recs) - 0.6) < 0.1


def test_check_disjoint_catches_a_leak():
    recs = [{"player_id": "A", "fingerprint": "f1"}, {"player_id": "A", "fingerprint": "f2"}]
    with pytest.raises(AssertionError):
        splits.check_disjoint(recs, ["train", "test"])


def test_folds_cover_every_player_once():
    recs = _records()
    folds = splits.assign_folds(recs, 4)
    splits.check_disjoint(recs, folds)
    assert set(folds) == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        splits.assign_folds(recs[:5], 4)                       # one player -> one group


def test_per_class_subsets_are_nested_and_capped():
    rng = np.random.default_rng(0)
    labels = rng.choice(["a", "b", "c"], 200, p=[0.6, 0.35, 0.05])
    ids = [f"c{i}" for i in range(200)]
    subsets = splits.per_class_subsets(ids, labels, [4, 8, 16], seed=1)
    assert set(subsets[4]) <= set(subsets[8]) <= set(subsets[16])
    by_id = dict(zip(ids, labels))
    n_c = int((labels == "c").sum())
    for k, subset in subsets.items():
        counts = Counter(by_id[i] for i in subset)
        assert counts["a"] == counts["b"] == k
        assert counts["c"] == min(k, n_c)                          # a small class gives all it has
    assert splits.per_class_subsets(ids, labels, [8], seed=1)[8] == subsets[8]


def test_budget_ladder_uses_the_smallest_fold():
    folds = [["a"] * 20 + ["b"] * 9 + ["c"] * 3 + ["d"] * 1,
             ["a"] * 30 + ["b"] * 12 + ["c"] * 5]
    # per class, the smallest training fold has a 20, b 9, c 3, d 0; half of 4 classes reach 9
    assert splits.budget_ladder(folds, ["a", "b", "c", "d"]) == [2, 4, 8]
    assert splits.budget_ladder(folds, ["c", "d"]) == [2]                # one of two classes reaches 3
    assert splits.budget_ladder(folds, ["d"]) == []
