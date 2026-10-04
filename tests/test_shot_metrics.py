import numpy as np
import pytest

from cricketfm.evaluation import shot_metrics as sm

LABELS = ["cover_drive", "pull", "other"]


def test_unknown_true_label_is_an_error_and_unknown_prediction_is_invalid():
    with pytest.raises(ValueError):
        sm.encode(["hook"], ["pull"], LABELS)
    t, p = sm.encode(["pull"], ["banana"], LABELS)
    assert p[0] == len(LABELS)


def test_report_matches_hand_computation():
    y_true = ["cover_drive"] * 4 + ["pull"] * 4 + ["other"] * 2
    y_pred = ["cover_drive", "cover_drive", "cover_drive", "pull",
              "pull", "pull", "cover_drive", sm.INVALID,
              "other", "pull"]
    r = sm.report(y_true, y_pred, LABELS, n_boot=10)
    m = {k: v["value"] for k, v in r["metrics"].items()}
    assert m["accuracy"] == pytest.approx(6 / 10)
    assert m["invalid_rate"] == pytest.approx(1 / 10)
    recalls = [3 / 4, 2 / 4, 1 / 2]
    assert m["balanced_accuracy"] == pytest.approx(np.mean(recalls))
    p = [3 / 4, 2 / 4, 1 / 1]                                   # cover 3/4 predicted right, pull 2/4, other 1/1
    f1 = [2 * pi * ri / (pi + ri) for pi, ri in zip(p, recalls)]
    assert m["macro_f1"] == pytest.approx(np.mean(f1))
    assert r["confusion"]["rows"][1] == [1, 2, 0, 1]             # pull row: 1 cover, 2 pull, 0 other, 1 invalid


def test_ece():
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.2])
    correct = np.array([1, 1, 1, 0, 0])
    # bin 9: acc 0.75 vs conf 0.9 (4 of 5 clips); bin 2: acc 0 vs conf 0.2 (1 of 5)
    assert sm.ece(conf, correct) == pytest.approx(0.8 * 0.15 + 0.2 * 0.2)
    assert np.isnan(sm.ece(np.array([np.nan]), np.array([1])))


def test_compare_identical_systems():
    y = ["pull", "other", "cover_drive", "pull"]
    out = sm.compare(y, y, y, LABELS, n_boot=20)
    assert out["accuracy"]["diff"] == 0


def test_per_fold_and_majority_baseline():
    y_true = ["pull", "pull", "other", "cover_drive", "pull", "other"]
    folds = [0, 0, 0, 1, 1, 1]
    maj = sm.majority_predictions(y_true, folds)
    # fold 0 trains on fold 1 (cover_drive, pull, other: a three-way tie -> cover_drive); fold 1 on pull, pull, other
    assert maj == ["cover_drive"] * 3 + ["pull"] * 3
    out = sm.per_fold(y_true, y_true, LABELS, folds, players=["a", "a", "b", "c", "c", "c"])
    assert out["folds"]["0"]["accuracy"] == 1 and out["folds"]["0"]["players"] == 2
    assert out["across_folds"]["accuracy"] == {"mean": 1.0, "sd": 0.0, "min": 1.0, "max": 1.0}


def test_permutation_test_separates_signal_from_chance():
    rng = np.random.default_rng(0)
    y = rng.choice(LABELS, 120).tolist()
    perfect = sm.permutation_test(y, y, LABELS, n_perm=500)
    assert perfect["p_value"] == pytest.approx(1 / 501) and perfect["observed"] == 1
    noise = sm.permutation_test(y, rng.choice(LABELS, 120).tolist(), LABELS, n_perm=500)
    assert noise["p_value"] > 0.05


def test_within_player_null_discounts_answers_that_track_the_batter():
    # Each batter plays one stroke and the model answers by batter, not by stroke: it beats the
    # across-clip null, but within a batter its answers carry nothing, so the within-player null
    # matches it exactly.
    y = ["pull"] * 20 + ["cover_drive"] * 20 + ["other"] * 20
    players = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
    across = sm.permutation_test(y, y, LABELS, n_perm=200)
    within = sm.permutation_test(y, y, LABELS, groups=players, n_perm=200)
    assert across["p_value"] < 0.01
    assert within["p_value"] == 1.0 and within["shuffled_within"] == "3 groups"
