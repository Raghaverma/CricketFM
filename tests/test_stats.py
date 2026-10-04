import numpy as np
import pytest

from cricketfm.evaluation import stats


def test_cluster_bootstrap_is_wider_than_instance_bootstrap_on_clustered_data():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 25)                      # 20 clips x 25 near-identical frames
    values = np.repeat(rng.normal(0, 1, 20), 25) + rng.normal(0, 0.05, 500)
    stat = lambda i: {"mean": values[i].mean()}
    naive = stats.bootstrap(stat, 500, None, n_boot=500)["mean"]["ci"]
    clustered = stats.bootstrap(stat, 500, groups, n_boot=500)["mean"]["ci"]
    assert (clustered[1] - clustered[0]) > 3 * (naive[1] - naive[0])


def test_bootstrap_point_value_uses_every_instance():
    values = np.arange(10.0)
    out = stats.bootstrap(lambda i: {"mean": values[i].mean()}, 10, n_boot=20)
    assert out["mean"]["value"] == pytest.approx(4.5)


def test_paired_difference_sign_and_interval():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.1, 200)
    b = a + 0.05                                               # cand better by exactly 0.05 everywhere
    out = stats.paired(lambda i: {"m": a[i].mean()}, lambda i: {"m": b[i].mean()}, 200, n_boot=200)["m"]
    assert out["diff"] == pytest.approx(0.05)
    assert out["ci"][0] == pytest.approx(0.05) and out["ci"][1] == pytest.approx(0.05)


def test_undefined_resamples_are_counted():
    values = np.array([np.nan] * 9 + [1.0])
    out = stats.bootstrap(lambda i: {"m": np.nanmean(values[i]) if np.isfinite(values[i]).any() else np.nan},
                          10, n_boot=200)
    assert out["m"]["resamples_undefined"] > 0


def test_describe_clusters():
    d = stats.describe_clusters(["a", "a", "b", "c", "c", "c"])
    assert d == {"clusters": 3, "instances": 6, "cluster_size_min": 1, "cluster_size_median": 2.0, "cluster_size_max": 3}
