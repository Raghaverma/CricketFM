import numpy as np
import pytest

from cricketfm import skeleton
from cricketfm.evaluation import pose_metrics as pm


def _synthetic(n: int, seed: int):
    """n single-person images: labels, predictions at mixed error levels, joint scores."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((n, 17, 3))
    gt[..., 0] = rng.uniform(100, 500, (n, 17))
    gt[..., 1] = rng.uniform(100, 700, (n, 17))
    gt[..., 2] = rng.choice([0, 1, 2], (n, 17), p=[0.15, 0.25, 0.6])
    gt[:, 0, 2] = 2                                       # at least one labeled joint each
    gt[gt[..., 2] == 0, :2] = 0
    boxes = np.column_stack([gt[..., 0].min(1), gt[..., 1].min(1), np.full(n, 400.0), np.full(n, 600.0)])
    area = boxes[:, 2] * boxes[:, 3] * pm.AREA_FROM_BOX
    noise = rng.uniform(0, 40, (n, 1, 1))                 # per-person error scale: some good, some bad
    pred = (gt[..., :2] + rng.normal(0, 1, (n, 17, 2)) * noise).round(3)
    scores = rng.uniform(0, 1, (n, 17))
    return gt, boxes, area, pred, scores


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matched_ap_equals_pycocotools_on_single_person_images(seed):
    gt, boxes, area, pred, scores = _synthetic(150, seed)
    ids = np.arange(1, len(gt) + 1)
    coco = {"images": [{"id": int(i), "width": 1000, "height": 1000} for i in ids],
            "annotations": [{"id": int(i), "image_id": int(i), "category_id": 1, "iscrowd": 0, "area": float(a),
                             "bbox": b.tolist(), "keypoints": g.ravel().tolist(), "num_keypoints": int((g[:, 2] > 0).sum())}
                            for i, a, b, g in zip(ids, area, boxes, gt)],
            "categories": [{"id": 1, "name": "person"}]}
    reference = pm.coco_ap(coco, pm.detections(ids, pred, scores))
    ours = pm.matched_ap(pm.oks(pred, gt, area), pm.person_scores(scores))
    for key in ("AP", "AP50", "AP75"):
        assert ours[key] == pytest.approx(reference[key], abs=1e-9)


def test_oks_perfect_prediction_and_subsets():
    gt, boxes, area, _, _ = _synthetic(5, 3)
    assert np.allclose(pm.oks(gt[..., :2], gt, area), 1.0)
    unlabeled = np.flatnonzero(gt[0, :, 2] == 0)
    if len(unlabeled):
        assert np.isnan(pm.oks(gt[..., :2], gt, area, joints=unlabeled[:1])[0])


def test_pck_is_strict_and_skips_unlabeled():
    gt = np.zeros((1, 17, 3))
    gt[0, :, 2] = 2
    gt[0, 16, 2] = 0
    pred = gt[..., :2].copy()
    pred[0, 0, 0] += 10.0                                  # exactly at the threshold: not a hit
    pred[0, 1, 0] += 9.9
    hits = pm.pck_hits(pred, gt, np.array([100.0]), 0.1)
    assert hits[0, 0] == 0 and hits[0, 1] == 1 and np.isnan(hits[0, 16])


def test_angle_error_recovers_a_known_rotation():
    gt = np.zeros((1, 17, 3))
    gt[..., 2] = 2
    s, e, w = (skeleton.INDEX[n] for n in ("left_shoulder", "left_elbow", "left_wrist"))
    gt[0, s, :2], gt[0, e, :2], gt[0, w, :2] = (0, 0), (100, 0), (200, 0)      # straight arm: 180 degrees
    pred = gt[..., :2].copy()
    theta = np.radians(30)
    pred[0, w] = gt[0, e, :2] + 100 * np.array([np.cos(theta), np.sin(theta)])  # bent by 30
    err = pm.angle_errors(pred, gt)[0, list(skeleton.ANGLES).index("left_elbow")]
    assert err == pytest.approx(30.0, abs=1e-6)


def test_keypoint_bias_recovers_a_constant_offset():
    gt, boxes, area, _, _ = _synthetic(40, 4)
    pred = gt[..., :2] + np.array([0.0, 30.0])            # everything 30 px low
    bias, err = pm.keypoint_bias(pred, gt, boxes)
    expected = 30.0 / pm.box_size(boxes)[0]               # all boxes are 400x600 -> size 600
    assert np.allclose(bias[:, 1], expected) and np.allclose(bias[:, 0], 0)
    assert np.allclose(err, expected)


def test_aggregate_splits_visible_and_occluded():
    gt, boxes, area, pred, scores = _synthetic(60, 5)
    t = pm.per_person(pred, scores, gt, boxes, area)
    out = pm.aggregate(t)
    hits = t[pm.PRIMARY_PCK]
    assert out["pck_visible"] == pytest.approx(np.nanmean(hits[t["visibility"] == 2]))
    assert out["pck_occluded"] == pytest.approx(np.nanmean(hits[t["visibility"] == 1]))
    assert set(skeleton.GROUPS) <= {k.removeprefix("pck_") for k in out}


def test_per_person_refuses_unlabeled_people():
    gt, boxes, area, pred, scores = _synthetic(3, 6)
    gt[1, :, 2] = 0
    with pytest.raises(ValueError):
        pm.per_person(pred, scores, gt, boxes, area)


def test_compare_identical_models_has_zero_difference():
    gt, boxes, area, pred, scores = _synthetic(30, 7)
    t = pm.per_person(pred, scores, gt, boxes, area)
    diff = pm.compare(t, t, groups=np.arange(30) // 3, n_boot=50)
    assert diff["AP"]["diff"] == 0 and diff["AP"]["ci"] == [0.0, 0.0]
