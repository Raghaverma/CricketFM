import cv2
import numpy as np
import pytest

from cricketfm import skeleton
from cricketfm.models.pose import sapiens as sp


def test_goliath_mapping_by_name():
    m = skeleton.Mapping.by_name("goliath308", sp.GOLIATH308)
    assert m.index.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def test_mapping_refuses_duplicates_and_missing_names():
    with pytest.raises(ValueError):
        skeleton.Mapping.by_name("dup", list(skeleton.COCO17) + ["nose"])
    with pytest.raises(ValueError):
        skeleton.Mapping.by_name("short", skeleton.COCO17[:-1])


def test_flip_and_mirror():
    assert (skeleton.FLIP[skeleton.FLIP] == np.arange(17)).all()
    assert skeleton.COCO17[skeleton.FLIP[skeleton.INDEX["left_wrist"]]] == "right_wrist"
    kp = np.random.default_rng(0).uniform(0, 100, (17, 2))
    assert np.allclose(skeleton.mirror(skeleton.mirror(kp, 640), 640), kp)


def test_center_scale_pads_and_fixes_aspect():
    center, scale = sp.center_scale(np.array([[100, 50, 80, 200]]))
    assert center[0].tolist() == [140, 150]
    assert scale[0, 1] == pytest.approx(250) and scale[0, 0] == pytest.approx(250 * 0.75)


def test_warp_puts_a_frame_point_where_the_matrix_says():
    frame = np.zeros((720, 1280, 3), np.uint8)
    p = (400.0, 300.0)
    cv2.circle(frame, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1)
    box = np.array([[300, 100, 200, 400]], np.float32)
    crop = sp.warp_crops(frame, box)[0, ..., 0].astype(float)
    c, s = sp.center_scale(box)
    u = sp.udp_warp_matrix(c[0], s[0], (768, 1024)) @ np.array([*p, 1.0])
    ys, xs = np.nonzero(crop > 0.5 * crop.max())
    assert abs(xs.mean() - u[0]) < 1.5 and abs(ys.mean() - u[1]) < 1.5


def test_decode_round_trip_recovers_a_subpixel_frame_point():
    """Frame point -> crop (UDP warp) -> heatmap (aligned corners) -> Gaussian -> decode -> frame."""
    box = np.array([[300, 100, 200, 400]], np.float32)
    c, s = sp.center_scale(box)
    m = sp.udp_warp_matrix(c[0], s[0], (768, 1024))
    hh, hw = sp.HEATMAP_HW
    yy, xx = np.mgrid[0:hh, 0:hw]
    for p in [(371.3, 222.7), (450.0, 410.25), (330.6, 150.1)]:
        u = m @ np.array([*p, 1.0])
        h = u * np.array([(hw - 1) / 767, (hh - 1) / 1023])
        heat = np.exp(-((xx - h[0]) ** 2 + (yy - h[1]) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
        kp_hm, score = sp.udp_decode(heat[None, None])
        back = sp.to_frame(kp_hm, box)[0, 0]
        assert np.abs(back - np.array(p)).max() < 0.25, (p, back)
        assert score[0, 0] == pytest.approx(heat.max())
