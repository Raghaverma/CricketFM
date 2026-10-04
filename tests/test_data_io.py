import json

import cv2
import numpy as np
import pytest

from cricketfm import taxonomy
from cricketfm.data import clips, pose_gt, sampling, video
from cricketfm.evaluation import latency


def _write_video(path, n: int):
    """MJPG video whose frame i is uniformly grey level 8 * i, so a frame's content names its index."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), 8 * i, np.uint8))
    w.release()


def test_read_frames_returns_frames_by_decoded_index(tmp_path):
    path = tmp_path / "v.avi"
    _write_video(path, 20)
    assert video.count_frames(path) == 20
    frames = video.read_frames(path, [17, 3, 3, 0])
    assert sorted(frames) == [0, 3, 17]
    for i, f in frames.items():
        assert abs(f.mean() - 8 * i) < 2
    with pytest.raises(RuntimeError):
        video.read_frames(path, [25])


def test_content_fingerprint_tracks_content(tmp_path):
    a, b = tmp_path / "a.avi", tmp_path / "b.avi"
    _write_video(a, 10)
    _write_video(b, 11)
    assert video.content_fingerprint(a) == video.content_fingerprint(a)
    assert video.content_fingerprint(a) != video.content_fingerprint(b)


def test_motion_trace_and_contact_sheet(tmp_path):
    path = tmp_path / "v.avi"
    _write_video(path, 20)                                      # grey level 8 * i: every step is 8 levels
    trace, dots, thumbs = sampling.motion_trace(path, fps=30)   # d = 3 frames at 30 fps
    assert dots.tolist() == [0] * 20                            # grey frames: no overlay
    assert len(trace) == 20
    assert np.isnan(trace[:3]).all() and np.isnan(trace[-3:]).all()
    assert np.allclose(trace[3:17], 6 * 8 / 255, atol=2 / 255)  # frames i - 3 and i + 3 differ by 48 levels
    assert [i for i, _ in thumbs] == [0]                        # one thumbnail a second
    sheet = sampling.contact_sheet(thumbs, "v")
    assert sheet.shape == (40 + 2 * 270, 3 * 480, 3)


def test_overlay_dots_finds_drawn_joints_not_red_objects():
    from cricketfm.data import overlay

    frame = np.full((720, 1280, 3), 90, np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 400), (40, 40, 200), -1)       # a red shirt: one big blob
    cv2.circle(frame, (900, 600), 6, (20, 20, 200), -1)                 # a ball
    assert overlay.dots(frame) == 1
    for k in range(6):                                                  # six joint dots, as the app draws them
        cv2.circle(frame, (600 + 30 * k, 200 + 25 * k), 8, (8, 0, 246), -1)
    assert overlay.dots(frame) >= overlay.MIN_DOTS


def test_overlay_mask_widens_by_the_guard():
    dots = np.zeros(20, np.uint8)
    dots[10] = 5
    dots[3] = 1                                                         # below MIN_DOTS: not an overlay frame
    assert np.flatnonzero(sampling.overlay_mask(dots, 2)).tolist() == [8, 9, 10, 11, 12]
    assert not sampling.overlay_mask(np.zeros(5, np.uint8), 2).any()


def test_priority_frames_alternate_bins_and_keep_the_gap():
    rng = np.random.default_rng(0)
    motion = rng.uniform(0, 0.1, 400).astype(np.float32)
    motion[:5] = np.nan
    motion[100:150] = motion[300:350] = 1.0                     # two bursts of action, a quarter of the frames
    frames = sampling.priority_frames(motion, 8, 30, rng)
    idx = sorted(i for i, _ in frames)
    assert len(frames) == 8 and all(b - a >= 30 for a, b in zip(idx, idx[1:]))
    assert {b for _, b in frames[:2]} == {"high", "low"}         # the bins alternate
    assert all(motion[i] == 1.0 for i, b in frames if b == "high")
    assert all(i >= 5 for i in idx)                             # no frame without a motion value
    assert sampling.priority_frames(np.full(10, np.nan), 4, 1, rng) == []
    static = sampling.priority_frames(np.zeros(100, np.float32), 4, 10, rng)
    assert sorted(b for _, b in static) == ["high", "high", "low", "low"]   # ties still give a top quartile
    firsts = [sampling.priority_frames(motion, 1, 30, np.random.default_rng(s))[0][1] for s in range(200)]
    assert 70 < firsts.count("high") < 130                     # a quota of one leans neither way


def test_waterfill_gives_small_children_everything():
    rng = np.random.default_rng(0)
    assert sampling.waterfill(10, [2, 100, 100], rng) == [2, 4, 4]
    assert sampling.waterfill(10, [1, 2, 3], rng) == [1, 2, 3]  # not enough to go round
    assert sum(sampling.waterfill(7, [5, 5, 5], rng)) == 7


def test_allocate_balances_cells_then_players_and_caps_players():
    tree = {"striker": {"side_on/right/daylight": {"A": {"a1": 16, "a2": 16}, "B": {"b1": 16}},
                        "rear/left/poor": {"C": {"c1": 3}}}}
    out = sampling.allocate(tree, 20, per_player=120, rng=np.random.default_rng(0))
    assert out["c1"] == 3                                       # the rare cell gives all it has
    assert out["a1"] + out["a2"] + out["b1"] == 17
    assert abs((out["a1"] + out["a2"]) - out["b1"]) <= 1        # players share their cell equally
    capped = sampling.allocate(tree, 40, per_player=5, rng=np.random.default_rng(0))
    assert capped["a1"] + capped["a2"] == 5 and capped["b1"] == 5


def test_systematic_sample_is_proportional():
    picks = sampling.systematic(800, 80, np.random.default_rng(0))
    assert len(set(picks.tolist())) == 80 and picks.min() >= 0 and picks.max() < 800
    strata = np.repeat(np.arange(4), [400, 200, 150, 50])        # 800 items sorted by stratum
    assert np.bincount(strata[picks]).tolist() == [40, 20, 15, 5]


def test_pose_gt_loader(tmp_path):
    kps = np.zeros((17, 3))
    kps[:5] = [[10, 10, 2]] * 5
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "video_id": "v", "fingerprint": "f", "player_id": "p",
                        "frame_index": 0, "role": "striker"}],
            "annotations": [{"id": 1, "image_id": 1, "bbox": [0, 0, 50, 100], "keypoints": kps.ravel().tolist()},
                            {"id": 2, "image_id": 1, "bbox": [0, 0, 50, 100], "keypoints": [0] * 51}],
            "categories": [{"id": 1, "name": "person"}]}
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(coco))
    gt = pose_gt.load(path)
    assert gt.ann_ids.tolist() == [1]                               # the unlabeled person is dropped
    assert gt.area[0] == pytest.approx(50 * 100 * 0.53)
    assert gt.column("role").tolist() == ["striker"]
    coco["annotations"][0]["num_keypoints"] = 4
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError):
        pose_gt.load(path)


def test_clip_validation(tmp_path):
    tax = taxonomy.load()
    good = {"clip_id": "c1", "video_id": "v", "fingerprint": "f", "player_id": "p", "start_frame": 0,
            "end_frame": 90, "contact_frame": 45, "label": "hook", "batting_hand": "left", "view": "front_on",
            "flags": ["motion_blur"], "annotator": "rv"}
    bad = dict(good, clip_id="c2", contact_frame=95, label="helicopter", view="drone", flags=["fog"])
    path = tmp_path / "clips.jsonl"
    path.write_text(json.dumps(good) + "\n")
    assert clips.load(path, tax)[0]["class"] == "other"
    path.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="line 2") as err:
        clips.load(path, tax)
    assert "flags" in str(err.value)
    assert clips.problems(dict(good, flags=[]), tax) == []           # no conditions is a valid answer
    assert clips.problems({k: v for k, v in good.items() if k != "flags"}, tax) == ["missing flags"]


def test_latency_summary_and_stage_clock():
    s = latency.summarize([float(i) for i in range(1, 101)], batch_size=4)
    assert s["p50_ms"] == pytest.approx(50.5) and s["p95_ms"] == 95 and s["p99_ms"] == 99
    assert s["fps"] == pytest.approx(1000 / 50.5 * 4)
    clock = latency.StageClock("cpu")
    for _ in range(3):
        with clock("work"):
            sum(range(1000))
    assert clock.summary()["work"]["num_iters"] == 3


def test_merge_restores_metadata_by_file_name():
    sampled = {"images": [{"id": 7, "file_name": "job1_000042.jpg", "video_id": "job1", "fingerprint": "f",
                           "player_id": "p", "frame_index": 42, "role": "striker"}],
               "categories": [{"id": 1, "name": "person"}]}
    export = {"images": [{"id": 1, "file_name": "images/job1_000042.jpg", "width": 1, "height": 1}],
              "annotations": [{"id": 3, "image_id": 1}]}
    merged = pose_gt.merge(export, sampled)
    assert merged["images"][0] == {**sampled["images"][0], "id": 1}
    export["images"][0]["file_name"] = "other.jpg"
    with pytest.raises(ValueError):
        pose_gt.merge(export, sampled)


def test_pose_eval_strata(tmp_path):
    from cricketfm.evaluation.pose_eval import strata

    kps = np.zeros((17, 3))
    kps[:8] = [[10, 10, 2]] * 8
    kps[8:10] = [[10, 10, 1]] * 2                                  # 2 of 10 labeled joints occluded
    images = [{"id": i, "file_name": f"{i}.jpg", "video_id": "v", "fingerprint": "f", "player_id": "p",
               "frame_index": i, "role": "striker", "view": "side_on", "hand": "left", "lighting": "indoor",
               "motion_bin": "high"} for i in (1, 2)]
    anns = [{"id": 1, "image_id": 1, "bbox": [0, 0, 50, 100], "keypoints": kps.ravel().tolist(),
             "attributes": {"occluded": False, "phase": "stroke", "hand": "right"}},
            {"id": 2, "image_id": 2, "bbox": [0, 0, 50, 600], "keypoints": kps.ravel().tolist()}]
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"images": images, "annotations": anns, "categories": []}))
    s = strata(pose_gt.load(path))
    assert s["box_height_px"].tolist() == ["<128", ">=512"]
    assert s["occluded_share"].tolist() == ["up to 25%", "up to 25%"]
    assert s["hand"].tolist() == ["right", "left"]                  # the person's tag wins over the video's
    assert s["phase"].tolist() == ["stroke", "None"]                # CVAT shape attribute; untagged -> "None"
