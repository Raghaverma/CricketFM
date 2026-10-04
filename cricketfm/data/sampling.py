"""Pick and extract frames for pose annotation (PoseFM, docs/PROTOCOL.md §3.3).

    python -m cricketfm.data.sampling motion    # one decode per video: motion trace + contact sheet
    #   fill view, batting_hand / bowling_arm and lighting in data/videos.csv from data/pose/sheets/
    python -m cricketfm.data.sampling select    # 800 stratified frames -> data/pose/images/, to_annotate.json

The set is built for hard, varied poses rather than the footage's natural mix, which is mostly a
batter standing side-on in stance. Neither lever below looks at a pose model's output: selecting
on, say, low ViTPose++ confidence would build a test set adversarial to one side of the
comparison.

  - Equal allocation down a tree, role -> cell (camera view x handedness x lighting) -> player
    -> video. Each node splits its quota equally among its children, capped by what each child
    has, and what a small child can't take is split among the rest (waterfill). Rare cells give
    everything they have before common ones give more.
  - Motion. `motion` records, per decoded frame, the mean absolute difference between the
    greyscale frames 0.1 s before and after it (160 px wide): a model-free proxy for action
    phase. A video's frames are taken alternately from its top quartile of motion (backlift to
    follow-through, the delivery stride) and from the rest (stance, run-up, between balls).

Frames that carry a burned-in app overlay (data/overlay.py), and those within --overlay-guard-s
of one, are never drawn: they show another model's pose. `motion` counts the overlay's dots in
every frame in the same decode.

Within a video, frames are at least --min-gap-s apart and at most --per-video, drawn with a seed
from the video id. Distance from camera and occlusion need the labels, so they are reporting
strata (evaluation/pose_eval.py), not sampling ones.

A --double share of the frames (80 of 800) goes to a second, blind annotator: a systematic sample
over the selection sorted by stratum, so it spreads across strata in proportion.

Writes data/pose/images/<video_id>_<frame>.jpg, decoded by data/video.read_frames, and
data/pose/to_annotate.json: a COCO file with the images and their metadata and no annotations,
to import into the annotation tool (docs/ANNOTATION.md). Its `reserve` lists, per video, the next
frames in that video's draw order, which replace frames the annotators skip.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from cricketfm import skeleton
from cricketfm.data import overlay
from cricketfm.data.clips import HANDS, VIEWS
from cricketfm.data.video import open_video, read_frames
from cricketfm.paths import AUTOCLIP_DATA, DATA, rel

ROLE = {"batsman": "striker", "bowler": "bowler"}
HAND_COLUMN = {"striker": "batting_hand", "bowler": "bowling_arm"}   # handedness of the person labeled
LIGHTING = ("daylight", "indoor", "floodlit", "poor")
# A video can show several people (a nets session) or none clearly enough to tell: hand is then
# "mixed" or "unknown", and each labeled person's own hand comes from the annotation (docs/ANNOTATION.md).
ALLOWED = {"view": VIEWS, "hand": HANDS + ("mixed", "unknown"), "lighting": LIGHTING}
MOTION_WIDTH = 160                 # px; greyscale frames this wide for the motion proxy
MOTION_DT_S = 0.1                  # motion of frame i compares frames i - 0.1 s and i + 0.1 s
HIGH_MOTION_QUANTILE = 0.75        # a video's top quartile of motion is its "high" bin
SHEET_COLS, SHEET_ROWS, SHEET_TILE = 3, 2, (480, 270)


def eligible(videos: Path, modes) -> list[dict]:
    """videos.csv rows the sample may use: the chosen modes, readable, one copy of each duplicate
    upload, not marked usable=no."""
    with open(videos, newline="") as f:
        return [r for r in csv.DictReader(f)
                if r["mode"] in modes and not r["duplicate_of"] and not r["error"]
                and r.get("usable", "").strip().lower() not in ("no", "n", "false", "0")]


def _fps(row: dict) -> float:
    return float(row["fps_avg"] or row["fps_nominal"] or 30)


def _seed(base: int, video_id: str) -> int:
    return base + int(hashlib.sha1(video_id.encode()).hexdigest()[:8], 16)


def _fit(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """img scaled to fit inside size (w, h), centred on black."""
    w, h = size
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def motion_trace(path: Path, fps: float, threads: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray]]]:
    """One sequential decode. Returns the motion of every decoded frame i (mean absolute
    difference / 255 between greyscale frames i - d and i + d, d = 0.1 s and at least one frame;
    NaN within d of either end), the overlay dots in every frame (overlay.dots), and one thumbnail
    a second, as (frame index, image)."""
    d = max(1, round(MOTION_DT_S * fps))
    stride = max(1, round(fps))
    window: deque[np.ndarray] = deque(maxlen=2 * d + 1)
    motion, dots, thumbs, n = [], [], [], 0
    cap = open_video(path, threads)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (MOTION_WIDTH, max(1, round(h * MOTION_WIDTH / w))), interpolation=cv2.INTER_AREA)
            window.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16))
            if len(window) == window.maxlen:                     # complete for frame n - d
                motion.append(np.abs(window[-1] - window[0]).mean() / 255)
            dots.append(min(overlay.dots(frame), 255))
            if n % stride == 0:
                thumbs.append((n, _fit(frame, SHEET_TILE)))
            n += 1
    finally:
        cap.release()
    trace = np.full(n, np.nan, np.float32)
    trace[d:d + len(motion)] = motion
    return trace, np.array(dots, np.uint8), thumbs


def contact_sheet(thumbs: list[tuple[int, np.ndarray]], title: str) -> np.ndarray:
    """SHEET_COLS x SHEET_ROWS thumbnails spread evenly over the video, under a title strip:
    enough to judge camera view, handedness and lighting without opening the video."""
    w, h = SHEET_TILE
    sheet = np.zeros((40 + SHEET_ROWS * h, SHEET_COLS * w, 3), np.uint8)
    cv2.putText(sheet, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    picks = np.unique(np.linspace(0, len(thumbs) - 1, SHEET_COLS * SHEET_ROWS).round().astype(int)) if thumbs else []
    for k, i in enumerate(picks):
        idx, img = thumbs[i]
        y, x = 40 + (k // SHEET_COLS) * h, (k % SHEET_COLS) * w
        sheet[y:y + h, x:x + w] = img
        cv2.putText(sheet, f"frame {idx}", (x + 8, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return sheet


def _motion_job(job: tuple[str, str, str, float, str, int]) -> tuple[str, int]:
    video_id, path, title, fps, out, threads = job
    os.nice(10)                                    # yield the CPU to AutoClipping's production jobs,
    cv2.setNumThreads(1)                           # and keep each worker to about `threads` cores
    trace, dots, thumbs = motion_trace(Path(path), fps, threads)
    out = Path(out)
    cv2.imwrite(str(out / "sheets" / f"{video_id}.jpg"), contact_sheet(thumbs, title), [cv2.IMWRITE_JPEG_QUALITY, 85])
    np.save(out / "overlay" / f"{video_id}.npy", dots)
    tmp = out / "motion" / f"{video_id}.tmp.npy"
    np.save(tmp, trace)
    tmp.replace(out / "motion" / f"{video_id}.npy")   # the trace marks the video done, so it is written last
    return video_id, len(trace)


def priority_frames(motion: np.ndarray, k: int, min_gap: int, rng: np.random.Generator) -> list[tuple[int, str]]:
    """Up to k frames, pairwise at least min_gap apart, in the order they are taken: alternately a
    random frame from the video's top quartile of motion ("high", by rank, ties broken at random,
    so static footage still has one) and from the rest ("low"), starting with a random one of the
    two; when one bin runs out the other continues. A video's quota takes a prefix of this list,
    so any quota is about half high-motion, and an odd one is as likely to lean either way.
    Frames without a motion value are never taken."""
    valid = np.flatnonzero(np.isfinite(motion))
    if len(valid) == 0:
        return []
    ranked = valid[np.lexsort((rng.random(len(valid)), -motion[valid]))]       # most motion first
    n_high = math.ceil(len(valid) * (1 - HIGH_MOTION_QUANTILE))
    queues = {"high": list(rng.permutation(ranked[:n_high])), "low": list(rng.permutation(ranked[n_high:]))}
    chosen: list[tuple[int, str]] = []
    turn = int(rng.integers(2))
    while len(chosen) < k and (queues["high"] or queues["low"]):
        b = ("high", "low")[turn % 2]
        turn += 1
        while queues[b]:
            i = int(queues[b].pop())
            if all(abs(i - j) >= min_gap for j, _ in chosen):
                chosen.append((i, b))
                break
    return chosen


def waterfill(quota: int, capacity: list[int], rng: np.random.Generator) -> list[int]:
    """Split quota among children as equally as their capacities allow: a child gets its equal
    share or all it has, whichever is less, and the rest is shared again among the others. Units
    left after equal shares go to a seeded random subset of the children that can take them."""
    alloc = [0] * len(capacity)
    remaining = quota
    open_ = [i for i, c in enumerate(capacity) if c > 0]
    while remaining > 0 and open_:
        share = remaining // len(open_)
        if share == 0:
            for i in rng.permutation(open_)[:remaining]:
                alloc[i] += 1
            break
        for i in open_:
            give = min(share, capacity[i] - alloc[i])
            alloc[i] += give
            remaining -= give
        open_ = [i for i in open_ if alloc[i] < capacity[i]]
    return alloc


def allocate(tree: dict, quota: int, per_player: int, rng: np.random.Generator) -> dict[str, int]:
    """tree = {role: {cell: {player: {video_id: frames available}}}} -> frames per video, by
    waterfill at every level. A player offers at most per_player frames to each cell; one who
    appears in several cells can exceed it in total (select reports the largest total)."""
    def cap(node, depth: int) -> int:
        if not isinstance(node, dict):
            return int(node)
        total = sum(cap(child, depth + 1) for child in node.values())
        return min(total, per_player) if depth == 3 else total

    out: dict[str, int] = {}

    def split(node: dict, q: int, depth: int) -> None:
        keys = sorted(node)
        for key, a in zip(keys, waterfill(q, [cap(node[k], depth + 1) for k in keys], rng)):
            if isinstance(node[key], dict):
                split(node[key], a, depth + 1)
            elif a:
                out[key] = a

    split(tree, quota, 0)
    return out


def systematic(n_items: int, n_pick: int, rng: np.random.Generator) -> np.ndarray:
    """n_pick of range(n_items), evenly spaced from a random start. Over a list sorted by stratum,
    every stratum gets its proportional share to within one."""
    if n_pick <= 0:
        return np.zeros(0, int)
    step = n_items / n_pick
    return np.floor(rng.random() * step + step * np.arange(n_pick)).astype(int)


def overlay_mask(dots: np.ndarray, guard: int) -> np.ndarray:
    """Frames with a burned-in overlay (at least overlay.MIN_DOTS dots), widened by `guard` frames
    each side: the frames next to one often carry part of it (a text box, a fading skeleton)."""
    flagged = dots >= overlay.MIN_DOTS
    if guard > 0 and flagged.any():
        flagged = np.convolve(flagged.astype(np.int32), np.ones(2 * guard + 1, np.int32), mode="same") > 0
    return flagged


def _cmd_motion(args) -> None:
    rows = eligible(args.videos, args.modes)
    for sub in ("motion", "overlay", "sheets"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    done = lambda vid: all((args.out / sub / f"{vid}.npy").is_file() for sub in ("motion", "overlay"))  # noqa: E731
    jobs = [(r["video_id"], str(args.jobs_dir / r["video_id"] / "source.mp4"),
             f"{r['video_id']}  {r['mode']}  player {r['player_id'][:8]}", _fps(r), str(args.out), args.decode_threads)
            for r in rows if args.redo or not done(r["video_id"])]
    print(f"[sampling] motion: {len(rows)} eligible videos, {len(jobs)} to decode with {args.workers} workers "
          f"x {args.decode_threads} decode threads")
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_motion_job, job) for job in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            vid, n = fut.result()
            print(f"[sampling] {k}/{len(jobs)} {vid}: {n} frames", flush=True)
    print(f"[sampling] traces -> {rel(args.out / 'motion')}, contact sheets -> {rel(args.out / 'sheets')}")


def _cmd_select(args) -> None:
    out_json = args.out / "to_annotate.json"
    if out_json.exists() and not args.force:
        raise SystemExit(f"{rel(out_json)} exists (annotation may be under way); pass --force to redraw")
    rows = eligible(args.videos, args.modes)
    bad, missing = [], []
    for r in rows:
        role = ROLE[r["mode"]]
        r["_role"] = role
        r["_strata"] = {"view": r.get("view", "").strip(), "hand": r.get(HAND_COLUMN[role], "").strip(),
                        "lighting": r.get("lighting", "").strip()}
        for field, value in r["_strata"].items():
            if value not in ALLOWED[field]:
                bad.append(f"{r['video_id']} ({r['mode']}): {HAND_COLUMN[role] if field == 'hand' else field}={value!r}")
                r["_strata"][field] = "unknown"
        if not all((args.out / sub / f"{r['video_id']}.npy").is_file() for sub in ("motion", "overlay")):
            missing.append(r["video_id"])
    if missing:
        raise SystemExit(f"{len(missing)} videos have no motion or overlay trace; run `python -m cricketfm.data.sampling motion`")
    if bad and not args.allow_unfilled:
        raise SystemExit(f"{len(bad)} per-video fields in {rel(args.videos)} are empty or not one of {ALLOWED} "
                         f"(fill them from {rel(args.out / 'sheets')}, or pass --allow-unfilled to sample "
                         f"them as 'unknown'):\n  " + "\n  ".join(bad[:20]))

    traces, plists, tree, masked = {}, {}, {}, {}
    for r in rows:
        vid = r["video_id"]
        traces[vid] = np.load(args.out / "motion" / f"{vid}.npy")
        flagged = overlay_mask(np.load(args.out / "overlay" / f"{vid}.npy"), round(args.overlay_guard_s * _fps(r)))
        masked[vid] = int(flagged.sum())
        clean = np.where(flagged, np.nan, traces[vid])
        gap = max(1, round(args.min_gap_s * _fps(r)))
        plists[vid] = priority_frames(clean, args.per_video, gap, np.random.default_rng(_seed(args.seed, vid)))
        r["_cell"] = "/".join(r["_strata"][f] for f in ("view", "hand", "lighting"))
        tree.setdefault(r["_role"], {}).setdefault(r["_cell"], {}).setdefault(r["player_id"], {})[vid] = len(plists[vid])
    quota = allocate(tree, args.frames, args.per_player, np.random.default_rng(args.seed))

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images, reserve = [], {}
    for r in rows:
        vid, q = r["video_id"], quota.get(r["video_id"], 0)
        if not q:
            continue
        chosen = dict(plists[vid][:q])
        reserve[vid] = [[f, b] for f, b in plists[vid][q:q + args.reserve]]
        for idx, frame in sorted(read_frames(args.jobs_dir / vid / "source.mp4", chosen).items()):
            name = f"{vid}_{idx:06d}.jpg"
            cv2.imwrite(str(images_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            images.append({"file_name": name, "width": frame.shape[1], "height": frame.shape[0],
                           "video_id": vid, "fingerprint": r["fingerprint"], "player_id": r["player_id"],
                           "frame_index": idx, "frames_in_video": len(traces[vid]), "role": r["_role"],
                           **r["_strata"], "cell": r["_cell"], "motion": round(float(traces[vid][idx]), 5),
                           "motion_bin": chosen[idx]})
    order = sorted(range(len(images)), key=lambda i: tuple(images[i][k] for k in
                   ("role", "cell", "motion_bin", "player_id", "video_id", "frame_index")))
    images = [images[i] for i in order]
    double = set(systematic(len(images), round(args.double * len(images)), np.random.default_rng(args.seed)).tolist())
    for k, im in enumerate(images):
        im["id"] = k + 1
        im["double_annotate"] = k in double

    counts = {role: {"frames": sum(im["role"] == role for im in images),
                     "players": len({im["player_id"] for im in images if im["role"] == role}),
                     "videos": len({im["video_id"] for im in images if im["role"] == role}),
                     "cells": dict(sorted(Counter(im["cell"] for im in images if im["role"] == role).items())),
                     "motion_bins": dict(Counter(im["motion_bin"] for im in images if im["role"] == role)),
                     "most_frames_one_player": max(Counter(im["player_id"] for im in images if im["role"] == role).values()),
                     "double_annotate": sum(im["double_annotate"] for im in images if im["role"] == role),
                     "overlay_frames_excluded": sum(masked[r["video_id"]] for r in rows if r["_role"] == role),
                     "frames_in_videos": sum(len(traces[r["video_id"]]) for r in rows if r["_role"] == role)}
              for role in sorted({im["role"] for im in images})}
    category = {"id": 1, "name": "person", "supercategory": "person", "keypoints": list(skeleton.COCO17),
                "skeleton": [[a + 1, b + 1] for a, b in skeleton.EDGES]}
    settings = {k: v for k, v in vars(args).items() if k not in ("cmd", "func", "force")}
    out_json.write_text(json.dumps({"images": images, "annotations": [], "categories": [category], "reserve": reserve,
                                    "sampling": settings | {"counts": counts, "unfilled_fields": len(bad)}},
                                   indent=1, default=str))
    if len(images) < args.frames:
        print(f"[sampling] only {len(images)} of {args.frames} frames available under the caps")
    stale = {f.name for f in images_dir.glob("*.jpg")} - {im["file_name"] for im in images}
    if stale:
        print(f"[sampling] {len(stale)} images in {rel(images_dir)} are from an earlier draw, not this one: "
              f"remove them before importing the folder into the annotation tool")
    for role, c in counts.items():
        print(f"[sampling] {role}: {c['frames']} frames, {c['players']} players (largest {c['most_frames_one_player']}), "
              f"{c['videos']} videos, motion {c['motion_bins']}, {c['double_annotate']} double-annotated; "
              f"{c['overlay_frames_excluded']} of {c['frames_in_videos']} frames excluded as overlay")
        for cell, n in c["cells"].items():
            print(f"[sampling]   {cell:32s} {n}")
    print(f"[sampling] -> {rel(images_dir)}, {rel(out_json)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=Path, default=DATA / "videos.csv")
    p.add_argument("--jobs-dir", type=Path, default=AUTOCLIP_DATA / "jobs")
    p.add_argument("--modes", nargs="+", default=["batsman", "bowler"], choices=sorted(ROLE))
    p.add_argument("--out", type=Path, default=DATA / "pose")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("motion", help="motion trace and contact sheet per video (CPU, cached)")
    m.add_argument("--workers", type=int, default=3, help="the box also serves production: keep workers x threads well under its cores")
    m.add_argument("--decode-threads", type=int, default=2)
    m.add_argument("--redo", action="store_true", help="recompute traces that exist")
    m.set_defaults(func=_cmd_motion)
    s = sub.add_parser("select", help="draw the stratified sample and extract its frames")
    s.add_argument("--frames", type=int, default=800)
    s.add_argument("--per-video", type=int, default=16)
    s.add_argument("--per-player", type=int, default=120)
    s.add_argument("--min-gap-s", type=float, default=1.0)
    s.add_argument("--double", type=float, default=0.10, help="share of frames a second annotator labels, blind")
    s.add_argument("--reserve", type=int, default=4, help="replacement frames recorded per video")
    s.add_argument("--overlay-guard-s", type=float, default=0.5, help="also skip frames this close to an overlay frame")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--allow-unfilled", action="store_true", help="sample videos with empty per-video fields as 'unknown'")
    s.add_argument("--force", action="store_true", help="overwrite an existing to_annotate.json")
    s.set_defaults(func=_cmd_select)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
