"""Video access under one rule: a frame is identified by its decoded index.

Most batting footage on this box is variable-frame-rate phone video (20 of 24 sampled stance
jobs), where container timestamps and nominal fps disagree with what a decoder yields;
AutoClipping saw clip times drift by more than a second on 21 of 54 batting sources. So every
frame reference in CricketFM (pose labels, clip boundaries, contact frames) is the index of the
frame in a sequential cv2 decode from the start of the file, and frames for annotation are
written out as images by that same decode, never by seeking. Seconds appear only as nominal
durations (e.g. a sampling gap), converted with the average frame rate.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


def content_fingerprint(path: Path) -> str:
    """sha1 of the first 16 MB + file size: the TensorRT repo's fingerprint (calibration/
    real_corpus.py), so the two repos name the same footage the same way. Re-uploads of one video
    under different job ids share it."""
    with open(path, "rb") as f:
        head = f.read(16 << 20)
    return f"{hashlib.sha1(head).hexdigest()[:16]}-{Path(path).stat().st_size}"


def _rate(text: str | None) -> float | None:
    try:
        r = Fraction(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return float(r) if r > 0 else None


def probe(path: Path) -> dict:
    """Container metadata from ffprobe, no decoding. VFR when the nominal (r_frame_rate) and
    average frame rates differ by more than 1%."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,codec_name,r_frame_rate,avg_frame_rate,nb_frames,duration"
                          ":stream_side_data=rotation:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    info = json.loads(out.stdout)
    s = info["streams"][0]
    nominal, avg = _rate(s.get("r_frame_rate")), _rate(s.get("avg_frame_rate"))
    rotation = 0
    for side in s.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(side["rotation"])
    duration = s.get("duration") or info.get("format", {}).get("duration")
    return {
        "width": s.get("width"), "height": s.get("height"), "codec": s.get("codec_name"),
        "rotation": rotation,
        "fps_nominal": round(nominal, 3) if nominal else None,
        "fps_avg": round(avg, 3) if avg else None,
        "vfr": bool(nominal and avg and abs(nominal - avg) / nominal > 0.01),
        "duration_s": round(float(duration), 3) if duration else None,
        "frames_container": int(s["nb_frames"]) if s.get("nb_frames", "").isdigit() else None,
    }


def open_video(path: Path, threads: int | None = None) -> cv2.VideoCapture:
    """threads caps FFmpeg's decode threads (default: FFmpeg's choice, about one per core); it
    changes speed, not the decoded frames."""
    cap = (cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads]) if threads
           else cv2.VideoCapture(str(path)))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    return cap


def count_frames(path: Path) -> int:
    """Frames cv2 actually decodes (container counts disagree on phone footage)."""
    cap = open_video(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def read_frames(path: Path, indices, threads: int | None = None) -> dict[int, np.ndarray]:
    """BGR frames at the given decoded indices, by one sequential pass (grab() skips the
    conversion of frames that aren't wanted). Raises if the video ends before the last index.
    threads caps the decoder's threads (open_video)."""
    wanted = sorted(set(int(i) for i in indices))
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    cap = open_video(path, threads)
    try:
        for i in range(wanted[-1] + 1):
            if not cap.grab():
                raise RuntimeError(f"{path} ended at frame {i}, before frame {wanted[-1]}")
            if i in wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"{path}: could not decode frame {i}")
                out[i] = frame
    finally:
        cap.release()
    return out
