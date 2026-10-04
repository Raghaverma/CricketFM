"""Inventory of the cricket footage on this box: AutoClipping's job videos.

    python -m cricketfm.data.inventory [--jobs-dir ../AutoClipping/data/jobs]

Reads each job's job_meta.json and source.mp4 (ffprobe + content fingerprint; nothing is
decoded or modified) and writes:
  data/videos.csv                 one row per video, private (gitignored). Columns after `notes`
                                  are for a person to fill in (batting hand, camera view, ...) and
                                  survive re-runs.
  results/data_inventory.json     aggregate counts only (committed).

Identity comes from the job metadata's pseudonymous player_id; display names and video URLs are
never copied. Duplicate uploads (same content fingerprint under different job ids) are marked
with duplicate_of, and every split treats them as one video.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from cricketfm.data.video import content_fingerprint, probe
from cricketfm.paths import AUTOCLIP_DATA, DATA, RESULTS, rel

AUTO_COLUMNS = ["video_id", "mode", "analysis_type", "player_id", "fingerprint", "duplicate_of", "width",
                "height", "rotation", "codec", "fps_nominal", "fps_avg", "vfr", "duration_s", "frames_container", "error"]
HUMAN_COLUMNS = ["content", "rotation", "batting_hand", "bowling_arm", "view", "lighting", "session", "usable", "notes"]


def scan(jobs_dir: Path) -> list[dict]:
    rows, first_seen = [], {}
    for meta_path in sorted(jobs_dir.glob("*/job_meta.json")):
        meta = json.loads(meta_path.read_text())
        source = meta_path.parent / "source.mp4"
        if not source.is_file():
            continue
        fp = content_fingerprint(source)
        row = {"video_id": meta["job_id"], "mode": meta["mode"], "analysis_type": meta["meta"].get("analysis_type"),
               "player_id": meta["meta"].get("player_id"), "fingerprint": fp, "duplicate_of": first_seen.get(fp, "")}
        first_seen.setdefault(fp, meta["job_id"])
        try:
            row.update(probe(source))
        except RuntimeError as exc:      # e.g. a truncated upload with no moov atom
            row["error"] = str(exc).splitlines()[-1][-200:]
        rows.append(row)
    return rows


def merge_human_columns(rows: list[dict], existing: Path) -> None:
    if not existing.is_file():
        return
    with open(existing, newline="") as f:
        old = {r["video_id"]: r for r in csv.DictReader(f)}
    for row in rows:
        for col in HUMAN_COLUMNS:
            row[col] = old.get(row["video_id"], {}).get(col, "")


def summarize(rows: list[dict]) -> dict:
    unreadable = [r for r in rows if r.get("error")]
    unique = [r for r in rows if not r["duplicate_of"] and not r.get("error")]
    out = {"videos": len(rows), "unreadable": len(unreadable),
           "duplicate_uploads": sum(bool(r["duplicate_of"]) for r in rows), "by_mode": {}}
    for mode in sorted({r["mode"] for r in unique}):
        rs = [r for r in unique if r["mode"] == mode]
        per_player = Counter(r["player_id"] for r in rs)
        durations = [r["duration_s"] for r in rs if r["duration_s"]]
        out["by_mode"][mode] = {
            "videos": len(rs),
            "analysis_types": dict(Counter(r["analysis_type"] for r in rs)),
            "players": len(per_player),
            "videos_per_player": sorted(per_player.values(), reverse=True),
            "duration_s_total": round(sum(durations), 1),
            "duration_s_min_median_max": [min(durations), sorted(durations)[len(durations) // 2], max(durations)]
            if durations else None,
            "vfr_videos": sum(bool(r["vfr"]) for r in rs),
            "resolutions": dict(Counter(f"{r['width']}x{r['height']}" for r in rs).most_common()),
            "fps_nominal": dict(Counter(str(r["fps_nominal"]) for r in rs).most_common()),
        }
    players = defaultdict(set)
    for r in unique:
        players[r["player_id"]].add(r["mode"])
    out["players_in_both_modes"] = sum(len(m) > 1 for m in players.values())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs-dir", type=Path, default=AUTOCLIP_DATA / "jobs")
    p.add_argument("--csv", type=Path, default=DATA / "videos.csv")
    p.add_argument("--summary", type=Path, default=RESULTS / "data_inventory.json")
    args = p.parse_args()

    rows = scan(args.jobs_dir)
    merge_human_columns(rows, args.csv)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUTO_COLUMNS + HUMAN_COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows)
    summary = {"source": "AutoClipping job videos", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **summarize(rows)}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[inventory] {len(rows)} videos -> {rel(args.csv)}; summary -> {rel(args.summary)}")
    for mode, m in summary["by_mode"].items():
        print(f"[inventory] {mode}: {m['videos']} unique videos, {m['players']} players, "
              f"{m['duration_s_total'] / 60:.1f} min, {m['vfr_videos']} VFR")


if __name__ == "__main__":
    main()
