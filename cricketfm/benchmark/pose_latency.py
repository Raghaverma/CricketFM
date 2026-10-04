"""PoseFM cost: latency, throughput, memory and energy of each pose model (docs/PROTOCOL.md §4.3, Q1).

    python -m cricketfm.benchmark.pose_latency --models vitpose-plus-l sapiens-0.3b-goliath

Two measurements per model, both on the same COCO val2017 people (public, so the inputs can be
named in the result file), each model with its own preprocessing:

  model latency  forward pass only, on a batch of B distinct preprocessed crops already on the
                 GPU, at every batch size in configs/benchmark.yaml. Distinct crops, not B copies
                 of one: the TensorRT repo's Stage 4 lesson. Timed with evaluation/latency.time_calls
                 after discarded warm-up calls.
  end to end     frame + person box -> COCO-17 keypoints, one person per frame (the cricket case:
                 the striker), split into preprocess / forward / decode with a sync at each
                 boundary, and timed again without the inner syncs. The two totals must agree
                 within 10% (the TensorRT repo's check for a misplaced sync). Video decode and
                 person detection are identical for every pose model and are not included.

Every model x batch cell is one unit run under benchmark/runner.guarded(): it starts only when
the shared box is quiet and is re-run if anything overlapped it; the models are released while
a production job runs. Cells are interleaved across rounds (all models at batch 1, 2, ... then
again), because the power-capped L4's clocks drift a few percent between runs.

Memory per cell: the model's weights, its input batch, and the activation peak above both
(torch.cuda.max_memory_allocated minus what was allocated at the start of the cell), so two
models loaded side by side don't inflate each other's numbers. Energy per image is the cell's
mean NVML board power x its mean latency / B.

Output: results/pose_latency/<tag>.json (committed).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import cv2
import numpy as np
import torch
import yaml

from cricketfm.benchmark import env, runner
from cricketfm.evaluation.latency import StageClock, sync_noise_floor_ms, time_calls
from cricketfm.models.pose import base
from cricketfm.paths import CONFIGS, COCO_DIR, RESULTS, rel

CONFIG = CONFIGS / "benchmark.yaml"
E2E_AGREEMENT = 0.10


def coco_people(n: int) -> list[dict]:
    """The first n non-crowd COCO val2017 people with >= 10 labeled joints, one per image."""
    gt = json.loads((COCO_DIR / "annotations" / "person_keypoints_val2017.json").read_text())
    images = {im["id"]: im for im in gt["images"]}
    out, seen = [], set()
    for a in sorted(gt["annotations"], key=lambda a: (a["image_id"], a["id"])):
        if a["num_keypoints"] >= 10 and not a["iscrowd"] and a["image_id"] not in seen:
            seen.add(a["image_id"])
            out.append({"file_name": images[a["image_id"]]["file_name"], "annotation_id": a["id"],
                        "box": np.array([a["bbox"]], np.float32)})
            if len(out) == n:
                break
    for p in out:
        p["frame"] = cv2.imread(str(COCO_DIR / "val2017" / p["file_name"]), cv2.IMREAD_COLOR)
    return out


class Models(runner.GpuResources):
    def __init__(self, names: list[str], precision: str):
        self.names, self.precision, self.models = names, precision, {}

    def load(self) -> None:
        self.models = {n: base.load(n, precision=self.precision) for n in self.names}

    def release(self) -> None:
        self.models = {}


def weight_mb(model) -> float:
    m = getattr(model, "model")
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 2**20


def model_cell(models: Models, name: str, people: list[dict], b: int, cfg: dict):
    def fn():
        m = models.models[name]
        torch.cuda.synchronize()
        x = torch.cat([m.preprocess(p["frame"], p["box"]) for p in people[:b]])
        with torch.inference_mode():
            for _ in range(cfg["warmup"]):
                m.forward(x)
            torch.cuda.synchronize()
            base_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            stats = time_calls(lambda: m.forward(x), b, "cuda", cfg["iters"], warmup=0)
        stats["memory_mb"] = {"weights": round(weight_mb(m), 1), "input": round(x.numel() * x.element_size() / 2**20, 1),
                              "activation_peak": round((torch.cuda.max_memory_allocated() - base_alloc) / 2**20, 1)}
        return None, stats
    return fn


def e2e_cell(models: Models, name: str, people: list[dict]):
    def fn():
        m = models.models[name]
        with torch.inference_mode():
            for p in people[:5]:                                   # warm-up
                m.predict(p["frame"], p["box"])
            clock = StageClock("cuda")
            for p in people:
                with clock("total_synced"):
                    with clock("preprocess"):
                        x = m.preprocess(p["frame"], p["box"])
                    with clock("forward"):
                        raw = m.forward(x)
                    with clock("decode"):
                        m.decode(raw, p["box"])
            plain = []
            for p in people:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                m.predict(p["frame"], p["box"])
                torch.cuda.synchronize()
                plain.append((time.perf_counter() - t0) * 1000)
        summary = clock.summary()
        stages = {k: round(v["mean_ms"], 3) for k, v in summary.items()}
        plain_mean = statistics.mean(plain)
        return None, {"mean_ms": stages, "plain_total_mean_ms": round(plain_mean, 3),
                      "plain_total_p95_ms": sorted(plain)[max(0, int(0.95 * len(plain)) - 1)],
                      "frames_per_s": round(1000 / plain_mean, 2),
                      "synced_vs_plain": round(stages["total_synced"] / plain_mean, 4)}
    return fn


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=cfg["batch_sizes"])
    p.add_argument("--rounds", type=int, default=cfg["rounds"])
    p.add_argument("--iters", type=int, default=cfg["iters"])
    p.add_argument("--tag", default=time.strftime("%Y%m%dT%H%M%S"))
    args = p.parse_args()
    cfg.update(iters=args.iters)
    torch.backends.cudnn.benchmark = cfg["cudnn_benchmark"]

    people = coco_people(max(max(args.batch_sizes), cfg["e2e_frames"]))
    models = Models(args.models, cfg["precision"])
    log = lambda msg: print(msg, flush=True)
    cells: dict[str, dict] = {n: {"model_latency": {}, "end_to_end": []} for n in args.models}
    for r in range(args.rounds):
        for b in args.batch_sizes:
            for name in args.models:
                try:
                    _, st = runner.guarded(model_cell(models, name, people, b, cfg), label=f"{name} b{b} r{r}",
                                           log=log, resources=models)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    st = {"oom": True}
                cells[name]["model_latency"].setdefault(str(b), []).append(st)
                if "mean_ms" in st:
                    log(f"[pose_latency] r{r} {name:22s} b{b:<2d} mean {st['mean_ms']:8.2f} ms  p95 {st['p95_ms']:8.2f}"
                        f"  {st['fps']:7.1f} img/s  act {st['memory_mb']['activation_peak']:7.0f} MB")
        for name in args.models:
            _, st = runner.guarded(e2e_cell(models, name, people[:cfg["e2e_frames"]]), label=f"{name} e2e r{r}",
                                   log=log, resources=models)
            cells[name]["end_to_end"].append(st)
            log(f"[pose_latency] r{r} {name:22s} e2e {st['plain_total_mean_ms']:.2f} ms/frame {st['mean_ms']}")

    summary = {}
    for name, c in cells.items():
        rows = {}
        for b, runs in c["model_latency"].items():
            ok = [s for s in runs if "mean_ms" in s]
            if not ok:
                rows[b] = {"oom": True}
                continue
            mean = statistics.median(s["mean_ms"] for s in ok)
            power = statistics.median(s["gpu"]["power_mean_w"] for s in ok)
            rows[b] = {"mean_ms": round(mean, 3), "p95_ms": round(statistics.median(s["p95_ms"] for s in ok), 3),
                       "img_per_s": round(1000 / mean * int(b), 2), "power_w": round(power, 1),
                       "joules_per_img": round(power * mean / 1000 / int(b), 4),
                       "sm_clock_mhz": round(statistics.median(s["gpu"]["sm_clock_mean_mhz"] for s in ok)),
                       "memory_mb": ok[-1]["memory_mb"],
                       "round_spread_pct": round((max(s["mean_ms"] for s in ok) / min(s["mean_ms"] for s in ok) - 1) * 100, 2)}
        e2e = c["end_to_end"]
        agree = all(abs(s["synced_vs_plain"] - 1) <= E2E_AGREEMENT for s in e2e)
        summary[name] = {"model": models.models[name].describe() if models.models else name,
                         "model_latency": rows,
                         "end_to_end_ms_per_frame": round(statistics.median(s["plain_total_mean_ms"] for s in e2e), 3),
                         "end_to_end_stages_ms": e2e[-1]["mean_ms"], "end_to_end_sync_check_passed": agree}

    out = {"tag": args.tag, "summary": summary, "cells": cells,
           "protocol": {**cfg, "rounds": args.rounds, "batch_sizes": args.batch_sizes,
                        "inputs": f"first {len(people)} COCO val2017 people (>= 10 labeled joints, one per image), GT boxes",
                        "sync_noise_floor_ms": round(sync_noise_floor_ms("cuda"), 4)},
           "environment": env.fingerprint()}
    path = RESULTS / "pose_latency" / f"{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    log(f"[pose_latency] wrote {rel(path)}")


if __name__ == "__main__":
    main()
