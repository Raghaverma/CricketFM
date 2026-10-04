"""One CSV row per model x measurement, with the record fields the plan asks for (its §7):
model, model_version, input_resolution, GPU, VRAM, batch_size, precision, preprocessing_time,
inference_time, postprocessing_time, total_latency, FPS, peak_VRAM, accuracy.

    python -m cricketfm.benchmark.table results/pose_latency/<tag>.json      # -> <tag>.csv beside it

Two kinds of row, told apart by `measurement`:
  model       forward pass on a preprocessed batch; inference_time is ms per batch, FPS images/s,
              peak_VRAM = weights + input + activation peak (MB)
  end_to_end  frame + box -> keypoints, one person per frame; the three stage times and
              total_latency are ms per frame
Accuracy is the model's COCO val2017 AP from results/coco_check/<model>.json when it exists
(cricket accuracy comes from pose_eval once the labels do).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cricketfm.paths import RESULTS, rel

COLUMNS = ["model", "model_version", "measurement", "input_resolution", "GPU", "VRAM_mb", "batch_size", "precision",
           "preprocessing_time_ms", "inference_time_ms", "postprocessing_time_ms", "total_latency_ms", "FPS",
           "peak_VRAM_mb", "joules_per_image", "accuracy_coco_ap"]


def rows(result: dict) -> list[dict]:
    envr = result["environment"]
    out = []
    for name, s in result["summary"].items():
        m = s["model"] if isinstance(s["model"], dict) else {"name": name}
        coco = RESULTS / "coco_check" / f"{name}.json"
        common = {"model": name, "model_version": m.get("checkpoint", ""),
                  "input_resolution": "x".join(map(str, m.get("input_hw", []))), "GPU": envr.get("gpu_name", ""),
                  "VRAM_mb": envr.get("gpu_memory_mb", ""), "precision": result["protocol"]["precision"],
                  "accuracy_coco_ap": round(json.loads(coco.read_text())["coco_ap"]["AP"], 4) if coco.is_file() else ""}
        for b, cell in s["model_latency"].items():
            if cell.get("oom"):
                out.append({**common, "measurement": "model", "batch_size": b, "inference_time_ms": "OOM"})
                continue
            mem = cell["memory_mb"]
            out.append({**common, "measurement": "model", "batch_size": b, "inference_time_ms": cell["mean_ms"],
                        "FPS": cell["img_per_s"], "peak_VRAM_mb": round(mem["weights"] + mem["input"] + mem["activation_peak"], 1),
                        "joules_per_image": cell["joules_per_img"]})
        st = s["end_to_end_stages_ms"]
        out.append({**common, "measurement": "end_to_end", "batch_size": 1, "preprocessing_time_ms": st["preprocess"],
                    "inference_time_ms": st["forward"], "postprocessing_time_ms": st["decode"],
                    "total_latency_ms": s["end_to_end_ms_per_frame"], "FPS": round(1000 / s["end_to_end_ms_per_frame"], 2)})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result", type=Path)
    args = p.parse_args()
    out = args.result.with_suffix(".csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows(json.loads(args.result.read_text())))
    print(f"[table] wrote {rel(out)}")


if __name__ == "__main__":
    main()
