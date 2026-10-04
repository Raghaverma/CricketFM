"""The interface every pose model under test implements, and the registry that builds them.

A top-down pose model is three stages, timed separately by the benchmark (evaluation/latency.py):

    preprocess(frame_bgr, boxes)  frame + person boxes -> model input on the device
    forward(x)                    model input -> raw output (heatmaps), on the device
    decode(raw, boxes)            raw output -> COCO-17 keypoints in frame px (n, 17, 2), scores (n, 17)

Each model keeps its own reference pre- and post-processing (crop padding, aspect handling,
normalization, heatmap decoding) at its native input size. Pushing Sapiens through ViTPose's
256x192 crop would measure a handicapped Sapiens, not Sapiens. What every model shares is the
input (the same frames and the same person boxes) and the output (COCO-17 via skeleton.Mapping).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from cricketfm.paths import CONFIGS
from cricketfm.skeleton import Mapping

REGISTRY = CONFIGS / "pose_models.yaml"
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class PoseModel:
    name: str
    family: str
    input_hw: tuple[int, int]
    mapping: Mapping
    device: str
    dtype: torch.dtype

    def preprocess(self, frame_bgr: np.ndarray, boxes_xywh: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, raw: torch.Tensor, boxes_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray, boxes_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All three stages for one frame: COCO-17 keypoints (n, 17, 2) and scores (n, 17)."""
        if len(boxes_xywh) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        return self.decode(self.forward(self.preprocess(frame_bgr, boxes_xywh)), boxes_xywh)

    def describe(self) -> dict:
        """What a result file records about the model."""
        return {"name": self.name, "family": self.family, "input_hw": list(self.input_hw),
                "native_skeleton": self.mapping.source, "dtype": str(self.dtype).replace("torch.", ""),
                "device": self.device}


@torch.inference_mode()
def pose_people(model: PoseModel, image_paths: list[Path], boxes_xywh: np.ndarray,
                batch: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """COCO-17 keypoints (N, 17, 2) and scores (N, 17) for N people, person i given by the image
    it is in and its box. Each image is read once; crops from consecutive images are batched.
    Before every batch it yields to AutoClipping: if a production job holds a GPU slot, the
    activation memory is released and the run waits (Sapiens at batch 16 holds ~11 GB)."""
    from cricketfm.benchmark import gpu_guard

    rows_by_image = defaultdict(list)
    for row, path in enumerate(image_paths):
        rows_by_image[str(path)].append(row)
    boxes_xywh = np.asarray(boxes_xywh, np.float32)
    pred = np.zeros((len(image_paths), 17, 2), np.float32)
    scores = np.zeros((len(image_paths), 17), np.float32)
    pending: list[tuple[torch.Tensor, list[int]]] = []

    def flush() -> None:
        if not pending:
            return
        if gpu_guard.prod_lock_holders():
            torch.cuda.empty_cache()
            gpu_guard.wait_for_no_prod()
        x = torch.cat([p[0] for p in pending])
        rows = [r for p in pending for r in p[1]]
        k, s = model.decode(model.forward(x), boxes_xywh[rows])
        pred[rows], scores[rows] = k, s
        pending.clear()

    for path, rows in rows_by_image.items():
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(path)
        pending.append((model.preprocess(frame, boxes_xywh[rows]), rows))
        if sum(len(p[1]) for p in pending) >= batch:
            flush()
    flush()
    return pred, scores


def registry(path: Path = REGISTRY) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text())


def load(name: str, device: str = "cuda", precision: str = "fp16") -> PoseModel:
    spec = registry().get(name)
    if spec is None:
        raise SystemExit(f"unknown pose model {name!r}; configured: {sorted(registry())}")
    dtype = DTYPES[precision]
    if spec["family"] == "vitpose":
        from cricketfm.models.pose.vitpose import ViTPose
        return ViTPose(name, spec, device, dtype)
    if spec["family"] == "sapiens":
        from cricketfm.models.pose.sapiens import Sapiens
        return Sapiens(name, spec, device, dtype)
    raise SystemExit(f"{name}: unknown family {spec['family']!r}")
