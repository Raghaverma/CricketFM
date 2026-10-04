"""Sapiens (Meta) pose models from their TorchScript exports, without mmpose.

Processing follows Sapiens' own evaluation configs (pose/configs/sapiens_pose/*-1024x768.py),
not its lite demo: the demo fixes the crop to the whole frame's aspect ratio and then resizes,
which stretches people on 16:9 footage. The config path, reimplemented here from mmpose's source:
  - box -> center and scale, padded 1.25 (GetBBoxCenterScale), aspect fixed to w/h = 0.75;
  - UDP affine warp to 768x1024 (cv2.INTER_LINEAR, zero border) (TopdownAffine, use_udp=True);
  - RGB, normalized with the training config's mean/std;
  - the traced network: (n, 3, 1024, 768) -> heatmaps (n, K, 256, 192);
  - UDP decode: argmax, then DARK-UDP refinement on a Gaussian-blurred (kernel 11) log heatmap,
    scaled by (heatmap size - 1), then mapped back through the crop's center and scale.
Scores are the heatmap maxima. No flip test, as for ViTPose (the published Sapiens numbers use one).

Only the channels the COCO-17 mapping reads are decoded: for a Goliath model, 17 of 308.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from cricketfm import skeleton
from cricketfm.models.pose.base import PoseModel

INPUT_HW = (1024, 768)
HEATMAP_HW = (256, 192)
PADDING = 1.25
MEAN = np.array([123.675, 116.28, 103.53], np.float32)     # RGB, from the training config's data_preprocessor
STD = np.array([58.395, 57.12, 57.375], np.float32)
BLUR_KERNEL = 11
GOLIATH308 = json.loads((Path(__file__).with_name("goliath308.json")).read_text())["names"]


# ---- crop (mmpose GetBBoxCenterScale + TopdownAffine with UDP) ----------------------------------

def center_scale(boxes_xywh: np.ndarray, aspect: float = INPUT_HW[1] / INPUT_HW[0]) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
    center = b[:, :2] + b[:, 2:] / 2
    w, h = b[:, 2:3] * PADDING, b[:, 3:4] * PADDING
    scale = np.where(w > h * aspect, np.hstack([w, w / aspect]), np.hstack([h * aspect, h]))
    return center, scale


def udp_warp_matrix(center: np.ndarray, scale: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    """mmpose get_udp_warp_matrix with rot = 0: x' = (x - (cx - sw / 2)) * (W_out - 1) / sw."""
    sx, sy = (out_wh[0] - 1) / scale[0], (out_wh[1] - 1) / scale[1]
    return np.array([[sx, 0.0, sx * (-center[0] + 0.5 * scale[0])],
                     [0.0, sy, sy * (-center[1] + 0.5 * scale[1])]], np.float32)


def warp_crops(frame_bgr: np.ndarray, boxes_xywh: np.ndarray) -> np.ndarray:
    """uint8 BGR (n, 1024, 768, 3) crops."""
    center, scale = center_scale(boxes_xywh)
    out_wh = (INPUT_HW[1], INPUT_HW[0])
    return np.stack([cv2.warpAffine(frame_bgr, udp_warp_matrix(c, s, out_wh), out_wh, flags=cv2.INTER_LINEAR)
                     for c, s in zip(center, scale)])


# ---- decode (mmpose UDPHeatmap.decode, gaussian heatmaps) ---------------------------------------

def _blur(heatmaps: np.ndarray, kernel: int = BLUR_KERNEL) -> np.ndarray:
    """mmpose gaussian_blur on (N, K, H, W): zero-pad, cv2 blur, crop, restore each map's max."""
    border = (kernel - 1) // 2
    n, k, h, w = heatmaps.shape
    out = np.empty_like(heatmaps)
    padded = np.zeros((h + 2 * border, w + 2 * border), np.float32)
    for i in range(n):
        for j in range(k):
            origin_max = heatmaps[i, j].max()
            padded[border:-border, border:-border] = heatmaps[i, j]
            blurred = cv2.GaussianBlur(padded, (kernel, kernel), 0)[border:-border, border:-border]
            out[i, j] = blurred * (origin_max / max(blurred.max(), 1e-12))
    return out


def udp_decode(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, K, H, W) float32 heatmaps -> keypoints (N, K, 2) in heatmap px after DARK-UDP
    refinement, and scores (N, K) = heatmap maxima. Vectorized form of mmpose's
    get_heatmap_maximum + refine_keypoints_dark_udp."""
    n, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, k, -1)
    idx = flat.argmax(-1)
    scores = np.take_along_axis(flat, idx[..., None], -1)[..., 0]
    y, x = np.divmod(idx, w)
    x = np.where(scores > 0, x, -1)
    y = np.where(scores > 0, y, -1)
    log = np.log(np.clip(_blur(heatmaps), 1e-3, 50.0))
    pad = np.pad(log, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    ni, ki = np.meshgrid(np.arange(n), np.arange(k), indexing="ij")
    px, py = x + 1, y + 1

    def at(dy: int, dx: int) -> np.ndarray:
        return pad[ni, ki, py + dy, px + dx]

    i_, ix1, ix1_, iy1, iy1_ = at(0, 0), at(0, 1), at(0, -1), at(1, 0), at(-1, 0)
    ix1y1, ix1_y1_ = at(1, 1), at(-1, -1)
    dx, dy = 0.5 * (ix1 - ix1_), 0.5 * (iy1 - iy1_)
    dxx, dyy = ix1 - 2 * i_ + ix1_, iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.stack([np.stack([dxx, dxy], -1), np.stack([dxy, dyy], -1)], -2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    keypoints = np.stack([x, y], -1).astype(np.float32)
    keypoints -= np.einsum("nkij,nkj->nki", hessian, np.stack([dx, dy], -1)).astype(np.float32)
    return keypoints, scores


def to_frame(keypoints_hm: np.ndarray, boxes_xywh: np.ndarray, heatmap_hw: tuple[int, int] = HEATMAP_HW) -> np.ndarray:
    """Heatmap px -> frame px: UDP scaling to the input (x / (W_hm - 1) * W_in), then mmpose's
    keypoints / input_size * scale + center - scale / 2, which composes to the expression below."""
    center, scale = center_scale(boxes_xywh)
    norm = keypoints_hm / np.array([heatmap_hw[1] - 1, heatmap_hw[0] - 1], np.float32)
    return (norm * scale[:, None] + center[:, None] - 0.5 * scale[:, None]).astype(np.float32)


class Sapiens(PoseModel):
    family = "sapiens"
    input_hw = INPUT_HW

    def __init__(self, name: str, spec: dict, device: str, dtype: torch.dtype):
        self.name, self.spec, self.device, self.dtype = name, spec, device, dtype
        names = skeleton.COCO17 if spec["skeleton"] == "coco17" else GOLIATH308
        self.mapping = skeleton.Mapping.by_name(spec["skeleton"], names)
        self.path = hf_hub_download(spec["repo"], spec["file"])
        self.model = torch.jit.load(self.path, map_location=device).eval().to(dtype)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def preprocess(self, frame_bgr: np.ndarray, boxes_xywh: np.ndarray) -> torch.Tensor:
        crops = torch.from_numpy(warp_crops(frame_bgr, boxes_xywh)).to(self.device)
        x = crops.permute(0, 3, 1, 2).flip(1).float()          # BGR -> RGB, (n, 3, 1024, 768)
        return ((x - self.mean) / self.std).to(self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def decode(self, raw: torch.Tensor, boxes_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        heatmaps = raw[:, torch.from_numpy(self.mapping.index).to(raw.device)].float().cpu().numpy()
        keypoints, scores = udp_decode(heatmaps)
        return to_frame(keypoints, boxes_xywh, heatmaps.shape[2:]), scores.astype(np.float32)

    def describe(self) -> dict:
        return {**super().describe(), "checkpoint": f"{self.spec['repo']}/{self.spec['file']}",
                "params": sum(p.numel() for p in self.model.parameters()),
                "processing": "mmpose eval-config path: UDP affine crop (padding 1.25, w/h 0.75) + "
                              "DARK-UDP decode (blur 11), no flip test"}
