"""ViTPose++ through Hugging Face transformers, held to the TensorRT repo's reference path.

The same processing that repo's Stage 10 scored on COCO val2017 (ViTPose++-L, AP 0.809): HF's
VitPoseImageProcessor for the crop (box padding 1.25, affine warp to 256x192, ImageNet
normalization), the model with an MoE `dataset_index` (0 = the COCO expert production uses), and
HF's post_process_pose_estimation (DARK) on fp32 heatmaps, called on at most 128 crops at a time:
it builds a flat float32 index that misdecodes from ~300 crops per call (TensorRT repo, Stage 6).
No flip test.
"""
from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

from cricketfm.models.pose.base import PoseModel
from cricketfm.skeleton import Mapping

HF_DECODE_CHUNK = 128


class ViTPose(PoseModel):
    family = "vitpose"
    input_hw = (256, 192)

    def __init__(self, name: str, spec: dict, device: str, dtype: torch.dtype):
        self.name, self.spec, self.device, self.dtype = name, spec, device, dtype
        self.dataset_index = spec.get("dataset_index", 0)
        self.processor = AutoProcessor.from_pretrained(spec["checkpoint"])
        self.model = VitPoseForPoseEstimation.from_pretrained(spec["checkpoint"], dtype=dtype).to(device).eval()
        self.mapping = Mapping.identity()       # the COCO expert's head is COCO-17

    def preprocess(self, frame_bgr: np.ndarray, boxes_xywh: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pv = self.processor(rgb, boxes=[np.asarray(boxes_xywh, np.float32)], return_tensors="pt")["pixel_values"]
        return pv.to(device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = torch.full((x.shape[0],), self.dataset_index, dtype=torch.long, device=x.device)
        return self.model(pixel_values=x, dataset_index=index).heatmaps

    def decode(self, raw: torch.Tensor, boxes_xywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hm = raw.float().cpu()
        xy, scores = [], []
        for k in range(0, len(hm), HF_DECODE_CHUNK):
            boxes = [[b.tolist()] for b in np.asarray(boxes_xywh, np.float32)[k:k + HF_DECODE_CHUNK]]
            poses = self.processor.post_process_pose_estimation(SimpleNamespace(heatmaps=hm[k:k + HF_DECODE_CHUNK]),
                                                                boxes=boxes)
            xy += [p[0]["keypoints"].numpy() for p in poses]
            scores += [p[0]["scores"].numpy().reshape(-1) for p in poses]
        return np.stack(xy).astype(np.float32), np.stack(scores).astype(np.float32)

    def describe(self) -> dict:
        return {**super().describe(), "checkpoint": self.spec["checkpoint"], "dataset_index": self.dataset_index,
                "params": sum(p.numel() for p in self.model.parameters()),
                "processing": "HF VitPoseImageProcessor crop (padding 1.25) + HF DARK decode, no flip test"}
