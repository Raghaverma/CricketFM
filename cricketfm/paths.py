"""Where things live. Every path the package uses is derived here, so a result file can always
name its inputs relative to the repo instead of by a private absolute path."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"                  # gitignored except data/README.md: footage and labels are private
RESULTS = REPO_ROOT / "results"            # committed aggregates (JSON), the source of every figure
RAW = RESULTS / "raw"                      # per-frame / per-clip outputs, gitignored
CHECKPOINTS = REPO_ROOT / "checkpoints"    # gitignored

# COCO val2017 (images + person_keypoints_val2017.json): the public sanity set for every pose
# adapter. Defaults to data/coco, which can be a symlink to the TensorRT repo's copy.
COCO_DIR = Path(os.environ.get("CRICKETFM_COCO_DIR", DATA / "coco"))

# AutoClipping (the production clipper on this box) keeps its job videos and its GPU lockfiles
# here; benchmark/gpu_guard.py watches those locks.
AUTOCLIP_DATA = Path(os.environ.get("AUTOCLIP_DATA_DIR", REPO_ROOT.parent / "AutoClipping" / "data"))


def rel(path: Path) -> str:
    """`path` relative to the repo when it is inside it, else as given."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
