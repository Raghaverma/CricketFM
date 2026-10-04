"""The environment a number was measured in, recorded inside every result file.

Beyond the TensorRT repo's fingerprint (python, torch, CUDA, cuDNN, transformers, GPU, driver)
this records the L4's power limit and the clocks at the start, because the power-capped card
drifts a few percent between runs, and the foreign GPU processes already resident, because the
box is shared.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from cricketfm.paths import REPO_ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else "not a git repo"
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def fingerprint(device: str = "cuda") -> dict:
    import numpy
    import torch
    import transformers

    fp = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        own = os.getpid()
        fp.update({
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_memory_mb": pynvml.nvmlDeviceGetMemoryInfo(h).total >> 20,
            "nvidia_driver": pynvml.nvmlSystemGetDriverVersion(),
            "power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
            "sm_clock_mhz_at_start": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "sm_clock_max_mhz": pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM),
            "foreign_gpu_processes_at_start": {
                p.pid: (p.usedGpuMemory or 0) >> 20
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h) if p.pid != own},
        })
    return fp
