"""Run timed measurements as short, contention-guarded units that are re-run until clean.

Ported from the TensorRT repo (pipeline/runner.py: guarded(), GpuResources), minus its video
chunking. A unit (one model x batch size cell of a latency sweep, say) starts only once the box is
quiet (gpu_guard.wait_until_quiet), runs under gpu_guard.GpuMonitor, and is simply re-run if
anything overlapped it, with no attempt limit: waiting is cheap, a contaminated number is not.
While a production job runs, the models are released so their VRAM can't be what it fails to
allocate, and reloaded (and re-warmed) once the box is quiet again.
"""
from __future__ import annotations

import os
import threading
import time

from cricketfm.benchmark import gpu_guard

_last_clean_end = float("-inf")

# Set by the running unit's GpuMonitor the moment a prod job or a new GPU process appears; long
# measurements poll it and stop early by raising Aborted.
ABORT = threading.Event()


class Aborted(Exception):
    """A measurement stopped early because the box stopped being quiet."""


class GpuResources:
    """What a benchmark holds on the GPU, so it can be given back while a prod job runs (a prod
    job takes ~7 GB and up to 3 run at once on this L4). Subclasses implement load() (build and
    warm up) and release()."""

    loaded = False
    warm = None     # optional callable run after every (re)load: a unit after a reload isn't measured cold

    def load(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def ensure_loaded(self, log=print) -> bool:
        """Load (and warm) if released. Returns True if it loaded now."""
        if self.loaded:
            return False
        t = time.perf_counter()
        self.load()
        self.loaded = True
        if self.warm is not None:
            self.warm()
        log(f"[runner] GPU state loaded + warmed up ({time.perf_counter() - t:.0f}s)")
        return True

    def ensure_released(self, log=print) -> None:
        if self.loaded:
            import gc

            import torch
            self.release()
            gc.collect()
            torch.cuda.empty_cache()
            self.loaded = False
            log(f"[runner] released GPU memory while the box is busy "
                f"({torch.cuda.memory_reserved() / 2**20:.0f} MB still reserved by PyTorch)")


def guarded(fn, own_pids: set[int] | None = None, label: str = "", log=print,
            resources: GpuResources | None = None):
    """Run fn() -> (result, stats dict) until one run overlaps nothing on the box. Returns
    (result, stats) with stats["gpu"] = the clean window's monitor report and
    stats["contaminated_attempts"] = the discarded ones."""
    global _last_clean_end
    own = own_pids or {os.getpid()}
    discarded = []
    while True:
        # A unit that follows a clean one within seconds skips the quiet window: that unit's
        # monitor just watched the box, and this one's will catch whatever starts now.
        if discarded or time.monotonic() - _last_clean_end > 10 or gpu_guard.prod_lock_holders():
            gpu_guard.wait_until_quiet(own, log=log, on_prod=None if resources is None
                                       else lambda: resources.ensure_released(log))
        ABORT.clear()
        if resources is not None and resources.ensure_loaded(log):
            # Something may have started while we reloaded; it must not count as resident.
            gpu_guard.wait_until_quiet(own, quiet_s=10.0, log=log)
        aborted = False
        with gpu_guard.GpuMonitor(own, abort=ABORT) as mon:
            try:
                result, stats = fn()
            except Aborted:
                aborted = True
        rep = mon.report()
        if aborted and rep["clean"]:
            rep["clean"] = False
            rep["aborted_without_cause"] = True
        if rep["clean"]:
            _last_clean_end = time.monotonic()
            stats["gpu"] = rep
            stats["contaminated_attempts"] = discarded
            return result, stats
        discarded.append(rep)
        if resources is not None and (rep["prod_lock_holders"] or rep["new_foreign_gpu_pids"]):
            resources.ensure_released(log)
        why = ["stopped early"] if aborted else []
        if rep["prod_lock_holders"]:
            why.append(f"prod job {rep['prod_lock_holders']}")
        if rep["new_foreign_gpu_pids"] or rep["foreign_pids_that_grew_vram"]:
            why.append(f"GPU pids {rep['new_foreign_gpu_pids'] + rep['foreign_pids_that_grew_vram']}")
        if rep["foreign_cpu_cores_mean"] > gpu_guard.FOREIGN_CPU_MAX_CORES:
            why.append(f"foreign CPU {rep['foreign_cpu_cores_mean']:.1f} cores")
        log(f"[runner] {label}: overlapped {', '.join(why)} -- discarded, re-running (attempt {len(discarded) + 1})")
