"""Timing, done the TensorRT repo's way so the two repos' numbers can be read side by side.

time_calls() is a port of TensorRT/baseline.py's: each call is bracketed by device syncs and
timed with perf_counter, after warm-up calls that are discarded (the first calls pay for CUDA
context creation, cuDNN algorithm search and allocator growth). Percentiles use that repo's
nearest-rank convention, so a p95 here and a p95 there are the same statistic.

StageClock times the stages of one end-to-end call (video decode, preprocess, forward, decode)
with a sync at every boundary. The TensorRT repo's Stage 1 found such stage sums within 1% of an
unsynced end-to-end run; StageClock users check the same (pose_latency.py does).
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager

import torch


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def summarize(latencies_ms: list[float], batch_size: int = 1) -> dict:
    n = len(latencies_ms)
    s = sorted(latencies_ms)
    mean = statistics.mean(s)
    return {
        "num_iters": n,
        "batch_size": batch_size,
        "mean_ms": mean,
        "std_ms": statistics.pstdev(s),
        "p50_ms": statistics.median(s),
        "p95_ms": s[max(0, int(0.95 * n) - 1)],
        "p99_ms": s[max(0, int(0.99 * n) - 1)],
        "min_ms": s[0],
        "max_ms": s[-1],
        "fps": 1000.0 / mean * batch_size,
    }


def time_calls(call_fn, batch_size: int, device: str, num_iters: int, warmup: int) -> dict:
    """Sync-bracketed wall-clock timing of a zero-argument callable; warm-up calls discarded."""
    for _ in range(warmup):
        call_fn()
    _sync(device)
    latencies_ms = []
    for _ in range(num_iters):
        _sync(device)
        t0 = time.perf_counter()
        call_fn()
        _sync(device)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return summarize(latencies_ms, batch_size)


def sync_noise_floor_ms(device: str, num_iters: int = 1000) -> float:
    """Median time of an empty sync-bracketed call: the resolution below which a stage time
    means nothing (the TensorRT repo measured 0.018 ms on this L4)."""
    return time_calls(lambda: None, 1, device, num_iters, warmup=10)["p50_ms"]


class StageClock:
    """`with clock("forward"): ...` records each stage's wall time (ms), syncing the device on
    both sides of the stage so asynchronous GPU work is charged to the stage that launched it."""

    def __init__(self, device: str):
        self.device = device
        self.ms: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def __call__(self, stage: str):
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.ms[stage].append((time.perf_counter() - t0) * 1000)

    def summary(self, batch_size: int = 1) -> dict[str, dict]:
        return {stage: summarize(v, batch_size) for stage, v in self.ms.items()}
