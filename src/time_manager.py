"""
Per-stage latency profiler.

    timer = TimeManager(enabled=True, warmup_frames=10)
    with timer.measure("detection"):
        ...
    timer.next_frame()
    timer.print_report()   # mean / P95 per stage, bottleneck, estimated FPS

The first `warmup_frames` frames are ignored (model loading and caches make
them slow and unrepresentative).
"""
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class TimeManager:
    def __init__(self, enabled=True, warmup_frames=0):
        self.enabled = enabled
        self.warmup_frames = warmup_frames
        self.timings = defaultdict(list)  # stage -> [(frame_index, ms), ...]
        self.frame_index = 0

    @contextmanager
    def measure(self, stage):
        """Time the code inside the `with` block and record it under `stage`."""
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if self.frame_index >= self.warmup_frames:
                self.timings[stage].append((self.frame_index, elapsed_ms))

    def next_frame(self):
        self.frame_index += 1

    def get_stats(self):
        """Per stage: mean, median, p95, max, total (ms) and count."""
        stats = {}
        for stage, records in self.timings.items():
            values = np.array([elapsed for _, elapsed in records])
            if len(values) == 0:
                continue
            stats[stage] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
                "total": float(np.sum(values)),
                "count": len(values),
            }
        return stats

    def print_report(self):
        """Print mean / P95 per stage (slowest first), the bottleneck and estimated FPS."""
        stats = self.get_stats()
        if not stats:
            print("No profiling data available.")
            return

        print("\n===== PROFILING REPORT =====")
        for stage, data in sorted(stats.items(), key=lambda item: item[1]["total"], reverse=True):
            print(f"{stage:<18} Mean: {data['mean']:7.2f} ms | P95: {data['p95']:7.2f} ms")

        stages = {name: data for name, data in stats.items() if name != "total_frame"}
        if stages:
            bottleneck = max(stages, key=lambda name: stages[name]["total"])
            print(f"\nMain bottleneck: {bottleneck}")

        if "total_frame" in stats and stats["total_frame"]["mean"] > 0:
            print(f"Estimated FPS: {1000 / stats['total_frame']['mean']:.2f}")
