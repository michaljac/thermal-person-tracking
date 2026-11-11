
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


class TimeManager:
    def __init__(self, enabled=True, warmup_frames=0):
        self.enabled = enabled
        self.warmup_frames = warmup_frames
        self.timings = defaultdict(list)
        self.frame_index = 0

    @contextmanager
    def measure(self, stage):
        if not self.enabled:
            yield
            return

        start = time.perf_counter()

        try:
            yield
        finally:
            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            if self.frame_index >= self.warmup_frames:
                self.timings[stage].append(
                    (self.frame_index, elapsed_ms)
                )

    def next_frame(self):
        self.frame_index += 1

    def get_stats(self):
        stats = {}

        for stage, records in self.timings.items():
            values = np.array(
                [elapsed for _, elapsed in records]
            )

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
        stats = self.get_stats()

        if not stats:
            print("No profiling data available.")
            return

        print("\n===== PROFILING REPORT =====")

        for stage, data in sorted(
            stats.items(),
            key=lambda item: item[1]["total"],
            reverse=True
        ):
            print(
                f"{stage:<18} "
                f"Mean: {data['mean']:7.2f} ms | "
                f"P95: {data['p95']:7.2f} ms"
            )

        stages = {
            name: data
            for name, data in stats.items()
            if name != "total_frame"
        }

        if stages:
            bottleneck = max(
                stages,
                key=lambda name: stages[name]["total"]
            )

            print(f"\nMain bottleneck: {bottleneck}")

        if "total_frame" in stats:
            mean_ms = stats["total_frame"]["mean"]

            if mean_ms > 0:
                print(
                    f"Estimated FPS: {1000 / mean_ms:.2f}"
                )

    def save_figures(self, output_dir, dpi=150):
        if not self.enabled:
            return

        stats = self.get_stats()

        if not stats:
            print("No profiling data to plot.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Exclude aggregate timing from stage comparison
        stages = [
            stage for stage in stats
            if stage != "total_frame"
        ]

        if stages:
            stages.sort(
                key=lambda s: stats[s]["mean"],
                reverse=True
            )

            # Figure 1: Average latency
            means = [stats[s]["mean"] for s in stages]

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.barh(stages, means)
            ax.invert_yaxis()
            ax.set_xlabel("Average latency (ms)")
            ax.set_title("Average Processing Time per Stage")
            ax.grid(axis="x", alpha=0.3)

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_dir, "average_latency.png"
                ),
                dpi=dpi
            )
            plt.close(fig)

            # Figure 2: Mean / P95 / Max
            x = np.arange(len(stages))
            width = 0.25

            fig, ax = plt.subplots(figsize=(11, 5))

            ax.bar(
                x - width,
                [stats[s]["mean"] for s in stages],
                width,
                label="Mean"
            )

            ax.bar(
                x,
                [stats[s]["p95"] for s in stages],
                width,
                label="P95"
            )

            ax.bar(
                x + width,
                [stats[s]["max"] for s in stages],
                width,
                label="Max"
            )

            ax.set_xticks(x)
            ax.set_xticklabels(
                stages, rotation=30, ha="right"
            )
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Latency Statistics by Stage")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_dir, "latency_statistics.png"
                ),
                dpi=dpi
            )
            plt.close(fig)

        # Figure 3: Per-frame latency
        fig, ax = plt.subplots(figsize=(12, 5))

        for stage, records in self.timings.items():
            if not records:
                continue

            frame_ids = [r[0] for r in records]
            latencies = [r[1] for r in records]

            ax.plot(
                frame_ids,
                latencies,
                label=stage,
                linewidth=1
            )

        ax.set_xlabel("Frame index")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Processing Latency Over Frames")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir, "latency_over_frames.png"
            ),
            dpi=dpi
        )
        plt.close(fig)

        print(f"\nProfiling figures saved to: {output_dir}")
