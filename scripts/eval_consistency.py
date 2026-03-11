#!/usr/bin/env python3
"""Evaluate within-pitcher metric consistency across clips.

Groups pipeline results by pitcher (from pitchers.json), computes
mean, std dev, and coefficient of variation for each metric.

Usage:
    python scripts/eval_consistency.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-pitch metric consistency")
    parser.add_argument("--pitchers", type=Path, default=Path("data/ground_truth/pitchers.json"))
    parser.add_argument("--outputs", type=Path, default=Path("data/outputs"))
    args = parser.parse_args()

    if not args.pitchers.exists():
        print(f"Error: {args.pitchers} not found")
        sys.exit(1)

    pitchers = json.loads(args.pitchers.read_text())

    for pitcher_id, info in pitchers.items():
        clips = info["clips"]
        print(f"\n{'=' * 70}")
        print(f"Pitcher: {pitcher_id} ({info.get('description', '')}) — {len(clips)} clips")
        print(f"{'=' * 70}")

        # Load metrics from each clip
        all_metrics: dict[str, list[float]] = {}
        loaded = 0
        for clip_stem in clips:
            results_path = args.outputs / f"validate_{clip_stem}" / "results.json"
            if not results_path.exists():
                print(f"  Warning: {results_path} not found, skipping")
                continue
            data = json.loads(results_path.read_text())
            loaded += 1
            for key, val in data.get("metrics_raw", {}).items():
                if val is not None:
                    all_metrics.setdefault(key, []).append(val)

        if loaded < 2:
            print(f"  Need at least 2 clips for consistency analysis (found {loaded})")
            continue

        print(f"\n{'Metric':35s} {'Values':>25s} {'Mean':>8s} {'Std':>8s} {'CV%':>6s} {'Status':>10s}")
        print("-" * 100)

        for key, values in sorted(all_metrics.items()):
            if len(values) < 2:
                continue
            mean = np.mean(values)
            std = np.std(values)
            cv = (std / abs(mean) * 100) if abs(mean) > 0.1 else float("inf")
            vals_str = ", ".join(f"{v:.1f}" for v in values)

            if cv > 15:
                status = "UNSTABLE"
            elif cv > 8:
                status = "variable"
            else:
                status = "stable"

            print(f"{key:35s} {vals_str:>25s} {mean:8.1f} {std:8.1f} {cv:5.1f}% {status:>10s}")


if __name__ == "__main__":
    main()
