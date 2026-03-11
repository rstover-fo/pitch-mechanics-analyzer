#!/usr/bin/env python3
"""Evaluate event detection accuracy against ground truth labels.

Compares pipeline-detected event frames to human-labeled ground truth
and computes MAE, bias, and hit rate per event type.

Usage:
    python scripts/eval_events.py
    python scripts/eval_events.py --ground-truth data/ground_truth/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


EVENT_NAMES = ["leg_lift", "foot_plant", "max_er", "ball_release"]


def load_ground_truth(gt_dir: Path) -> dict[str, dict]:
    """Load all ground truth JSON files from directory."""
    labels = {}
    for f in sorted(gt_dir.glob("*.json")):
        if f.name == "pitchers.json":
            continue
        data = json.loads(f.read_text())
        video_stem = Path(data.get("video", f.stem)).stem
        labels[video_stem] = data
    return labels


def load_pipeline_results(outputs_dir: Path) -> dict[str, dict]:
    """Load all pipeline results.json files."""
    results = {}
    for results_file in sorted(outputs_dir.glob("validate_*/results.json")):
        data = json.loads(results_file.read_text())
        video_stem = Path(data["video"]).stem
        results[video_stem] = data
    return results


def evaluate(gt_labels: dict, pipeline_results: dict) -> None:
    """Compare pipeline results to ground truth and print accuracy report."""
    matched_clips = set(gt_labels.keys()) & set(pipeline_results.keys())
    if not matched_clips:
        print("No matching clips found between ground truth and pipeline results.")
        sys.exit(1)

    print(f"Evaluating {len(matched_clips)} clips: {', '.join(sorted(matched_clips))}")
    print()

    errors: dict[str, list[int]] = {evt: [] for evt in EVENT_NAMES}
    missed: dict[str, int] = {evt: 0 for evt in EVENT_NAMES}
    total: dict[str, int] = {evt: 0 for evt in EVENT_NAMES}

    # Per-clip detail
    for clip in sorted(matched_clips):
        gt = gt_labels[clip].get("events", {})
        det = pipeline_results[clip].get("events", {})
        fps = pipeline_results[clip].get("fps", 30.0)

        print(f"  {clip} (fps={fps}):")
        for evt in EVENT_NAMES:
            gt_frame = gt.get(evt)
            det_frame = det.get(evt)
            if gt_frame is None:
                continue  # Not labeled
            total[evt] += 1
            if det_frame is None:
                missed[evt] += 1
                print(f"    {evt:15s}: MISSED (gt={gt_frame})")
            else:
                err = det_frame - gt_frame
                errors[evt].append(err)
                ms = err / fps * 1000
                print(f"    {evt:15s}: det={det_frame} gt={gt_frame} err={err:+d} frames ({ms:+.0f}ms)")
        print()

    # Summary
    print("=" * 65)
    print(f"{'Event':15s} {'MAE':>8s} {'Bias':>8s} {'MAE(ms)':>8s} {'Bias(ms)':>9s} {'Hit%':>6s}")
    print("-" * 65)
    for evt in EVENT_NAMES:
        errs = errors[evt]
        n = total[evt]
        if n == 0:
            print(f"{evt:15s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>9s} {'--':>6s}")
            continue
        hit_pct = (n - missed[evt]) / n * 100
        if errs:
            mae = np.mean(np.abs(errs))
            bias = np.mean(errs)
            # Assume ~30fps for ms conversion
            fps_avg = 30.0
            print(f"{evt:15s} {mae:8.1f} {bias:+8.1f} {mae/fps_avg*1000:8.0f} {bias/fps_avg*1000:+9.0f} {hit_pct:5.0f}%")
        else:
            print(f"{evt:15s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>9s} {hit_pct:5.0f}%")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate event detection accuracy")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth"),
                        help="Directory with ground truth JSON files")
    parser.add_argument("--outputs", type=Path, default=Path("data/outputs"),
                        help="Directory with pipeline output directories")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    results = load_pipeline_results(args.outputs)
    evaluate(gt, results)


if __name__ == "__main__":
    main()
