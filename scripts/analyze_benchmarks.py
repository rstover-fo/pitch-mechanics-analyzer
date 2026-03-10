#!/usr/bin/env python3
"""Analyze Driveline OBP benchmark data and generate distribution reports.

Usage:
    python scripts/analyze_benchmarks.py
    python scripts/analyze_benchmarks.py --level college
    python scripts/analyze_benchmarks.py --level high_school --output data/outputs/hs_benchmarks.html
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biomechanics.benchmarks import (
    OBPBenchmarks,
    TIMING_SEQUENCE_METRICS,
    ARM_MECHANICS_METRICS,
    TRUNK_MECHANICS_METRICS,
    HIP_SHOULDER_SEPARATION_METRICS,
    LEAD_LEG_METRICS,
    KINETIC_METRICS,
)
from src.viz.plots import plot_benchmark_distributions


def main():
    parser = argparse.ArgumentParser(description="Analyze OBP pitching biomechanics benchmarks")
    parser.add_argument("--data-path", type=Path, default=None,
                        help="Path to OBP data directory (default: data/obp/)")
    parser.add_argument("--level", type=str, default=None,
                        choices=["college", "high_school", "milb", "independent"],
                        help="Filter to specific playing level")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output HTML file for interactive charts")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Output CSV file for benchmark summary table")
    args = parser.parse_args()

    # Load OBP data
    obp = OBPBenchmarks(obp_data_path=args.data_path)
    obp.load()

    # Print dataset summary
    print("\n" + "=" * 60)
    print("OBP PITCHING BIOMECHANICS BENCHMARK ANALYSIS")
    print("=" * 60)

    if obp.merged_df is not None and "playing_level" in obp.merged_df.columns:
        print("\nDataset composition:")
        for level, count in obp.merged_df["playing_level"].value_counts().items():
            print(f"  {level}: {count} pitches")

    if obp.merged_df is not None and "pitch_speed_mph" in obp.merged_df.columns:
        speeds = obp.merged_df["pitch_speed_mph"].dropna()
        print(f"\nPitch speed range: {speeds.min():.1f} - {speeds.max():.1f} mph")
        print(f"Pitch speed median: {speeds.median():.1f} mph")

    # Generate summary table
    level_label = args.level or "ALL LEVELS"
    print(f"\n--- Benchmarks: {level_label} ---\n")

    summary = obp.summary_table(playing_level=args.level)

    # Print grouped summaries
    metric_groups = {
        "TIMING SEQUENCE": TIMING_SEQUENCE_METRICS,
        "ARM MECHANICS": ARM_MECHANICS_METRICS,
        "TRUNK MECHANICS": TRUNK_MECHANICS_METRICS,
        "HIP-SHOULDER SEPARATION": HIP_SHOULDER_SEPARATION_METRICS,
        "LEAD LEG": LEAD_LEG_METRICS,
        "KINETICS (JOINT LOADING)": KINETIC_METRICS,
    }

    for group_name, group_metrics in metric_groups.items():
        group_df = summary[summary["metric"].isin(group_metrics)]
        if group_df.empty:
            continue

        print(f"\n{group_name}")
        print("-" * 80)
        print(f"{'Metric':<40} {'P25':>8} {'P50':>8} {'P75':>8} {'Unit':>8}")
        print("-" * 80)

        for _, row in group_df.iterrows():
            print(f"{row['display_name']:<40} {row['p25']:>8.1f} {row['p50']:>8.1f} {row['p75']:>8.1f} {row['unit']:>8}")

    # Save outputs
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.csv, index=False)
        print(f"\nSaved benchmark CSV to {args.csv}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig = plot_benchmark_distributions(summary, title=f"OBP Benchmarks — {level_label}")
        fig.write_html(str(args.output))
        print(f"Saved interactive chart to {args.output}")

    # Show available metrics count
    available = obp.get_available_metrics()
    print(f"\nTotal available numeric metrics: {len(available)}")


if __name__ == "__main__":
    main()
