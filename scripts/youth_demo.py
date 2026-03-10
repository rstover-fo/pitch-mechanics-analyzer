#!/usr/bin/env python3
"""Demo: Youth pitcher normalization pipeline.

Shows how OBP benchmarks are adjusted for a youth pitcher and generates
a coaching report using the three-tier normalization framework.

Usage:
    python scripts/youth_demo.py
    python scripts/youth_demo.py --age 10 --height 138 --weight 32
    python scripts/youth_demo.py --age 14 --height 170 --weight 55
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biomechanics.benchmarks import OBPBenchmarks
from src.biomechanics.youth_normalizer import (
    YouthPitcherProfile,
    YouthNormalizer,
    MetricTier,
)
from src.coaching.insights import generate_youth_report_offline


def main():
    parser = argparse.ArgumentParser(description="Youth pitcher normalization demo")
    parser.add_argument("--age", type=float, default=12.0, help="Pitcher age in years")
    parser.add_argument("--height", type=float, default=152.0, help="Height in cm")
    parser.add_argument("--weight", type=float, default=41.0, help="Weight in kg")
    parser.add_argument("--throws", type=str, default="R", choices=["R", "L"])
    args = parser.parse_args()

    # Create pitcher profile
    pitcher = YouthPitcherProfile(
        age=args.age,
        height_cm=args.height,
        weight_kg=args.weight,
        throws=args.throws,
    )

    print(f"\n{'=' * 60}")
    print("YOUTH PITCHER NORMALIZATION DEMO")
    print(f"{'=' * 60}")
    print(f"\nPitcher: {pitcher.age:.0f} years old, {pitcher.height_cm:.0f}cm, {pitcher.weight_kg:.0f}kg, {pitcher.throws}HP")
    print(f"Dev stage: {pitcher.dev_stage.value.replace('_', ' ').title()}")
    print(f"Size vs CDC 50th: {pitcher.size_relative_to_cdc_50th:.0%}")
    print(f"Maturity offset: {pitcher.maturity_offset_estimate:+.1f} years")
    print(f"Effective dev age: {pitcher.effective_dev_age:.1f}")
    print(f"Estimated arm length: {pitcher.estimated_arm_length_m:.3f}m")

    # Load OBP benchmarks
    obp = OBPBenchmarks().load()

    # Create normalizer
    norm = YouthNormalizer(obp, pitcher)

    # Show adjusted benchmarks
    print(f"\n{'─' * 60}")
    print("TIER 1: BODY-POSITION ANGLES (direct comparison, widened range)")
    print(f"{'─' * 60}")
    print(f"{'Metric':<40} {'OBP P50':>8} {'Youth P25':>9} {'Youth P50':>9} {'Youth P75':>9} {'Coach?':>7}")
    print(f"{'─' * 60}")

    adjusted = norm.get_adjusted_benchmarks()
    tier1 = [b for b in adjusted if b.tier == MetricTier.TIER_1_ANGLE]
    for b in tier1[:12]:
        coach = "YES" if b.coaching_relevant else "info"
        print(f"{b.display_name:<40} {b.obp_p50:>8.1f} {b.youth_p25:>9.1f} {b.youth_p50:>9.1f} {b.youth_p75:>9.1f} {coach:>7}")

    print(f"\n{'─' * 60}")
    print("TIER 2: SCALED METRICS (adjusted for body size)")
    print(f"{'─' * 60}")
    print(f"{'Metric':<40} {'OBP P50':>8} {'Scale':>7} {'Youth P50':>9} {'Coach?':>7}")
    print(f"{'─' * 60}")

    tier2 = [b for b in adjusted if b.tier != MetricTier.TIER_1_ANGLE]
    for b in tier2[:10]:
        coach = "YES" if b.coaching_relevant else "info"
        print(f"{b.display_name:<40} {b.obp_p50:>8.1f} {b.scale_factor:>7.3f} {b.youth_p50:>9.1f} {coach:>7}")

    # Simulate a pitcher's metrics and compare
    print(f"\n{'=' * 60}")
    print("SIMULATED ANALYSIS: Typical 12-year-old mechanics")
    print(f"{'=' * 60}")

    # Simulated values for a typical 12-year-old with decent mechanics
    simulated_metrics = {
        "elbow_flexion_fp": 88.0,                       # Close to 90° target
        "shoulder_abduction_fp": 82.0,                   # Slightly low
        "shoulder_horizontal_abduction_fp": 30.0,        # A bit wide
        "shoulder_external_rotation_fp": 55.0,           # Slightly high
        "rotation_hip_shoulder_separation_fp": 22.0,     # Low — common in youth
        "max_rotation_hip_shoulder_separation": 25.0,    # Low peak sep
        "max_shoulder_external_rotation": 155.0,         # Lower layback
        "torso_anterior_tilt_fp": -3.0,                  # Slight forward lean
        "torso_lateral_tilt_fp": -8.0,                   # Some lateral tilt
        "torso_anterior_tilt_br": 28.0,                  # Not enough forward lean at release
    }

    comparisons = norm.compare(simulated_metrics)

    print(f"\n{'Metric':<40} {'Value':>7} {'Flag':<18} {'ASMI Target':>12} {'Note'}")
    print(f"{'─' * 100}")
    for c in comparisons:
        target = f"{c.asmi_target:.0f}° ±{c.asmi_tolerance:.0f}" if c.asmi_target else ""
        note = c.coaching_note or ""
        relevant = "" if c.coaching_relevant else " [info only]"
        print(f"{c.display_name:<40} {c.measured_value:>7.1f} {c.flag_emoji} {(c.flag or ''):.<15} {target:>12} {note}{relevant}")

    # Generate offline report
    ctx = norm.generate_youth_report_context()
    report = generate_youth_report_offline(comparisons, ctx)
    print(f"\n{'=' * 60}")
    print(report)

    # Show coaching priorities
    priorities = norm.get_coaching_priorities()
    print(f"\n{'=' * 60}")
    print(f"COACHING PRIORITIES FOR {pitcher.dev_stage.value.upper().replace('_', ' ')}")
    print(f"{'=' * 60}")
    print(f"\nFocus: {priorities['focus']}")
    print("\nEmphasize:")
    for item in priorities["emphasis"]:
        print(f"  ✓ {item}")
    if priorities["de_emphasize"]:
        print("\nDe-emphasize:")
        for item in priorities["de_emphasize"]:
            print(f"  ✗ {item}")
    print("\nInjury watch:")
    for item in priorities["injury_watch"]:
        print(f"  ⚠️ {item}")


if __name__ == "__main__":
    main()
