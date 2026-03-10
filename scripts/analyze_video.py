#!/usr/bin/env python3
"""Analyze a pitching video end-to-end.

Full pipeline: video → pose estimation → event detection → feature extraction →
benchmark comparison → coaching report.

Usage:
    python scripts/analyze_video.py --video path/to/video.mp4
    python scripts/analyze_video.py --video path/to/video.mp4 --throws R --backend mediapipe
    python scripts/analyze_video.py --video path/to/video.mp4 --coaching --level high_school
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Analyze pitching mechanics from video")
    parser.add_argument("--video", type=Path, required=True, help="Path to pitching video")
    parser.add_argument("--throws", type=str, default="R", choices=["R", "L"],
                        help="Pitcher handedness (R=right, L=left)")
    parser.add_argument("--backend", type=str, default="yolov8", choices=["yolov8", "mediapipe"],
                        help="Pose estimation backend")
    parser.add_argument("--level", type=str, default=None,
                        choices=["college", "high_school", "milb", "independent"],
                        help="OBP comparison level")
    parser.add_argument("--coaching", action="store_true",
                        help="Generate Claude API coaching report")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for reports and charts")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    output_dir = args.output_dir or Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Pose estimation
    print(f"\n[1/5] Running pose estimation ({args.backend})...")
    from src.pose.estimator import extract_poses
    pose_seq = extract_poses(args.video, backend=args.backend)
    print(f"  Extracted {len(pose_seq.frames)} frames with keypoints")

    # Step 2: Event detection
    print("\n[2/5] Detecting delivery events...")
    from src.biomechanics.events import detect_events
    kp_df = pose_seq.to_dataframe()
    events = detect_events(kp_df, fps=pose_seq.video_info.fps, pitcher_throws=args.throws)

    phases = events.phase_durations()
    for phase, duration in phases.items():
        if duration is not None:
            print(f"  {phase}: {duration:.3f}s")

    # Step 3: Feature extraction
    print("\n[3/5] Extracting biomechanical features...")
    from src.biomechanics.features import extract_metrics
    keypoints_dict = pose_seq.to_keypoints_dict()
    metrics = extract_metrics(keypoints_dict, events, pitcher_throws=args.throws)
    obp_metrics = metrics.to_obp_comparison_dict()
    print(f"  Computed {len(obp_metrics)} OBP-comparable metrics")

    for name, value in obp_metrics.items():
        print(f"  {name}: {value:.1f}")

    # Step 4: Benchmark comparison
    print("\n[4/5] Comparing to OBP benchmarks...")
    from src.biomechanics.benchmarks import OBPBenchmarks
    obp = OBPBenchmarks().load()
    comparisons = obp.compare_to_benchmarks(obp_metrics, playing_level=args.level)

    for c in comparisons:
        flag = c["flag"] or "normal"
        print(f"  {c['display_name']}: {c['value']:.1f} → {c['percentile_rank']:.0f}th pct [{flag}]")

    # Step 5: Generate reports
    print("\n[5/5] Generating reports...")

    # Visualization
    from src.viz.plots import plot_pitcher_comparison, plot_percentile_gauges
    radar_path = output_dir / "radar_comparison.html"
    gauge_path = output_dir / "percentile_gauges.html"

    plot_pitcher_comparison(comparisons, output_path=radar_path)
    print(f"  Saved radar chart: {radar_path}")

    plot_percentile_gauges(comparisons, output_path=gauge_path)
    print(f"  Saved gauge chart: {gauge_path}")

    # Coaching report
    if args.coaching:
        print("\n  Generating Claude coaching report...")
        from src.coaching.insights import generate_coaching_report
        report = generate_coaching_report(
            comparisons=comparisons,
            pitcher_context=f"youth {args.throws}HP pitcher",
        )
        report_path = output_dir / "coaching_report.txt"
        report_path.write_text(report)
        print(f"  Saved coaching report: {report_path}")
        print("\n" + report)
    else:
        # Offline fallback
        from src.coaching.insights import generate_report_offline
        report = generate_report_offline(comparisons)
        print("\n" + report)

    print("\nDone!")


if __name__ == "__main__":
    main()
