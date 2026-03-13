#!/usr/bin/env python3
"""CLI wrapper for the pitch mechanics analysis pipeline.

Usage:
    python scripts/validate_pose.py --video path/to/clip.mp4
    python scripts/validate_pose.py --video path/to/clip.mp4 --throws L --backend mediapipe
    python scripts/validate_pose.py --video path/to/clip.mp4 --age 12 --height 60 --weight 95
    python scripts/validate_pose.py --video path/to/clip.mp4 --no-3d
"""

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PipelineConfig, PitchAnalysisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pitching mechanics from video")
    parser.add_argument("--video", type=str, required=True, help="Path to pitching video")
    parser.add_argument("--throws", type=str, default="R", choices=["R", "L"])
    parser.add_argument("--backend", type=str, default="yolov8", choices=["yolov8", "mediapipe"])
    parser.add_argument("--model-size", type=str, default="m", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--roi", type=str, default=None,
                        help="x,y,width,height (pixel coordinates of the pitcher's bounding region)")
    parser.add_argument("--no-open", action="store_true", help="Don't open report in browser")
    parser.add_argument("--no-3d", action="store_true",
                        help="Force 2D-only mode (skip MotionBERT 3D lifting)")
    parser.add_argument("--age", type=int, default=None, help="Pitcher age (youth mode)")
    parser.add_argument("--height", type=float, default=None, help="Height in inches (youth mode)")
    parser.add_argument("--weight", type=float, default=None, help="Weight in lbs (youth mode)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: video not found: {video_path}")
        sys.exit(1)

    youth_flags = [args.age, args.height, args.weight]
    if any(youth_flags) and not all(youth_flags):
        print("Error: --age, --height, and --weight must all be provided together")
        sys.exit(1)

    roi = None
    if args.roi:
        x, y, w, h = (int(v) for v in args.roi.split(","))
        roi = (x, y, x + w, y + h)  # Convert x,y,w,h to x1,y1,x2,y2 for estimator

    config = PipelineConfig(
        backend=args.backend,
        model_size=args.model_size,
        confidence_threshold=args.confidence,
        throws=args.throws,
        roi=roi,
        no_3d=args.no_3d,
        age=args.age,
        height_inches=args.height,
        weight_lbs=args.weight,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    pipeline = PitchAnalysisPipeline(config)

    def on_progress(stage: str, pct: float) -> None:
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"\r  {stage:.<40s} [{bar}] {pct:5.1%}", end="", flush=True)
        if pct >= 1.0:
            print()

    result = pipeline.run(video_path, progress_callback=on_progress)

    if result.validation_warnings:
        print(f"\n⚠ {len(result.validation_warnings)} validation warning(s):")
        for w in result.validation_warnings:
            print(f"  - [{w['severity']}] {w['message']}")

    if result.output_dir and not args.no_open:
        report_path = result.output_dir / "report.html"
        if report_path.exists():
            print(f"\nOpening report: {report_path}")
            webbrowser.open(str(report_path))


if __name__ == "__main__":
    main()
