#!/usr/bin/env python3
"""Validate pose estimation pipeline with full diagnostic HTML report.

Runs video -> pose estimation -> event detection -> metrics extraction,
then assembles an HTML report with annotated video, trajectory plots,
key frame captures, and metrics comparison.

Usage:
    python scripts/validate_pose.py --video path/to/clip.mp4
    python scripts/validate_pose.py --video path/to/clip.mp4 --throws L --backend mediapipe
    python scripts/validate_pose.py --video path/to/clip.mp4 --model-size l --confidence 0.4
"""

import argparse
import base64
import json
import os
import sys
import webbrowser
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose.estimator import extract_poses, load_video, PoseSequence, VideoInfo
from src.biomechanics.events import (
    DeliveryEvents,
    approximate_shoulder_er_2d,
    detect_ball_release,
    detect_foot_plant_from_keypoints,
    detect_leg_lift,
    detect_max_external_rotation,
    find_delivery_anchor,
)
from src.biomechanics.features import angle_between_points, extract_metrics
from src.biomechanics.benchmarks import METRIC_DISPLAY_NAMES, OBPBenchmarks
from src.biomechanics.youth_normalizer import YouthPitcherProfile, YouthNormalizer
from src.biomechanics.validation import validate_pipeline_output
from src.coaching.insights import (
    generate_coaching_report,
    generate_youth_coaching_report,
    generate_report_offline,
    generate_youth_report_offline,
    load_prompt,
)
from src.viz.skeleton import draw_angle_arc, draw_skeleton
from src.viz.trajectories import (
    plot_confidence_heatmap,
    plot_joint_trajectory,
    plot_wrist_speed,
)
from src.viz.plots import plot_pitcher_comparison, plot_percentile_gauges
from src.viz.report import build_report_html


def encode_frame_as_base64(frame: np.ndarray) -> str:
    """Encode a BGR frame as a base64-encoded PNG string."""
    _, buf = cv2.imencode(".png", frame)
    return base64.b64encode(buf).decode("utf-8")


def _to_int_tuple(arr: np.ndarray) -> tuple[int, int]:
    """Convert a numpy (x, y) array to an int tuple for OpenCV drawing."""
    return (int(arr[0]), int(arr[1]))


def _smooth_moving_avg(data: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply a simple moving average with edge padding."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate pose estimation pipeline and generate HTML diagnostic report"
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to pitching video")
    parser.add_argument(
        "--throws", type=str, default="R", choices=["R", "L"],
        help="Pitcher handedness (R or L)",
    )
    parser.add_argument(
        "--backend", type=str, default="yolov8", choices=["yolov8", "mediapipe"],
        help="Pose estimation backend",
    )
    parser.add_argument(
        "--model-size", type=str, default="m",
        help="YOLOv8 model size (n/s/m/l/x)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Minimum keypoint confidence threshold",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: data/outputs/validate_<video_stem>/)",
    )
    parser.add_argument(
        "--roi", type=str, default=None,
        help="Region of interest as x1,y1,x2,y2 (pixels). Selects the person "
             "closest to this region instead of highest confidence.",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't auto-open the report in a browser",
    )
    parser.add_argument(
        "--age", type=int, default=None,
        help="Pitcher age in years (requires --height and --weight)",
    )
    parser.add_argument(
        "--height", type=float, default=None,
        help="Pitcher height in inches (requires --age and --weight)",
    )
    parser.add_argument(
        "--weight", type=float, default=None,
        help="Pitcher weight in pounds (requires --age and --height)",
    )
    args = parser.parse_args()

    # Validate youth profile flags: all-or-nothing
    youth_flags = [args.age, args.height, args.weight]
    provided_count = sum(1 for v in youth_flags if v is not None)
    if 0 < provided_count < 3:
        print("Error: --age, --height, and --weight must all be provided together.")
        sys.exit(1)

    youth_profile = None
    if provided_count == 3:
        if not (6 <= args.age <= 25):
            print(f"Error: --age must be between 6 and 25 (got {args.age})")
            sys.exit(1)
        if not (36 <= args.height <= 84):
            print(f"Error: --height must be between 36 and 84 inches (got {args.height})")
            sys.exit(1)
        if not (40 <= args.weight <= 350):
            print(f"Error: --weight must be between 40 and 350 lbs (got {args.weight})")
            sys.exit(1)
        height_cm = args.height * 2.54
        weight_kg = args.weight * 0.4536
        youth_profile = {
            "age": args.age,
            "height_in": args.height,
            "weight_lbs": args.weight,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
        }

    # Parse ROI if provided
    roi = None
    if args.roi:
        parts = [int(x.strip()) for x in args.roi.split(",")]
        if len(parts) != 4:
            print("Error: --roi must be x1,y1,x2,y2 (4 integers)")
            sys.exit(1)
        roi = tuple(parts)

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    video_stem = args.video.stem
    output_dir = args.output_dir or Path(f"data/outputs/validate_{video_stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    key_frames_dir = output_dir / "key_frames"
    key_frames_dir.mkdir(parents=True, exist_ok=True)

    # Determine sides from handedness
    lead_side = "left" if args.throws == "R" else "right"
    throw_side = "right" if args.throws == "R" else "left"

    # =========================================================================
    # Stage 1: Pose Estimation -> Annotated Video
    # =========================================================================
    print("=" * 60)
    print("Stage 1: Pose Estimation")
    print("=" * 60)

    video_info = load_video(args.video)
    print(f"  Video: {args.video.name}")
    print(f"  Resolution: {video_info.width}x{video_info.height}")
    print(f"  FPS: {video_info.fps:.1f}")
    print(f"  Frames: {video_info.total_frames}")
    print(f"  Duration: {video_info.duration_sec:.2f}s")

    backend_kwargs = {}
    if args.backend == "yolov8":
        backend_kwargs["model_size"] = args.model_size
        backend_kwargs["confidence"] = args.confidence
        if roi is not None:
            backend_kwargs["roi"] = roi

    print(f"\n  Running {args.backend} pose estimation...")
    pose_seq = extract_poses(args.video, backend=args.backend, **backend_kwargs)

    if not pose_seq.frames:
        print("Error: No poses detected in the video. Check video quality and confidence threshold.")
        sys.exit(1)

    print(f"  Detected poses in {len(pose_seq.frames)} / {video_info.total_frames} frames")

    # Build a lookup from frame_idx -> PoseFrame for annotated video
    pose_by_frame = {pf.frame_idx: pf for pf in pose_seq.frames}

    annotated_path = output_dir / "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(annotated_path),
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )

    cap = cv2.VideoCapture(str(args.video))
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in pose_by_frame:
            pf = pose_by_frame[frame_idx]
            kp_tuples = {name: _to_int_tuple(pos) for name, pos in pf.keypoints.items()}
            bbox_tuple = None
            if pf.bbox is not None:
                bbox_tuple = (
                    int(pf.bbox[0]), int(pf.bbox[1]),
                    int(pf.bbox[2]), int(pf.bbox[3]),
                )
            frame = draw_skeleton(frame, kp_tuples, pf.confidence, bbox=bbox_tuple)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Annotated video saved: {annotated_path}")

    # =========================================================================
    # Stage 2: Event Detection -> Trajectory Plots
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 2: Event Detection")
    print("=" * 60)

    fps = video_info.fps
    timestamps = np.array([pf.timestamp for pf in pose_seq.frames])

    # Lead knee Y trajectory (inverted: negate Y since lower pixel Y = higher position)
    lead_knee_traj = pose_seq.get_joint_trajectory(f"{lead_side}_knee")
    lead_knee_y_inverted = -lead_knee_traj[:, 1]

    # Lead ankle trajectory (X for foot plant detection, Y for plotting)
    lead_ankle_traj = pose_seq.get_joint_trajectory(f"{lead_side}_ankle")
    lead_ankle_x_raw = lead_ankle_traj[:, 0]
    lead_ankle_y_raw = lead_ankle_traj[:, 1]
    lead_ankle_y_inverted = -lead_ankle_y_raw

    # Wrist speed: pixels/frame * fps = pixels/second, then smooth
    throw_wrist = f"{throw_side}_wrist"
    wrist_speed_raw = pose_seq.get_joint_speed(throw_wrist) * fps
    wrist_speed = _smooth_moving_avg(wrist_speed_raw, window=3)

    # Approximate shoulder ER per frame
    shoulder_er_series = np.zeros(len(pose_seq.frames))
    for i, pf in enumerate(pose_seq.frames):
        shoulder_key = f"{throw_side}_shoulder"
        elbow_key = f"{throw_side}_elbow"
        wrist_key = f"{throw_side}_wrist"
        l_hip_key = "left_hip"
        r_hip_key = "right_hip"

        has_all = all(
            k in pf.keypoints
            for k in [shoulder_key, elbow_key, wrist_key, l_hip_key, r_hip_key]
        )
        if has_all:
            hip_center = (pf.keypoints[l_hip_key] + pf.keypoints[r_hip_key]) / 2
            shoulder_er_series[i] = approximate_shoulder_er_2d(
                pf.keypoints[shoulder_key],
                pf.keypoints[elbow_key],
                pf.keypoints[wrist_key],
                hip_center,
            )

    # Detect events using anchor-based approach:
    # 1. Find MER (anchor) from shoulder ER rise-then-drop pattern
    # 2. Ball release = peak wrist speed shortly after MER
    # 3. Foot plant = ankle stabilization before MER
    # 4. Leg lift = peak knee height before foot plant
    events = DeliveryEvents(fps=fps)

    events.max_external_rotation = find_delivery_anchor(
        shoulder_er_series, wrist_speed, fps=fps,
    )

    if events.max_external_rotation is not None:
        events.ball_release = detect_ball_release(
            wrist_speed, after_frame=events.max_external_rotation,
        )

        events.foot_plant = detect_foot_plant_from_keypoints(
            lead_ankle_y_raw, lead_ankle_x=lead_ankle_x_raw, fps=fps,
            before_frame=events.max_external_rotation,
        )

        events.leg_lift_apex = detect_leg_lift(
            lead_knee_y_inverted,
            before_frame=events.foot_plant,
        )
    else:
        print("  WARNING: Could not find delivery anchor (MER). Falling back to independent detection.")
        events.leg_lift_apex = detect_leg_lift(lead_knee_y_inverted)
        events.foot_plant = detect_foot_plant_from_keypoints(
            lead_ankle_y_raw, lead_ankle_x=lead_ankle_x_raw, fps=fps,
        )
        events.max_external_rotation = detect_max_external_rotation(
            shoulder_er_series, after_frame=events.foot_plant,
        )
        events.ball_release = detect_ball_release(
            wrist_speed, after_frame=events.max_external_rotation,
        )

    # Print detected events
    event_names = {
        "Leg Lift": events.leg_lift_apex,
        "Foot Plant": events.foot_plant,
        "Max ER": events.max_external_rotation,
        "Ball Release": events.ball_release,
    }
    for name, frame in event_names.items():
        if frame is not None:
            t = frame / fps
            print(f"  {name}: frame {frame} ({t:.3f}s)")
        else:
            print(f"  {name}: not detected")

    # Convert event frames to timestamps for plot annotations
    def _event_time(frame_idx: int | None) -> float | None:
        if frame_idx is None:
            return None
        return frame_idx / fps

    # Generate trajectory plots
    trajectory_plots_html: list[str] = []

    # 1. Lead knee Y (inverted) with Leg Lift + Foot Plant events
    knee_events = {}
    if events.leg_lift_apex is not None:
        knee_events["Leg Lift"] = _event_time(events.leg_lift_apex)
    if events.foot_plant is not None:
        knee_events["Foot Plant"] = _event_time(events.foot_plant)
    fig_knee = plot_joint_trajectory(
        lead_knee_y_inverted, timestamps,
        f"{lead_side.title()} Knee Y (inverted)",
        events=knee_events,
    )
    trajectory_plots_html.append(fig_knee.to_html(full_html=False, include_plotlyjs=False))

    # 2. Lead ankle Y (inverted) with Foot Plant event
    ankle_events = {}
    if events.foot_plant is not None:
        ankle_events["Foot Plant"] = _event_time(events.foot_plant)
    fig_ankle = plot_joint_trajectory(
        lead_ankle_y_inverted, timestamps,
        f"{lead_side.title()} Ankle Y (inverted)",
        events=ankle_events,
    )
    trajectory_plots_html.append(fig_ankle.to_html(full_html=False, include_plotlyjs=False))

    # 3. Wrist speed with Max ER + Ball Release events
    wrist_events = {}
    if events.max_external_rotation is not None:
        wrist_events["Max ER"] = _event_time(events.max_external_rotation)
    if events.ball_release is not None:
        wrist_events["Ball Release"] = _event_time(events.ball_release)
    fig_wrist = plot_wrist_speed(wrist_speed, timestamps, events=wrist_events)
    trajectory_plots_html.append(fig_wrist.to_html(full_html=False, include_plotlyjs=False))

    # 4. Confidence heatmap for key joints
    heatmap_joints = [
        f"{throw_side}_shoulder", f"{throw_side}_elbow", f"{throw_side}_wrist",
        f"{lead_side}_hip", f"{lead_side}_knee", f"{lead_side}_ankle",
    ]
    conf_matrix = np.zeros((len(heatmap_joints), len(pose_seq.frames)))
    for col_idx, pf in enumerate(pose_seq.frames):
        for row_idx, joint in enumerate(heatmap_joints):
            conf_matrix[row_idx, col_idx] = pf.confidence.get(joint, 0.0)

    fig_heatmap = plot_confidence_heatmap(conf_matrix, timestamps)
    # Replace default labels with actual joint names
    fig_heatmap.data[0].y = heatmap_joints
    trajectory_plots_html.append(fig_heatmap.to_html(full_html=False, include_plotlyjs=False))

    print(f"  Generated {len(trajectory_plots_html)} trajectory plots")

    # =========================================================================
    # Stage 3: Key Frame Extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 3: Key Frame Extraction")
    print("=" * 60)

    key_frame_images: dict[str, str] = {}

    for event_label, event_frame in event_names.items():
        if event_frame is None:
            continue

        # Find the PoseFrame closest to the event frame index
        # Events are indexed into pose_seq.frames, so map back to actual video frames
        if event_frame >= len(pose_seq.frames):
            continue
        pf = pose_seq.frames[event_frame]
        actual_frame_idx = pf.frame_idx

        # Seek to that frame in the video
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # Draw skeleton overlay
        kp_tuples = {name: _to_int_tuple(pos) for name, pos in pf.keypoints.items()}
        frame = draw_skeleton(frame, kp_tuples, pf.confidence)

        # At Foot Plant: draw elbow flexion angle arc
        if event_label == "Foot Plant":
            shoulder_key = f"{throw_side}_shoulder"
            elbow_key = f"{throw_side}_elbow"
            wrist_key = f"{throw_side}_wrist"
            if all(k in pf.keypoints for k in [shoulder_key, elbow_key, wrist_key]):
                angle = angle_between_points(
                    pf.keypoints[shoulder_key],
                    pf.keypoints[elbow_key],
                    pf.keypoints[wrist_key],
                )
                frame = draw_angle_arc(
                    frame,
                    vertex=_to_int_tuple(pf.keypoints[elbow_key]),
                    point_a=_to_int_tuple(pf.keypoints[shoulder_key]),
                    point_b=_to_int_tuple(pf.keypoints[wrist_key]),
                    angle_deg=angle,
                    label=f"{angle:.0f} deg",
                    color=(0, 255, 255),
                )

        # Add text label with event name and frame number
        label_text = f"{event_label} (frame {actual_frame_idx})"
        cv2.putText(
            frame, label_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Save PNG to key_frames/ directory
        png_path = key_frames_dir / f"{event_label.lower().replace(' ', '_')}.png"
        cv2.imwrite(str(png_path), frame)

        # Encode as base64 for HTML embedding
        key_frame_images[event_label] = encode_frame_as_base64(frame)
        print(f"  Captured: {event_label} at frame {actual_frame_idx}")

    # =========================================================================
    # Stage 4: Metrics Extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 4: Metrics Extraction")
    print("=" * 60)

    keypoints_dict = pose_seq.to_keypoints_dict()
    metrics = extract_metrics(keypoints_dict, events, pitcher_throws=args.throws)
    obp_dict = metrics.to_obp_comparison_dict()

    # Build metrics display rows
    metrics_to_show = [
        ("elbow_flexion_fp", "Elbow Flexion @ FP", "deg"),
        ("shoulder_abduction_fp", "Shoulder Abduction @ FP", "deg"),
        ("shoulder_horizontal_abduction_fp", "Shoulder Horiz. Abduction @ FP", "deg"),
        ("torso_anterior_tilt_fp", "Forward Trunk Tilt @ FP", "deg"),
        ("torso_lateral_tilt_fp", "Lateral Trunk Tilt @ FP", "deg"),
        ("lead_knee_angle_fp", "Lead Knee Angle @ FP", "deg"),
        ("hip_shoulder_separation_fp", "Hip-Shoulder Separation @ FP", "deg"),
        ("max_shoulder_external_rotation", "Peak Shoulder ER", "deg"),
        ("torso_anterior_tilt_br", "Forward Trunk Tilt @ BR", "deg"),
        ("arm_slot_angle", "Arm Slot Angle", "deg"),
        ("lead_knee_angle_br", "Lead Knee Angle @ BR", "deg"),
    ]

    # Map internal metric names to OBP names for comparison lookup
    internal_to_obp = {
        "elbow_flexion_fp": "elbow_flexion_fp",
        "shoulder_abduction_fp": "shoulder_abduction_fp",
        "shoulder_horizontal_abduction_fp": "shoulder_horizontal_abduction_fp",
        "torso_anterior_tilt_fp": "torso_anterior_tilt_fp",
        "torso_lateral_tilt_fp": "torso_lateral_tilt_fp",
        "lead_knee_angle_fp": None,
        "hip_shoulder_separation_fp": "rotation_hip_shoulder_separation_fp",
        "max_shoulder_external_rotation": "max_shoulder_external_rotation",
        "torso_anterior_tilt_br": "torso_anterior_tilt_br",
        "arm_slot_angle": None,
        "lead_knee_angle_br": None,
    }

    # =========================================================================
    # Stage 5: OBP Benchmark Comparison
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 5: OBP Benchmark Comparison")
    print("=" * 60)

    obp = None
    obp_comparisons: list[dict] = []
    poi_path = Path("data/obp/poi_metrics.csv")
    if poi_path.exists() and obp_dict:
        try:
            obp = OBPBenchmarks().load()
            obp_comparisons = obp.compare_to_benchmarks(obp_dict)
            print(f"  Compared {len(obp_comparisons)} metrics against OBP benchmarks")
            for c in obp_comparisons:
                pct = c["percentile_rank"]
                pct_str = f"{pct:.0f}th" if pct is not None else "N/A"
                print(f"    {c['display_name']}: {c['value']:.1f} → {pct_str} percentile [{c['flag']}]")
        except Exception as exc:
            print(f"  Warning: Could not load OBP benchmarks: {exc}")
    else:
        print("  Skipped: no OBP data or no metrics to compare")

    # Build a lookup from OBP metric name to comparison result
    obp_comp_map = {c["metric"]: c for c in obp_comparisons}

    # Generate percentile charts
    percentile_charts_html: list[str] = []
    if obp_comparisons:
        try:
            radar_fig = plot_pitcher_comparison(obp_comparisons)
            percentile_charts_html.append(
                radar_fig.to_html(full_html=False, include_plotlyjs=False)
            )
            gauges_fig = plot_percentile_gauges(obp_comparisons)
            percentile_charts_html.append(
                gauges_fig.to_html(full_html=False, include_plotlyjs=False)
            )
            print(f"  Generated {len(percentile_charts_html)} percentile charts")
        except Exception as exc:
            print(f"  Warning: Could not generate percentile charts: {exc}")

    # Build metrics rows with percentile data
    metrics_rows: list[dict] = []
    for metric_key, display_name, unit in metrics_to_show:
        value = getattr(metrics, metric_key, None)
        obp_key = internal_to_obp.get(metric_key)
        comp = obp_comp_map.get(obp_key) if obp_key else None

        median = comp["benchmark_median"] if comp else None
        percentile = comp["percentile_rank"] if comp else None

        metrics_rows.append({
            "metric": display_name,
            "value": f"{value:.1f}" if value is not None else "--",
            "unit": unit,
            "obp_median": f"{median:.1f}" if median is not None else "--",
            "percentile": f"{percentile:.0f}" if percentile is not None else "--",
            "status": "ok" if value is not None else "missing",
        })

    computed_count = sum(1 for r in metrics_rows if r["status"] == "ok")
    print(f"  Metrics: {computed_count} / {len(metrics_to_show)} computed")

    # =========================================================================
    # Stage 6: Youth Normalization (conditional)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 6: Youth Normalization")
    print("=" * 60)

    youth_comparisons = None
    youth_context = None
    if youth_profile and obp is not None:
        profile = YouthPitcherProfile(
            age=youth_profile["age"],
            height_cm=youth_profile["height_cm"],
            weight_kg=youth_profile["weight_kg"],
            throws=args.throws,
        )
        normalizer = YouthNormalizer(obp, profile)
        youth_comparisons = normalizer.compare(obp_dict)
        youth_context = normalizer.generate_youth_report_context()
        youth_profile["developmental_stage"] = normalizer.dev_stage.value
        print(f"  Pitcher: {youth_profile['age']}yo, {youth_profile['height_in']}in, {youth_profile['weight_lbs']}lbs")
        print(f"  Developmental stage: {normalizer.dev_stage.value.replace('_', ' ').title()}")
        print(f"  Youth comparisons: {len(youth_comparisons)} metrics")
    else:
        if youth_profile:
            print("  Skipped: OBP data not available")
        else:
            print("  Skipped: no youth profile provided (adult comparison path)")

    # =========================================================================
    # Stage 7: Coaching Report
    # =========================================================================
    print("\n" + "=" * 60)
    print("Stage 7: Coaching Report")
    print("=" * 60)

    coaching_text = ""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    # Build additional context for non-OBP metrics
    additional_lines = []
    if metrics.arm_slot_angle is not None:
        additional_lines.append(
            f"Arm slot angle: {metrics.arm_slot_angle:.1f}° "
            "(camera angle dependent — from front-quarter view, reads ~15-22° low)"
        )
    if metrics.lead_knee_angle_fp is not None:
        additional_lines.append(f"Lead knee angle at foot plant: {metrics.lead_knee_angle_fp:.1f}°")
    if metrics.lead_knee_angle_br is not None:
        additional_lines.append(f"Lead knee angle at ball release: {metrics.lead_knee_angle_br:.1f}°")
    if metrics.stride_length_pct_height is not None:
        additional_lines.append(
            f"Stride length: {metrics.stride_length_pct_height:.0f}% of height "
            "(ASMI target: 75-85%)"
        )
    additional_context = "\n".join(additional_lines) if additional_lines else None

    # Append measurement caveats from prompt file
    caveats = load_prompt("measurement_caveats")
    if caveats and additional_context:
        additional_context += f"\n\nMEASUREMENT CAVEATS:\n{caveats}"
    elif caveats:
        additional_context = f"MEASUREMENT CAVEATS:\n{caveats}"

    if youth_comparisons and youth_context:
        # Youth coaching path
        if api_key:
            try:
                print("  Generating youth coaching report via Claude API...")
                coaching_text = generate_youth_coaching_report(
                    youth_comparisons, youth_context,
                    additional_context=additional_context,
                )
                print("  Youth coaching report generated (API)")
            except Exception as exc:
                print(f"  Warning: API call failed ({exc}), using offline fallback")
                coaching_text = generate_youth_report_offline(youth_comparisons, youth_context)
        else:
            coaching_text = generate_youth_report_offline(youth_comparisons, youth_context)
            print("  Youth coaching report generated (offline)")
    elif obp_comparisons:
        # Adult coaching path
        if api_key:
            try:
                print("  Generating coaching report via Claude API...")
                coaching_text = generate_coaching_report(
                    obp_comparisons,
                    additional_context=additional_context,
                )
                print("  Coaching report generated (API)")
            except Exception as exc:
                print(f"  Warning: API call failed ({exc}), using offline fallback")
                coaching_text = generate_report_offline(obp_comparisons)
        else:
            coaching_text = generate_report_offline(obp_comparisons)
            print("  Coaching report generated (offline)")
    else:
        print("  Skipped: no benchmark comparisons available")

    # =========================================================================
    # Save Pipeline Results as JSON
    # =========================================================================
    avg_confidence = float(np.mean([
        conf
        for pf in pose_seq.frames
        for conf in pf.confidence.values()
    ]))

    # Run sanity checks
    validation_warnings = validate_pipeline_output(
        events, avg_confidence=avg_confidence, metrics=metrics.__dict__,
    )
    if validation_warnings:
        print("\n  Validation warnings:")
        for w in validation_warnings:
            print(f"    [{w['severity']}] {w['code']}: {w['message']}")
    else:
        print("\n  No validation warnings")

    pipeline_output = {
        "video": args.video.name,
        "backend": args.backend,
        "pitcher_throws": args.throws,
        "fps": fps,
        "total_frames": video_info.total_frames,
        "events": {
            "leg_lift": events.leg_lift_apex,
            "foot_plant": events.foot_plant,
            "max_er": events.max_external_rotation,
            "ball_release": events.ball_release,
        },
        "metrics": {
            row["metric"]: row["value"]
            for row in metrics_rows
            if row["status"] == "ok"
        },
        "metrics_raw": {
            k: getattr(metrics, k)
            for k in [
                "elbow_flexion_fp", "torso_anterior_tilt_fp", "lead_knee_angle_fp",
                "max_shoulder_external_rotation", "torso_anterior_tilt_br",
                "arm_slot_angle", "lead_knee_angle_br",
            ]
            if getattr(metrics, k) is not None
        },
        "diagnostics": {
            "frames_with_poses": len(pose_seq.frames),
            "avg_confidence": float(avg_confidence),
        },
        "warnings": validation_warnings,
    }

    # Enrich results.json with Phase 5 data
    if obp_comparisons:
        pipeline_output["obp_comparisons"] = [
            {k: v for k, v in c.items()}
            for c in obp_comparisons
        ]
    if coaching_text:
        pipeline_output["coaching_report"] = coaching_text
    if youth_profile:
        pipeline_output["youth_profile"] = {
            "age": youth_profile["age"],
            "height_in": youth_profile["height_in"],
            "weight_lbs": youth_profile["weight_lbs"],
            "developmental_stage": youth_profile.get("developmental_stage"),
        }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(pipeline_output, indent=2, default=str))
    print(f"  Results saved: {results_path}")

    # =========================================================================
    # Assembly: Build HTML Report
    # =========================================================================
    print("\n" + "=" * 60)
    print("Assembling Report")
    print("=" * 60)

    events_detected = sum(1 for v in event_names.values() if v is not None)

    diagnostics = {
        "frames_processed": video_info.total_frames,
        "frames_with_poses": len(pose_seq.frames),
        "avg_confidence": f"{avg_confidence:.3f}",
        "events_detected": f"{events_detected} / 4",
        "metrics_computed": f"{computed_count} / {len(metrics_to_show)}",
        "warnings": "; ".join(w["message"] for w in validation_warnings) if validation_warnings else "none",
    }

    report_html = build_report_html(
        video_filename=args.video.name,
        video_rel_path="annotated_video.mp4",
        fps=fps,
        frame_count=video_info.total_frames,
        backend=args.backend,
        pitcher_throws=args.throws,
        trajectory_plots_html=trajectory_plots_html,
        key_frame_images=key_frame_images,
        metrics_rows=metrics_rows,
        diagnostics=diagnostics,
        coaching_html=coaching_text,
        percentile_charts_html=percentile_charts_html,
        pitcher_profile=youth_profile,
    )

    report_path = output_dir / "report.html"
    report_path.write_text(report_html)
    print(f"  Report saved: {report_path}")

    if not args.no_open:
        webbrowser.open(str(report_path.resolve()))
        print("  Opened report in browser")

    print("\nDone.")


if __name__ == "__main__":
    main()
