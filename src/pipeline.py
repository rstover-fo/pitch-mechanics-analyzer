"""PitchAnalysisPipeline: composable end-to-end pipeline for pitch mechanics analysis.

Encapsulates the full video → report workflow:
  1. Pose estimation (YOLOv8 / MediaPipe)
  2. Delivery event detection (anchor-based)
  3. Key frame extraction with skeleton overlay
  4. Biomechanical metric extraction
  5. OBP benchmark comparison
  6. Youth normalization (optional)
  7. Coaching report generation (Claude API / offline)
  8. HTML report assembly

Designed to be called from CLI scripts, desktop apps, or APIs.
Zero CLI dependencies — progress is reported via an optional callback.
"""

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.biomechanics.benchmarks import OBPBenchmarks
from src.biomechanics.events import (
    DeliveryEvents,
    approximate_shoulder_er_2d,
    detect_ball_release,
    detect_foot_plant_from_keypoints,
    detect_leg_lift,
    detect_max_external_rotation,
    find_delivery_anchor,
)
from src.biomechanics.features import PitcherMetrics, angle_between_points, extract_metrics
from src.biomechanics.validation import validate_pipeline_output
from src.biomechanics.youth_normalizer import YouthNormalizer, YouthPitcherProfile
from src.coaching.insights import (
    generate_coaching_report,
    generate_report_offline,
    generate_youth_coaching_report,
    generate_youth_report_offline,
    load_prompt,
)
from src.pose.estimator import PoseSequence, extract_poses, load_video
from src.viz.plots import plot_percentile_gauges, plot_pitcher_comparison
from src.viz.report import build_report_html
from src.viz.skeleton import draw_angle_arc, draw_skeleton
from src.viz.trajectories import (
    plot_confidence_heatmap,
    plot_joint_trajectory,
    plot_wrist_speed,
)


# Type alias for the progress callback.
ProgressCallback = Callable[[str, float], None]


@dataclass
class PipelineConfig:
    """Configuration for the pitch analysis pipeline."""

    backend: str = "yolov8"
    model_size: str = "m"
    confidence_threshold: float = 0.3
    target_fps: Optional[float] = None
    throws: str = "R"
    camera_view: str = "auto"
    roi: Optional[tuple[int, int, int, int]] = None
    # Youth mode
    age: Optional[int] = None
    height_inches: Optional[float] = None
    weight_lbs: Optional[float] = None
    # Output
    output_dir: Optional[Path] = None
    generate_video: bool = True
    generate_report: bool = True


@dataclass
class PipelineResult:
    """Results from a complete pipeline run."""

    pose_sequence: PoseSequence
    events: DeliveryEvents
    metrics: PitcherMetrics
    benchmark_comparisons: list = field(default_factory=list)
    youth_comparisons: Optional[list] = None
    coaching_report: str = ""
    report_html: str = ""
    output_dir: Optional[Path] = None
    validation_warnings: list = field(default_factory=list)


def _encode_frame_as_base64(frame: np.ndarray) -> str:
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


class PitchAnalysisPipeline:
    """End-to-end pitch mechanics analysis pipeline.

    Usage::

        config = PipelineConfig(throws="R", backend="yolov8")
        pipeline = PitchAnalysisPipeline(config)
        result = pipeline.run(Path("pitch.mp4"))
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        """Execute the full pipeline: video -> report.

        Args:
            video_path: Path to the pitching video.
            progress_callback: Optional ``(stage, progress)`` callback.
                Stages: ``"pose_estimation"``, ``"event_detection"``,
                ``"feature_extraction"``, ``"benchmarking"``,
                ``"coaching"``, ``"report_generation"``.

        Returns:
            :class:`PipelineResult` with all pipeline outputs.
        """
        cb = progress_callback or (lambda _s, _p: None)
        cfg = self.config
        video_path = Path(video_path)

        # Resolve output directory
        output_dir = cfg.output_dir or Path(f"data/outputs/validate_{video_path.stem}")
        output_dir.mkdir(parents=True, exist_ok=True)
        key_frames_dir = output_dir / "key_frames"
        key_frames_dir.mkdir(parents=True, exist_ok=True)

        # Determine sides from handedness
        lead_side = "left" if cfg.throws == "R" else "right"
        throw_side = "right" if cfg.throws == "R" else "left"

        # Stage 1: Pose estimation
        cb("pose_estimation", 0.0)
        pose_seq = self.extract_poses(video_path)
        video_info = pose_seq.video_info
        cb("pose_estimation", 1.0)

        # Generate annotated video
        if cfg.generate_video:
            self._write_annotated_video(video_path, pose_seq, video_info, output_dir)

        # Stage 2: Event detection
        cb("event_detection", 0.0)
        events = self.detect_events(pose_seq)
        cb("event_detection", 1.0)

        # Stage 3: Key frame extraction
        event_names = {
            "Leg Lift": events.leg_lift_apex,
            "Foot Plant": events.foot_plant,
            "Max ER": events.max_external_rotation,
            "Ball Release": events.ball_release,
        }
        key_frame_images = self._extract_key_frames(
            video_path, pose_seq, event_names, throw_side, key_frames_dir,
        )

        # Build trajectory plots
        trajectory_plots_html = self._build_trajectory_plots(
            pose_seq, events, lead_side, throw_side,
        )

        # Stage 4: Feature extraction
        cb("feature_extraction", 0.0)
        metrics = self.extract_metrics(pose_seq, events)
        cb("feature_extraction", 1.0)

        # Stage 5: Benchmark comparison
        cb("benchmarking", 0.0)
        obp_comparisons = self.compare_benchmarks(metrics)
        percentile_charts_html = self._build_percentile_charts(obp_comparisons)
        cb("benchmarking", 1.0)

        # Stage 5b: Youth normalization
        youth_comparisons = None
        youth_context = None
        youth_profile_dict = None
        obp = self._load_obp_benchmarks()
        if self._has_youth_profile() and obp is not None:
            youth_comparisons, youth_context, youth_profile_dict = self.normalize_youth(
                metrics, obp,
            )

        # Stage 6: Coaching report
        cb("coaching", 0.0)
        coaching_text = self._generate_coaching(
            metrics, obp_comparisons, youth_comparisons, youth_context,
        )
        cb("coaching", 1.0)

        # Validation
        avg_confidence = float(np.mean([
            conf
            for pf in pose_seq.frames
            for conf in pf.confidence.values()
        ]))
        validation_warnings = validate_pipeline_output(
            events, avg_confidence=avg_confidence, metrics=metrics.__dict__,
        )

        # Build metrics display rows
        metrics_rows = self._build_metrics_rows(metrics, obp_comparisons)

        # Save results JSON
        self._save_results_json(
            output_dir, video_path, video_info, events, event_names,
            metrics, metrics_rows, pose_seq, avg_confidence,
            validation_warnings, obp_comparisons, coaching_text,
            youth_profile_dict,
        )

        # Stage 7: HTML report
        cb("report_generation", 0.0)
        report_html = ""
        if cfg.generate_report:
            computed_count = sum(1 for r in metrics_rows if r["status"] == "ok")
            events_detected = sum(1 for v in event_names.values() if v is not None)
            diagnostics = {
                "frames_processed": video_info.total_frames,
                "frames_with_poses": len(pose_seq.frames),
                "avg_confidence": f"{avg_confidence:.3f}",
                "events_detected": f"{events_detected} / 4",
                "metrics_computed": f"{computed_count} / {len(metrics_rows)}",
                "warnings": (
                    "; ".join(w["message"] for w in validation_warnings)
                    if validation_warnings
                    else "none"
                ),
            }
            report_html = build_report_html(
                video_filename=video_path.name,
                video_rel_path="annotated_video.mp4",
                fps=video_info.fps,
                frame_count=video_info.total_frames,
                backend=cfg.backend,
                pitcher_throws=cfg.throws,
                trajectory_plots_html=trajectory_plots_html,
                key_frame_images=key_frame_images,
                metrics_rows=metrics_rows,
                diagnostics=diagnostics,
                coaching_html=coaching_text,
                percentile_charts_html=percentile_charts_html,
                pitcher_profile=youth_profile_dict,
            )
            report_path = output_dir / "report.html"
            report_path.write_text(report_html)
        cb("report_generation", 1.0)

        return PipelineResult(
            pose_sequence=pose_seq,
            events=events,
            metrics=metrics,
            benchmark_comparisons=obp_comparisons,
            youth_comparisons=youth_comparisons,
            coaching_report=coaching_text,
            report_html=report_html,
            output_dir=output_dir,
            validation_warnings=validation_warnings,
        )

    # ------------------------------------------------------------------
    # Individual pipeline stages (publicly callable)
    # ------------------------------------------------------------------

    def extract_poses(self, video_path: Path) -> PoseSequence:
        """Stage 1: Run pose estimation on a video."""
        cfg = self.config
        kwargs: dict = {}
        if cfg.backend == "yolov8":
            kwargs["model_size"] = cfg.model_size
            kwargs["confidence"] = cfg.confidence_threshold
            if cfg.roi is not None:
                kwargs["roi"] = cfg.roi
        if cfg.target_fps is not None:
            kwargs["target_fps"] = cfg.target_fps
        return extract_poses(video_path, backend=cfg.backend, **kwargs)

    def detect_events(self, pose_seq: PoseSequence) -> DeliveryEvents:
        """Stage 2: Detect delivery events using anchor-based approach."""
        cfg = self.config
        fps = pose_seq.video_info.fps
        lead_side = "left" if cfg.throws == "R" else "right"
        throw_side = "right" if cfg.throws == "R" else "left"

        # Build trajectory arrays
        lead_knee_traj = pose_seq.get_joint_trajectory(f"{lead_side}_knee")
        lead_knee_y_inverted = -lead_knee_traj[:, 1]

        lead_ankle_traj = pose_seq.get_joint_trajectory(f"{lead_side}_ankle")
        lead_ankle_x_raw = lead_ankle_traj[:, 0]
        lead_ankle_y_raw = lead_ankle_traj[:, 1]

        wrist_speed_raw = pose_seq.get_joint_speed(f"{throw_side}_wrist") * fps
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

        # Anchor-based detection
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
                lead_knee_y_inverted, before_frame=events.foot_plant,
            )
        else:
            # Fallback: independent detection without anchor
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

        return events

    def extract_metrics(
        self, pose_seq: PoseSequence, events: DeliveryEvents,
    ) -> PitcherMetrics:
        """Stage 4: Extract biomechanical metrics."""
        keypoints_dict = pose_seq.to_keypoints_dict()
        return extract_metrics(
            keypoints_dict, events,
            pitcher_throws=self.config.throws,
            camera_view=self.config.camera_view,
        )

    def compare_benchmarks(self, metrics: PitcherMetrics) -> list[dict]:
        """Stage 5: Compare metrics to OBP benchmarks."""
        obp_dict = metrics.to_obp_comparison_dict()
        poi_path = Path("data/obp/poi_metrics.csv")
        if not poi_path.exists() or not obp_dict:
            return []
        try:
            obp = OBPBenchmarks().load()
            return obp.compare_to_benchmarks(obp_dict)
        except Exception:
            return []

    def normalize_youth(
        self, metrics: PitcherMetrics, obp: OBPBenchmarks,
    ) -> tuple[list, dict, dict]:
        """Stage 5b: Youth-normalize benchmark comparisons.

        Returns:
            Tuple of (youth_comparisons, youth_context, youth_profile_dict).
        """
        cfg = self.config
        height_cm = cfg.height_inches * 2.54
        weight_kg = cfg.weight_lbs * 0.4536
        profile = YouthPitcherProfile(
            age=cfg.age,
            height_cm=height_cm,
            weight_kg=weight_kg,
            throws=cfg.throws,
        )
        normalizer = YouthNormalizer(obp, profile)
        obp_dict = metrics.to_obp_comparison_dict()
        youth_comparisons = normalizer.compare(obp_dict)
        youth_context = normalizer.generate_youth_report_context()
        youth_profile_dict = {
            "age": cfg.age,
            "height_in": cfg.height_inches,
            "weight_lbs": cfg.weight_lbs,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "developmental_stage": normalizer.dev_stage.value,
        }
        return youth_comparisons, youth_context, youth_profile_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_youth_profile(self) -> bool:
        cfg = self.config
        return all(v is not None for v in [cfg.age, cfg.height_inches, cfg.weight_lbs])

    def _load_obp_benchmarks(self) -> Optional[OBPBenchmarks]:
        poi_path = Path("data/obp/poi_metrics.csv")
        if not poi_path.exists():
            return None
        try:
            return OBPBenchmarks().load()
        except Exception:
            return None

    def _write_annotated_video(
        self,
        video_path: Path,
        pose_seq: PoseSequence,
        video_info,
        output_dir: Path,
    ) -> None:
        """Write annotated video with skeleton overlays."""
        pose_by_frame = {pf.frame_idx: pf for pf in pose_seq.frames}
        annotated_path = output_dir / "annotated_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(annotated_path), fourcc,
            video_info.fps, (video_info.width, video_info.height),
        )
        cap = cv2.VideoCapture(str(video_path))
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

    def _extract_key_frames(
        self,
        video_path: Path,
        pose_seq: PoseSequence,
        event_names: dict[str, Optional[int]],
        throw_side: str,
        key_frames_dir: Path,
    ) -> dict[str, str]:
        """Extract and annotate key frames for each detected event."""
        key_frame_images: dict[str, str] = {}

        for event_label, event_frame in event_names.items():
            if event_frame is None:
                continue
            if event_frame >= len(pose_seq.frames):
                continue

            pf = pose_seq.frames[event_frame]
            actual_frame_idx = pf.frame_idx

            cap = cv2.VideoCapture(str(video_path))
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

            # Add text label
            label_text = f"{event_label} (frame {actual_frame_idx})"
            cv2.putText(
                frame, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )

            # Save PNG
            png_path = key_frames_dir / f"{event_label.lower().replace(' ', '_')}.png"
            cv2.imwrite(str(png_path), frame)

            key_frame_images[event_label] = _encode_frame_as_base64(frame)

        return key_frame_images

    def _build_trajectory_plots(
        self,
        pose_seq: PoseSequence,
        events: DeliveryEvents,
        lead_side: str,
        throw_side: str,
    ) -> list[str]:
        """Build Plotly trajectory plot HTML fragments."""
        fps = pose_seq.video_info.fps
        timestamps = np.array([pf.timestamp for pf in pose_seq.frames])

        lead_knee_traj = pose_seq.get_joint_trajectory(f"{lead_side}_knee")
        lead_knee_y_inverted = -lead_knee_traj[:, 1]

        lead_ankle_traj = pose_seq.get_joint_trajectory(f"{lead_side}_ankle")
        lead_ankle_y_inverted = -lead_ankle_traj[:, 1]

        wrist_speed_raw = pose_seq.get_joint_speed(f"{throw_side}_wrist") * fps
        wrist_speed = _smooth_moving_avg(wrist_speed_raw, window=3)

        def _event_time(frame_idx):
            if frame_idx is None:
                return None
            return frame_idx / fps

        plots: list[str] = []

        # 1. Lead knee Y
        knee_events = {}
        if events.leg_lift_apex is not None:
            knee_events["Leg Lift"] = _event_time(events.leg_lift_apex)
        if events.foot_plant is not None:
            knee_events["Foot Plant"] = _event_time(events.foot_plant)
        fig_knee = plot_joint_trajectory(
            lead_knee_y_inverted, timestamps,
            f"{lead_side.title()} Knee Y (inverted)", events=knee_events,
        )
        plots.append(fig_knee.to_html(full_html=False, include_plotlyjs=False))

        # 2. Lead ankle Y
        ankle_events = {}
        if events.foot_plant is not None:
            ankle_events["Foot Plant"] = _event_time(events.foot_plant)
        fig_ankle = plot_joint_trajectory(
            lead_ankle_y_inverted, timestamps,
            f"{lead_side.title()} Ankle Y (inverted)", events=ankle_events,
        )
        plots.append(fig_ankle.to_html(full_html=False, include_plotlyjs=False))

        # 3. Wrist speed
        wrist_events = {}
        if events.max_external_rotation is not None:
            wrist_events["Max ER"] = _event_time(events.max_external_rotation)
        if events.ball_release is not None:
            wrist_events["Ball Release"] = _event_time(events.ball_release)
        fig_wrist = plot_wrist_speed(wrist_speed, timestamps, events=wrist_events)
        plots.append(fig_wrist.to_html(full_html=False, include_plotlyjs=False))

        # 4. Confidence heatmap
        heatmap_joints = [
            f"{throw_side}_shoulder", f"{throw_side}_elbow", f"{throw_side}_wrist",
            f"{lead_side}_hip", f"{lead_side}_knee", f"{lead_side}_ankle",
        ]
        conf_matrix = np.zeros((len(heatmap_joints), len(pose_seq.frames)))
        for col_idx, pf in enumerate(pose_seq.frames):
            for row_idx, joint in enumerate(heatmap_joints):
                conf_matrix[row_idx, col_idx] = pf.confidence.get(joint, 0.0)
        fig_heatmap = plot_confidence_heatmap(conf_matrix, timestamps)
        fig_heatmap.data[0].y = heatmap_joints
        plots.append(fig_heatmap.to_html(full_html=False, include_plotlyjs=False))

        return plots

    def _build_percentile_charts(self, obp_comparisons: list[dict]) -> list[str]:
        """Build Plotly percentile chart HTML fragments."""
        if not obp_comparisons:
            return []
        charts: list[str] = []
        try:
            radar_fig = plot_pitcher_comparison(obp_comparisons)
            charts.append(radar_fig.to_html(full_html=False, include_plotlyjs=False))
            gauges_fig = plot_percentile_gauges(obp_comparisons)
            charts.append(gauges_fig.to_html(full_html=False, include_plotlyjs=False))
        except Exception:
            pass
        return charts

    def _generate_coaching(
        self,
        metrics: PitcherMetrics,
        obp_comparisons: list[dict],
        youth_comparisons: Optional[list],
        youth_context: Optional[dict],
    ) -> str:
        """Generate coaching report (API or offline fallback)."""
        additional_context = self._build_additional_context(metrics)
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if youth_comparisons and youth_context:
            if api_key:
                try:
                    return generate_youth_coaching_report(
                        youth_comparisons, youth_context,
                        additional_context=additional_context,
                    )
                except Exception:
                    return generate_youth_report_offline(youth_comparisons, youth_context)
            return generate_youth_report_offline(youth_comparisons, youth_context)

        if obp_comparisons:
            if api_key:
                try:
                    return generate_coaching_report(
                        obp_comparisons, additional_context=additional_context,
                    )
                except Exception:
                    return generate_report_offline(obp_comparisons)
            return generate_report_offline(obp_comparisons)

        return ""

    def _build_additional_context(self, metrics: PitcherMetrics) -> Optional[str]:
        """Build additional context string for coaching prompts."""
        lines: list[str] = []
        if metrics.arm_slot_angle is not None:
            lines.append(
                f"Arm slot angle: {metrics.arm_slot_angle:.1f}\u00b0 "
                "(camera angle dependent \u2014 from front-quarter view, reads ~15-22\u00b0 low)"
            )
        if metrics.lead_knee_angle_fp is not None:
            lines.append(f"Lead knee angle at foot plant: {metrics.lead_knee_angle_fp:.1f}\u00b0")
        if metrics.lead_knee_angle_br is not None:
            lines.append(f"Lead knee angle at ball release: {metrics.lead_knee_angle_br:.1f}\u00b0")
        if metrics.stride_length_pct_height is not None:
            lines.append(
                f"Stride length: {metrics.stride_length_pct_height:.0f}% of height "
                "(ASMI target: 75-85%)"
            )
        additional = "\n".join(lines) if lines else None

        caveats = load_prompt("measurement_caveats")
        if caveats and additional:
            additional += f"\n\nMEASUREMENT CAVEATS:\n{caveats}"
        elif caveats:
            additional = f"MEASUREMENT CAVEATS:\n{caveats}"

        return additional

    def _build_metrics_rows(
        self, metrics: PitcherMetrics, obp_comparisons: list[dict],
    ) -> list[dict]:
        """Build display-ready metrics rows with percentile data."""
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

        obp_comp_map = {c["metric"]: c for c in obp_comparisons}
        rows: list[dict] = []

        for metric_key, display_name, unit in metrics_to_show:
            value = getattr(metrics, metric_key, None)
            obp_key = internal_to_obp.get(metric_key)
            comp = obp_comp_map.get(obp_key) if obp_key else None
            median = comp["benchmark_median"] if comp else None
            percentile = comp["percentile_rank"] if comp else None

            rows.append({
                "metric": display_name,
                "value": f"{value:.1f}" if value is not None else "--",
                "unit": unit,
                "obp_median": f"{median:.1f}" if median is not None else "--",
                "percentile": f"{percentile:.0f}" if percentile is not None else "--",
                "status": "ok" if value is not None else "missing",
            })

        return rows

    def _save_results_json(
        self,
        output_dir: Path,
        video_path: Path,
        video_info,
        events: DeliveryEvents,
        event_names: dict,
        metrics: PitcherMetrics,
        metrics_rows: list[dict],
        pose_seq: PoseSequence,
        avg_confidence: float,
        validation_warnings: list,
        obp_comparisons: list[dict],
        coaching_text: str,
        youth_profile_dict: Optional[dict],
    ) -> None:
        """Save pipeline results as JSON."""
        pipeline_output = {
            "video": video_path.name,
            "backend": self.config.backend,
            "pitcher_throws": self.config.throws,
            "fps": video_info.fps,
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

        if obp_comparisons:
            pipeline_output["obp_comparisons"] = [
                {k: v for k, v in c.items()} for c in obp_comparisons
            ]
        if coaching_text:
            pipeline_output["coaching_report"] = coaching_text
        if youth_profile_dict:
            pipeline_output["youth_profile"] = {
                "age": youth_profile_dict["age"],
                "height_in": youth_profile_dict["height_in"],
                "weight_lbs": youth_profile_dict["weight_lbs"],
                "developmental_stage": youth_profile_dict.get("developmental_stage"),
            }

        results_path = output_dir / "results.json"
        results_path.write_text(json.dumps(pipeline_output, indent=2, default=str))
