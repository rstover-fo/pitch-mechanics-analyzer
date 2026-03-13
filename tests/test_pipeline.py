"""End-to-end pipeline integration test.

Synthetic PoseSequence → events → metrics → benchmarks → report.
No real video, no API calls, no OBP data files required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biomechanics.events import DeliveryEvents, detect_events
from src.biomechanics.features import extract_metrics
from src.biomechanics.validation import validate_pipeline_output
from src.coaching.insights import generate_report_offline
from src.viz.report import build_report_html
from src.pose.estimator import PoseSequence, PoseFrame, VideoInfo


def _build_synthetic_pose_sequence(n_frames=60, fps=30.0):
    """Build a PoseSequence with synthetic keypoint data (no video needed)."""
    rng = np.random.RandomState(99)
    video_info = VideoInfo(
        path=Path("/tmp/synthetic.mp4"),
        width=640,
        height=480,
        fps=fps,
        total_frames=n_frames,
        duration_sec=n_frames / fps,
    )

    frames = []
    for i in range(n_frames):
        t = i / fps
        kp = {
            "right_shoulder": np.array([320.0 + rng.randn() * 2, 200.0 + rng.randn() * 2]),
            "right_elbow":    np.array([360.0 + rng.randn() * 2, 220.0 + rng.randn() * 2]),
            "right_wrist":    np.array([400.0 + rng.randn() * 2, 240.0 + rng.randn() * 2]),
            "right_hip":      np.array([310.0, 350.0 + rng.randn()]),
            "right_knee":     np.array([310.0, 450.0 + rng.randn()]),
            "right_ankle":    np.array([310.0, 540.0 + rng.randn()]),
            "left_shoulder":  np.array([280.0 + rng.randn() * 2, 200.0 + rng.randn() * 2]),
            "left_elbow":     np.array([240.0 + rng.randn() * 2, 220.0 + rng.randn() * 2]),
            "left_wrist":     np.array([200.0 + rng.randn() * 2, 240.0 + rng.randn() * 2]),
            "left_hip":       np.array([290.0, 350.0 + rng.randn()]),
            "left_knee":      np.array([270.0, 450.0 + rng.randn()]),
            "left_ankle":     np.array([250.0, 540.0 + rng.randn()]),
        }
        conf = {joint: 0.9 for joint in kp}
        frames.append(PoseFrame(
            frame_idx=i,
            timestamp=t,
            keypoints=kp,
            confidence=conf,
        ))

    return PoseSequence(video_info=video_info, frames=frames)


def _make_fake_comparisons():
    """Create fake benchmark comparison dicts for offline report."""
    return [
        {
            "metric": "elbow_flexion_fp",
            "display_name": "Elbow Flexion @ FP",
            "value": 92.0,
            "unit": "deg",
            "percentile_rank": 55.0,
            "benchmark_median": 90.0,
            "playing_level": "all",
            "n_samples": 500,
            "flag": "average",
        },
        {
            "metric": "torso_anterior_tilt_fp",
            "display_name": "Trunk Forward Tilt @ FP",
            "value": 20.0,
            "unit": "deg",
            "percentile_rank": 45.0,
            "benchmark_median": 22.0,
            "playing_level": "all",
            "n_samples": 500,
            "flag": "average",
        },
        {
            "metric": "max_shoulder_external_rotation",
            "display_name": "Peak Shoulder ER",
            "value": 170.0,
            "unit": "deg",
            "percentile_rank": 72.0,
            "benchmark_median": 165.0,
            "playing_level": "all",
            "n_samples": 500,
            "flag": "above_average",
        },
    ]


class TestPipelineIntegration:
    """Full pipeline: PoseSequence → events → metrics → validation → report."""

    def test_end_to_end_produces_html_report(self):
        """Full pipeline runs without error and produces valid HTML."""
        # Step 1: Synthetic PoseSequence
        pose_seq = _build_synthetic_pose_sequence(n_frames=60, fps=30.0)
        keypoints_dict = pose_seq.to_keypoints_dict()

        # Step 2: Create realistic events (skip event detection on synthetic
        # data since it needs very specific signal shapes — use known frames)
        events = DeliveryEvents(
            leg_lift_apex=8,
            foot_plant=22,
            max_external_rotation=32,
            ball_release=38,
            fps=30.0,
        )

        # Step 3: Extract metrics
        metrics = extract_metrics(
            keypoints=keypoints_dict,
            events=events,
            pitcher_throws="R",
            camera_view="side",
        )
        assert metrics.elbow_flexion_fp is not None
        assert metrics.lead_knee_angle_fp is not None
        assert metrics.max_arm_speed is not None

        # Step 4: Validate pipeline output
        metrics_dict = {
            "elbow_flexion_fp": metrics.elbow_flexion_fp,
            "lead_knee_angle_fp": metrics.lead_knee_angle_fp,
        }
        if metrics.max_shoulder_external_rotation is not None:
            metrics_dict["max_shoulder_external_rotation"] = metrics.max_shoulder_external_rotation

        warnings = validate_pipeline_output(
            events=events,
            avg_confidence=0.9,
            metrics=metrics_dict,
        )
        # Warnings should be a list (may or may not have entries)
        assert isinstance(warnings, list)
        for w in warnings:
            assert "code" in w
            assert "severity" in w
            assert "message" in w

        # Step 5: Generate offline coaching report (no API key needed)
        comparisons = _make_fake_comparisons()
        coaching_text = generate_report_offline(comparisons)
        assert isinstance(coaching_text, str)
        assert len(coaching_text) > 50

        # Step 6: Build HTML report
        metrics_rows = [
            {"metric": "Elbow Flexion @ FP", "value": f"{metrics.elbow_flexion_fp:.1f}", "unit": "deg",
             "obp_median": "90", "percentile": "55th", "status": "ok"},
            {"metric": "Lead Knee @ FP", "value": f"{metrics.lead_knee_angle_fp:.1f}", "unit": "deg",
             "obp_median": "150", "percentile": "--", "status": "ok"},
        ]
        html = build_report_html(
            video_filename="synthetic_test.mp4",
            video_rel_path="synthetic_test.mp4",
            fps=30.0,
            frame_count=60,
            backend="synthetic",
            pitcher_throws="R",
            trajectory_plots_html=[],
            key_frame_images={},
            metrics_rows=metrics_rows,
            diagnostics={"test": "true"},
            coaching_html=coaching_text,
        )
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "Pitch Mechanics Report" in html
        assert "Coaching Report" in html
        assert "Elbow Flexion" in html

    def test_pipeline_with_missing_events(self):
        """Pipeline should handle missing events gracefully."""
        pose_seq = _build_synthetic_pose_sequence(n_frames=60, fps=30.0)
        keypoints_dict = pose_seq.to_keypoints_dict()

        events = DeliveryEvents(fps=30.0)  # All events None

        metrics = extract_metrics(
            keypoints=keypoints_dict,
            events=events,
            pitcher_throws="R",
        )
        # Most metrics will be None, but extraction shouldn't crash
        assert metrics.elbow_flexion_fp is None
        assert metrics.lead_knee_angle_fp is None
        # max_arm_speed still works (event-independent)
        assert metrics.max_arm_speed is not None

        warnings = validate_pipeline_output(events=events, avg_confidence=0.9)
        # Should have warnings about missing events
        assert len(warnings) > 0
        codes = [w["code"] for w in warnings]
        assert "event_not_detected" in codes

    def test_pipeline_lhp(self):
        """Pipeline works for left-handed pitchers."""
        pose_seq = _build_synthetic_pose_sequence(n_frames=60, fps=30.0)
        keypoints_dict = pose_seq.to_keypoints_dict()

        events = DeliveryEvents(
            leg_lift_apex=8,
            foot_plant=22,
            max_external_rotation=32,
            ball_release=38,
            fps=30.0,
        )

        metrics = extract_metrics(
            keypoints=keypoints_dict,
            events=events,
            pitcher_throws="L",
        )
        assert metrics.pitcher_throws == "L"
        assert metrics.elbow_flexion_fp is not None
        assert metrics.lead_knee_angle_fp is not None

    def test_pose_sequence_to_keypoints_dict(self):
        """PoseSequence.to_keypoints_dict() produces correct shape."""
        pose_seq = _build_synthetic_pose_sequence(n_frames=40, fps=30.0)
        kp_dict = pose_seq.to_keypoints_dict()

        assert "right_wrist" in kp_dict
        assert "left_ankle" in kp_dict
        assert kp_dict["right_wrist"].shape == (40, 2)
        assert kp_dict["left_ankle"].shape == (40, 2)

    def test_validation_warnings_reasonable(self):
        """Validation warnings are well-formed and have expected structure."""
        events = DeliveryEvents(
            leg_lift_apex=10,
            foot_plant=25,
            max_external_rotation=35,
            ball_release=40,
            fps=30.0,
        )
        warnings = validate_pipeline_output(
            events=events,
            avg_confidence=0.9,
            metrics={"elbow_flexion_fp": 90.0, "lead_knee_angle_fp": 150.0},
        )
        for w in warnings:
            assert w["severity"] in ("error", "warning")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
