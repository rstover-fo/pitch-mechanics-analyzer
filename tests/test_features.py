"""Tests for feature extraction (features.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biomechanics.events import DeliveryEvents
from src.biomechanics.features import (
    extract_metrics,
    compute_hip_shoulder_separation,
    PitcherMetrics,
)


class TestExtractMetrics:
    """Tests for the extract_metrics() pipeline."""

    def test_all_metrics_populated(self, synthetic_keypoints, delivery_events):
        """With good input data and events, all 2D-feasible metrics should be populated."""
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=delivery_events,
            pitcher_throws="R",
        )
        assert metrics.pitcher_throws == "R"
        assert metrics.elbow_flexion_fp is not None
        assert metrics.torso_anterior_tilt_fp is not None
        assert metrics.lead_knee_angle_fp is not None
        assert metrics.hip_shoulder_separation_fp is not None
        assert metrics.max_shoulder_external_rotation is not None
        assert metrics.torso_anterior_tilt_br is not None
        assert metrics.lead_knee_angle_br is not None
        assert metrics.arm_slot_angle is not None
        assert metrics.max_hip_shoulder_separation is not None
        assert metrics.max_arm_speed is not None

    def test_3d_metrics_remain_none(self, synthetic_keypoints, delivery_events):
        """Metrics requiring 3D should always be None from 2D extraction."""
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=delivery_events,
        )
        assert metrics.shoulder_abduction_fp is None
        assert metrics.shoulder_horizontal_abduction_fp is None
        assert metrics.max_trunk_rotation_velo is None

    def test_missing_foot_plant_event(self, synthetic_keypoints):
        """If foot_plant is None, foot-plant metrics should be None."""
        events = DeliveryEvents(
            leg_lift_apex=10,
            foot_plant=None,
            max_external_rotation=35,
            ball_release=40,
            fps=30.0,
        )
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=events,
        )
        assert metrics.elbow_flexion_fp is None
        assert metrics.lead_knee_angle_fp is None
        assert metrics.hip_shoulder_separation_fp is None
        assert metrics.torso_anterior_tilt_fp is None
        # Ball release metrics should still work
        assert metrics.torso_anterior_tilt_br is not None
        assert metrics.arm_slot_angle is not None

    def test_missing_ball_release_event(self, synthetic_keypoints):
        """If ball_release is None, release metrics should be None."""
        events = DeliveryEvents(
            leg_lift_apex=10,
            foot_plant=25,
            max_external_rotation=35,
            ball_release=None,
            fps=30.0,
        )
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=events,
        )
        assert metrics.torso_anterior_tilt_br is None
        assert metrics.arm_slot_angle is None
        assert metrics.lead_knee_angle_br is None
        # Foot plant metrics should still work
        assert metrics.elbow_flexion_fp is not None

    def test_missing_mer_event(self, synthetic_keypoints):
        """If max_external_rotation is None, MER metric should be None."""
        events = DeliveryEvents(
            leg_lift_apex=10,
            foot_plant=25,
            max_external_rotation=None,
            ball_release=40,
            fps=30.0,
        )
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=events,
        )
        assert metrics.max_shoulder_external_rotation is None

    def test_all_events_none(self, synthetic_keypoints):
        """With no events detected, only max_arm_speed should survive."""
        events = DeliveryEvents(fps=30.0)
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=events,
        )
        assert metrics.elbow_flexion_fp is None
        assert metrics.lead_knee_angle_fp is None
        assert metrics.max_shoulder_external_rotation is None
        assert metrics.torso_anterior_tilt_br is None
        # max_arm_speed doesn't depend on events
        assert metrics.max_arm_speed is not None

    def test_rhp_side_selection(self, synthetic_keypoints, delivery_events):
        """RHP: throwing arm = right, lead leg = left."""
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=delivery_events,
            pitcher_throws="R",
        )
        # Lead knee angle should use left side joints
        assert metrics.lead_knee_angle_fp is not None

    def test_lhp_side_selection(self, synthetic_keypoints, delivery_events):
        """LHP: throwing arm = left, lead leg = right."""
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=delivery_events,
            pitcher_throws="L",
        )
        # Should still compute metrics (just with swapped sides)
        assert metrics.pitcher_throws == "L"
        assert metrics.lead_knee_angle_fp is not None
        assert metrics.elbow_flexion_fp is not None

    def test_angles_in_plausible_range(self, synthetic_keypoints, delivery_events):
        """All angle metrics should be in 0-180 degree range."""
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=delivery_events,
        )
        for field_name in [
            "elbow_flexion_fp", "lead_knee_angle_fp", "lead_knee_angle_br",
            "torso_anterior_tilt_fp", "torso_anterior_tilt_br",
            "hip_shoulder_separation_fp", "max_hip_shoulder_separation",
            "max_shoulder_external_rotation", "arm_slot_angle",
        ]:
            val = getattr(metrics, field_name)
            if val is not None:
                assert 0 <= val <= 180, f"{field_name}={val} out of range"

    def test_max_arm_speed_positive(self, synthetic_keypoints, delivery_events):
        """Max arm speed should be a positive value."""
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=delivery_events,
        )
        assert metrics.max_arm_speed is not None
        assert metrics.max_arm_speed > 0

    def test_max_arm_speed_with_nan_frames(self, delivery_events):
        """max_arm_speed should ignore NaN frames and still return a valid value."""
        rng = np.random.RandomState(99)
        n = 60
        wrist = np.column_stack([
            np.linspace(400, 500, n) + rng.randn(n),
            np.linspace(240, 200, n) + rng.randn(n),
        ])
        # Inject NaN in a few frames
        wrist[10] = [np.nan, np.nan]
        wrist[30] = [np.nan, np.nan]

        kp = {
            "right_shoulder": np.tile([320, 200], (n, 1)).astype(float),
            "right_elbow": np.tile([360, 220], (n, 1)).astype(float),
            "right_wrist": wrist,
            "right_hip": np.tile([310, 350], (n, 1)).astype(float),
            "right_knee": np.tile([310, 450], (n, 1)).astype(float),
            "right_ankle": np.tile([310, 540], (n, 1)).astype(float),
            "left_shoulder": np.tile([280, 200], (n, 1)).astype(float),
            "left_elbow": np.tile([240, 220], (n, 1)).astype(float),
            "left_wrist": np.tile([200, 240], (n, 1)).astype(float),
            "left_hip": np.tile([290, 350], (n, 1)).astype(float),
            "left_knee": np.tile([270, 450], (n, 1)).astype(float),
            "left_ankle": np.tile([250, 540], (n, 1)).astype(float),
        }
        metrics = extract_metrics(keypoints=kp, events=delivery_events, pitcher_throws="R")
        assert metrics.max_arm_speed is not None
        assert not np.isnan(metrics.max_arm_speed)
        assert metrics.max_arm_speed > 0

    def test_max_arm_speed_all_nan(self, delivery_events):
        """When ALL wrist positions are NaN, max_arm_speed should be None."""
        n = 60
        wrist = np.full((n, 2), np.nan)
        kp = {
            "right_shoulder": np.tile([320, 200], (n, 1)).astype(float),
            "right_elbow": np.tile([360, 220], (n, 1)).astype(float),
            "right_wrist": wrist,
            "right_hip": np.tile([310, 350], (n, 1)).astype(float),
            "right_knee": np.tile([310, 450], (n, 1)).astype(float),
            "right_ankle": np.tile([310, 540], (n, 1)).astype(float),
            "left_shoulder": np.tile([280, 200], (n, 1)).astype(float),
            "left_elbow": np.tile([240, 220], (n, 1)).astype(float),
            "left_wrist": np.tile([200, 240], (n, 1)).astype(float),
            "left_hip": np.tile([290, 350], (n, 1)).astype(float),
            "left_knee": np.tile([270, 450], (n, 1)).astype(float),
            "left_ankle": np.tile([250, 540], (n, 1)).astype(float),
        }
        metrics = extract_metrics(keypoints=kp, events=delivery_events, pitcher_throws="R")
        assert metrics.max_arm_speed is None

    def test_max_hip_shoulder_separation_zero_is_valid(self, delivery_events):
        """When hips and shoulders are always parallel, max_hip_shoulder_separation should be 0.0, not None."""
        n = 60
        # All joints static — hip and shoulder lines perfectly parallel
        kp = {
            "right_shoulder": np.tile([320, 200], (n, 1)).astype(float),
            "right_elbow": np.tile([360, 220], (n, 1)).astype(float),
            "right_wrist": np.tile([400, 240], (n, 1)).astype(float),
            "right_hip": np.tile([320, 350], (n, 1)).astype(float),
            "right_knee": np.tile([310, 450], (n, 1)).astype(float),
            "right_ankle": np.tile([310, 540], (n, 1)).astype(float),
            "left_shoulder": np.tile([280, 200], (n, 1)).astype(float),
            "left_elbow": np.tile([240, 220], (n, 1)).astype(float),
            "left_wrist": np.tile([200, 240], (n, 1)).astype(float),
            "left_hip": np.tile([280, 350], (n, 1)).astype(float),
            "left_knee": np.tile([270, 450], (n, 1)).astype(float),
            "left_ankle": np.tile([250, 540], (n, 1)).astype(float),
        }
        metrics = extract_metrics(keypoints=kp, events=delivery_events, pitcher_throws="R")
        assert metrics.max_hip_shoulder_separation is not None
        assert metrics.max_hip_shoulder_separation == pytest.approx(0.0, abs=1.0)

    def test_max_hip_shoulder_sep_gte_fp(self, synthetic_keypoints, delivery_events):
        """Max hip-shoulder separation should be >= separation at foot plant."""
        metrics = extract_metrics(
            keypoints=synthetic_keypoints,
            events=delivery_events,
        )
        if metrics.hip_shoulder_separation_fp is not None and metrics.max_hip_shoulder_separation is not None:
            assert metrics.max_hip_shoulder_separation >= metrics.hip_shoulder_separation_fp - 1e-6


class TestComputeHipShoulderSeparation:
    """Tests for the compute_hip_shoulder_separation helper."""

    def test_aligned_returns_zero(self):
        """When hip and shoulder lines are parallel, separation = 0."""
        sep = compute_hip_shoulder_separation(
            left_hip=np.array([0.0, 0.0]),
            right_hip=np.array([10.0, 0.0]),
            left_shoulder=np.array([0.0, -10.0]),
            right_shoulder=np.array([10.0, -10.0]),
        )
        assert sep < 1.0  # Nearly zero

    def test_perpendicular_returns_90(self):
        """When hip and shoulder lines are perpendicular, separation = 90."""
        sep = compute_hip_shoulder_separation(
            left_hip=np.array([0.0, 0.0]),
            right_hip=np.array([10.0, 0.0]),
            left_shoulder=np.array([5.0, -10.0]),
            right_shoulder=np.array([5.0, 0.0]),
        )
        assert abs(sep - 90.0) < 1.0

    def test_typical_separation(self):
        """30-degree separation (typical at foot plant)."""
        angle_rad = np.radians(30)
        sep = compute_hip_shoulder_separation(
            left_hip=np.array([0.0, 0.0]),
            right_hip=np.array([10.0, 0.0]),
            left_shoulder=np.array([0.0, -10.0]),
            right_shoulder=np.array([10 * np.cos(angle_rad), -10.0 + 10 * np.sin(angle_rad)]),
        )
        assert 20 <= sep <= 40


class TestPitcherMetricsOBP:
    """Tests for to_obp_comparison_dict."""

    def test_excludes_none_values(self):
        m = PitcherMetrics(elbow_flexion_fp=90.0, lead_knee_angle_fp=150.0)
        d = m.to_obp_comparison_dict()
        assert "elbow_flexion_fp" in d
        assert d["elbow_flexion_fp"] == 90.0
        # shoulder_abduction_fp is None, should be excluded
        assert "shoulder_abduction_fp" not in d

    def test_hip_shoulder_sep_mapped(self):
        m = PitcherMetrics(hip_shoulder_separation_fp=30.0, max_hip_shoulder_separation=45.0)
        d = m.to_obp_comparison_dict()
        assert d["rotation_hip_shoulder_separation_fp"] == 30.0
        assert d["max_rotation_hip_shoulder_separation"] == 45.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
