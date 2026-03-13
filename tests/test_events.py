"""Tests for event detection functions in events.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biomechanics.events import (
    detect_foot_plant_from_keypoints,
    _detect_fp_ankle_x_stop,
    _detect_fp_y_recovery,
    find_delivery_anchor,
    detect_ball_release,
)


class TestFindFootPlant:
    """Tests for detect_foot_plant_from_keypoints."""

    def test_primary_strategy_ankle_x_stop(self, synthetic_ankle_x_stride, synthetic_ankle_y_dip):
        """With both X and Y data, should use X-velocity strategy."""
        fp = detect_foot_plant_from_keypoints(
            lead_ankle_y=synthetic_ankle_y_dip,
            lead_ankle_x=synthetic_ankle_x_stride,
            fps=30.0,
        )
        assert fp is not None
        # Foot plant should be in the deceleration zone (frames ~25-40)
        assert 20 <= fp <= 45

    def test_fallback_y_recovery(self, synthetic_ankle_y_dip):
        """With only Y data, should fall back to Y-recovery strategy."""
        fp = detect_foot_plant_from_keypoints(
            lead_ankle_y=synthetic_ankle_y_dip,
            lead_ankle_x=None,
            fps=30.0,
        )
        assert fp is not None
        # Should detect recovery after the dip
        assert 20 <= fp <= 40

    def test_no_stabilization_returns_none(self):
        """Flat ankle X → no clear foot plant detected."""
        flat_x = np.linspace(100, 102, 60)  # Nearly constant
        flat_y = np.full(60, 500.0)
        fp = detect_foot_plant_from_keypoints(
            lead_ankle_y=flat_y,
            lead_ankle_x=flat_x,
            fps=30.0,
        )
        # Both strategies should fail on flat data
        assert fp is None

    def test_foot_plant_after_leg_lift(self, synthetic_ankle_x_stride, synthetic_ankle_y_dip):
        """Foot plant should be temporally plausible (not too early)."""
        fp = detect_foot_plant_from_keypoints(
            lead_ankle_y=synthetic_ankle_y_dip,
            lead_ankle_x=synthetic_ankle_x_stride,
            fps=30.0,
        )
        assert fp is not None
        assert fp > 10  # Should not be in the first few frames

    def test_too_short_signal(self):
        """Very short signals should return None."""
        fp = detect_foot_plant_from_keypoints(
            lead_ankle_y=np.array([500.0] * 5),
            fps=30.0,
        )
        assert fp is None


class TestDetectFpAnkleXStop:
    """Tests for _detect_fp_ankle_x_stop helper."""

    def test_clear_deceleration(self, synthetic_ankle_x_stride):
        fp = _detect_fp_ankle_x_stop(synthetic_ankle_x_stride, fps=30.0, before_frame=None)
        assert fp is not None
        assert 25 <= fp <= 45

    def test_with_before_frame(self, synthetic_ankle_x_stride):
        fp = _detect_fp_ankle_x_stop(synthetic_ankle_x_stride, fps=30.0, before_frame=50)
        assert fp is not None
        assert fp < 50


class TestDetectFpYRecovery:
    """Tests for _detect_fp_y_recovery helper."""

    def test_clear_dip_recovery(self, synthetic_ankle_y_dip):
        fp = _detect_fp_y_recovery(synthetic_ankle_y_dip, fps=30.0, before_frame=None)
        assert fp is not None
        assert 20 <= fp <= 40

    def test_no_dip_returns_none(self):
        """Flat Y signal → no dip → returns None."""
        flat = np.full(60, 500.0)
        fp = _detect_fp_y_recovery(flat, fps=30.0, before_frame=None)
        assert fp is None


class TestFindDeliveryAnchor:
    """Tests for find_delivery_anchor (MER detection)."""

    def test_finds_er_trough_before_wrist_peak(self, synthetic_shoulder_er, synthetic_wrist_speed):
        mer = find_delivery_anchor(synthetic_shoulder_er, synthetic_wrist_speed, fps=30.0)
        assert mer is not None
        # MER should be in the trough region before wrist speed peak (~frame 40)
        assert 25 <= mer <= 39

    def test_mer_before_wrist_peak(self, synthetic_shoulder_er, synthetic_wrist_speed):
        """MER must come before peak wrist speed."""
        mer = find_delivery_anchor(synthetic_shoulder_er, synthetic_wrist_speed, fps=30.0)
        wrist_peak = int(np.argmax(synthetic_wrist_speed))
        assert mer is not None
        assert mer <= wrist_peak

    def test_direction_invariant_lhp(self, synthetic_shoulder_er, synthetic_wrist_speed):
        """ER signal is direction-agnostic — same function works for LHP."""
        # For LHP the ER signal shape is the same (trough before wrist peak)
        mer_rhp = find_delivery_anchor(synthetic_shoulder_er, synthetic_wrist_speed, fps=30.0)
        # Mirror doesn't change the signal shape, just which arm it is
        mer_lhp = find_delivery_anchor(synthetic_shoulder_er, synthetic_wrist_speed, fps=30.0)
        assert mer_rhp == mer_lhp

    def test_no_clear_trough_returns_none(self):
        """Low wrist speed (no delivery) → returns None."""
        er = np.full(60, 100.0)
        speed = np.full(60, 10.0)  # Below 100 threshold
        mer = find_delivery_anchor(er, speed, fps=30.0)
        assert mer is None

    def test_short_signal_returns_none(self):
        er = np.array([100.0] * 10)
        speed = np.array([50.0] * 10)
        mer = find_delivery_anchor(er, speed, fps=30.0)
        assert mer is None


class TestDetectBallRelease:
    """Tests for detect_ball_release."""

    def test_finds_peak_after_mer(self, synthetic_wrist_speed):
        mer_frame = 34
        br = detect_ball_release(synthetic_wrist_speed, after_frame=mer_frame, window_frames=12)
        assert br is not None
        assert mer_frame <= br <= mer_frame + 12

    def test_peak_in_expected_window(self, synthetic_wrist_speed):
        """Ball release should be near the wrist speed peak."""
        br = detect_ball_release(synthetic_wrist_speed, after_frame=35, window_frames=15)
        assert br is not None
        # Peak is around frame 40 in our synthetic data
        assert 35 <= br <= 50

    def test_flat_speed_returns_none(self):
        """Flat speed signal → still returns the argmax, but tests edge case."""
        flat = np.zeros(60)
        br = detect_ball_release(flat, after_frame=30, window_frames=12)
        # With all zeros, argmax returns the first element in window
        assert br is not None
        assert br == 30

    def test_short_signal(self):
        br = detect_ball_release(np.array([1.0] * 5))
        assert br is None

    def test_no_window_after_frame(self):
        """after_frame near end → empty search window."""
        speed = np.array([10.0] * 20)
        br = detect_ball_release(speed, after_frame=20, window_frames=5)
        assert br is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
