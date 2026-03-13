"""Tests for 3D angle calculations.

TDD: Written BEFORE implementation. Tests define the expected API.
All angles in degrees.
"""

import numpy as np
import pytest

from src.biomechanics.angles_3d import (
    compute_hip_shoulder_separation_3d,
    compute_shoulder_abduction_3d,
    compute_shoulder_horizontal_abduction_3d,
    compute_torso_lateral_tilt_3d,
)


class TestHipShoulderSeparation:
    """Hip-shoulder separation: angle between hip line and shoulder line projected onto transverse plane."""

    def test_aligned_is_zero(self):
        """Hips and shoulders aligned → 0° separation."""
        l_hip = np.array([-1.0, 0.0, 0.0])
        r_hip = np.array([1.0, 0.0, 0.0])
        l_sho = np.array([-1.0, -1.0, 0.0])  # Directly above hips (Y is up or down doesn't matter)
        r_sho = np.array([1.0, -1.0, 0.0])
        result = compute_hip_shoulder_separation_3d(l_hip, r_hip, l_sho, r_sho)
        assert abs(result) < 2.0  # Within 2° of zero

    def test_90_degree_rotation(self):
        """Shoulders rotated 90° from hips → ~90° separation."""
        l_hip = np.array([-1.0, 0.0, 0.0])
        r_hip = np.array([1.0, 0.0, 0.0])
        l_sho = np.array([0.0, -1.0, -1.0])  # Shoulders rotated 90° around Y axis
        r_sho = np.array([0.0, -1.0, 1.0])
        result = compute_hip_shoulder_separation_3d(l_hip, r_hip, l_sho, r_sho)
        assert abs(result - 90.0) < 2.0

    def test_returns_positive_angle(self):
        """Separation angle should always be positive."""
        l_hip = np.array([-1.0, 0.0, 0.0])
        r_hip = np.array([1.0, 0.0, 0.0])
        l_sho = np.array([-0.7, -1.0, -0.7])
        r_sho = np.array([0.7, -1.0, 0.7])
        result = compute_hip_shoulder_separation_3d(l_hip, r_hip, l_sho, r_sho)
        assert result >= 0


class TestShoulderAbduction:
    """Shoulder abduction: angle of upper arm from trunk in coronal plane."""

    def test_arms_at_sides_near_zero(self):
        """Arms hanging at sides → ~0° abduction."""
        shoulder = np.array([1.0, -1.0, 0.0])
        elbow = np.array([1.0, 0.0, 0.0])  # Directly below shoulder
        hip_center = np.array([0.0, 0.0, 0.0])
        shoulder_center = np.array([0.0, -1.0, 0.0])
        result = compute_shoulder_abduction_3d(shoulder, elbow, hip_center, shoulder_center)
        assert result < 10.0

    def test_t_pose_90(self):
        """Arms out to sides (T-pose) → ~90° abduction."""
        shoulder = np.array([1.0, -1.0, 0.0])
        elbow = np.array([2.0, -1.0, 0.0])  # Directly lateral from shoulder
        hip_center = np.array([0.0, 0.0, 0.0])
        shoulder_center = np.array([0.0, -1.0, 0.0])
        result = compute_shoulder_abduction_3d(shoulder, elbow, hip_center, shoulder_center)
        assert abs(result - 90.0) < 5.0


class TestShoulderHorizontalAbduction:
    """Horizontal abduction: arm position relative to shoulder line in transverse plane."""

    def test_arm_forward_near_zero(self):
        """Arm pointing forward → ~0° horizontal abduction."""
        shoulder = np.array([1.0, -1.0, 0.0])
        elbow = np.array([1.0, -1.0, -1.0])  # Forward (negative Z = toward camera)
        l_sho = np.array([-1.0, -1.0, 0.0])
        r_sho = np.array([1.0, -1.0, 0.0])
        result = compute_shoulder_horizontal_abduction_3d(shoulder, elbow, l_sho, r_sho)
        assert abs(result) < 10.0

    def test_arm_inline_with_shoulders_90(self):
        """Arm pointing laterally along shoulder line → ~90° horizontal abduction."""
        shoulder = np.array([1.0, -1.0, 0.0])
        elbow = np.array([2.0, -1.0, 0.0])  # Laterally along shoulder line
        l_sho = np.array([-1.0, -1.0, 0.0])
        r_sho = np.array([1.0, -1.0, 0.0])
        result = compute_shoulder_horizontal_abduction_3d(shoulder, elbow, l_sho, r_sho)
        assert abs(result - 90.0) < 5.0


class TestTorsoLateralTilt:
    """Torso lateral tilt: trunk lean in coronal plane."""

    def test_upright_near_zero(self):
        """Vertical trunk → ~0° lateral tilt."""
        hip_center = np.array([0.0, 0.0, 0.0])
        shoulder_center = np.array([0.0, -1.0, 0.0])
        l_sho = np.array([-1.0, -1.0, 0.0])
        r_sho = np.array([1.0, -1.0, 0.0])
        result = compute_torso_lateral_tilt_3d(hip_center, shoulder_center, l_sho, r_sho)
        assert abs(result) < 2.0

    def test_leaning_30(self):
        """30° lateral lean → ~30° tilt."""
        hip_center = np.array([0.0, 0.0, 0.0])
        # Lean 30° to the right: shoulder_center offset in X
        angle_rad = np.radians(30)
        shoulder_center = np.array([np.sin(angle_rad), -np.cos(angle_rad), 0.0])
        l_sho = np.array([shoulder_center[0] - 1, shoulder_center[1], 0.0])
        r_sho = np.array([shoulder_center[0] + 1, shoulder_center[1], 0.0])
        result = compute_torso_lateral_tilt_3d(hip_center, shoulder_center, l_sho, r_sho)
        assert abs(result - 30.0) < 3.0

    def test_returns_nonnegative(self):
        """Tilt angle should be non-negative (magnitude only)."""
        hip_center = np.array([0.0, 0.0, 0.0])
        shoulder_center = np.array([-0.5, -1.0, 0.0])
        l_sho = np.array([-1.5, -1.0, 0.0])
        r_sho = np.array([0.5, -1.0, 0.0])
        result = compute_torso_lateral_tilt_3d(hip_center, shoulder_center, l_sho, r_sho)
        assert result >= 0


class TestGeometrySanity:
    """Generic geometry checks that apply to all angle functions."""

    def test_perpendicular_vectors(self):
        """Hip and shoulder lines at 90° → 90° separation."""
        l_hip = np.array([0.0, 0.0, -1.0])
        r_hip = np.array([0.0, 0.0, 1.0])
        l_sho = np.array([-1.0, -1.0, 0.0])
        r_sho = np.array([1.0, -1.0, 0.0])
        result = compute_hip_shoulder_separation_3d(l_hip, r_hip, l_sho, r_sho)
        assert abs(result - 90.0) < 2.0

    def test_parallel_vectors(self):
        """Hip and shoulder lines parallel → 0° separation."""
        l_hip = np.array([-1.0, 0.0, 0.0])
        r_hip = np.array([1.0, 0.0, 0.0])
        l_sho = np.array([-1.0, -1.5, 0.0])
        r_sho = np.array([1.0, -1.5, 0.0])
        result = compute_hip_shoulder_separation_3d(l_hip, r_hip, l_sho, r_sho)
        assert result < 2.0
