"""3D angle calculations for biomechanical analysis.

All functions take 3D joint positions as numpy arrays and return angles in degrees.
Coordinate system: MotionBERT camera frame (X=right, Y=down, Z=depth).
"""

import numpy as np


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in degrees."""
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def compute_hip_shoulder_separation_3d(
    l_hip: np.ndarray, r_hip: np.ndarray,
    l_sho: np.ndarray, r_sho: np.ndarray,
) -> float:
    """Angle between hip line and shoulder line projected onto transverse (XZ) plane.

    Returns angle in degrees (always non-negative).
    """
    hip_vec = r_hip - l_hip
    sho_vec = r_sho - l_sho

    # Project onto transverse plane (XZ — remove Y component)
    hip_xz = np.array([hip_vec[0], hip_vec[2]])
    sho_xz = np.array([sho_vec[0], sho_vec[2]])

    return _angle_between(hip_xz, sho_xz)


def compute_shoulder_abduction_3d(
    shoulder: np.ndarray, elbow: np.ndarray,
    hip_center: np.ndarray, shoulder_center: np.ndarray,
) -> float:
    """Angle of upper arm from trunk axis in the coronal plane.

    0° = arm at side, 90° = arm horizontal (T-pose).
    """
    # Trunk downward direction (from shoulders toward hips)
    trunk_down = hip_center - shoulder_center
    upper_arm = elbow - shoulder

    return _angle_between(trunk_down, upper_arm)


def compute_shoulder_horizontal_abduction_3d(
    shoulder: np.ndarray, elbow: np.ndarray,
    l_sho: np.ndarray, r_sho: np.ndarray,
) -> float:
    """Arm angle relative to forward direction in the transverse plane.

    0° = arm pointing forward, 90° = arm along shoulder line.
    """
    sho_line = r_sho - l_sho
    upper_arm = elbow - shoulder

    # Forward direction = perpendicular to shoulder line in XZ plane
    # Camera frame: Y=down, so up=[0,-1,0]. Cross with up gives forward.
    forward = np.cross(sho_line, np.array([0, -1, 0]))
    forward_norm = forward / (np.linalg.norm(forward) + 1e-8)

    # Project upper arm onto transverse plane (remove Y)
    arm_xz = np.array([upper_arm[0], 0, upper_arm[2]])
    forward_xz = np.array([forward_norm[0], 0, forward_norm[2]])

    return _angle_between(arm_xz, forward_xz)


def compute_torso_lateral_tilt_3d(
    hip_center: np.ndarray, shoulder_center: np.ndarray,
    l_sho: np.ndarray, r_sho: np.ndarray,
) -> float:
    """Trunk lean angle from vertical in the coronal plane.

    0° = upright, positive = leaning.
    """
    trunk = shoulder_center - hip_center

    # Vertical direction (Y axis — camera frame Y=down, so vertical = [0, -1, 0])
    vertical = np.array([0, -1, 0])

    angle = _angle_between(trunk, vertical)
    return abs(angle)
