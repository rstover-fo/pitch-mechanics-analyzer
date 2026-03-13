"""Key frame overlay renderer with color-coded glowing rings on joints.

Maps each visible joint to a grade metric based on pitcher handedness,
then draws a colored glowing ring:
  - Green (#27AE60)  → metric within ASMI target
  - Amber (#F5A623)  → within tolerance band (working on it)
  - Red   (#E74C3C)  → outside tolerance (focus area)

Returns the annotated frame as a base64 PNG string.
"""

from typing import Optional

import cv2
import numpy as np

from src.biomechanics.features import PitcherMetrics


# ── Grade rules: metric → (ideal, tolerance) ────────────────────────────
# Green  if abs(value − ideal) ≤ tolerance
# Amber  if abs(value − ideal) ≤ 2 × tolerance
# Red    otherwise

GRADE_RULES: dict[str, tuple[float, float]] = {
    "elbow_flexion_fp":           (90.0,  15.0),
    "shoulder_abduction_fp":      (90.0,  15.0),
    "torso_anterior_tilt_fp":     (30.0,  10.0),
    "hip_shoulder_separation_fp": (30.0,  10.0),
    "stride_length_pct_height":   (80.0,  10.0),
    "lead_knee_angle_fp":         (160.0, 15.0),
}

# ── Joint-to-grade mapping (keyed by handedness) ────────────────────────
# Each entry: joint_name → metric_name
# At foot plant, these joints light up with the corresponding metric grade.

_RHP_JOINT_TO_GRADE: dict[str, str] = {
    "right_elbow":    "elbow_flexion_fp",
    "right_shoulder": "shoulder_abduction_fp",
    "left_hip":       "hip_shoulder_separation_fp",
    "right_hip":      "hip_shoulder_separation_fp",
    "left_knee":      "lead_knee_angle_fp",
    "left_ankle":     "stride_length_pct_height",
}

_LHP_JOINT_TO_GRADE: dict[str, str] = {
    "left_elbow":     "elbow_flexion_fp",
    "left_shoulder":  "shoulder_abduction_fp",
    "right_hip":      "hip_shoulder_separation_fp",
    "left_hip":       "hip_shoulder_separation_fp",
    "right_knee":     "lead_knee_angle_fp",
    "right_ankle":    "stride_length_pct_height",
}

# Torso tilt maps to the trunk region (approximate via mid-shoulder)
_TORSO_JOINTS = ("left_shoulder", "right_shoulder")

# ── Colors (BGR for OpenCV) ─────────────────────────────────────────────
_GREEN = (96, 174, 39)     # #27AE60 → BGR
_AMBER = (35, 166, 245)    # #F5A623 → BGR
_RED   = (60, 76, 231)     # #E74C3C → BGR


def _grade_color(metric_name: str, metrics: PitcherMetrics) -> tuple[int, int, int]:
    """Return BGR color for a metric based on its grade rule."""
    value = getattr(metrics, metric_name, None)
    if value is None:
        return _AMBER  # unknown → amber
    ideal, tol = GRADE_RULES.get(metric_name, (0.0, 999.0))
    diff = abs(value - ideal)
    if diff <= tol:
        return _GREEN
    elif diff <= 2 * tol:
        return _AMBER
    else:
        return _RED


def _draw_glow_ring(
    frame: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 18,
    thickness: int = 3,
) -> np.ndarray:
    """Draw a glowing ring on a frame using additive blending."""
    overlay = frame.copy()
    # Outer glow (larger, more transparent)
    cv2.circle(overlay, center, radius + 6, color, 2, cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    # Middle glow
    overlay = frame.copy()
    cv2.circle(overlay, center, radius + 3, color, 2, cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    # Core ring
    cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)
    return frame


def render_graded_overlay(
    frame: np.ndarray,
    keypoints: dict[str, np.ndarray],
    confidence: dict[str, float],
    metrics: PitcherMetrics,
    throws: str = "R",
    min_confidence: float = 0.3,
) -> np.ndarray:
    """Render color-coded glowing rings on joints for the foot-plant frame.

    Args:
        frame: BGR image (will be copied, not modified in place).
        keypoints: Joint name → (x, y) numpy array from PoseFrame.
        confidence: Joint name → confidence score.
        metrics: Extracted pitcher metrics (used for grading).
        throws: "R" or "L" pitcher handedness.
        min_confidence: Skip joints below this confidence.

    Returns:
        Annotated BGR frame with glowing rings.
    """
    result = frame.copy()
    joint_map = _RHP_JOINT_TO_GRADE if throws == "R" else _LHP_JOINT_TO_GRADE

    # Scale ring radius to frame size (roughly 1.5% of frame height)
    h = frame.shape[0]
    ring_radius = max(12, int(h * 0.015))

    for joint_name, metric_name in joint_map.items():
        if joint_name not in keypoints:
            continue
        if confidence.get(joint_name, 0.0) < min_confidence:
            continue

        pt = keypoints[joint_name]
        center = (int(pt[0]), int(pt[1]))
        color = _grade_color(metric_name, metrics)
        result = _draw_glow_ring(result, center, color, radius=ring_radius)

    # Torso tilt: draw on mid-point between shoulders
    if all(j in keypoints for j in _TORSO_JOINTS):
        ls = keypoints["left_shoulder"]
        rs = keypoints["right_shoulder"]
        ls_conf = confidence.get("left_shoulder", 0.0)
        rs_conf = confidence.get("right_shoulder", 0.0)
        if ls_conf >= min_confidence and rs_conf >= min_confidence:
            mid = ((ls + rs) / 2).astype(int)
            center = (int(mid[0]), int(mid[1]))
            color = _grade_color("torso_anterior_tilt_fp", metrics)
            result = _draw_glow_ring(result, center, color, radius=ring_radius)

    return result
