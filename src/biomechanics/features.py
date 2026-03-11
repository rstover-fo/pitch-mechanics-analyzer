"""Extract biomechanical features aligned to OBP metric definitions.

Computes pitching mechanics metrics from pose estimation keypoints at
detected delivery events. The goal is to produce metrics that are
comparable to the OBP POI dataset, even though they're derived from
2D camera data rather than 3D motion capture.

Limitations of camera-based analysis vs. motion capture:
  - 2D projection loses depth information (especially hip rotation)
  - Lower spatial resolution than marker-based systems
  - No force plate data (GRF metrics unavailable)
  - Joint moment estimation requires inverse dynamics (not feasible from video alone)

Metrics we CAN estimate from side-view video:
  - Elbow flexion angle
  - Shoulder abduction / horizontal abduction
  - Shoulder external rotation (approximate from 2D)
  - Trunk forward tilt and lateral tilt
  - Hip-shoulder separation (approximate)
  - Lead knee angle at foot plant and release
  - Stride length (relative to body height)
  - Arm slot angle at release

Metrics we CANNOT estimate (require force plates or 3D):
  - Joint moments (elbow varus, shoulder IR moment)
  - Ground reaction forces
  - Energy flow / power metrics
  - Precise rotational velocities (need 3D)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.biomechanics.events import DeliveryEvents


@dataclass
class PitcherMetrics:
    """Extracted biomechanical metrics for a single pitch.

    Values are in degrees (angles) or deg/s (velocities).
    None indicates the metric couldn't be computed from available data.
    """
    # Identity
    pitcher_throws: str = "R"
    camera_view: str = "side"

    # At foot plant
    elbow_flexion_fp: Optional[float] = None
    shoulder_abduction_fp: Optional[float] = None
    shoulder_horizontal_abduction_fp: Optional[float] = None
    torso_anterior_tilt_fp: Optional[float] = None
    torso_lateral_tilt_fp: Optional[float] = None
    lead_knee_angle_fp: Optional[float] = None
    hip_shoulder_separation_fp: Optional[float] = None

    # Peak values
    max_shoulder_external_rotation: Optional[float] = None
    max_hip_shoulder_separation: Optional[float] = None

    # At ball release
    torso_anterior_tilt_br: Optional[float] = None
    torso_lateral_tilt_br: Optional[float] = None
    lead_knee_angle_br: Optional[float] = None
    arm_slot_angle: Optional[float] = None

    # Derived / additional
    stride_length_pct_height: Optional[float] = None

    # Velocities (approximate from frame-to-frame changes)
    max_trunk_rotation_velo: Optional[float] = None
    max_arm_speed: Optional[float] = None

    def to_obp_comparison_dict(self) -> dict[str, float]:
        """Convert to dict with OBP-compatible metric names for benchmark comparison.

        Only includes metrics that have values and map to OBP metrics.
        """
        mapping = {
            "elbow_flexion_fp": self.elbow_flexion_fp,
            "shoulder_abduction_fp": self.shoulder_abduction_fp,
            "shoulder_horizontal_abduction_fp": self.shoulder_horizontal_abduction_fp,
            "torso_anterior_tilt_fp": self.torso_anterior_tilt_fp,
            "torso_lateral_tilt_fp": self.torso_lateral_tilt_fp,
            "max_shoulder_external_rotation": self.max_shoulder_external_rotation,
            "rotation_hip_shoulder_separation_fp": self.hip_shoulder_separation_fp,
            "max_rotation_hip_shoulder_separation": self.max_hip_shoulder_separation,
            "torso_anterior_tilt_br": self.torso_anterior_tilt_br,
            "torso_lateral_tilt_br": self.torso_lateral_tilt_br,
        }
        return {k: v for k, v in mapping.items() if v is not None}


def angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle at point b formed by segments ba and bc.

    Args:
        a, b, c: 2D points as (x, y) arrays.

    Returns:
        Angle in degrees.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def compute_elbow_flexion(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
    """Compute elbow flexion angle (degrees).

    Full extension = ~180°, typical at foot plant = 80-100°.
    """
    return angle_between_points(shoulder, elbow, wrist)


def compute_trunk_tilt(
    hip_center: np.ndarray,
    shoulder_center: np.ndarray,
    vertical: np.ndarray | None = None,
) -> float:
    """Compute trunk forward tilt from vertical (degrees).

    0° = perfectly upright, positive = forward lean.
    """
    if vertical is None:
        dim = hip_center.shape[-1]
        vertical = np.array([0, -1, 0]) if dim == 3 else np.array([0, -1])
    trunk_vec = shoulder_center - hip_center
    trunk_norm = trunk_vec / (np.linalg.norm(trunk_vec) + 1e-8)
    cos_angle = np.dot(trunk_norm, vertical)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def compute_arm_slot(
    shoulder: np.ndarray,
    release_point: np.ndarray,
) -> float:
    """Compute arm slot angle at ball release.

    Measured from horizontal: 0° = sidearm, 90° = overhand.
    """
    arm_vec = release_point - shoulder
    dim = shoulder.shape[-1]
    horizontal = np.array([1, 0, 0]) if dim == 3 else np.array([1, 0])
    cos_angle = np.dot(arm_vec / (np.linalg.norm(arm_vec) + 1e-8), horizontal)
    angle_from_horiz = np.degrees(np.arccos(np.clip(abs(cos_angle), 0, 1)))
    return float(angle_from_horiz)


def compute_stride_length(
    rear_ankle_at_fp: np.ndarray,
    lead_ankle_at_fp: np.ndarray,
    body_height_pixels: float,
) -> float:
    """Compute stride length as percentage of body height.

    Elite pitchers typically stride 75-85% of body height.
    """
    stride_px = np.linalg.norm(lead_ankle_at_fp - rear_ankle_at_fp)
    return float(stride_px / body_height_pixels * 100)


def extract_metrics(
    keypoints: dict[str, np.ndarray],
    events: DeliveryEvents,
    pitcher_throws: str = "R",
    camera_view: str = "side",
    use_3d: bool = False,
) -> PitcherMetrics:
    """Extract all available biomechanical metrics from keypoints at detected events.

    Args:
        keypoints: Dict mapping joint names to (N_frames, D) arrays.
                   D=2 for (x, y) or D=3 for (x, y, z).
        events: Detected delivery events with frame indices.
        pitcher_throws: "R" or "L".
        camera_view: "side" or "behind".
        use_3d: If True, compute additional 3D-specific metrics.

    Returns:
        PitcherMetrics with all computable metrics filled in.
    """
    metrics = PitcherMetrics(pitcher_throws=pitcher_throws, camera_view=camera_view)

    # Determine throwing / glove sides
    throw_side = "right" if pitcher_throws == "R" else "left"
    lead_side = "left" if pitcher_throws == "R" else "right"

    # Helper to get joint position at a specific frame
    def at(joint: str, frame: int) -> Optional[np.ndarray]:
        key = f"{joint}"
        if key in keypoints and frame < len(keypoints[key]):
            return keypoints[key][frame]
        return None

    # --- Metrics at foot plant ---
    if events.foot_plant is not None:
        fp = events.foot_plant

        shoulder = at(f"{throw_side}_shoulder", fp)
        elbow = at(f"{throw_side}_elbow", fp)
        wrist = at(f"{throw_side}_wrist", fp)
        if all(p is not None for p in [shoulder, elbow, wrist]):
            metrics.elbow_flexion_fp = compute_elbow_flexion(shoulder, elbow, wrist)

        throw_hip = at(f"{throw_side}_hip", fp)
        lead_hip = at(f"{lead_side}_hip", fp)
        throw_shoulder = at(f"{throw_side}_shoulder", fp)
        lead_shoulder = at(f"{lead_side}_shoulder", fp)
        if all(p is not None for p in [throw_hip, lead_hip, throw_shoulder, lead_shoulder]):
            hip_center = (throw_hip + lead_hip) / 2
            shoulder_center = (throw_shoulder + lead_shoulder) / 2
            metrics.torso_anterior_tilt_fp = compute_trunk_tilt(hip_center, shoulder_center)

        lead_hip_pt = at(f"{lead_side}_hip", fp)
        lead_knee_pt = at(f"{lead_side}_knee", fp)
        lead_ankle_pt = at(f"{lead_side}_ankle", fp)
        if all(p is not None for p in [lead_hip_pt, lead_knee_pt, lead_ankle_pt]):
            metrics.lead_knee_angle_fp = angle_between_points(lead_hip_pt, lead_knee_pt, lead_ankle_pt)

    # --- Peak values ---
    if events.max_external_rotation is not None:
        mer = events.max_external_rotation
        shoulder = at(f"{throw_side}_shoulder", mer)
        elbow = at(f"{throw_side}_elbow", mer)
        wrist = at(f"{throw_side}_wrist", mer)
        if all(p is not None for p in [shoulder, elbow, wrist]):
            # Approximate shoulder ER from 2D as the angle behind the shoulder line
            metrics.max_shoulder_external_rotation = compute_elbow_flexion(shoulder, elbow, wrist)

    # --- Metrics at ball release ---
    if events.ball_release is not None:
        br = events.ball_release

        throw_hip = at(f"{throw_side}_hip", br)
        lead_hip = at(f"{lead_side}_hip", br)
        throw_shoulder = at(f"{throw_side}_shoulder", br)
        lead_shoulder = at(f"{lead_side}_shoulder", br)
        if all(p is not None for p in [throw_hip, lead_hip, throw_shoulder, lead_shoulder]):
            hip_center = (throw_hip + lead_hip) / 2
            shoulder_center = (throw_shoulder + lead_shoulder) / 2
            metrics.torso_anterior_tilt_br = compute_trunk_tilt(hip_center, shoulder_center)

        shoulder = at(f"{throw_side}_shoulder", br)
        wrist = at(f"{throw_side}_wrist", br)
        if shoulder is not None and wrist is not None:
            metrics.arm_slot_angle = compute_arm_slot(shoulder, wrist)

        lead_knee_pt = at(f"{lead_side}_knee", br)
        lead_hip_pt = at(f"{lead_side}_hip", br)
        lead_ankle_pt = at(f"{lead_side}_ankle", br)
        if all(p is not None for p in [lead_hip_pt, lead_knee_pt, lead_ankle_pt]):
            metrics.lead_knee_angle_br = angle_between_points(lead_hip_pt, lead_knee_pt, lead_ankle_pt)

    # --- 3D-specific metrics ---
    if use_3d and events.foot_plant is not None:
        from src.biomechanics.angles_3d import (
            compute_hip_shoulder_separation_3d,
            compute_shoulder_abduction_3d,
            compute_shoulder_horizontal_abduction_3d,
            compute_torso_lateral_tilt_3d,
        )

        fp = events.foot_plant
        throw_side = "right" if pitcher_throws == "R" else "left"
        lead_side = "left" if pitcher_throws == "R" else "right"

        l_hip = at("left_hip", fp)
        r_hip = at("right_hip", fp)
        l_sho = at("left_shoulder", fp)
        r_sho = at("right_shoulder", fp)

        if all(p is not None for p in [l_hip, r_hip, l_sho, r_sho]):
            metrics.hip_shoulder_separation_fp = compute_hip_shoulder_separation_3d(
                l_hip, r_hip, l_sho, r_sho
            )
            hip_center = (l_hip + r_hip) / 2
            shoulder_center = (l_sho + r_sho) / 2
            metrics.torso_lateral_tilt_fp = compute_torso_lateral_tilt_3d(
                hip_center, shoulder_center, l_sho, r_sho
            )

        shoulder = at(f"{throw_side}_shoulder", fp)
        elbow = at(f"{throw_side}_elbow", fp)
        if all(p is not None for p in [shoulder, elbow, l_hip, r_hip, l_sho, r_sho]):
            hip_center = (l_hip + r_hip) / 2
            shoulder_center = (l_sho + r_sho) / 2
            metrics.shoulder_abduction_fp = compute_shoulder_abduction_3d(
                shoulder, elbow, hip_center, shoulder_center
            )
            metrics.shoulder_horizontal_abduction_fp = compute_shoulder_horizontal_abduction_3d(
                shoulder, elbow, l_sho, r_sho
            )

    # --- 3D-specific: torso lateral tilt at ball release ---
    if use_3d and events.ball_release is not None:
        from src.biomechanics.angles_3d import compute_torso_lateral_tilt_3d

        br = events.ball_release
        l_hip = at("left_hip", br)
        r_hip = at("right_hip", br)
        l_sho = at("left_shoulder", br)
        r_sho = at("right_shoulder", br)
        if all(p is not None for p in [l_hip, r_hip, l_sho, r_sho]):
            hip_center = (l_hip + r_hip) / 2
            shoulder_center = (l_sho + r_sho) / 2
            metrics.torso_lateral_tilt_br = compute_torso_lateral_tilt_3d(
                hip_center, shoulder_center, l_sho, r_sho
            )

    return metrics
