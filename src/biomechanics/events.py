"""Detect key biomechanical events in the pitching delivery from keypoint time series.

Events detected:
  - Leg lift apex (peak knee height of drive leg)
  - Foot contact (~10% bodyweight threshold in Driveline convention)
  - Foot plant (~100% bodyweight, or peak stride extension)
  - Max external rotation (MER / layback)
  - Ball release
  - Max internal rotation (MIR / follow-through)

For camera-based (non-force-plate) analysis, we approximate these from
joint positions and velocities rather than force plate data.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DeliveryEvents:
    """Frame indices and timestamps for key events in the pitching delivery."""
    leg_lift_apex: Optional[int] = None
    foot_contact: Optional[int] = None
    foot_plant: Optional[int] = None
    max_external_rotation: Optional[int] = None
    ball_release: Optional[int] = None
    max_internal_rotation: Optional[int] = None
    fps: float = 30.0

    def time_at(self, frame_idx: Optional[int]) -> Optional[float]:
        """Convert frame index to seconds."""
        if frame_idx is None:
            return None
        return frame_idx / self.fps

    def phase_durations(self) -> dict[str, Optional[float]]:
        """Compute durations of key phases in seconds."""
        return {
            "windup_to_foot_plant": self._delta(self.leg_lift_apex, self.foot_plant),
            "foot_plant_to_mer": self._delta(self.foot_plant, self.max_external_rotation),
            "mer_to_release": self._delta(self.max_external_rotation, self.ball_release),
            "arm_cocking": self._delta(self.foot_plant, self.max_external_rotation),
            "arm_acceleration": self._delta(self.max_external_rotation, self.ball_release),
            "arm_deceleration": self._delta(self.ball_release, self.max_internal_rotation),
        }

    def _delta(self, start: Optional[int], end: Optional[int]) -> Optional[float]:
        if start is None or end is None:
            return None
        return (end - start) / self.fps


def detect_leg_lift(
    lead_knee_y: np.ndarray,
    before_frame: Optional[int] = None,
) -> Optional[int]:
    """Detect leg lift apex as the frame where lead knee reaches peak height.

    Args:
        lead_knee_y: Y-coordinate of lead knee over time.
                     Convention: higher values = higher position (screen coords may need inverting).
        before_frame: Only search before this frame (e.g., before foot plant).
    """
    if len(lead_knee_y) < 10:
        return None
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(lead_knee_y, size=5)

    if before_frame is not None:
        # Search within 1.5s before the reference frame
        window_start = max(0, before_frame - 45)
        region = smoothed[window_start:before_frame]
        if len(region) == 0:
            return None
        return int(np.argmax(region) + window_start)

    return int(np.argmax(smoothed))


def detect_foot_plant_from_keypoints(
    lead_ankle_y: np.ndarray,
    lead_ankle_vy: Optional[np.ndarray] = None,
    fps: float = 30.0,
    before_frame: Optional[int] = None,
) -> Optional[int]:
    """Detect foot plant using the stride drop-then-rise pattern in ankle Y.

    In screen coordinates: ankle Y drops during leg lift/stride (foot rises),
    then rises back to baseline when the foot plants (foot comes down).
    Foot plant = where ankle Y recovers to near-baseline after the stride dip.

    Args:
        lead_ankle_y: Raw Y-coordinate of lead ankle (screen coords: higher = lower).
        lead_ankle_vy: Y-velocity of lead ankle (unused, kept for API compat).
        fps: Video frame rate.
        before_frame: Only search before this frame (e.g., before MER).
    """
    if len(lead_ankle_y) < 15:
        return None

    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(lead_ankle_y, size=5)

    # Search window: up to 2s before the reference frame
    end = before_frame if before_frame is not None else len(smoothed)
    start = max(0, end - int(fps * 2.0))
    region = smoothed[start:end]

    if len(region) < 10:
        return None

    # Step 1: Find the baseline ankle Y (standing level) from the early part of the window
    early_portion = region[: max(5, len(region) // 4)]
    baseline = np.median(early_portion)

    # Step 2: Find the stride dip — minimum ankle Y (foot at highest point during stride)
    dip_idx = int(np.argmin(region))
    dip_value = region[dip_idx]
    dip_depth = baseline - dip_value

    # If the dip is too shallow (< 5% of baseline), no real stride detected
    if dip_depth < baseline * 0.05:
        return None

    # Step 3: From the dip, look forward for where ankle Y recovers to near-baseline
    # Foot plant = first frame after the dip where ankle Y rises to within 15% of baseline
    recovery_threshold = baseline - dip_depth * 0.15
    for i in range(dip_idx, len(region)):
        if region[i] >= recovery_threshold:
            return int(i + start)

    # Fallback: frame closest to MER where ankle is near max Y
    return int(np.argmax(region[dip_idx:]) + dip_idx + start)


def detect_max_external_rotation(
    shoulder_er_angle: np.ndarray,
    after_frame: Optional[int] = None,
) -> Optional[int]:
    """Detect max external rotation (layback) as peak shoulder ER angle.

    Args:
        shoulder_er_angle: Shoulder external rotation angle over time (degrees).
        after_frame: Only search after this frame (e.g., after foot plant).
    """
    if len(shoulder_er_angle) < 10:
        return None

    search = shoulder_er_angle
    offset = 0
    if after_frame is not None:
        search = shoulder_er_angle[after_frame:]
        offset = after_frame

    return int(np.argmax(search) + offset)


def detect_ball_release(
    wrist_velo: np.ndarray,
    after_frame: Optional[int] = None,
    window_frames: int = 12,
) -> Optional[int]:
    """Approximate ball release as peak wrist/hand velocity near MER.

    Args:
        wrist_velo: Wrist velocity magnitude over time.
        after_frame: Only search after this frame (e.g., after MER).
        window_frames: Max frames after after_frame to search (default ~0.4s at 30fps).
    """
    if len(wrist_velo) < 10:
        return None

    search = wrist_velo
    offset = 0
    if after_frame is not None:
        end = min(len(wrist_velo), after_frame + window_frames)
        search = wrist_velo[after_frame:end]
        offset = after_frame

    if len(search) == 0:
        return None

    return int(np.argmax(search) + offset)


def find_delivery_anchor(
    shoulder_er: np.ndarray,
    wrist_speed: np.ndarray,
    fps: float = 30.0,
) -> Optional[int]:
    """Find the MER frame using ER trough before peak wrist speed.

    From a 2D front-quarter camera angle, the approximate_shoulder_er_2d
    signal is inverted relative to true 3D shoulder ER: as the arm lays
    back further, the forearm aligns with the trunk and the measured angle
    DECREASES. True MER therefore appears as the ER MINIMUM just before
    the wrist speed explosion (arm acceleration phase).

    Strategy: find the global wrist speed peak (reliable anchor), then
    find the ER trough in the 0.5s window before it.

    Args:
        shoulder_er: Shoulder ER angle over time (degrees, 2D approximation).
        wrist_speed: Wrist speed over time (pixels/second).
        fps: Video frame rate.

    Returns:
        Frame index of MER, or None if no delivery detected.
    """
    if len(shoulder_er) < 20:
        return None

    from scipy.ndimage import uniform_filter1d

    smoothed_er = uniform_filter1d(shoulder_er, size=5)
    smoothed_ws = uniform_filter1d(wrist_speed, size=5)

    # Find the peak wrist speed — this is the most reliable delivery marker
    peak_ws_frame = int(np.argmax(smoothed_ws))
    if smoothed_ws[peak_ws_frame] < 100:
        return None

    # Search for ER trough in 0.5s window before peak wrist speed
    window_start = max(0, peak_ws_frame - int(fps * 0.5))
    window_end = peak_ws_frame

    region = smoothed_er[window_start:window_end]
    if len(region) == 0:
        return None

    return int(np.argmin(region) + window_start)


def approximate_shoulder_er_2d(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    hip_center: np.ndarray,
) -> float:
    """Approximate shoulder external rotation from 2D side-view keypoints.

    Measures the angle of the forearm (elbow->wrist vector) relative to the
    trunk line (hip_center->shoulder vector). From a side view, high ER
    (layback) appears as the forearm angled behind and above the shoulder.

    Args:
        shoulder: Throwing shoulder (x, y), shape (2,).
        elbow: Throwing elbow (x, y), shape (2,).
        wrist: Throwing wrist (x, y), shape (2,).
        hip_center: Midpoint of hips (x, y), shape (2,).

    Returns:
        Approximate ER angle in degrees (0-180).
    """
    forearm = wrist - elbow
    trunk = shoulder - hip_center

    dot = np.dot(forearm, trunk)
    mag_forearm = np.linalg.norm(forearm)
    mag_trunk = np.linalg.norm(trunk)

    cos_angle = dot / (mag_forearm * mag_trunk + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def detect_events(
    keypoints_df: pd.DataFrame,
    fps: float = 30.0,
    pitcher_throws: str = "R",
) -> DeliveryEvents:
    """Run full event detection pipeline on a keypoint time series DataFrame.

    Expected columns depend on the pose estimation backend, but should include
    at minimum: lead_knee_y, lead_ankle_y, shoulder_er_angle, wrist_speed.

    This is a stub that will be fleshed out once the pose estimation module
    provides standardized keypoint DataFrames.

    Args:
        keypoints_df: DataFrame with per-frame keypoint data.
        fps: Video frame rate.
        pitcher_throws: "R" or "L" to determine lead/drive sides.

    Returns:
        DeliveryEvents with detected frame indices.
    """
    events = DeliveryEvents(fps=fps)

    # Map lead/drive side based on handedness
    # Right-handed pitcher: lead leg = left, drive leg = right
    lead_side = "left" if pitcher_throws == "R" else "right"

    # These column name patterns will be standardized by the pose module
    lead_knee_col = f"{lead_side}_knee_y"
    lead_ankle_col = f"{lead_side}_ankle_y"

    if lead_knee_col in keypoints_df.columns:
        events.leg_lift_apex = detect_leg_lift(keypoints_df[lead_knee_col].values)

    if lead_ankle_col in keypoints_df.columns:
        events.foot_plant = detect_foot_plant_from_keypoints(
            keypoints_df[lead_ankle_col].values, fps=fps
        )

    if "shoulder_er_angle" in keypoints_df.columns:
        events.max_external_rotation = detect_max_external_rotation(
            keypoints_df["shoulder_er_angle"].values,
            after_frame=events.foot_plant,
        )

    if "wrist_speed" in keypoints_df.columns:
        events.ball_release = detect_ball_release(
            keypoints_df["wrist_speed"].values,
            after_frame=events.max_external_rotation,
        )

    return events
