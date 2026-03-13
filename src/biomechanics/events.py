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
    lead_ankle_x: Optional[np.ndarray] = None,
    fps: float = 30.0,
    before_frame: Optional[int] = None,
) -> Optional[int]:
    """Detect foot plant as the frame where the stride foot stops moving forward.

    During the stride, the lead ankle moves forward (X increases for RHP)
    and downward. At foot plant, forward motion stops abruptly. This is
    more reliable than Y-recovery-to-baseline because it doesn't require
    knowing the ground plane position.

    Falls back to Y-recovery if ankle X is not provided.

    Args:
        lead_ankle_y: Raw Y-coordinate of lead ankle (screen coords).
        lead_ankle_x: Raw X-coordinate of lead ankle. If provided, uses
            X-velocity stabilization (preferred). If None, uses Y-recovery.
        fps: Video frame rate.
        before_frame: Only search before this frame (e.g., before MER).
    """
    if len(lead_ankle_y) < 15:
        return None

    from scipy.ndimage import uniform_filter1d

    # Primary: ankle X stabilization (foot stops moving forward)
    if lead_ankle_x is not None:
        result = _detect_fp_ankle_x_stop(lead_ankle_x, fps, before_frame)
        if result is not None:
            return result

    # Fallback: ankle Y recovery to baseline
    return _detect_fp_y_recovery(lead_ankle_y, fps, before_frame)


def _detect_fp_ankle_x_stop(
    lead_ankle_x: np.ndarray,
    fps: float,
    before_frame: Optional[int],
) -> Optional[int]:
    """Detect foot plant by finding where ankle X velocity drops to near zero."""
    from scipy.ndimage import uniform_filter1d

    smoothed_x = uniform_filter1d(lead_ankle_x, size=5)
    vx = np.gradient(smoothed_x)
    smoothed_vx = uniform_filter1d(vx, size=3)

    end = before_frame if before_frame is not None else len(smoothed_vx)
    start = max(0, end - int(fps * 1.5))
    region = smoothed_vx[start:end]

    if len(region) < 10:
        return None

    # Find peak forward velocity during stride
    peak_idx = int(np.argmax(region))
    peak_vx = region[peak_idx]

    if peak_vx < 3:
        return None

    # Foot plant = where forward velocity drops below 30% of peak
    threshold = peak_vx * 0.30
    for i in range(peak_idx, len(region)):
        if region[i] < threshold:
            return int(i + start)

    return None


def _detect_fp_y_recovery(
    lead_ankle_y: np.ndarray,
    fps: float,
    before_frame: Optional[int],
) -> Optional[int]:
    """Fallback: detect foot plant via ankle Y recovery to baseline."""
    from scipy.ndimage import uniform_filter1d

    smoothed = uniform_filter1d(lead_ankle_y, size=5)

    end = before_frame if before_frame is not None else len(smoothed)
    start = max(0, end - int(fps * 2.0))
    region = smoothed[start:end]

    if len(region) < 10:
        return None

    early_portion = region[: max(5, len(region) // 4)]
    baseline = np.median(early_portion)
    dip_idx = int(np.argmin(region))
    dip_value = region[dip_idx]
    dip_depth = baseline - dip_value

    if dip_depth < baseline * 0.05:
        return None

    recovery_threshold = baseline - dip_depth * 0.15
    for i in range(dip_idx, len(region)):
        if region[i] >= recovery_threshold:
            return int(i + start)

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


def detect_events_from_pose_sequence(
    pose_sequence,
    fps: float = 30.0,
    pitcher_throws: str = "R",
) -> DeliveryEvents:
    """Run full event detection pipeline on a PoseSequence.

    Uses the anchor-based approach: find MER first (most reliable marker),
    then detect other events relative to it.

    Args:
        pose_sequence: PoseSequence from the pose estimation module.
        fps: Video frame rate.
        pitcher_throws: "R" or "L" to determine lead/drive sides.

    Returns:
        DeliveryEvents with detected frame indices.
    """
    from scipy.ndimage import uniform_filter1d

    lead_side = "left" if pitcher_throws == "R" else "right"
    throw_side = "right" if pitcher_throws == "R" else "left"

    # Build trajectory arrays from pose sequence
    lead_knee_traj = pose_sequence.get_joint_trajectory(f"{lead_side}_knee")
    lead_knee_y_inverted = -lead_knee_traj[:, 1]

    lead_ankle_traj = pose_sequence.get_joint_trajectory(f"{lead_side}_ankle")
    lead_ankle_x_raw = lead_ankle_traj[:, 0]
    lead_ankle_y_raw = lead_ankle_traj[:, 1]

    # Wrist speed: pixels/frame * fps = pixels/second, then smooth
    throw_wrist = f"{throw_side}_wrist"
    wrist_speed_raw = pose_sequence.get_joint_speed(throw_wrist) * fps
    kernel = np.ones(3) / 3
    wrist_speed = np.convolve(wrist_speed_raw, kernel, mode="same")

    # Approximate shoulder ER per frame
    shoulder_er_series = np.zeros(len(pose_sequence.frames))
    for i, pf in enumerate(pose_sequence.frames):
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

    # Anchor-based detection:
    # 1. Find MER (anchor) via ER trough before peak wrist speed
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


def detect_events(
    keypoints_df: pd.DataFrame,
    fps: float = 30.0,
    pitcher_throws: str = "R",
) -> DeliveryEvents:
    """Run full event detection pipeline on a keypoint time series DataFrame.

    Expected columns depend on the pose estimation backend, but should include
    at minimum: lead_knee_y, lead_ankle_y, shoulder_er_angle, wrist_speed.

    Args:
        keypoints_df: DataFrame with per-frame keypoint data.
        fps: Video frame rate.
        pitcher_throws: "R" or "L" to determine lead/drive sides.

    Returns:
        DeliveryEvents with detected frame indices.
    """
    events = DeliveryEvents(fps=fps)

    lead_side = "left" if pitcher_throws == "R" else "right"
    throw_side = "right" if pitcher_throws == "R" else "left"

    lead_knee_col = f"{lead_side}_knee_y"
    lead_ankle_y_col = f"{lead_side}_ankle_y"
    lead_ankle_x_col = f"{lead_side}_ankle_x"
    wrist_speed_col = f"{throw_side}_wrist_speed"

    # Build shoulder ER series if individual keypoint columns exist
    shoulder_er_series = None
    shoulder_col = f"{throw_side}_shoulder"
    elbow_col = f"{throw_side}_elbow"
    wrist_col = f"{throw_side}_wrist"
    has_keypoints = all(
        f"{col}_{axis}" in keypoints_df.columns
        for col in [shoulder_col, elbow_col, wrist_col, "left_hip", "right_hip"]
        for axis in ["x", "y"]
    )
    if has_keypoints:
        shoulder_er_series = np.zeros(len(keypoints_df))
        for i in range(len(keypoints_df)):
            row = keypoints_df.iloc[i]
            shoulder = np.array([row[f"{shoulder_col}_x"], row[f"{shoulder_col}_y"]])
            elbow = np.array([row[f"{elbow_col}_x"], row[f"{elbow_col}_y"]])
            wrist = np.array([row[f"{wrist_col}_x"], row[f"{wrist_col}_y"]])
            hip_center = np.array([
                (row["left_hip_x"] + row["right_hip_x"]) / 2,
                (row["left_hip_y"] + row["right_hip_y"]) / 2,
            ])
            shoulder_er_series[i] = approximate_shoulder_er_2d(
                shoulder, elbow, wrist, hip_center,
            )
    elif "shoulder_er_angle" in keypoints_df.columns:
        shoulder_er_series = keypoints_df["shoulder_er_angle"].values

    # Build wrist speed series
    wrist_speed = None
    if wrist_speed_col in keypoints_df.columns:
        wrist_speed = keypoints_df[wrist_speed_col].values * fps
        kernel = np.ones(3) / 3
        wrist_speed = np.convolve(wrist_speed, kernel, mode="same")
    elif "wrist_speed" in keypoints_df.columns:
        wrist_speed = keypoints_df["wrist_speed"].values

    # Anchor-based detection when we have both ER and wrist speed
    if shoulder_er_series is not None and wrist_speed is not None:
        events.max_external_rotation = find_delivery_anchor(
            shoulder_er_series, wrist_speed, fps=fps,
        )

    if events.max_external_rotation is not None:
        if wrist_speed is not None:
            events.ball_release = detect_ball_release(
                wrist_speed, after_frame=events.max_external_rotation,
            )
        if lead_ankle_y_col in keypoints_df.columns:
            lead_ankle_x = (
                keypoints_df[lead_ankle_x_col].values
                if lead_ankle_x_col in keypoints_df.columns
                else None
            )
            events.foot_plant = detect_foot_plant_from_keypoints(
                keypoints_df[lead_ankle_y_col].values,
                lead_ankle_x=lead_ankle_x,
                fps=fps,
                before_frame=events.max_external_rotation,
            )
        if lead_knee_col in keypoints_df.columns:
            events.leg_lift_apex = detect_leg_lift(
                keypoints_df[lead_knee_col].values,
                before_frame=events.foot_plant,
            )
    else:
        # Fallback: independent detection
        if lead_knee_col in keypoints_df.columns:
            events.leg_lift_apex = detect_leg_lift(keypoints_df[lead_knee_col].values)
        if lead_ankle_y_col in keypoints_df.columns:
            lead_ankle_x = (
                keypoints_df[lead_ankle_x_col].values
                if lead_ankle_x_col in keypoints_df.columns
                else None
            )
            events.foot_plant = detect_foot_plant_from_keypoints(
                keypoints_df[lead_ankle_y_col].values,
                lead_ankle_x=lead_ankle_x,
                fps=fps,
            )
        if shoulder_er_series is not None:
            events.max_external_rotation = detect_max_external_rotation(
                shoulder_er_series, after_frame=events.foot_plant,
            )
        if wrist_speed is not None:
            events.ball_release = detect_ball_release(
                wrist_speed, after_frame=events.max_external_rotation,
            )

    return events
