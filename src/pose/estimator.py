"""Video-to-keypoints pipeline using YOLOv8-pose or MediaPipe.

Handles video ingestion, frame extraction, pose estimation, and
keypoint normalization into a standardized format for downstream
biomechanical analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# YOLOv8-pose keypoint indices (COCO format)
YOLO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# MediaPipe Pose landmark indices (subset relevant to pitching)
MEDIAPIPE_KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# Joints needed for pitching biomechanics
PITCHING_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


@dataclass
class VideoInfo:
    """Metadata about an input video."""
    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float


@dataclass
class PoseFrame:
    """Keypoint detections for a single frame."""
    frame_idx: int
    timestamp: float
    keypoints: dict[str, np.ndarray]    # joint_name -> (x, y)
    confidence: dict[str, float]         # joint_name -> confidence score
    bbox: Optional[np.ndarray] = None    # Person bounding box [x1, y1, x2, y2]


@dataclass
class PoseSequence:
    """Full sequence of pose detections across a video."""
    video_info: VideoInfo
    frames: list[PoseFrame] = field(default_factory=list)

    def get_joint_trajectory(self, joint: str) -> np.ndarray:
        """Get (N, 2) array of joint positions across all frames."""
        positions = []
        for frame in self.frames:
            if joint in frame.keypoints:
                positions.append(frame.keypoints[joint])
            else:
                positions.append(np.array([np.nan, np.nan]))
        return np.array(positions)

    def get_joint_speed(self, joint: str) -> np.ndarray:
        """Compute speed (pixels/frame) of a joint across frames."""
        traj = self.get_joint_trajectory(joint)
        diff = np.diff(traj, axis=0)
        speeds = np.linalg.norm(diff, axis=1)
        return np.concatenate([[0], speeds])  # Pad first frame

    def to_keypoints_dict(self) -> dict[str, np.ndarray]:
        """Convert to dict format expected by feature extraction.

        Returns:
            Dict mapping joint names to (N_frames, 2) arrays.
        """
        result = {}
        for joint in PITCHING_JOINTS:
            result[joint] = self.get_joint_trajectory(joint)
        return result

    def to_dataframe(self):
        """Convert to a pandas DataFrame for event detection."""
        import pandas as pd

        records = []
        for frame in self.frames:
            row = {"frame_idx": frame.frame_idx, "timestamp": frame.timestamp}
            for joint, pos in frame.keypoints.items():
                row[f"{joint}_x"] = pos[0]
                row[f"{joint}_y"] = pos[1]
            for joint, conf in frame.confidence.items():
                row[f"{joint}_conf"] = conf
            records.append(row)

        df = pd.DataFrame(records)

        # Add derived columns for event detection
        for side in ["left", "right"]:
            knee_y = f"{side}_knee_y"
            ankle_y = f"{side}_ankle_y"
            if knee_y in df.columns and ankle_y in df.columns:
                pass  # Already present

            # Compute wrist speed
            wrist_x = f"{side}_wrist_x"
            wrist_y = f"{side}_wrist_y"
            if wrist_x in df.columns and wrist_y in df.columns:
                dx = df[wrist_x].diff().fillna(0)
                dy = df[wrist_y].diff().fillna(0)
                df[f"{side}_wrist_speed"] = np.sqrt(dx**2 + dy**2)

        return df


def load_video(video_path: str | Path) -> VideoInfo:
    """Load video metadata without reading all frames."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        duration_sec=0,
    )
    info.duration_sec = info.total_frames / info.fps if info.fps > 0 else 0

    cap.release()
    return info


def _select_person_by_roi(
    boxes: np.ndarray,
    roi: tuple[int, int, int, int],
) -> int:
    """Select the detected person whose bbox center is closest to the ROI center.

    Args:
        boxes: (N, 4) array of bounding boxes [x1, y1, x2, y2].
        roi: Region of interest as (x1, y1, x2, y2) in pixels.

    Returns:
        Index of the best-matching person detection.
    """
    roi_cx = (roi[0] + roi[2]) / 2
    roi_cy = (roi[1] + roi[3]) / 2

    best_idx = 0
    best_dist = float("inf")
    for i, box in enumerate(boxes):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def extract_poses_yolo(
    video_path: str | Path,
    model_size: str = "n",
    confidence: float = 0.5,
    target_fps: Optional[float] = None,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> PoseSequence:
    """Run YOLOv8-pose on a video and return standardized keypoint sequence.

    Args:
        video_path: Path to input video file.
        model_size: YOLOv8 model variant (n=nano, s=small, m=medium, l=large, x=xlarge).
        confidence: Minimum detection confidence.
        target_fps: Resample to this FPS. None = use original.
        roi: Region of interest (x1, y1, x2, y2) in pixels. When set,
             selects the person whose bbox center is closest to the ROI
             center instead of the highest-confidence detection.

    Returns:
        PoseSequence with per-frame keypoint detections.
    """
    from ultralytics import YOLO

    video_info = load_video(video_path)
    model = YOLO(f"yolov8{model_size}-pose.pt")

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0

    # Frame skip for target FPS
    skip = 1
    if target_fps and target_fps < video_info.fps:
        skip = max(1, int(video_info.fps / target_fps))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        results = model.predict(frame, conf=confidence, verbose=False)

        if len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints
            if len(kpts.data) > 0:
                # Select which person to track
                person_idx = 0
                if roi is not None and results[0].boxes is not None:
                    all_boxes = results[0].boxes.xyxy.cpu().numpy()
                    if len(all_boxes) > 1:
                        person_idx = _select_person_by_roi(all_boxes, roi)

                person_kpts = kpts.data[person_idx].cpu().numpy()  # (17, 3)

                keypoints = {}
                confidences = {}
                for name, idx in YOLO_KEYPOINTS.items():
                    if name in PITCHING_JOINTS or name == "nose":
                        keypoints[name] = person_kpts[idx, :2]
                        confidences[name] = float(person_kpts[idx, 2])

                # Get bounding box for the selected person
                bbox = None
                if results[0].boxes is not None and len(results[0].boxes) > person_idx:
                    bbox = results[0].boxes[person_idx].xyxy[0].cpu().numpy()

                frames.append(PoseFrame(
                    frame_idx=frame_idx,
                    timestamp=frame_idx / video_info.fps,
                    keypoints=keypoints,
                    confidence=confidences,
                    bbox=bbox,
                ))

        frame_idx += 1

    cap.release()
    return PoseSequence(video_info=video_info, frames=frames)


def extract_poses_mediapipe(
    video_path: str | Path,
    target_fps: Optional[float] = None,
) -> PoseSequence:
    """Run MediaPipe Pose on a video and return standardized keypoint sequence.

    MediaPipe provides 33 landmarks with full-body coverage.
    Lighter weight than YOLOv8 but may be less robust for fast motion.

    Args:
        video_path: Path to input video file.
        target_fps: Resample to this FPS. None = use original.

    Returns:
        PoseSequence with per-frame keypoint detections.
    """
    import mediapipe as mp

    video_info = load_video(video_path)
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0

    skip = 1
    if target_fps and target_fps < video_info.fps:
        skip = max(1, int(video_info.fps / target_fps))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = {}
            confidences = {}

            for name, idx in MEDIAPIPE_KEYPOINTS.items():
                if name in PITCHING_JOINTS or name == "nose":
                    lm = landmarks[idx]
                    keypoints[name] = np.array([
                        lm.x * video_info.width,
                        lm.y * video_info.height,
                    ])
                    confidences[name] = lm.visibility

            frames.append(PoseFrame(
                frame_idx=frame_idx,
                timestamp=frame_idx / video_info.fps,
                keypoints=keypoints,
                confidence=confidences,
            ))

        frame_idx += 1

    cap.release()
    mp_pose.close()
    return PoseSequence(video_info=video_info, frames=frames)


def extract_poses(
    video_path: str | Path,
    backend: str = "yolov8",
    **kwargs,
) -> PoseSequence:
    """Unified entry point for pose estimation.

    Args:
        video_path: Path to input video.
        backend: "yolov8" or "mediapipe".
        **kwargs: Backend-specific arguments.

    Returns:
        PoseSequence with standardized keypoint data.
    """
    if backend == "yolov8":
        return extract_poses_yolo(video_path, **kwargs)
    elif backend == "mediapipe":
        return extract_poses_mediapipe(video_path, **kwargs)
    else:
        raise ValueError(f"Unknown pose backend: {backend}. Use 'yolov8' or 'mediapipe'.")
