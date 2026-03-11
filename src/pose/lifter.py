"""3D pose lifting using MotionBERT-Lite.

Converts 2D COCO keypoints from YOLOv8-pose into 3D positions using a
pre-trained temporal transformer. The 3D keypoints are returned as a
separate dict (NOT stored in PoseSequence) because they live in
camera-normalized coordinates, not pixel coordinates.

Usage:
    from src.pose.lifter import is_3d_available, lift_to_3d
    if is_3d_available():
        kpts_3d = lift_to_3d(pose_sequence)
        # kpts_3d: dict[str, np.ndarray] mapping joint names to (T, 3) arrays
"""

import copy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from vendor.motionbert.DSTformer import DSTformer
from vendor.motionbert.utils_data import crop_scale, flip_data

from src.pose.estimator import (
    PITCHING_JOINTS,
    YOLO_KEYPOINTS,
    PoseSequence,
)


CHECKPOINT_PATH = Path("data/models/motionbert_lite.bin")

# H36M joint indices (17 joints)
H36M_JOINTS = {
    "root": 0, "right_hip": 1, "right_knee": 2, "right_ankle": 3,
    "left_hip": 4, "left_knee": 5, "left_ankle": 6,
    "spine": 7, "neck": 8, "nose": 9, "head": 10,
    "left_shoulder": 11, "left_elbow": 12, "left_wrist": 13,
    "right_shoulder": 14, "right_elbow": 15, "right_wrist": 16,
}

# Reverse mapping for h36m_to_pitching_joints
_H36M_PITCHING_MAP = {
    "left_shoulder": 11, "right_shoulder": 14,
    "left_elbow": 12, "right_elbow": 15,
    "left_wrist": 13, "right_wrist": 16,
    "left_hip": 4, "right_hip": 1,
    "left_knee": 5, "right_knee": 2,
    "left_ankle": 6, "right_ankle": 3,
    "spine": 7, "neck": 8, "root": 0,
}


def is_3d_available(checkpoint_path: Path = CHECKPOINT_PATH) -> bool:
    """Check if MotionBERT checkpoint is available."""
    return checkpoint_path.exists()


def coco_to_h36m(kpts_coco: np.ndarray, confs: np.ndarray) -> np.ndarray:
    """Map COCO 17 keypoints to H36M 17 keypoints with confidence.

    Args:
        kpts_coco: (T, 17, 2) array of (x, y) positions in COCO order.
        confs: (T, 17) confidence scores.

    Returns:
        (T, 17, 3) array in H36M order where channels are (x, y, confidence).
    """
    T = kpts_coco.shape[0]
    y = np.zeros((T, 17, 3))

    # Direct mappings: COCO index → H36M index
    direct = [
        (12, 1),   # r_hip
        (14, 2),   # r_knee
        (16, 3),   # r_ankle
        (11, 4),   # l_hip
        (13, 5),   # l_knee
        (15, 6),   # l_ankle
        (0, 9),    # nose
        (5, 11),   # l_shoulder
        (7, 12),   # l_elbow
        (9, 13),   # l_wrist
        (6, 14),   # r_shoulder
        (8, 15),   # r_elbow
        (10, 16),  # r_wrist
    ]
    for coco_idx, h36m_idx in direct:
        y[:, h36m_idx, :2] = kpts_coco[:, coco_idx, :]
        y[:, h36m_idx, 2] = confs[:, coco_idx]

    # Synthesized joints (averages)
    # Root = avg(l_hip, r_hip)
    y[:, 0, :2] = (kpts_coco[:, 11, :] + kpts_coco[:, 12, :]) * 0.5
    y[:, 0, 2] = (confs[:, 11] + confs[:, 12]) * 0.5

    # Neck = avg(l_shoulder, r_shoulder)
    y[:, 8, :2] = (kpts_coco[:, 5, :] + kpts_coco[:, 6, :]) * 0.5
    y[:, 8, 2] = (confs[:, 5] + confs[:, 6]) * 0.5

    # Spine = avg(root, neck)
    y[:, 7, :2] = (y[:, 0, :2] + y[:, 8, :2]) * 0.5
    y[:, 7, 2] = (y[:, 0, 2] + y[:, 8, 2]) * 0.5

    # Head = avg(l_eye, r_eye)
    y[:, 10, :2] = (kpts_coco[:, 1, :] + kpts_coco[:, 2, :]) * 0.5
    y[:, 10, 2] = (confs[:, 1] + confs[:, 2]) * 0.5

    return y


def h36m_to_pitching_joints(h36m_frame: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a single H36M 17-joint frame to pitching joint dict.

    Args:
        h36m_frame: (17, 3) array of 3D positions.

    Returns:
        Dict mapping joint names to (3,) position arrays.
        Includes 12 standard pitching joints plus spine, neck, root.
    """
    return {name: h36m_frame[idx].copy() for name, idx in _H36M_PITCHING_MAP.items()}


def load_motionbert(
    checkpoint_path: Path = CHECKPOINT_PATH, device: str = "cpu"
) -> nn.Module:
    """Load MotionBERT-Lite model from checkpoint."""
    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=256,
        dim_rep=512,
        depth=5,
        num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        maxlen=243,
        num_joints=17,
    )

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint["model_pos"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def _pose_sequence_to_coco_arrays(
    pose_seq: PoseSequence,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract COCO-ordered keypoints and confidence from PoseSequence.

    Returns:
        kpts: (T, 17, 2) array in COCO order.
        confs: (T, 17) confidence scores.
    """
    T = len(pose_seq.frames)
    coco_names = sorted(YOLO_KEYPOINTS.keys(), key=lambda n: YOLO_KEYPOINTS[n])
    kpts = np.zeros((T, 17, 2))
    confs = np.zeros((T, 17))
    for t, frame in enumerate(pose_seq.frames):
        for name in coco_names:
            idx = YOLO_KEYPOINTS[name]
            if name in frame.keypoints:
                kpts[t, idx, :] = frame.keypoints[name][:2]
            if name in frame.confidence:
                confs[t, idx] = frame.confidence[name]
    return kpts, confs


def infer_3d(
    model: nn.Module, kpts_h36m: np.ndarray, device: str = "cpu", flip: bool = True
) -> np.ndarray:
    """Run MotionBERT inference to lift 2D→3D.

    Args:
        model: Loaded MotionBERT model.
        kpts_h36m: (T, 17, 3) H36M keypoints with confidence channel.
        device: torch device string.
        flip: Enable test-time flip augmentation.

    Returns:
        (T, 17, 3) array of 3D positions in camera-normalized coordinates.

    Raises:
        ValueError: If T > 243 (MotionBERT's max sequence length).
    """
    T = kpts_h36m.shape[0]
    if T > 243:
        raise ValueError(
            f"Sequence length {T} exceeds MotionBERT max of 243 frames. "
            f"Trim video or reduce fps before lifting."
        )

    # Normalize to [-1, 1]
    motion = crop_scale(kpts_h36m)

    # Pad to 243 if shorter
    if T < 243:
        padded = np.zeros((243, 17, 3))
        padded[:T] = motion
        # Replicate last frame for padding
        padded[T:] = motion[T - 1]
        motion = padded

    motion_tensor = torch.FloatTensor(motion).unsqueeze(0).to(device)  # (1, 243, 17, 3)

    with torch.no_grad():
        pred = model(motion_tensor)  # (1, 243, 17, 3)

        if flip:
            motion_flipped = flip_data(motion)
            flipped_tensor = torch.FloatTensor(motion_flipped).unsqueeze(0).to(device)
            pred_flip = model(flipped_tensor)
            pred_flip = torch.FloatTensor(flip_data(pred_flip.cpu().numpy())).to(device)
            pred = (pred + pred_flip) / 2

    result = pred[0].cpu().numpy()[:T]  # Trim padding
    return result


def lift_to_3d(
    pose_sequence: PoseSequence, model: nn.Module | None = None
) -> dict[str, np.ndarray]:
    """Lift 2D PoseSequence to 3D keypoints.

    Returns 3D keypoints as a separate dict (NOT modifying PoseSequence)
    because 3D positions are in camera-normalized coordinates, not pixels.

    Args:
        pose_sequence: 2D pose data from YOLOv8.
        model: Pre-loaded MotionBERT model. If None, loads from CHECKPOINT_PATH.

    Returns:
        Dict mapping joint names to (T, 3) arrays in camera-normalized coords.
        Empty dict if checkpoint is missing (falsy value for easy if-checking).
    """
    if model is None:
        if not is_3d_available():
            return {}
        model = load_motionbert()

    device = next(model.parameters()).device.type

    kpts_coco, confs = _pose_sequence_to_coco_arrays(pose_sequence)
    kpts_h36m = coco_to_h36m(kpts_coco, confs)
    pts_3d = infer_3d(model, kpts_h36m, device=device)

    # Convert to per-joint dict
    T = pts_3d.shape[0]
    result: dict[str, np.ndarray] = {}
    for name, idx in _H36M_PITCHING_MAP.items():
        result[name] = pts_3d[:, idx, :]

    return result
