"""Tests for 3D pose lifting (COCO→H36M mapping, model loading, inference).

TDD: Written BEFORE implementation. Tests define the expected API.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

# These imports will fail until lifter.py is created — that's TDD.
from src.pose.lifter import (
    CHECKPOINT_PATH,
    H36M_JOINTS,
    coco_to_h36m,
    h36m_to_pitching_joints,
    infer_3d,
    is_3d_available,
    lift_to_3d,
)
from src.pose.estimator import (
    PITCHING_JOINTS,
    YOLO_KEYPOINTS,
    PoseFrame,
    PoseSequence,
    VideoInfo,
)


# --- Fixtures ---

@pytest.fixture
def coco_keypoints() -> np.ndarray:
    """Create synthetic COCO keypoints (T=10, 17, 2) with known positions."""
    T = 10
    kpts = np.zeros((T, 17, 2))
    # Place joints at recognizable positions
    kpts[:, 0, :] = [100, 50]   # nose
    kpts[:, 1, :] = [95, 45]    # left_eye
    kpts[:, 2, :] = [105, 45]   # right_eye
    kpts[:, 3, :] = [90, 50]    # left_ear
    kpts[:, 4, :] = [110, 50]   # right_ear
    kpts[:, 5, :] = [80, 100]   # left_shoulder
    kpts[:, 6, :] = [120, 100]  # right_shoulder
    kpts[:, 7, :] = [70, 150]   # left_elbow
    kpts[:, 8, :] = [130, 150]  # right_elbow
    kpts[:, 9, :] = [60, 200]   # left_wrist
    kpts[:, 10, :] = [140, 200]  # right_wrist
    kpts[:, 11, :] = [85, 200]  # left_hip
    kpts[:, 12, :] = [115, 200]  # right_hip
    kpts[:, 13, :] = [80, 300]  # left_knee
    kpts[:, 14, :] = [120, 300]  # right_knee
    kpts[:, 15, :] = [80, 400]  # left_ankle
    kpts[:, 16, :] = [120, 400]  # right_ankle
    return kpts


@pytest.fixture
def coco_confidence() -> np.ndarray:
    """Confidence scores for COCO keypoints (T=10, 17)."""
    return np.ones((10, 17)) * 0.9


@pytest.fixture
def sample_pose_sequence() -> PoseSequence:
    """Create a minimal PoseSequence for testing lift_to_3d."""
    video_info = VideoInfo(
        path=Path("test.mp4"), width=640, height=480,
        fps=30.0, total_frames=10, duration_sec=0.333,
    )
    frames = []
    for i in range(10):
        kpts = {}
        confs = {}
        for name, idx in YOLO_KEYPOINTS.items():
            kpts[name] = np.array([100.0 + idx * 10, 200.0 + idx * 5])
            confs[name] = 0.9
        frames.append(PoseFrame(frame_idx=i, timestamp=i / 30.0, keypoints=kpts, confidence=confs))
    return PoseSequence(video_info=video_info, frames=frames)


# --- COCO → H36M mapping tests ---

class TestCocoToH36m:
    def test_output_shape(self, coco_keypoints, coco_confidence):
        """Output should be (T, 17, 3) where 3 = (x, y, confidence)."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        assert result.shape == (10, 17, 3)

    def test_root_is_hip_average(self, coco_keypoints, coco_confidence):
        """H36M root (index 0) = average of COCO left_hip and right_hip."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        expected_xy = (coco_keypoints[0, 11, :] + coco_keypoints[0, 12, :]) / 2
        np.testing.assert_allclose(result[0, 0, :2], expected_xy)

    def test_neck_is_shoulder_average(self, coco_keypoints, coco_confidence):
        """H36M neck (index 8) = average of COCO left_shoulder and right_shoulder."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        expected_xy = (coco_keypoints[0, 5, :] + coco_keypoints[0, 6, :]) / 2
        np.testing.assert_allclose(result[0, 8, :2], expected_xy)

    def test_spine_is_root_neck_midpoint(self, coco_keypoints, coco_confidence):
        """H36M spine (index 7) = average of root and neck."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        root = result[0, 0, :2]
        neck = result[0, 8, :2]
        np.testing.assert_allclose(result[0, 7, :2], (root + neck) / 2)

    def test_right_shoulder_mapped(self, coco_keypoints, coco_confidence):
        """H36M right_shoulder (14) = COCO right_shoulder (6)."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        np.testing.assert_allclose(result[0, 14, :2], coco_keypoints[0, 6, :])

    def test_right_elbow_mapped(self, coco_keypoints, coco_confidence):
        """H36M right_elbow (15) = COCO right_elbow (8)."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        np.testing.assert_allclose(result[0, 15, :2], coco_keypoints[0, 8, :])

    def test_right_wrist_mapped(self, coco_keypoints, coco_confidence):
        """H36M right_wrist (16) = COCO right_wrist (10)."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        np.testing.assert_allclose(result[0, 16, :2], coco_keypoints[0, 10, :])

    def test_confidence_channel_present(self, coco_keypoints, coco_confidence):
        """Third channel should contain confidence scores, not zeros."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        # Direct-mapped joints should have original confidence
        assert result[0, 14, 2] > 0  # right_shoulder conf
        # Synthesized joints should have averaged confidence
        assert result[0, 0, 2] > 0   # root conf (average of hip confs)

    def test_all_17_joints_nonzero(self, coco_keypoints, coco_confidence):
        """All 17 H36M joints should have non-zero positions."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        for j in range(17):
            assert not np.allclose(result[0, j, :2], 0), f"H36M joint {j} is zero"

    def test_temporal_consistency(self, coco_keypoints, coco_confidence):
        """Same input across frames → same output across frames."""
        result = coco_to_h36m(coco_keypoints, coco_confidence)
        for t in range(1, 10):
            np.testing.assert_allclose(result[t], result[0])


# --- H36M → Pitching Joints mapping ---

class TestH36mToPitchingJoints:
    def test_contains_12_standard_joints(self):
        """Result contains all 12 standard pitching joints."""
        h36m_frame = np.random.randn(17, 3)
        result = h36m_to_pitching_joints(h36m_frame)
        for joint in PITCHING_JOINTS:
            assert joint in result, f"Missing standard joint: {joint}"

    def test_extra_joints_present(self):
        """Result contains spine, neck, root for 3D body frame."""
        h36m_frame = np.random.randn(17, 3)
        result = h36m_to_pitching_joints(h36m_frame)
        for joint in ["spine", "neck", "root"]:
            assert joint in result, f"Missing extra joint: {joint}"

    def test_values_are_3d(self):
        """All returned arrays should be shape (3,)."""
        h36m_frame = np.random.randn(17, 3)
        result = h36m_to_pitching_joints(h36m_frame)
        for name, arr in result.items():
            assert arr.shape == (3,), f"Joint {name} has shape {arr.shape}, expected (3,)"

    def test_roundtrip_positions(self, coco_keypoints, coco_confidence):
        """COCO → H36M → pitching joints preserves positions for shared joints."""
        h36m = coco_to_h36m(coco_keypoints, coco_confidence)
        result = h36m_to_pitching_joints(h36m[0])
        # right_shoulder in H36M index 14, originally COCO index 6
        np.testing.assert_allclose(result["right_shoulder"][:2], coco_keypoints[0, 6, :])


# --- 3D availability ---

class TestIs3dAvailable:
    def test_missing_checkpoint_returns_false(self):
        """is_3d_available() returns False when checkpoint file doesn't exist."""
        with patch.object(Path, "exists", return_value=False):
            assert is_3d_available() is False


# --- lift_to_3d ---

class TestLiftTo3d:
    def test_no_checkpoint_returns_empty_dict(self, sample_pose_sequence):
        """lift_to_3d returns empty dict when checkpoint is missing."""
        with patch("src.pose.lifter.is_3d_available", return_value=False):
            result = lift_to_3d(sample_pose_sequence)
            assert result == {}
            assert not result  # Empty dict is falsy


class TestInfer3d:
    def test_rejects_sequence_over_243_frames(self):
        """infer_3d raises ValueError when T > 243."""
        import torch.nn as nn

        dummy_model = nn.Linear(1, 1)  # Won't be called
        kpts = np.random.randn(244, 17, 3)
        with pytest.raises(ValueError, match="exceeds MotionBERT max of 243"):
            infer_3d(dummy_model, kpts)


# --- PoseFrame 3D support ---

class TestPoseFrame3d:
    def test_is_3d_with_2d_keypoints(self):
        """is_3d returns False for standard 2D keypoints."""
        frame = PoseFrame(
            frame_idx=0, timestamp=0.0,
            keypoints={"right_shoulder": np.array([100.0, 200.0])},
            confidence={"right_shoulder": 0.9},
        )
        assert frame.is_3d is False

    def test_is_3d_with_3d_keypoints(self):
        """is_3d returns True for 3D keypoints."""
        frame = PoseFrame(
            frame_idx=0, timestamp=0.0,
            keypoints={"right_shoulder": np.array([100.0, 200.0, 50.0])},
            confidence={"right_shoulder": 0.9},
        )
        assert frame.is_3d is True

    def test_is_3d_empty_keypoints(self):
        """is_3d returns False for empty keypoints dict."""
        frame = PoseFrame(
            frame_idx=0, timestamp=0.0,
            keypoints={},
            confidence={},
        )
        assert frame.is_3d is False


class TestPoseSequence3d:
    def test_get_joint_trajectory_3d_nan_fill(self):
        """NaN fill should match 3D shape (3,) when frames contain 3D data."""
        video_info = VideoInfo(
            path=Path("test.mp4"), width=640, height=480,
            fps=30.0, total_frames=3, duration_sec=0.1,
        )
        frames = [
            PoseFrame(0, 0.0, {"shoulder": np.array([1.0, 2.0, 3.0])}, {"shoulder": 0.9}),
            PoseFrame(1, 0.033, {}, {}),  # Missing joint → should NaN-fill with (3,)
            PoseFrame(2, 0.066, {"shoulder": np.array([4.0, 5.0, 6.0])}, {"shoulder": 0.9}),
        ]
        seq = PoseSequence(video_info=video_info, frames=frames)
        traj = seq.get_joint_trajectory("shoulder")
        assert traj.shape == (3, 3)  # 3 frames, 3 dimensions
        assert np.isnan(traj[1]).all()
        assert traj[1].shape == (3,)

    def test_to_dataframe_includes_z(self):
        """to_dataframe includes {joint}_z columns for 3D data."""
        video_info = VideoInfo(
            path=Path("test.mp4"), width=640, height=480,
            fps=30.0, total_frames=1, duration_sec=0.033,
        )
        frames = [
            PoseFrame(0, 0.0, {"right_shoulder": np.array([1.0, 2.0, 3.0])}, {"right_shoulder": 0.9}),
        ]
        seq = PoseSequence(video_info=video_info, frames=frames)
        df = seq.to_dataframe()
        assert "right_shoulder_z" in df.columns
        assert df["right_shoulder_z"].iloc[0] == 3.0


# --- Regression tests ---

class TestExtractMetrics2dUnchanged:
    """Verify 2D metric extraction is identical to pre-3D behavior."""

    def test_elbow_flexion_2d(self):
        """2D extract_metrics produces same elbow flexion as before."""
        from src.biomechanics.features import extract_metrics
        from src.biomechanics.events import DeliveryEvents

        # Build simple 2D keypoints
        T = 10
        keypoints = {}
        for joint in PITCHING_JOINTS:
            keypoints[joint] = np.zeros((T, 2))

        # Set up a right-angle elbow at frame 5 (foot plant)
        keypoints["right_shoulder"][5] = [100, 100]
        keypoints["right_elbow"][5] = [100, 200]
        keypoints["right_wrist"][5] = [200, 200]

        events = DeliveryEvents(foot_plant=5, max_external_rotation=5, ball_release=7)
        metrics = extract_metrics(keypoints, events, pitcher_throws="R", use_3d=False)

        assert metrics.elbow_flexion_fp is not None
        assert abs(metrics.elbow_flexion_fp - 90.0) < 1.0

    def test_use_3d_false_no_3d_metrics(self):
        """With use_3d=False, 3D-specific metrics remain None."""
        from src.biomechanics.features import extract_metrics
        from src.biomechanics.events import DeliveryEvents

        T = 10
        keypoints = {}
        for joint in PITCHING_JOINTS:
            keypoints[joint] = np.random.randn(T, 2) * 100 + 300

        events = DeliveryEvents(foot_plant=5, max_external_rotation=6, ball_release=7)
        metrics = extract_metrics(keypoints, events, pitcher_throws="R", use_3d=False)

        # 3D-specific metrics should NOT be computed
        assert metrics.hip_shoulder_separation_fp is None
        assert metrics.shoulder_abduction_fp is None
        assert metrics.shoulder_horizontal_abduction_fp is None

    def test_use_3d_true_computes_3d_metrics(self):
        """With use_3d=True and 3D keypoints, 3D metrics are computed."""
        from src.biomechanics.features import extract_metrics
        from src.biomechanics.events import DeliveryEvents

        T = 10
        keypoints = {}
        joints_3d = PITCHING_JOINTS + ["spine", "neck", "root"]
        for joint in joints_3d:
            keypoints[joint] = np.random.randn(T, 3) * 0.3

        # Place hips and shoulders at known positions for frame 5
        keypoints["left_hip"][5] = [-0.2, 0.1, 0.0]
        keypoints["right_hip"][5] = [0.2, 0.1, 0.0]
        keypoints["left_shoulder"][5] = [-0.2, -0.5, -0.3]
        keypoints["right_shoulder"][5] = [0.2, -0.5, 0.3]
        keypoints["right_elbow"][5] = [0.5, -0.5, 0.5]

        events = DeliveryEvents(foot_plant=5, max_external_rotation=6, ball_release=7)
        metrics = extract_metrics(keypoints, events, pitcher_throws="R", use_3d=True)

        assert metrics.hip_shoulder_separation_fp is not None
        assert metrics.torso_lateral_tilt_fp is not None
        assert metrics.shoulder_abduction_fp is not None
