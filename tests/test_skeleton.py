"""Tests for skeleton drawing utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz.skeleton import (
    SKELETON_CONNECTIONS,
    confidence_color,
    draw_skeleton,
    draw_angle_arc,
)


class TestConfidenceColor:
    """Tests for BGR color mapping by confidence value."""

    def test_high_confidence_returns_green(self):
        assert confidence_color(0.9) == (0, 200, 0)
        assert confidence_color(0.71) == (0, 200, 0)

    def test_medium_confidence_returns_yellow(self):
        assert confidence_color(0.5) == (0, 200, 200)
        assert confidence_color(0.7) == (0, 200, 200)
        assert confidence_color(0.4) == (0, 200, 200)

    def test_low_confidence_returns_red(self):
        assert confidence_color(0.1) == (0, 0, 200)
        assert confidence_color(0.39) == (0, 0, 200)

    def test_boundary_values(self):
        # Exactly 0.7 should be yellow (0.4-0.7 range)
        assert confidence_color(0.7) == (0, 200, 200)
        # Exactly 0.4 should be yellow
        assert confidence_color(0.4) == (0, 200, 200)
        # Just above 0.7 should be green
        assert confidence_color(0.701) == (0, 200, 0)
        # Just below 0.4 should be red
        assert confidence_color(0.399) == (0, 0, 200)

    def test_extreme_values(self):
        assert confidence_color(1.0) == (0, 200, 0)
        assert confidence_color(0.0) == (0, 0, 200)


class TestSkeletonConnections:
    """Tests for SKELETON_CONNECTIONS structure."""

    def test_connections_is_list_of_tuples(self):
        assert isinstance(SKELETON_CONNECTIONS, list)
        for conn in SKELETON_CONNECTIONS:
            assert isinstance(conn, tuple)
            assert len(conn) == 2

    def test_throwing_arm_connections_present(self):
        connections_set = set(SKELETON_CONNECTIONS)
        assert ("right_shoulder", "right_elbow") in connections_set
        assert ("right_elbow", "right_wrist") in connections_set

    def test_glove_arm_connections_present(self):
        connections_set = set(SKELETON_CONNECTIONS)
        assert ("left_shoulder", "left_elbow") in connections_set
        assert ("left_elbow", "left_wrist") in connections_set

    def test_trunk_connections_present(self):
        connections_set = set(SKELETON_CONNECTIONS)
        assert ("left_shoulder", "right_shoulder") in connections_set
        assert ("left_hip", "right_hip") in connections_set
        assert ("left_shoulder", "left_hip") in connections_set
        assert ("right_shoulder", "right_hip") in connections_set

    def test_leg_connections_present(self):
        connections_set = set(SKELETON_CONNECTIONS)
        assert ("left_hip", "left_knee") in connections_set
        assert ("left_knee", "left_ankle") in connections_set
        assert ("right_hip", "right_knee") in connections_set
        assert ("right_knee", "right_ankle") in connections_set

    def test_total_connection_count(self):
        # 2 throwing arm + 2 glove arm + 1 shoulder line + 1 hip line
        # + 2 trunk sides + 4 legs = 12
        assert len(SKELETON_CONNECTIONS) == 12


class TestDrawSkeleton:
    """Tests for draw_skeleton rendering."""

    @pytest.fixture
    def synthetic_frame(self):
        """Create a 480x640 black frame."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_keypoints(self):
        """Keypoints dict with pixel coordinates for a rough stick figure."""
        return {
            "right_shoulder": (350, 150),
            "right_elbow": (400, 200),
            "right_wrist": (420, 260),
            "left_shoulder": (250, 150),
            "left_elbow": (200, 200),
            "left_wrist": (180, 260),
            "left_hip": (270, 300),
            "right_hip": (330, 300),
            "left_knee": (260, 380),
            "left_ankle": (255, 450),
            "right_knee": (340, 380),
            "right_ankle": (345, 450),
        }

    @pytest.fixture
    def sample_confidence(self):
        """All joints with high confidence."""
        return {
            "right_shoulder": 0.95,
            "right_elbow": 0.90,
            "right_wrist": 0.85,
            "left_shoulder": 0.92,
            "left_elbow": 0.88,
            "left_wrist": 0.80,
            "left_hip": 0.93,
            "right_hip": 0.91,
            "left_knee": 0.87,
            "left_ankle": 0.75,
            "right_knee": 0.89,
            "right_ankle": 0.78,
        }

    def test_returns_modified_frame(self, synthetic_frame, sample_keypoints, sample_confidence):
        result = draw_skeleton(synthetic_frame, sample_keypoints, sample_confidence)
        # Should not be all black — something was drawn
        assert result.sum() > 0

    def test_does_not_mutate_input(self, synthetic_frame, sample_keypoints, sample_confidence):
        original = synthetic_frame.copy()
        draw_skeleton(synthetic_frame, sample_keypoints, sample_confidence)
        np.testing.assert_array_equal(synthetic_frame, original)

    def test_returns_same_shape(self, synthetic_frame, sample_keypoints, sample_confidence):
        result = draw_skeleton(synthetic_frame, sample_keypoints, sample_confidence)
        assert result.shape == synthetic_frame.shape
        assert result.dtype == synthetic_frame.dtype

    def test_handles_missing_joints(self, synthetic_frame, sample_confidence):
        """Partial keypoints should not raise."""
        partial_keypoints = {
            "right_shoulder": (350, 150),
            "right_elbow": (400, 200),
        }
        partial_conf = {
            "right_shoulder": 0.9,
            "right_elbow": 0.8,
        }
        result = draw_skeleton(synthetic_frame, partial_keypoints, partial_conf)
        assert result.shape == synthetic_frame.shape

    def test_handles_low_confidence_joints(self, synthetic_frame, sample_keypoints):
        """Joints below min_confidence should be skipped."""
        low_conf = {k: 0.05 for k in sample_keypoints}
        result = draw_skeleton(synthetic_frame, sample_keypoints, low_conf, min_confidence=0.1)
        # All joints below threshold — frame should be unchanged (still black)
        assert result.sum() == 0

    def test_handles_empty_keypoints(self, synthetic_frame):
        """Empty keypoints should return a clean copy."""
        result = draw_skeleton(synthetic_frame, {}, {})
        assert result.shape == synthetic_frame.shape
        assert result.sum() == 0

    def test_bbox_drawn_when_provided(self, synthetic_frame, sample_keypoints, sample_confidence):
        result_no_bbox = draw_skeleton(synthetic_frame, sample_keypoints, sample_confidence)
        result_with_bbox = draw_skeleton(
            synthetic_frame, sample_keypoints, sample_confidence,
            bbox=(100, 50, 400, 460),
        )
        # The bbox version should have more non-zero pixels
        assert result_with_bbox.sum() > result_no_bbox.sum()


class TestDrawAngleArc:
    """Tests for draw_angle_arc rendering."""

    @pytest.fixture
    def frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_returns_modified_frame(self, frame):
        vertex = (300, 250)
        point_a = (350, 200)
        point_b = (350, 300)
        result = draw_angle_arc(frame, vertex, point_a, point_b, 90.0, "90°")
        assert result.sum() > 0

    def test_does_not_mutate_input(self, frame):
        original = frame.copy()
        draw_angle_arc(frame, (300, 250), (350, 200), (350, 300), 90.0, "90°")
        np.testing.assert_array_equal(frame, original)

    def test_custom_color(self, frame):
        result = draw_angle_arc(
            frame, (300, 250), (350, 200), (350, 300),
            90.0, "90°", color=(0, 255, 0),
        )
        assert result.sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
