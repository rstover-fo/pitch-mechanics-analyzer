"""Skeleton drawing utilities for pose validation and visualization.

Draws 2D stick-figure overlays from keypoint detections onto video frames.
All coordinates are in pixel space; colors are BGR (OpenCV convention).
"""

import math
from typing import Optional

import cv2
import numpy as np

# Bone connections: (joint_a, joint_b) pairs defining the skeleton topology.
SKELETON_CONNECTIONS: list[tuple[str, str]] = [
    # Throwing arm (right)
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Glove arm (left)
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Shoulder line
    ("left_shoulder", "right_shoulder"),
    # Hip line
    ("left_hip", "right_hip"),
    # Trunk (both sides)
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def confidence_color(conf: float) -> tuple[int, int, int]:
    """Map a 0-1 confidence score to a BGR color.

    Args:
        conf: Joint detection confidence in [0, 1].

    Returns:
        BGR color tuple:
            >0.7  -> green  (0, 200, 0)
            0.4-0.7 -> yellow (0, 200, 200)
            <0.4  -> red    (0, 0, 200)
    """
    if conf > 0.7:
        return (0, 200, 0)
    if conf >= 0.4:
        return (0, 200, 200)
    return (0, 0, 200)


def draw_skeleton(
    frame: np.ndarray,
    keypoints: dict[str, tuple[int, int]],
    confidence: dict[str, float],
    bbox: Optional[tuple[int, int, int, int]] = None,
    min_confidence: float = 0.1,
) -> np.ndarray:
    """Draw a pose skeleton overlay on a video frame.

    Args:
        frame: BGR image array (H, W, 3).
        keypoints: Mapping of joint name -> (x, y) pixel coordinates.
        confidence: Mapping of joint name -> detection confidence [0, 1].
        bbox: Optional bounding box as (x1, y1, x2, y2) drawn in light gray.
        min_confidence: Joints below this threshold are skipped.

    Returns:
        A new frame with the skeleton drawn on top. The input is not mutated.
    """
    canvas = frame.copy()

    # Determine which joints pass the confidence threshold
    visible = {
        joint
        for joint in keypoints
        if joint in confidence and confidence[joint] >= min_confidence
    }

    # Draw connections as 2px lines colored by the minimum confidence of both endpoints
    for joint_a, joint_b in SKELETON_CONNECTIONS:
        if joint_a not in visible or joint_b not in visible:
            continue
        pt_a = keypoints[joint_a]
        pt_b = keypoints[joint_b]
        min_conf = min(confidence[joint_a], confidence[joint_b])
        color = confidence_color(min_conf)
        cv2.line(canvas, pt_a, pt_b, color, thickness=2, lineType=cv2.LINE_AA)

    # Draw joints as 6px radius filled circles with 1px white outline
    for joint in visible:
        pt = keypoints[joint]
        color = confidence_color(confidence[joint])
        cv2.circle(canvas, pt, 6, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt, 6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Draw optional bounding box in light gray
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), thickness=1)

    return canvas


def draw_angle_arc(
    frame: np.ndarray,
    vertex: tuple[int, int],
    point_a: tuple[int, int],
    point_b: tuple[int, int],
    angle_deg: float,
    label: str,
    color: tuple[int, int, int] = (255, 255, 255),
    radius: int = 30,
) -> np.ndarray:
    """Draw an angle arc annotation between two line segments meeting at a vertex.

    Draws an arc at the vertex showing the angle between vertex->point_a and
    vertex->point_b, plus a text label offset from the vertex.

    Args:
        frame: BGR image array (H, W, 3).
        vertex: (x, y) pixel coordinates of the angle vertex.
        point_a: (x, y) pixel coordinates of the first ray endpoint.
        point_b: (x, y) pixel coordinates of the second ray endpoint.
        angle_deg: The angle value in degrees (used for the label, not recomputed).
        label: Text label to draw near the arc (e.g. "90°").
        color: BGR color for the arc and label.
        radius: Pixel radius of the arc.

    Returns:
        A new frame with the arc drawn. The input is not mutated.
    """
    canvas = frame.copy()

    vx, vy = vertex

    # Compute angles of both rays from the vertex (in degrees, OpenCV convention)
    angle_a = math.degrees(math.atan2(-(point_a[1] - vy), point_a[0] - vx))
    angle_b = math.degrees(math.atan2(-(point_b[1] - vy), point_b[0] - vx))

    # Ensure we draw the smaller arc (start < end for cv2.ellipse)
    start_angle = min(angle_a, angle_b)
    end_angle = max(angle_a, angle_b)

    # If the arc spans more than 180 degrees, go the other way
    if end_angle - start_angle > 180:
        start_angle, end_angle = end_angle, start_angle + 360

    cv2.ellipse(
        canvas,
        center=vertex,
        axes=(radius, radius),
        angle=0,
        startAngle=-end_angle,    # Negate because OpenCV y-axis is flipped
        endAngle=-start_angle,
        color=color,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    # Place label offset from vertex toward the bisector of the two rays
    bisect_angle = math.radians((angle_a + angle_b) / 2)
    label_offset = radius + 12
    label_x = int(vx + label_offset * math.cos(bisect_angle))
    label_y = int(vy - label_offset * math.sin(bisect_angle))

    cv2.putText(
        canvas,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )

    return canvas
