"""Trajectory plot generators for pose validation.

Generates time-series visualizations for:
  - Joint position trajectories (y-coordinate over time)
  - Wrist speed profiles
  - Keypoint confidence heatmaps across frames
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go


def _add_event_lines(fig: go.Figure, events: dict[str, float]) -> None:
    """Add vertical dashed red lines for detected pitching events.

    Args:
        fig: Plotly figure to add lines to.
        events: Mapping of event name to timestamp in seconds.
    """
    for event_name, timestamp_seconds in events.items():
        fig.add_vline(
            x=timestamp_seconds,
            line_dash="dash",
            line_color="red",
            annotation_text=event_name,
            annotation_position="top",
        )


def plot_joint_trajectory(
    y_data: np.ndarray,
    timestamps: np.ndarray,
    joint_name: str,
    events: Optional[dict[str, float]] = None,
    invert_y: bool = False,
) -> go.Figure:
    """Line chart of a single joint's y-coordinate over time.

    Args:
        y_data: Array of y-position values (pixels).
        timestamps: Array of timestamps in seconds.
        joint_name: Name of the joint (used as chart title).
        events: Optional mapping of event name -> timestamp_seconds
            for vertical marker lines.
        invert_y: If True, invert the y-axis (useful when pixel coords
            increase downward).

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=y_data,
        mode="lines",
        line=dict(color="#4A90D9", width=2),
        name=joint_name,
    ))

    if events:
        _add_event_lines(fig, events)

    fig.update_layout(
        title=joint_name,
        xaxis_title="Time (s)",
        yaxis_title="Position (px)",
        height=300,
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=40),
    )

    if invert_y:
        fig.update_yaxes(autorange="reversed")

    return fig


def plot_wrist_speed(
    speed: np.ndarray,
    timestamps: np.ndarray,
    events: Optional[dict[str, float]] = None,
) -> go.Figure:
    """Line chart of throwing-wrist speed over time with fill to zero.

    Args:
        speed: Array of speed values (pixels per second).
        timestamps: Array of timestamps in seconds.
        events: Optional mapping of event name -> timestamp_seconds
            for vertical marker lines.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=speed,
        mode="lines",
        line=dict(color="#FF6B35", width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 107, 53, 0.15)",
        name="Wrist Speed",
    ))

    if events:
        _add_event_lines(fig, events)

    fig.update_layout(
        title="Throwing Wrist Speed",
        xaxis_title="Time (s)",
        yaxis_title="Speed (px/s)",
        height=300,
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return fig


def plot_confidence_heatmap(
    confidence_data: np.ndarray,
    timestamps: np.ndarray,
) -> go.Figure:
    """Heatmap of keypoint detection confidence across joints and time.

    Args:
        confidence_data: 2D array of shape (num_joints, num_frames)
            with confidence values in [0, 1].
        timestamps: Array of timestamps in seconds (length = num_frames).

    Returns:
        Plotly Figure object.
    """
    num_joints = confidence_data.shape[0]

    # Default joint labels when none provided
    joint_labels = [f"Joint {i}" for i in range(num_joints)]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=confidence_data,
        x=timestamps,
        y=joint_labels,
        colorscale=[
            [0.0, "red"],
            [0.4, "yellow"],
            [0.7, "green"],
            [1.0, "green"],
        ],
        zmin=0,
        zmax=1,
        colorbar=dict(title="Confidence"),
    ))

    fig.update_layout(
        title="Keypoint Detection Confidence",
        xaxis_title="Time (s)",
        yaxis_title="Joint",
        height=max(200, num_joints * 25 + 100),
        template="plotly_white",
        margin=dict(l=100, r=20, t=40, b=40),
    )

    return fig
