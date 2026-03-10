"""Tests for trajectory plot generators."""

import numpy as np
import plotly.graph_objects as go

from src.viz.trajectories import (
    plot_confidence_heatmap,
    plot_joint_trajectory,
    plot_wrist_speed,
)


class TestPlotJointTrajectory:
    """Tests for plot_joint_trajectory."""

    def test_returns_plotly_figure_with_to_html(self):
        y_data = np.array([100.0, 105.0, 110.0, 108.0, 103.0])
        timestamps = np.array([0.0, 0.033, 0.066, 0.1, 0.133])
        fig = plot_joint_trajectory(y_data, timestamps, "Right Wrist Y")

        assert isinstance(fig, go.Figure)
        html = fig.to_html(full_html=False, include_plotlyjs=False)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_works_with_no_events(self):
        y_data = np.array([200.0, 210.0, 220.0])
        timestamps = np.array([0.0, 0.5, 1.0])
        fig = plot_joint_trajectory(y_data, timestamps, "Left Elbow Y", events=None)

        assert isinstance(fig, go.Figure)
        # Should have exactly one trace (the line) and no vertical lines
        assert len(fig.data) == 1

    def test_events_add_vertical_lines(self):
        y_data = np.array([100.0, 150.0, 200.0, 180.0, 160.0])
        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        events = {"Foot Plant": 0.1, "Ball Release": 0.3}
        fig = plot_joint_trajectory(
            y_data, timestamps, "Right Shoulder Y", events=events
        )

        assert isinstance(fig, go.Figure)
        # One data trace + vertical line shapes for each event
        shapes = fig.layout.shapes
        assert len(shapes) == 2

    def test_layout_properties(self):
        y_data = np.array([50.0, 60.0, 70.0])
        timestamps = np.array([0.0, 0.5, 1.0])
        fig = plot_joint_trajectory(y_data, timestamps, "Test Joint")

        assert fig.layout.title.text == "Test Joint"
        assert fig.layout.xaxis.title.text == "Time (s)"
        assert fig.layout.yaxis.title.text == "Position (px)"
        assert fig.layout.height == 300


class TestPlotWristSpeed:
    """Tests for plot_wrist_speed."""

    def test_returns_plotly_figure(self):
        speed = np.array([0.0, 500.0, 1200.0, 800.0, 200.0])
        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        fig = plot_wrist_speed(speed, timestamps)

        assert isinstance(fig, go.Figure)
        html = fig.to_html(full_html=False, include_plotlyjs=False)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_layout_properties(self):
        speed = np.array([0.0, 100.0, 200.0])
        timestamps = np.array([0.0, 0.5, 1.0])
        fig = plot_wrist_speed(speed, timestamps)

        assert fig.layout.title.text == "Throwing Wrist Speed"
        assert fig.layout.yaxis.title.text == "Speed (px/s)"
        assert fig.layout.height == 300

    def test_with_events(self):
        speed = np.array([0.0, 500.0, 1200.0, 800.0, 200.0])
        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        events = {"MER": 0.15, "Ball Release": 0.25}
        fig = plot_wrist_speed(speed, timestamps, events=events)

        assert isinstance(fig, go.Figure)
        shapes = fig.layout.shapes
        assert len(shapes) == 2


class TestPlotConfidenceHeatmap:
    """Tests for plot_confidence_heatmap."""

    def test_returns_plotly_figure(self):
        # 5 joints, 10 time frames
        confidence_data = np.random.rand(5, 10)
        timestamps = np.linspace(0.0, 1.0, 10)
        fig = plot_confidence_heatmap(confidence_data, timestamps)

        assert isinstance(fig, go.Figure)
        html = fig.to_html(full_html=False, include_plotlyjs=False)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_height_scales_with_joints(self):
        # 3 joints -> max(200, 3*25 + 100) = 200
        small_data = np.random.rand(3, 10)
        timestamps = np.linspace(0.0, 1.0, 10)
        fig_small = plot_confidence_heatmap(small_data, timestamps)
        assert fig_small.layout.height == 200

        # 20 joints -> max(200, 20*25 + 100) = 600
        large_data = np.random.rand(20, 10)
        fig_large = plot_confidence_heatmap(large_data, timestamps)
        assert fig_large.layout.height == 600
