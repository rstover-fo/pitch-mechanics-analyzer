"""Tests for empty pose sequence handling."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biomechanics.events import DeliveryEvents
from src.pipeline import PipelineConfig, PitchAnalysisPipeline
from src.pose.estimator import PoseSequence, VideoInfo
from src.viz.report import build_report_html


@pytest.fixture
def empty_pose_seq() -> PoseSequence:
    """PoseSequence with zero frames."""
    video_info = VideoInfo(
        path=Path("test.mp4"),
        width=1920,
        height=1080,
        fps=30.0,
        total_frames=100,
        duration_sec=3.33,
    )
    return PoseSequence(video_info=video_info, frames=[])


class TestDetectEventsEmptyPoses:
    """detect_events should return default DeliveryEvents for empty frames."""

    def test_returns_delivery_events_without_crash(self, empty_pose_seq: PoseSequence) -> None:
        pipeline = PitchAnalysisPipeline(PipelineConfig())
        events = pipeline.detect_events(empty_pose_seq)

        assert isinstance(events, DeliveryEvents)
        assert events.leg_lift_apex is None
        assert events.foot_plant is None
        assert events.max_external_rotation is None
        assert events.ball_release is None
        assert events.fps == 30.0


class TestBuildReportNoVideo:
    """build_report_html should skip video embed when video_rel_path is None."""

    def test_no_video_section_when_path_is_none(self) -> None:
        html = build_report_html(
            video_filename="test.mp4",
            video_rel_path=None,
            fps=30.0,
            frame_count=100,
            backend="yolov8",
            pitcher_throws="R",
            trajectory_plots_html=[],
            key_frame_images={},
            metrics_rows=[],
            diagnostics={},
        )
        assert "<video" not in html
        assert "annotated_video.mp4" not in html

    def test_video_section_present_when_path_given(self) -> None:
        html = build_report_html(
            video_filename="test.mp4",
            video_rel_path="annotated_video.mp4",
            fps=30.0,
            frame_count=100,
            backend="yolov8",
            pitcher_throws="R",
            trajectory_plots_html=[],
            key_frame_images={},
            metrics_rows=[],
            diagnostics={},
        )
        assert "<video" in html
        assert "annotated_video.mp4" in html
