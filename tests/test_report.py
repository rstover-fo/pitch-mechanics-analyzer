"""Tests for HTML report generator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz.report import build_report_html


class TestBuildReportHtml:
    """Tests for build_report_html output."""

    @pytest.fixture
    def minimal_args(self) -> dict:
        """Minimal valid arguments for build_report_html."""
        return dict(
            video_filename="test_pitch.mp4",
            video_rel_path="../uploads/test_pitch_annotated.mp4",
            fps=30.0,
            frame_count=120,
            backend="yolov8",
            pitcher_throws="R",
            trajectory_plots_html=[],
            key_frame_images={},
            metrics_rows=[],
            diagnostics={},
        )

    def test_returns_valid_html(self, minimal_args: dict) -> None:
        html = build_report_html(**minimal_args)
        assert isinstance(html, str)
        assert "<html" in html
        assert "</html>" in html

    def test_contains_video_filename_and_backend(self, minimal_args: dict) -> None:
        html = build_report_html(**minimal_args)
        assert "test_pitch.mp4" in html
        assert "yolov8" in html

    def test_contains_video_tag_with_rel_path(self, minimal_args: dict) -> None:
        html = build_report_html(**minimal_args)
        assert "<video" in html
        assert "../uploads/test_pitch_annotated.mp4" in html

    def test_key_frame_images_rendered(self, minimal_args: dict) -> None:
        minimal_args["key_frame_images"] = {
            "foot_plant": "AAAA",
            "ball_release": "BBBB",
        }
        html = build_report_html(**minimal_args)
        assert "foot_plant" in html
        assert "ball_release" in html
        assert "data:image/png;base64," in html

    def test_metrics_rows_rendered(self, minimal_args: dict) -> None:
        minimal_args["metrics_rows"] = [
            {
                "metric": "elbow_flexion_fp",
                "value": 92.3,
                "unit": "deg",
                "obp_median": 88.0,
                "status": "normal",
            },
        ]
        html = build_report_html(**minimal_args)
        assert "elbow_flexion_fp" in html
        assert "92.3" in html

    def test_metrics_missing_status_gets_warning_class(self, minimal_args: dict) -> None:
        minimal_args["metrics_rows"] = [
            {
                "metric": "missing_metric",
                "value": None,
                "unit": "deg",
                "obp_median": 50.0,
                "status": "missing",
            },
        ]
        html = build_report_html(**minimal_args)
        assert "missing_metric" in html
        assert "warning" in html

    def test_trajectory_plots_embedded(self, minimal_args: dict) -> None:
        plot_html = '<div id="plot-1"><p>Fake plot</p></div>'
        minimal_args["trajectory_plots_html"] = [plot_html]
        html = build_report_html(**minimal_args)
        assert plot_html in html

    def test_diagnostics_rendered(self, minimal_args: dict) -> None:
        minimal_args["diagnostics"] = {"pose_confidence": 0.87, "dropped_frames": 3}
        html = build_report_html(**minimal_args)
        assert "pose_confidence" in html
        assert "0.87" in html
        assert "dropped_frames" in html

    def test_duration_computed_from_fps_and_frames(self, minimal_args: dict) -> None:
        minimal_args["fps"] = 30.0
        minimal_args["frame_count"] = 150
        html = build_report_html(**minimal_args)
        # 150 / 30 = 5.0s
        assert "5.0" in html

    def test_dark_theme_styles(self, minimal_args: dict) -> None:
        html = build_report_html(**minimal_args)
        assert "#0a0a0a" in html
        assert "#e0e0e0" in html

    def test_plotly_cdn_included(self, minimal_args: dict) -> None:
        html = build_report_html(**minimal_args)
        assert "plotly" in html.lower()
        assert "<script" in html

    def test_mediapipe_backend(self, minimal_args: dict) -> None:
        minimal_args["backend"] = "mediapipe"
        html = build_report_html(**minimal_args)
        assert "mediapipe" in html

    def test_left_handed_pitcher(self, minimal_args: dict) -> None:
        minimal_args["pitcher_throws"] = "L"
        html = build_report_html(**minimal_args)
        assert "L" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
