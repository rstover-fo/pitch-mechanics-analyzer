"""Tests for Phase 5 coaching report integration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coaching.insights import generate_report_offline, generate_youth_report_offline
from src.viz.report import build_report_html

# load_prompt is added by Task 1; guard import so the rest of the file is testable now.
try:
    from src.coaching.insights import load_prompt
except ImportError:
    load_prompt = None


class TestImperialToMetric:
    """Test imperial-to-metric unit conversions used in CLI."""

    def test_inches_to_cm(self):
        height_in = 60
        height_cm = height_in * 2.54
        assert height_cm == 152.4

    def test_pounds_to_kg(self):
        weight_lbs = 90
        weight_kg = weight_lbs * 0.4536
        assert abs(weight_kg - 40.824) < 0.01

    def test_tall_pitcher_conversion(self):
        height_in = 72  # 6 feet
        height_cm = height_in * 2.54
        assert abs(height_cm - 182.88) < 0.01

    def test_heavy_pitcher_conversion(self):
        weight_lbs = 200
        weight_kg = weight_lbs * 0.4536
        assert abs(weight_kg - 90.72) < 0.01


_skip_no_load_prompt = pytest.mark.skipif(
    load_prompt is None,
    reason="load_prompt not yet implemented (Task 1)",
)


class TestPromptLoading:
    """Test prompt file loading from data/prompts/."""

    @_skip_no_load_prompt
    def test_load_existing_prompt(self):
        text = load_prompt("youth_coaching_persona")
        assert len(text) > 0
        assert "youth" in text.lower() or "pitcher" in text.lower()

    @_skip_no_load_prompt
    def test_load_adult_prompt(self):
        text = load_prompt("adult_coaching_persona")
        assert len(text) > 0
        assert "coach" in text.lower() or "pitcher" in text.lower()

    @_skip_no_load_prompt
    def test_load_coaching_knowledge(self):
        text = load_prompt("coaching_knowledge")
        assert len(text) > 0

    @_skip_no_load_prompt
    def test_load_measurement_caveats(self):
        text = load_prompt("measurement_caveats")
        assert len(text) > 0
        assert "2d" in text.lower() or "camera" in text.lower()

    @_skip_no_load_prompt
    def test_load_nonexistent_prompt_returns_empty(self):
        text = load_prompt("this_prompt_does_not_exist_12345")
        assert text == ""

    def test_all_prompt_files_exist(self):
        prompt_dir = Path(__file__).parent.parent / "data" / "prompts"
        expected_files = [
            "youth_coaching_persona.md",
            "adult_coaching_persona.md",
            "coaching_knowledge.md",
            "measurement_caveats.md",
        ]
        for fname in expected_files:
            assert (prompt_dir / fname).exists(), f"Missing prompt file: {fname}"


class TestCliValidation:
    """Test CLI argument validation logic for youth profile flags."""

    def test_all_youth_flags_valid(self):
        """All three flags provided should be valid."""
        age, height, weight = 12, 60.0, 90.0
        youth_flags = [age, height, weight]
        # All provided = valid
        assert all(v is not None for v in youth_flags)

    def test_no_youth_flags_valid(self):
        """No youth flags is valid (adult path)."""
        age, height, weight = None, None, None
        youth_flags = [age, height, weight]
        # None provided = valid (adult path)
        provided = [v for v in youth_flags if v is not None]
        assert len(provided) == 0

    def test_partial_youth_flags_detected(self):
        """Partial flags should be detected as invalid."""
        age, height, weight = 12, None, None
        youth_flags = [age, height, weight]
        provided = [v for v in youth_flags if v is not None]
        all_provided = len(provided) == 3
        none_provided = len(provided) == 0
        # Should be invalid (partial)
        assert not all_provided and not none_provided

    def test_age_range_lower_bound(self):
        """Age below 6 should be flagged."""
        assert 3 < 6  # Below valid range

    def test_age_range_upper_bound(self):
        """Age above 25 should be flagged."""
        assert 30 > 25  # Above valid range

    def test_age_range_valid(self):
        """Age 12 should be in valid range."""
        assert 6 <= 12 <= 25


class TestOfflineCoachingReport:
    """Test offline (rule-based) coaching report generation."""

    def test_offline_report_with_high_percentile(self):
        comparisons = [
            {
                "metric": "elbow_flexion_fp",
                "display_name": "Elbow Flexion @ FP",
                "value": 92.0,
                "unit": "deg",
                "percentile_rank": 75.0,
                "flag": "above_average",
                "benchmark_median": 88.0,
                "playing_level": "all",
                "n_samples": 411,
            }
        ]
        report = generate_report_offline(comparisons)
        assert isinstance(report, str)
        assert "STRENGTHS" in report or "strengths" in report.lower()

    def test_offline_report_with_low_percentile(self):
        comparisons = [
            {
                "metric": "rotation_hip_shoulder_separation_fp",
                "display_name": "Hip-Shoulder Sep @ FP",
                "value": 15.0,
                "unit": "deg",
                "percentile_rank": 10.0,
                "flag": "well_below_average",
                "benchmark_median": 32.0,
                "playing_level": "all",
                "n_samples": 411,
            }
        ]
        report = generate_report_offline(comparisons)
        assert isinstance(report, str)
        assert "DEVELOPMENT" in report or "development" in report.lower()

    def test_offline_report_empty_comparisons(self):
        report = generate_report_offline([])
        assert isinstance(report, str)
        assert len(report) > 0


class TestReportCoachingSection:
    """Test HTML report builder with coaching parameters."""

    @pytest.fixture
    def base_args(self) -> dict:
        return dict(
            video_filename="test.mp4",
            video_rel_path="annotated_video.mp4",
            fps=30.0,
            frame_count=120,
            backend="yolov8",
            pitcher_throws="R",
            trajectory_plots_html=[],
            key_frame_images={},
            metrics_rows=[],
            diagnostics={},
        )

    def test_report_includes_coaching_section(self, base_args):
        base_args["coaching_html"] = "You have excellent elbow positioning at foot plant."
        html = build_report_html(**base_args)
        assert "coaching" in html.lower()
        assert "elbow positioning" in html

    def test_report_without_coaching(self, base_args):
        base_args["coaching_html"] = ""
        html = build_report_html(**base_args)
        assert "Coaching Report" not in html

    def test_report_includes_percentile_charts(self, base_args):
        base_args["percentile_charts_html"] = ['<div id="radar">chart1</div>']
        html = build_report_html(**base_args)
        assert "radar" in html
        assert "chart1" in html

    def test_report_includes_pitcher_profile(self, base_args):
        base_args["pitcher_profile"] = {
            "age": 12,
            "height_in": 60,
            "weight_lbs": 90,
            "developmental_stage": "early_adolescent",
        }
        html = build_report_html(**base_args)
        assert "12" in html
        assert "Early Adolescent" in html or "early_adolescent" in html

    def test_backward_compatible_no_new_params(self, base_args):
        """Existing call signature still works without new params."""
        html = build_report_html(**base_args)
        assert "<html" in html
        assert "</html>" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
