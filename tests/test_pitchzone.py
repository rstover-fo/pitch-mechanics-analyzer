"""Tests for src/viz/pitchzone.py — PitchZone SVG generation.

Run with:  pytest tests/test_pitchzone.py -v
"""

import math
import xml.etree.ElementTree as ET

import pytest

from src.viz.pitchzone import (
    ZONE_BANDS,
    _GRADE_COLOR,
    _GREEN,
    _YELLOW,
    _RED,
    calculate_pitchzone_score,
    generate_pitchzone_svg,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────────────────

ALL_GREEN = {
    "shoulder_abduction_fp":      "green",
    "elbow_flexion_fp":           "green",
    "torso_anterior_tilt_fp":     "green",
    "hip_shoulder_separation_fp": "green",
    "stride_length_pct_height":   "green",
    "lead_knee_angle_fp":         "green",
}

ALL_RED = {k: "red" for k in ALL_GREEN}

ALL_YELLOW = {k: "yellow" for k in ALL_GREEN}

MIXED_GRADES = {
    "shoulder_abduction_fp":      "green",
    "elbow_flexion_fp":           "yellow",
    "torso_anterior_tilt_fp":     "red",
    "hip_shoulder_separation_fp": "green",
    "stride_length_pct_height":   "yellow",
    "lead_knee_angle_fp":         "green",
}


# ── Score calculation tests ─────────────────────────────────────────────────────────────────────

class TestCalculatePitchzoneScore:

    def test_all_green_is_100(self):
        assert calculate_pitchzone_score(ALL_GREEN) == 100

    def test_all_red_is_30(self):
        assert calculate_pitchzone_score(ALL_RED) == 30

    def test_all_yellow_is_65(self):
        assert calculate_pitchzone_score(ALL_YELLOW) == 65

    def test_mixed_grades_average(self):
        # 3 green (100) + 2 yellow (65) + 1 red (30) = (300+130+30)/6 = 460/6 ≈ 77
        score = calculate_pitchzone_score(MIXED_GRADES)
        assert score == 77

    def test_empty_grades_returns_default(self):
        # Should not raise; returns mid-range default
        score = calculate_pitchzone_score({})
        assert score == 65

    def test_single_green(self):
        assert calculate_pitchzone_score({"shoulder_abduction_fp": "green"}) == 100

    def test_single_red(self):
        assert calculate_pitchzone_score({"elbow_flexion_fp": "red"}) == 30

    def test_score_range(self):
        """Score must always be in [0, 100]."""
        for grades in [ALL_GREEN, ALL_RED, ALL_YELLOW, MIXED_GRADES, {}]:
            s = calculate_pitchzone_score(grades)
            assert 0 <= s <= 100, f"Score {s} out of range for grades {grades}"

    def test_unknown_grade_treated_as_yellow(self):
        grades = {"shoulder_abduction_fp": "purple"}  # unknown
        score = calculate_pitchzone_score(grades)
        assert score == 65  # yellow default


# ── SVG generation tests ────────────────────────────────────────────────────────────────────────

class TestGeneratePitchzoneSvg:

    def _parse_svg(self, svg: str) -> ET.Element:
        """Parse SVG string and return root element, asserting valid XML."""
        return ET.fromstring(svg)

    def test_returns_string(self):
        svg = generate_pitchzone_svg(ALL_GREEN)
        assert isinstance(svg, str)

    def test_svg_starts_with_svg_tag(self):
        svg = generate_pitchzone_svg(ALL_GREEN)
        assert svg.strip().startswith("<svg")

    def test_svg_is_valid_xml(self):
        """SVG must be parseable as XML (required for QWebEngineView)."""
        for grades in [ALL_GREEN, ALL_RED, ALL_YELLOW, MIXED_GRADES]:
            svg = generate_pitchzone_svg(grades)
            root = self._parse_svg(svg)
            assert root.tag.endswith("svg")

    def test_svg_has_viewbox(self):
        svg = generate_pitchzone_svg(ALL_GREEN)
        assert 'viewBox="' in svg

    def test_svg_no_javascript(self):
        """Pure SVG — no script tags allowed."""
        svg = generate_pitchzone_svg(ALL_GREEN)
        assert "<script" not in svg.lower()

    def test_svg_dark_background(self):
        """Background must be #0a0a0a (dark theme)."""
        svg = generate_pitchzone_svg(ALL_GREEN)
        assert "#0a0a0a" in svg

    def test_green_color_present_when_all_green(self):
        svg = generate_pitchzone_svg(ALL_GREEN)
        assert _GREEN in svg

    def test_red_color_present_when_all_red(self):
        svg = generate_pitchzone_svg(ALL_RED)
        assert _RED in svg

    def test_yellow_color_present_when_all_yellow(self):
        svg = generate_pitchzone_svg(ALL_YELLOW)
        assert _YELLOW in svg

    def test_mixed_grades_all_colors_present(self):
        svg = generate_pitchzone_svg(MIXED_GRADES)
        assert _GREEN in svg
        assert _YELLOW in svg
        assert _RED in svg

    def test_glow_filters_present(self):
        """SVG must contain glow filter definitions."""
        svg = generate_pitchzone_svg(ALL_GREEN)
        assert "feGaussianBlur" in svg
        assert "feFlood" in svg

    def test_contains_score_text(self):
        """Overall score number should appear in the SVG."""
        svg = generate_pitchzone_svg(ALL_GREEN)
        # Score is 100 for all-green
        assert ">100<" in svg

    def test_all_zone_labels_present(self):
        """All six zone dimension names should appear in the SVG."""
        svg = generate_pitchzone_svg(ALL_GREEN)
        expected_labels = [
            "Arm Height",
            "Elbow Bend",
            "Posture",
            "Hip Lead",
            "Stride",
            "Front Leg",
        ]
        for label in expected_labels:
            assert label in svg, f"Expected label '{label}' not found in SVG"

    def test_grade_words_present(self):
        """Grade words (Excellent/Good/Focus) should appear in SVG."""
        svg = generate_pitchzone_svg(MIXED_GRADES)
        assert "Excellent" in svg
        assert "Good" in svg
        assert "Focus" in svg

    def test_rhp_label_in_svg(self):
        svg = generate_pitchzone_svg(ALL_GREEN, throws="R")
        assert "RHP" in svg

    def test_lhp_label_in_svg(self):
        svg = generate_pitchzone_svg(ALL_GREEN, throws="L")
        assert "LHP" in svg

    def test_empty_grades_does_not_raise(self):
        """Empty grades dict should produce valid SVG with yellow defaults."""
        svg = generate_pitchzone_svg({})
        root = self._parse_svg(svg)
        assert root.tag.endswith("svg")

    def test_missing_grades_default_to_yellow(self):
        """Partially specified grades: missing ones should default to yellow."""
        partial = {"shoulder_abduction_fp": "green"}
        svg = generate_pitchzone_svg(partial)
        # Should still be valid XML
        self._parse_svg(svg)
        assert _YELLOW in svg  # other zones defaulted to yellow

    def test_custom_title_appears(self):
        svg = generate_pitchzone_svg(ALL_GREEN, title="My Custom Title")
        assert "My Custom Title" in svg

    def test_svg_size_attributes(self):
        """Default width/height attributes should be present."""
        svg = generate_pitchzone_svg(ALL_GREEN, width=700, height=600)
        assert 'width="700"' in svg
        assert 'height="600"' in svg

    def test_pitchzone_text_present(self):
        svg = generate_pitchzone_svg(ALL_GREEN, title="PitchZone")
        assert "PitchZone" in svg


# ── LHP mirror tests ──────────────────────────────────────────────────────────────────────────

class TestLhpMirror:

    def test_lhp_svg_valid_xml(self):
        svg = generate_pitchzone_svg(MIXED_GRADES, throws="L")
        ET.fromstring(svg)

    def test_lhp_svg_differs_from_rhp(self):
        """LHP and RHP SVGs should differ (mirror affects joint positions)."""
        rhp_svg = generate_pitchzone_svg(ALL_GREEN, throws="R")
        lhp_svg = generate_pitchzone_svg(ALL_GREEN, throws="L")
        assert rhp_svg != lhp_svg

    def test_lhp_same_colors_as_rhp(self):
        """Grade colors should be identical regardless of handedness."""
        rhp_svg = generate_pitchzone_svg(MIXED_GRADES, throws="R")
        lhp_svg = generate_pitchzone_svg(MIXED_GRADES, throws="L")
        # Both should contain the same set of grade colors
        for color in [_GREEN, _YELLOW, _RED]:
            assert (color in rhp_svg) == (color in lhp_svg)

    def test_lhp_label_present(self):
        svg = generate_pitchzone_svg(ALL_GREEN, throws="L")
        assert "LHP" in svg

    def test_lhp_no_rhp_label(self):
        svg = generate_pitchzone_svg(ALL_GREEN, throws="L")
        assert "RHP" not in svg


# ── Zone bands metadata tests ──────────────────────────────────────────────────────────────────

class TestZoneBandsMetadata:

    def test_all_expected_metrics_present(self):
        expected = {
            "shoulder_abduction_fp",
            "elbow_flexion_fp",
            "torso_anterior_tilt_fp",
            "hip_shoulder_separation_fp",
            "stride_length_pct_height",
            "lead_knee_angle_fp",
        }
        assert set(ZONE_BANDS.keys()) == expected

    def test_each_band_has_required_keys(self):
        required_keys = {"label", "region", "excellent", "good", "focus"}
        for metric, band in ZONE_BANDS.items():
            missing = required_keys - set(band.keys())
            assert not missing, f"Band '{metric}' missing keys: {missing}"

    def test_grade_color_map_complete(self):
        for grade in ("green", "yellow", "red"):
            assert grade in _GRADE_COLOR


# ── Edge case / robustness tests ────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_uppercase_grade_normalised(self):
        """Grades should be case-insensitive."""
        grades = {k: "GREEN" for k in ALL_GREEN}
        svg = generate_pitchzone_svg(grades)
        assert _GREEN in svg

    def test_none_grade_treated_as_yellow(self):
        """None values in grades should not crash."""
        grades = {k: None for k in ALL_GREEN}
        svg = generate_pitchzone_svg(grades)
        ET.fromstring(svg)  # must be valid XML

    def test_extra_keys_in_grades_ignored(self):
        """Extra metric keys not in ZONE_BANDS are silently ignored."""
        grades = dict(ALL_GREEN)
        grades["unknown_metric_xyz"] = "red"
        svg = generate_pitchzone_svg(grades)
        ET.fromstring(svg)

    def test_score_displayed_in_svg_all_red(self):
        svg = generate_pitchzone_svg(ALL_RED)
        assert ">30<" in svg

    def test_score_displayed_in_svg_all_yellow(self):
        svg = generate_pitchzone_svg(ALL_YELLOW)
        assert ">65<" in svg

    def test_metrics_kwarg_accepted(self):
        """metrics kwarg should be accepted without error."""
        metrics = {
            "shoulder_abduction_fp": 88.5,
            "elbow_flexion_fp": 92.0,
        }
        svg = generate_pitchzone_svg(ALL_GREEN, metrics=metrics, throws="R")
        assert svg.strip().startswith("<svg")
