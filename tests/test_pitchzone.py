"""Tests for src/viz/pitchzone.py — PitchZone Three.js visualization.

Run with:  pytest tests/test_pitchzone.py -v
"""

import pytest

from src.viz.pitchzone import (
    ZONE_BANDS,
    GRADE_RULES,
    _GRADE_COLOR,
    _GREEN,
    _YELLOW,
    _RED,
    calculate_pitchzone_score,
    generate_pitchzone_html,
    generate_pitchzone_svg,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────────────────────────

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

SAMPLE_METRICS = {
    "shoulder_abduction_fp": 88.0,
    "elbow_flexion_fp": 72.0,
    "torso_anterior_tilt_fp": 28.5,
    "hip_shoulder_separation_fp": 12.0,
    "stride_length_pct_height": 68.0,
    "lead_knee_angle_fp": 155.0,
}


# ── Score calculation tests ─────────────────────────────────────────────────────────────────────────────────────────

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
        score = calculate_pitchzone_score({})
        assert score == 65

    def test_single_green(self):
        assert calculate_pitchzone_score({"shoulder_abduction_fp": "green"}) == 100

    def test_single_red(self):
        assert calculate_pitchzone_score({"elbow_flexion_fp": "red"}) == 30

    def test_score_range(self):
        for grades in [ALL_GREEN, ALL_RED, ALL_YELLOW, MIXED_GRADES, {}]:
            s = calculate_pitchzone_score(grades)
            assert 0 <= s <= 100, f"Score {s} out of range for grades {grades}"

    def test_unknown_grade_treated_as_yellow(self):
        grades = {"shoulder_abduction_fp": "purple"}
        score = calculate_pitchzone_score(grades)
        assert score == 65


class TestGeneratePitchzoneHtml:

    def test_returns_string(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert isinstance(html, str)

    def test_html_starts_with_doctype(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_contains_canvas_element(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert '<canvas id="pz-canvas"' in html

    def test_contains_threejs(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert "THREE.WebGLRenderer" in html

    def test_dark_background(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert "#0a0a0a" in html

    def test_contains_score_text(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert ">100<" in html

    def test_all_zone_labels_present(self):
        html = generate_pitchzone_html(ALL_GREEN)
        expected_labels = ["Arm Height", "Elbow Bend", "Posture", "Hip Lead", "Stride", "Front Leg"]
        for label in expected_labels:
            assert label in html, f"Expected label '{label}' not found in HTML"

    def test_rhp_label(self):
        html = generate_pitchzone_html(ALL_GREEN, throws="R")
        assert "RHP" in html

    def test_lhp_label(self):
        html = generate_pitchzone_html(ALL_GREEN, throws="L")
        assert "LHP" in html

    def test_pitchzone_title_present(self):
        html = generate_pitchzone_html(ALL_GREEN, title="PitchZone")
        assert "PitchZone" in html

    def test_grade_colors_in_scene_js(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert "0x22c55e" in html

    def test_mixed_grades_all_colors_in_js(self):
        html = generate_pitchzone_html(MIXED_GRADES)
        assert "0x22c55e" in html
        assert "0xeab308" in html
        assert "0xef4444" in html

    def test_empty_grades_does_not_raise(self):
        html = generate_pitchzone_html({})
        assert "<canvas" in html

    def test_missing_grades_default_to_yellow(self):
        partial = {"shoulder_abduction_fp": "green"}
        html = generate_pitchzone_html(partial)
        assert "<canvas" in html
        assert '"yellow"' in html

    def test_extra_keys_ignored(self):
        grades = dict(ALL_GREEN)
        grades["unknown_metric_xyz"] = "red"
        html = generate_pitchzone_html(grades)
        assert "<canvas" in html

    def test_custom_width_height(self):
        html = generate_pitchzone_html(ALL_GREEN, width=800, height=500)
        assert 'width="800"' in html
        assert 'height="500"' in html


class TestRangeBands:

    def test_grade_rules_passed_to_js(self):
        html = generate_pitchzone_html(ALL_GREEN, metrics=SAMPLE_METRICS)
        assert "GRADE_RULES" in html
        assert "90.0" in html
        assert "160.0" in html

    def test_metrics_embedded_when_provided(self):
        html = generate_pitchzone_html(MIXED_GRADES, metrics=SAMPLE_METRICS)
        assert "METRICS" in html
        assert "88.0" in html
        assert "72.0" in html

    def test_no_metrics_still_renders(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert "GRADE_RULES" in html
        assert "addRangeBands" in html

    def test_range_band_function_present(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert "addRangeBands" in html
        assert "createRingSectorGeo" in html


class TestGeneratePitchzoneSvgCompat:

    def test_returns_iframe(self):
        result = generate_pitchzone_svg(ALL_GREEN)
        assert result.strip().startswith("<iframe")

    def test_iframe_contains_srcdoc(self):
        result = generate_pitchzone_svg(ALL_GREEN)
        assert 'srcdoc="' in result

    def test_iframe_sandbox(self):
        result = generate_pitchzone_svg(ALL_GREEN)
        assert 'sandbox="allow-scripts"' in result

    def test_iframe_accepts_metrics(self):
        result = generate_pitchzone_svg(ALL_GREEN, metrics=SAMPLE_METRICS)
        assert isinstance(result, str)
        assert "<iframe" in result

    def test_iframe_width_height(self):
        result = generate_pitchzone_svg(ALL_GREEN, width=800, height=500)
        assert 'width="800"' in result
        assert 'height="500"' in result


class TestLhpMirror:

    def test_lhp_html_valid(self):
        html = generate_pitchzone_html(MIXED_GRADES, throws="L")
        assert "<!DOCTYPE html>" in html
        assert "<canvas" in html

    def test_lhp_html_differs_from_rhp(self):
        rhp = generate_pitchzone_html(ALL_GREEN, throws="R")
        lhp = generate_pitchzone_html(ALL_GREEN, throws="L")
        assert rhp != lhp

    def test_lhp_mirror_value_negative(self):
        html = generate_pitchzone_html(ALL_GREEN, throws="L")
        assert "const MIRROR = -1;" in html

    def test_rhp_mirror_value_positive(self):
        html = generate_pitchzone_html(ALL_GREEN, throws="R")
        assert "const MIRROR = 1;" in html

    def test_lhp_same_colors_as_rhp(self):
        rhp = generate_pitchzone_html(MIXED_GRADES, throws="R")
        lhp = generate_pitchzone_html(MIXED_GRADES, throws="L")
        for color_hex in ["0x22c55e", "0xeab308", "0xef4444"]:
            assert (color_hex in rhp) == (color_hex in lhp)

    def test_lhp_label_present(self):
        html = generate_pitchzone_html(ALL_GREEN, throws="L")
        assert "LHP" in html

    def test_lhp_no_rhp_label(self):
        html = generate_pitchzone_html(ALL_GREEN, throws="L")
        import re
        m = re.search(r'id="title-block"[^>]*>(.*?)</div>\s*</div>', html, re.DOTALL)
        assert m is not None, "title-block div not found"
        title_text = m.group(1)
        assert "RHP" not in title_text


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

    def test_grade_rules_match_zone_bands(self):
        for metric in ZONE_BANDS:
            assert metric in GRADE_RULES, f"Missing GRADE_RULES for '{metric}'"

    def test_grade_rules_have_valid_tuples(self):
        for metric, (ideal, tol) in GRADE_RULES.items():
            assert isinstance(ideal, (int, float))
            assert isinstance(tol, (int, float))
            assert tol > 0, f"Tolerance for '{metric}' must be positive"


class TestEdgeCases:

    def test_none_grade_treated_as_yellow(self):
        grades = {k: None for k in ALL_GREEN}
        html = generate_pitchzone_html(grades)
        assert "<canvas" in html

    def test_score_displayed_all_red(self):
        html = generate_pitchzone_html(ALL_RED)
        assert ">30<" in html

    def test_score_displayed_all_yellow(self):
        html = generate_pitchzone_html(ALL_YELLOW)
        assert ">65<" in html

    def test_metrics_with_none_values(self):
        metrics = {k: None for k in ALL_GREEN}
        html = generate_pitchzone_html(ALL_GREEN, metrics=metrics)
        assert "<canvas" in html

    def test_partial_metrics(self):
        metrics = {"elbow_flexion_fp": 90.0}
        html = generate_pitchzone_html(ALL_GREEN, metrics=metrics)
        assert "<canvas" in html

    def test_mannequin_geometry_present(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert "CylinderGeometry" in html
        assert "SphereGeometry" in html
        assert "castShadow" in html

    def test_hip_shoulder_planes_present(self):
        html = generate_pitchzone_html(ALL_GREEN)
        assert "shoulderLine" in html
        assert "hipLine" in html
        assert "shoulderCenter" in html
