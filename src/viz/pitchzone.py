"""
pitchzone_v3.py
---------------
Generates a self-contained HTML string with an inline Three.js 3D scene
for pitcher mechanics visualization (PitchZone v3).

Public API
----------
ZONE_BANDS : dict
_GRADE_COLOR : dict
_GREEN, _YELLOW, _RED : str
calculate_pitchzone_score(grades) -> int
generate_pitchzone_svg(grades, metrics, throws, title, width, height) -> str
    NOTE: Returns HTML (not SVG) — backward-compatible name kept for report_parent.py
generate_pitchzone_html(grades, metrics, throws, title, width, height) -> str
"""

import json
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

_GREEN  = "#22c55e"
_YELLOW = "#eab308"
_RED    = "#ef4444"

_GRADE_COLOR: dict[str, str] = {
    "green":  _GREEN,
    "yellow": _YELLOW,
    "red":    _RED,
}

ZONE_BANDS: dict[str, dict] = {
    "shoulder_abduction_fp": {
        "label": "Arm Height",
        "region": "shoulder",
        "excellent": "Arm at perfect height",
        "good": "Nearly there",
        "focus": "Needs work",
    },
    "elbow_flexion_fp": {
        "label": "Elbow Bend",
        "region": "elbow",
        "excellent": "Perfect 'L' shape",
        "good": "Almost right",
        "focus": "Needs attention",
    },
    "torso_anterior_tilt_fp": {
        "label": "Posture",
        "region": "torso",
        "excellent": "Staying tall",
        "good": "Slight lean",
        "focus": "Too much lean",
    },
    "hip_shoulder_separation_fp": {
        "label": "Hip Lead",
        "region": "hip",
        "excellent": "Great separation",
        "good": "Getting there",
        "focus": "Opening together",
    },
    "stride_length_pct_height": {
        "label": "Stride",
        "region": "stride_leg",
        "excellent": "Great reach",
        "good": "Almost enough",
        "focus": "Need more reach",
    },
    "lead_knee_angle_fp": {
        "label": "Front Leg",
        "region": "lead_knee",
        "excellent": "Firm brace",
        "good": "Mostly firm",
        "focus": "Too soft",
    },
}

# Grade word → numeric value for score
_GRADE_VALUES = {"green": 100, "yellow": 65, "red": 30}

# ---------------------------------------------------------------------------
# Ideal ranges per metric: (ideal_value, tolerance)
# Green  if |value − ideal| ≤ tolerance
# Yellow if |value − ideal| ≤ 2 × tolerance
# Red    otherwise
# These match src/viz/overlay.py GRADE_RULES exactly.
# ---------------------------------------------------------------------------
GRADE_RULES: dict[str, tuple[float, float]] = {
    "elbow_flexion_fp":           (90.0,  15.0),
    "shoulder_abduction_fp":      (90.0,  15.0),
    "torso_anterior_tilt_fp":     (30.0,  10.0),
    "hip_shoulder_separation_fp": (30.0,  10.0),
    "stride_length_pct_height":   (80.0,  10.0),
    "lead_knee_angle_fp":         (160.0, 15.0),
}

# ---------------------------------------------------------------------------
# Lazy-loaded Three.js content
# ---------------------------------------------------------------------------

_THREE_JS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "three.min.js")
_THREE_JS_CONTENT: Optional[str] = None


def _load_threejs() -> str:
    global _THREE_JS_CONTENT
    if _THREE_JS_CONTENT is None:
        with open(_THREE_JS_PATH, "r", encoding="utf-8") as fh:
            _THREE_JS_CONTENT = fh.read()
    return _THREE_JS_CONTENT


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

def calculate_pitchzone_score(grades: dict[str, str]) -> int:
    """
    Average grade values: green=100, yellow=65, red=30.
    Unknown or missing keys treated as yellow (65).
    Returns 65 for empty input.
    """
    if not grades:
        return 65
    total = sum(_GRADE_VALUES.get(v, 65) for v in grades.values())
    return round(total / len(grades))
