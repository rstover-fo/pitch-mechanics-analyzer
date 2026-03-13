"""Self-contained HTML parent report generator.

Coaching-first layout aimed at parents and young players:
  1. Header with pitcher name / date
  2. PitchZone diagram (SwingAI-inspired 3D figure with colored zone bands + score gauge)
  3. Key frame overlay (actual foot-plant photo with graded joint rings)
  4. Stoplight grade report card
  5. Drill homework (max 3, only for yellow/red grades)
  6. Coach's note (Claude API with offline fallback)
  7. Growth snapshot (if youth profile provided)
  8. Injury watch callout
"""

import os
from html import escape
from typing import Optional

from src.biomechanics.features import PitcherMetrics
from src.viz.overlay import GRADE_RULES, _grade_color, _GREEN, _AMBER, _RED
from src.viz.pitchzone import generate_pitchzone_svg, ZONE_BANDS


# ── Drill prescriptions ────────────────────────────────────────────────────────────────────────
DRILL_MAP: dict[str, dict[str, str]] = {
    "elbow_flexion_fp": {
        "name": "Elbow Spiral Drill",
        "description": "Stand sideways to a wall. Bring throwing arm to 90° "
                       "angle and spiral forearm up/down 10×. Feel the 'L' shape.",
        "reps": "2 × 10 reps",
    },
    "shoulder_abduction_fp": {
        "name": "Arm Path Scarecrow",
        "description": "From set position, lift arms out to a 'T' and hold 3 sec. "
                       "Trains the arm to stay at shoulder height.",
        "reps": "2 × 8 reps",
    },
    "torso_anterior_tilt_fp": {
        "name": "Tall & Fall Drill",
        "description": "Stand on mound, stay tall through leg lift, then lead "
                       "with chest toward the target. No sinking!",
        "reps": "8 throws from stretch",
    },
    "hip_shoulder_separation_fp": {
        "name": "Hip Lead Drill",
        "description": "From leg lift, drive lead hip toward target while keeping "
                       "shoulders closed. Feel the 'coil' before arm fires.",
        "reps": "2 × 6 dry runs, then 6 throws",
    },
    "stride_length_pct_height": {
        "name": "Stride-Length Markers",
        "description": "Place a tape line at 80% of height from rubber. Practice "
                       "landing on the line consistently.",
        "reps": "10 throws to the marker",
    },
    "lead_knee_angle_fp": {
        "name": "Firm Front Side",
        "description": "On flat ground, stride and brace the lead leg straight "
                       "(don't collapse). Partner can push on lead shoulder to test brace.",
        "reps": "2 × 8 reps",
    },
}

# ── Display names for metrics ──────────────────────────────────────────────────────────────────
_DISPLAY_NAMES: dict[str, str] = {
    "elbow_flexion_fp":           "Elbow Angle",
    "shoulder_abduction_fp":      "Arm Height",
    "torso_anterior_tilt_fp":     "Trunk Posture",
    "hip_shoulder_separation_fp": "Hip-Shoulder Separation",
    "stride_length_pct_height":   "Stride Length",
    "lead_knee_angle_fp":         "Front Knee Brace",
}

# ── CSS ───────────────────────────────────────────────────────────────────────────────────
CSS = """\
body {
    background: #0a0a0a;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 2rem 1rem;
}
.container {
    max-width: 900px;
    margin: 0 auto;
}
h1 {
    color: #f0f0f0;
    font-size: 1.6rem;
    margin-bottom: 0.3rem;
}
.subtitle {
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* PitchZone section */
.pitchzone-section {
    margin-bottom: 2rem;
}
.pitchzone-section svg {
    width: 100%;
    height: auto;
    border-radius: 10px;
    display: block;
}
.section-label {
    text-align: center;
    color: #666;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.4rem;
}

/* Key frame */
.keyframe-section {
    margin-bottom: 2rem;
}
.keyframe-section img {
    width: 100%;
    max-width: 480px;
    display: block;
    margin: 0 auto;
    border-radius: 8px;
    border: 1px solid #333;
}

/* Stoplight report card */
.report-card {
    margin-bottom: 2rem;
}
.report-card h2 {
    color: #f0f0f0;
    border-bottom: 1px solid #333;
    padding-bottom: 0.4rem;
}
.grade-row {
    display: flex;
    align-items: center;
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid #1a1a1a;
}
.grade-dot {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    margin-right: 1rem;
    flex-shrink: 0;
}
.grade-dot.green  { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.grade-dot.amber  { background: #eab308; box-shadow: 0 0 6px #eab308; }
.grade-dot.red    { background: #ef4444; box-shadow: 0 0 6px #ef4444; }
.grade-name {
    flex: 1;
    font-weight: 500;
}
.grade-value {
    color: #aaa;
    font-size: 0.9rem;
    min-width: 80px;
    text-align: right;
}
.grade-target {
    color: #666;
    font-size: 0.8rem;
    min-width: 100px;
    text-align: right;
}

/* Drill homework */
.drills {
    background: #121225;
    border: 1px solid #2a2a4e;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
}
.drills h2 {
    color: #7eb8da;
    margin-top: 0;
    border-bottom: 1px solid #2a2a4e;
    padding-bottom: 0.4rem;
}
.drill-item {
    margin-bottom: 1rem;
}
.drill-item h3 {
    color: #e0e0e0;
    font-size: 1rem;
    margin: 0 0 0.3rem;
}
.drill-item p {
    color: #aaa;
    margin: 0.2rem 0;
    font-size: 0.9rem;
    line-height: 1.5;
}
.drill-item .reps {
    color: #4A90D9;
    font-weight: 600;
    font-size: 0.85rem;
}

/* Coach's note */
.coach-note {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    line-height: 1.7;
}
.coach-note h2 {
    color: #7eb8da;
    margin-top: 0;
    border-bottom: 1px solid #2a2a4e;
    padding-bottom: 0.4rem;
}
.coach-note p {
    margin: 0.5rem 0;
}

/* Growth snapshot */
.growth-snapshot {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    background: #141428;
    border-radius: 6px;
    padding: 1rem 1.5rem;
    margin-bottom: 2rem;
}
.growth-snapshot div {
    min-width: 80px;
}
.growth-snapshot .label {
    color: #888;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.growth-snapshot .value {
    color: #ddd;
    font-weight: 600;
    font-size: 1.1rem;
}

/* Injury watch */
.injury-watch {
    background: #1a1210;
    border: 1px solid #3a2a1a;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 2rem;
    font-size: 0.9rem;
    color: #d4a574;
}
.injury-watch h3 {
    color: #f5a623;
    margin: 0 0 0.5rem;
    font-size: 1rem;
}
.injury-watch p {
    margin: 0.3rem 0;
}

footer {
    text-align: center;
    color: #555;
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #222;
}
"""

# Keep _CSS as an alias for backward compat
_CSS = CSS


def _grade_class(metric_name: str, metrics: PitcherMetrics) -> str:
    """Return CSS class for the grade dot."""
    color = _grade_color(metric_name, metrics)
    if color == _GREEN:
        return "green"
    elif color == _AMBER:
        return "amber"
    return "red"


def _grade_to_pitchzone(css_class: str) -> str:
    """Convert CSS grade class to pitchzone grade string."""
    return {"green": "green", "amber": "yellow", "red": "red"}.get(css_class, "yellow")


def _format_value(metric_name: str, metrics: PitcherMetrics) -> str:
    """Format metric value for display."""
    value = getattr(metrics, metric_name, None)
    if value is None:
        return "--"
    if metric_name == "stride_length_pct_height":
        return f"{value:.0f}%"
    return f"{value:.0f}°"


def _format_target(metric_name: str) -> str:
    """Format ASMI target for display."""
    ideal, tol = GRADE_RULES[metric_name]
    if metric_name == "stride_length_pct_height":
        return f"Target: {ideal-tol:.0f}–{ideal+tol:.0f}%"
    return f"Target: {ideal-tol:.0f}–{ideal+tol:.0f}°"


def _format_height_imperial(inches: float) -> str:
    feet = int(inches // 12)
    remaining = int(inches % 12)
    return f"{feet}'{remaining}\""


def _generate_coach_note(
    metrics: PitcherMetrics,
    grades: dict[str, str],
    pitcher_name: str = "your pitcher",
) -> str:
    """Generate a coach's note using Claude API, with offline fallback.

    Args:
        metrics: Extracted pitcher metrics.
        grades: Metric name → grade class ("green", "amber", "red").
        pitcher_name: Player name for personalization.

    Returns:
        Plain text coaching note (2-3 sentences).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return _coach_note_api(metrics, grades, pitcher_name, api_key)
        except Exception:
            pass
    return _coach_note_offline(grades, pitcher_name)


def _coach_note_api(
    metrics: PitcherMetrics,
    grades: dict[str, str],
    pitcher_name: str,
    api_key: str,
) -> str:
    """Call Claude API for a parent-friendly coaching note."""
    import anthropic

    green_items = [_DISPLAY_NAMES[k] for k, v in grades.items() if v == "green"]
    focus_items = [_DISPLAY_NAMES[k] for k, v in grades.items() if v in ("amber", "red")]

    prompt = (
        f"Write a 2-3 sentence encouraging coaching note for a youth pitcher's parent. "
        f"The pitcher's name is {pitcher_name}. "
        f"Strengths (green): {', '.join(green_items) or 'none identified yet'}. "
        f"Areas to work on: {', '.join(focus_items) or 'looking great overall'}. "
        f"Keep it positive, specific, and parent-friendly. No jargon. "
        f"End with one actionable next step."
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _coach_note_offline(grades: dict[str, str], pitcher_name: str = "your pitcher") -> str:
    """Offline fallback coaching note based on grades."""
    green_items = [_DISPLAY_NAMES[k] for k, v in grades.items() if v == "green"]
    focus_items = [_DISPLAY_NAMES[k] for k, v in grades.items() if v in ("amber", "red")]

    parts: list[str] = []
    if green_items:
        parts.append(
            f"Great work — {pitcher_name} is showing solid mechanics in "
            f"{', '.join(green_items[:2])}."
        )
    else:
        parts.append(
            f"{pitcher_name.title()} is building a good foundation — "
            f"keep focusing on movement quality over velocity."
        )

    if focus_items:
        top_focus = focus_items[0]
        parts.append(
            f"The main area to focus on this week is {top_focus}. "
            f"Use the drill below to work on it during warm-ups."
        )
    else:
        parts.append("All positions look on target — keep up the great work!")

    return " ".join(parts)


def _build_injury_watch(metrics: PitcherMetrics) -> str:
    """Build injury watch HTML if any red-flag patterns are detected."""
    warnings: list[str] = []

    ef = metrics.elbow_flexion_fp
    if ef is not None and (ef < 60 or ef > 120):
        warnings.append(
            "Elbow angle at foot plant is outside the safe range (60–120°). "
            "This can increase stress on the elbow ligament."
        )

    sa = metrics.shoulder_abduction_fp
    if sa is not None and sa > 120:
        warnings.append(
            "Arm is too high at foot plant (above 120°). "
            "This 'high-elbow' pattern can stress the shoulder."
        )

    stride = metrics.stride_length_pct_height
    if stride is not None and stride < 60:
        warnings.append(
            "Stride is very short — the arm may have to work harder "
            "to compensate. Focus on getting more out of the legs."
        )

    if not warnings:
        return ""

    items = "".join(f"<p>{escape(w)}</p>" for w in warnings)
    return f"""<div class="injury-watch">
<h3>Injury Watch</h3>
{items}
<p>These flags don't mean injury is imminent — they highlight patterns worth
monitoring. If your pitcher reports arm pain, consult a sports medicine
professional.</p>
</div>
"""


def build_parent_report_html(
    pitcher_name: str,
    video_filename: str,
    throws: str,
    metrics: PitcherMetrics,
    foot_plant_overlay_b64: Optional[str] = None,
    pitcher_profile: Optional[dict] = None,
    analysis_date: str = "",
) -> str:
    """Build a self-contained HTML parent report with PitchZone visualization.

    Layout:
        1. Header (name, date, handedness)
        2. PitchZone diagram (3D mannequin + colored zone bands + score gauge)
        3. Key frame overlay (actual foot-plant photo, if provided)
        4. Stoplight grade report card
        5. Drill homework (max 3, only amber/red)
        6. Coach's note
        7. Growth snapshot (if pitcher_profile provided)
        8. Injury watch

    Args:
        pitcher_name: Player display name.
        video_filename: Source video filename.
        throws: "R" or "L".
        metrics: Extracted pitcher metrics.
        foot_plant_overlay_b64: Base64 PNG of foot-plant frame with graded overlay.
        pitcher_profile: Dict with age, height_in, weight_lbs, developmental_stage.
        analysis_date: ISO date string for the report header.

    Returns:
        Complete self-contained HTML string.
    """
    parts: list[str] = []

    # ── Document open ──────────────────────────────────────────────────────────────────
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pitching Report — {escape(pitcher_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
""")

    # ── 1. Header ────────────────────────────────────────────────────────────────────
    hand_label = "RHP" if throws == "R" else "LHP"
    date_str = f" · {escape(analysis_date)}" if analysis_date else ""
    parts.append(f'<h1>⚾ {escape(pitcher_name)} — Pitching Report</h1>\n')
    parts.append(
        f'<div class="subtitle">{escape(video_filename)}{date_str} · {hand_label}</div>\n'
    )

    # ── 2. Collect grades first (needed for PitchZone + report card) ─────
    css_grades: dict[str, str] = {}
    for metric_name in GRADE_RULES:
        css_grades[metric_name] = _grade_class(metric_name, metrics)

    # Convert to pitchzone grade strings (green/yellow/red)
    pz_grades: dict[str, str] = {
        m: _grade_to_pitchzone(g) for m, g in css_grades.items()
    }

    # ── 3. PitchZone diagram ──────────────────────────────────────────────────
    pitchzone_svg = generate_pitchzone_svg(
        grades=pz_grades,
        metrics={m: getattr(metrics, m, None) for m in GRADE_RULES},
        throws=throws,
        title="PitchZone",
    )
    parts.append('<div class="pitchzone-section">\n')
    parts.append(pitchzone_svg)
    parts.append('\n<div class="section-label">PitchZone · Foot Plant Analysis</div>\n')
    parts.append('</div>\n')

    # ── 4. Key frame overlay ──────────────────────────────────────────────────
    if foot_plant_overlay_b64:
        parts.append('<div class="keyframe-section">\n')
        parts.append(
            f'<img src="data:image/png;base64,{foot_plant_overlay_b64}" '
            f'alt="Foot Plant with Overlay">\n'
        )
        parts.append('<div class="section-label">Your Foot Plant</div>\n')
        parts.append('</div>\n')

    # ── 5. Stoplight grade report card ───────────────────────────────────────
    parts.append('<div class="report-card">\n<h2>Breakdown</h2>\n')
    for metric_name in GRADE_RULES:
        grade = css_grades[metric_name]
        display = _DISPLAY_NAMES.get(metric_name, metric_name)
        value_str = _format_value(metric_name, metrics)
        target_str = _format_target(metric_name)
        parts.append(f"""<div class="grade-row">
<span class="grade-dot {grade}"></span>
<span class="grade-name">{escape(display)}</span>
<span class="grade-value">{escape(value_str)}</span>
<span class="grade-target">{escape(target_str)}</span>
</div>
""")
    parts.append('</div>\n')

    # ── 6. Drill homework (max 3, only amber/red) ──────────────────────────
    focus_metrics = [m for m, g in css_grades.items() if g in ("amber", "red")][:3]
    if focus_metrics:
        parts.append('<div class="drills">\n<h2>This Week\'s Homework</h2>\n')
        for metric_name in focus_metrics:
            drill = DRILL_MAP.get(metric_name)
            if drill:
                parts.append(f"""<div class="drill-item">
<h3>{escape(drill['name'])}</h3>
<p>{escape(drill['description'])}</p>
<p class="reps">{escape(drill['reps'])}</p>
</div>
""")
        parts.append('</div>\n')

    # ── 7. Coach's note ─────────────────────────────────────────────────────────
    coach_text = _generate_coach_note(metrics, css_grades, pitcher_name)
    parts.append(f"""<div class="coach-note">
<h2>Coach's Note</h2>
<p>{escape(coach_text)}</p>
</div>
""")

    # ── 8. Growth snapshot ─────────────────────────────────────────────────────────
    if pitcher_profile:
        age = pitcher_profile.get("age", "")
        height_in = pitcher_profile.get("height_in")
        weight_lbs = pitcher_profile.get("weight_lbs", "")
        stage = pitcher_profile.get("developmental_stage", "")
        height_display = _format_height_imperial(height_in) if height_in else ""
        stage_display = stage.replace("_", " ").title() if stage else ""

        parts.append(f"""<h2>Growth Snapshot</h2>
<div class="growth-snapshot">
<div><span class="label">Age</span><br><span class="value">{escape(str(age))}</span></div>
<div><span class="label">Height</span><br><span class="value">{escape(height_display)}</span></div>
<div><span class="label">Weight</span><br><span class="value">{escape(str(weight_lbs))} lbs</span></div>
<div><span class="label">Stage</span><br><span class="value">{escape(stage_display)}</span></div>
</div>
""")

    # ── 9. Injury watch ───────────────────────────────────────────────────────────
    injury_html = _build_injury_watch(metrics)
    if injury_html:
        parts.append(injury_html)

    # ── Footer ────────────────────────────────────────────────────────────────────
    parts.append("""<footer>
Generated by Pitch Mechanics Analyzer. Movement analysis from video is approximate —
consult a qualified coach for a complete evaluation.
</footer>
""")

    parts.append('</div>\n</body>\n</html>')
    return "".join(parts)
