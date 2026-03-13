"""SwingAI-inspired PitchZone visualization for pitcher mechanics at foot plant.

Generates a self-contained inline SVG of a mannequin-style pitcher figure at
foot plant position with translucent colored zone bands (green/yellow/red) around
each graded body region.  Dark background (#0a0a0a), 3D-ish perspective floor grid,
SVG filter glow effects on zone bands.  Mirrored for LHP.

Overall PitchZone Score displayed as a donut arc gauge (0–100).
"""

from __future__ import annotations

import math
from typing import Optional


# ── Grade colours ─────────────────────────────────────────────────────────────────────────
_GREEN  = "#22c55e"
_YELLOW = "#eab308"
_RED    = "#ef4444"

_GRADE_COLOR: dict[str, str] = {
    "green":  _GREEN,
    "yellow": _YELLOW,
    "red":    _RED,
}

_GRADE_SCORE: dict[str, int] = {
    "green":  100,
    "yellow": 65,
    "red":    30,
}

_GRADE_WORD: dict[str, str] = {
    "green":  "Excellent",
    "yellow": "Good",
    "red":    "Focus",
}


# ── Zone band metadata ─────────────────────────────────────────────────────────────────────────
ZONE_BANDS: dict[str, dict] = {
    "shoulder_abduction_fp": {
        "label":    "Arm Height",
        "region":   "shoulder",
        "excellent": "Arm at perfect height",
        "good":      "Nearly there",
        "focus":     "Needs work",
    },
    "elbow_flexion_fp": {
        "label":    "Elbow Bend",
        "region":   "elbow",
        "excellent": "Perfect 'L' shape",
        "good":      "Almost right",
        "focus":     "Needs attention",
    },
    "torso_anterior_tilt_fp": {
        "label":    "Posture",
        "region":   "torso",
        "excellent": "Staying tall",
        "good":      "Slight lean",
        "focus":     "Too much lean",
    },
    "hip_shoulder_separation_fp": {
        "label":    "Hip Lead",
        "region":   "hip",
        "excellent": "Great separation",
        "good":      "Getting there",
        "focus":     "Opening together",
    },
    "stride_length_pct_height": {
        "label":    "Stride",
        "region":   "stride_leg",
        "excellent": "Great reach",
        "good":      "Almost enough",
        "focus":     "Need more reach",
    },
    "lead_knee_angle_fp": {
        "label":    "Front Leg",
        "region":   "lead_knee",
        "excellent": "Firm brace",
        "good":      "Mostly firm",
        "focus":     "Too soft",
    },
}

# Ordered list for consistent label placement
_BAND_ORDER = [
    "shoulder_abduction_fp",
    "elbow_flexion_fp",
    "torso_anterior_tilt_fp",
    "hip_shoulder_separation_fp",
    "stride_length_pct_height",
    "lead_knee_angle_fp",
]


# ── SVG canvas constants ──────────────────────────────────────────────────────────────────────────
_VW = 700   # viewBox width
_VH = 600   # viewBox height


# ── RHP mannequin joint positions (in figure sub-canvas 0,0 → 400,560) ────────
# Figure is centered at x=200, occupies y≈30–530 within the sub-canvas.
# Named after mechanical role, not body-side, to simplify mirroring.
#
#  throw_*  = throwing arm side  (right for RHP)
#  lead_*   = stride/glove side  (left  for RHP)
#
_RHP: dict[str, tuple[float, float]] = {
    # Head / neck
    "head":              (200, 48),
    "neck":              (200, 75),
    # Shoulders
    "throw_shoulder":    (168, 100),
    "lead_shoulder":     (232, 100),
    # Throwing arm: cocked-back position (arm up, elbow bent ~90°)
    "throw_elbow":       (132, 78),
    "throw_wrist":       (118, 50),
    # Glove arm: extended forward at ~shoulder height
    "lead_elbow":        (268, 112),
    "lead_wrist":        (298, 128),
    # Torso centerline
    "chest":             (200, 120),
    "torso_mid":         (200, 175),
    "pelvis":            (200, 220),
    # Hips
    "throw_hip":         (182, 228),
    "lead_hip":          (218, 228),
    # Throw leg (back leg / pivot leg): knee bent, foot planted behind
    "throw_knee":        (175, 308),
    "throw_ankle":       (168, 388),
    "throw_foot_toe":    (148, 402),
    "throw_foot_heel":   (172, 400),
    # Lead leg (stride leg): forward and planted at foot plant
    "lead_knee":         (295, 300),
    "lead_ankle":        (348, 372),
    "lead_foot_toe":     (372, 382),
    "lead_foot_heel":    (344, 376),
}

# Bone connections for the mannequin skeleton
_BONES: list[tuple[str, str]] = [
    # Spine / torso column
    ("neck",          "chest"),
    ("chest",         "torso_mid"),
    ("torso_mid",     "pelvis"),
    # Shoulders
    ("neck",          "throw_shoulder"),
    ("neck",          "lead_shoulder"),
    # Throwing arm
    ("throw_shoulder","throw_elbow"),
    ("throw_elbow",   "throw_wrist"),
    # Glove arm
    ("lead_shoulder", "lead_elbow"),
    ("lead_elbow",    "lead_wrist"),
    # Hips cross
    ("pelvis",        "throw_hip"),
    ("pelvis",        "lead_hip"),
    ("throw_hip",     "lead_hip"),
    # Pivot leg
    ("throw_hip",     "throw_knee"),
    ("throw_knee",    "throw_ankle"),
    ("throw_ankle",   "throw_foot_toe"),
    ("throw_ankle",   "throw_foot_heel"),
    # Lead leg
    ("lead_hip",      "lead_knee"),
    ("lead_knee",     "lead_ankle"),
    ("lead_ankle",    "lead_foot_toe"),
    ("lead_ankle",    "lead_foot_heel"),
]


def _mirror(joints: dict[str, tuple[float, float]], cx: float = 200.0
            ) -> dict[str, tuple[float, float]]:
    """Mirror joint x-coordinates around cx for LHP."""
    return {k: (2 * cx - x, y) for k, (x, y) in joints.items()}


# ── Score calculation ───────────────────────────────────────────────────────────────────────────

def calculate_pitchzone_score(grades: dict[str, str]) -> int:
    """Compute overall PitchZone score (0–100) as equal-weight average.

    Args:
        grades: metric_name → "green" / "yellow" / "red".
                Missing/unknown keys default to "yellow".
    Returns:
        Integer score 0–100.
    """
    if not grades:
        return 65  # default mid-range when no data
    scores = [_GRADE_SCORE.get(g, _GRADE_SCORE["yellow"]) for g in grades.values()]
    return round(sum(scores) / len(scores))


# ── SVG primitive helpers ────────────────────────────────────────────────────────────────────────

def _pt(joints: dict, name: str) -> tuple[float, float]:
    return joints[name]


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float = 0.5
          ) -> tuple[float, float]:
    return (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0]-a[0], b[1]-a[1])


def _perpendicular_offset(
    a: tuple[float, float], b: tuple[float, float], d: float
) -> tuple[float, float]:
    """Return a unit perpendicular to segment a→b, scaled by d."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    length = math.hypot(dx, dy) or 1
    return (-dy/length * d, dx/length * d)


def _ellipse(cx: float, cy: float, rx: float, ry: float,
             fill: str, opacity: float = 1.0, **kw) -> str:
    extra = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return (
        f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
        f'fill="{fill}" opacity="{opacity:.2f}" {extra}/>'
    )


def _path(d: str, fill: str = "none", stroke: str = "none",
          stroke_width: float = 1, opacity: float = 1.0, **kw) -> str:
    extra = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return (
        f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width:.1f}" opacity="{opacity:.2f}" {extra}/>'
    )


def _line_svg(x1: float, y1: float, x2: float, y2: float,
              stroke: str, width: float = 2, linecap: str = "round",
              opacity: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}" stroke-linecap="{linecap}" '
        f'opacity="{opacity:.2f}"/>'
    )


def _circle_svg(cx: float, cy: float, r: float, fill: str,
                opacity: float = 1.0, **kw) -> str:
    extra = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
        f'fill="{fill}" opacity="{opacity:.2f}" {extra}/>'
    )


def _text_svg(x: float, y: float, text: str, size: float = 12,
              fill: str = "#e0e0e0", anchor: str = "middle",
              weight: str = "normal", **kw) -> str:
    extra = " ".join(f'{k}="{v}"' for k, v in kw.items())
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="system-ui,sans-serif" '
        f'font-size="{size:.1f}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}" dominant-baseline="middle" {extra}>{text}</text>'
    )


# ── SVG filter definitions ────────────────────────────────────────────────────────────────────────

def _build_filters() -> str:
    """Return <defs> block with glow filters for each zone color."""
    filters = []
    for fid, color in [("glow-green", _GREEN), ("glow-yellow", _YELLOW), ("glow-red", _RED)]:
        filters.append(f"""
  <filter id="{fid}" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur stdDeviation="6" result="blur"/>
    <feFlood flood-color="{color}" flood-opacity="0.8" result="color"/>
    <feComposite in="color" in2="blur" operator="in" result="glow"/>
    <feMerge>
      <feMergeNode in="glow"/>
      <feMergeNode in="SourceGraphic"/>
    </feMerge>
  </filter>""")
    # Subtle body shadow
    filters.append("""
  <filter id="body-shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="3" dy="4" stdDeviation="5" flood-color="#000" flood-opacity="0.6"/>
  </filter>""")
    return "<defs>\n" + "\n".join(filters) + "\n</defs>"


def _glow_filter_id(color_key: str) -> str:
    return {"green": "glow-green", "yellow": "glow-yellow", "red": "glow-red"}.get(
        color_key, "glow-yellow"
    )


# ── Perspective floor grid ────────────────────────────────────────────────────────────────────────

def _build_floor_grid(ox: float = 350, oy: float = 500,
                      vp_x: float = 200, vp_y: float = 330) -> str:
    """
    Perspective grid on the bottom portion of the canvas.
    Lines converge toward vanishing point (vp_x, vp_y).
    Grid is drawn on a dark base rectangle.
    """
    parts: list[str] = []
    # Background floor plane
    parts.append(
        f'<polygon points="0,{oy} {_VW},{oy} {_VW},{_VH} 0,{_VH}" '
        f'fill="#080808"/>'
    )

    grid_color = "#1e1e1e"
    n_horiz = 7
    n_vert  = 10

    # Horizontal lines (depth lines)
    for i in range(n_horiz + 1):
        t = i / n_horiz
        y = oy + t * (_VH - oy)
        parts.append(_line_svg(0, y, _VW, y, grid_color, 0.8, opacity=0.6 + 0.4*t))

    # Vertical lines converging to VP
    left_x  = 0
    right_x = _VW
    bottom_y = _VH

    for i in range(n_vert + 1):
        t = i / n_vert
        bx = left_x + t * (right_x - left_x)
        # Line from bottom edge converging toward VP
        parts.append(_line_svg(bx, bottom_y, vp_x, vp_y, grid_color, 0.8, opacity=0.5))

    # Horizon glow line
    parts.append(
        f'<line x1="0" y1="{oy}" x2="{_VW}" y2="{oy}" '
        f'stroke="#2a2a3a" stroke-width="1.5" opacity="0.9"/>'
    )

    return "\n".join(parts)


# ── Mannequin body drawing ────────────────────────────────────────────────────────────────────────

def _build_mannequin(joints: dict[str, tuple[float, float]],
                     fig_ox: float, fig_oy: float) -> str:
    """
    Build the filled-polygon mannequin body using translated joint positions.
    Returns SVG fragment string.
    """
    # Translate all joints
    j: dict[str, tuple[float, float]] = {
        k: (x + fig_ox, y + fig_oy) for k, (x, y) in joints.items()
    }

    parts: list[str] = []

    # ── Color palette for mannequin body parts ─────────────────
    body_fill   = "#a8a8b0"   # light steel — base body color
    shadow_fill = "#787888"   # darker shaded faces
    hi_fill     = "#d0d0dc"   # highlight faces
    joint_fill  = "#c8c8d8"

    def _pts(*names: str) -> str:
        return " ".join(f"{j[n][0]:.1f},{j[n][1]:.1f}" for n in names)

    def _poly(*names: str, fill: str = body_fill,
              opacity: float = 1.0, filt: str = "") -> str:
        filt_attr = f' filter="url(#{filt})"' if filt else ""
        return (
            f'<polygon points="{_pts(*names)}" fill="{fill}" '
            f'opacity="{opacity:.2f}"{filt_attr}/>'
        )

    # ── Head ─────────────────────────────────────────────────────
    hx, hy = j["head"]
    nx, ny = j["neck"]
    head_r = 18
    head_ry = 20
    # Head sphere (slightly flattened ellipse)
    parts.append(_ellipse(hx, hy, head_r, head_ry, body_fill, 1.0,
                          filter="url(#body-shadow)"))
    # Highlight top
    parts.append(_ellipse(hx-4, hy-6, 8, 6, hi_fill, 0.5))
    # Neck
    nw = 8
    parts.append(
        f'<polygon points="{hx-nw/2:.1f},{hy+head_ry-4:.1f} '
        f'{hx+nw/2:.1f},{hy+head_ry-4:.1f} '
        f'{nx+nw/2:.1f},{ny:.1f} {nx-nw/2:.1f},{ny:.1f}" '
        f'fill="{body_fill}"/>'
    )

    # ── Torso ─────────────────────────────────────────────────────
    ts  = j["throw_shoulder"]
    ls  = j["lead_shoulder"]
    th  = j["throw_hip"]
    lh  = j["lead_hip"]
    pv  = j["pelvis"]
    chest = j["chest"]
    torso_mid = j["torso_mid"]

    # Main torso quad (neck to hips)
    shoulder_w = 30
    hip_w = 22
    parts.append(
        f'<polygon points="'
        f'{ts[0]-2:.1f},{ts[1]+4:.1f} '
        f'{ls[0]+2:.1f},{ls[1]+4:.1f} '
        f'{lh[0]+10:.1f},{lh[1]+4:.1f} '
        f'{th[0]-10:.1f},{th[1]+4:.1f}" '
        f'fill="{shadow_fill}" filter="url(#body-shadow)"/>'
    )
    parts.append(
        f'<polygon points="'
        f'{ts[0]-2:.1f},{ts[1]:.1f} '
        f'{ls[0]+2:.1f},{ls[1]:.1f} '
        f'{lh[0]+10:.1f},{lh[1]:.1f} '
        f'{th[0]-10:.1f},{th[1]:.1f}" '
        f'fill="{body_fill}"/>'
    )
    # Highlight stripe down the center of torso
    parts.append(
        f'<polygon points="'
        f'{(ts[0]+ls[0])/2-6:.1f},{ts[1]:.1f} '
        f'{(ts[0]+ls[0])/2+6:.1f},{ts[1]:.1f} '
        f'{(th[0]+lh[0])/2+4:.1f},{th[1]:.1f} '
        f'{(th[0]+lh[0])/2-4:.1f},{th[1]:.1f}" '
        f'fill="{hi_fill}" opacity="0.35"/>'
    )

    # ── Throwing arm ────────────────────────────────────────────────
    te = j["throw_elbow"]
    tw = j["throw_wrist"]
    # Upper arm (shoulder → elbow)
    _draw_limb_segment(parts, ts, te, 9, 7, body_fill, shadow_fill)
    # Forearm (elbow → wrist)
    _draw_limb_segment(parts, te, tw, 7, 5, shadow_fill, body_fill)
    # Elbow joint sphere
    parts.append(_circle_svg(*te, 9, joint_fill))

    # ── Glove arm ─────────────────────────────────────────────────
    le  = j["lead_elbow"]
    lw  = j["lead_wrist"]
    _draw_limb_segment(parts, ls, le, 9, 7, shadow_fill, body_fill)
    _draw_limb_segment(parts, le, lw, 7, 5, body_fill, shadow_fill)
    parts.append(_circle_svg(*le, 9, joint_fill))

    # ── Hips / pelvis block ─────────────────────────────────────────────
    parts.append(
        f'<polygon points="'
        f'{th[0]-10:.1f},{th[1]:.1f} '
        f'{lh[0]+10:.1f},{lh[1]:.1f} '
        f'{lh[0]+8:.1f},{lh[1]+16:.1f} '
        f'{th[0]-8:.1f},{th[1]+16:.1f}" '
        f'fill="{shadow_fill}" opacity="0.9"/>'
    )

    # ── Throwing (pivot) leg ──────────────────────────────────────────────
    tk  = j["throw_knee"]
    ta  = j["throw_ankle"]
    tft = j["throw_foot_toe"]
    tfh = j["throw_foot_heel"]
    _draw_limb_segment(parts, th, tk, 11, 9, shadow_fill, body_fill)
    _draw_limb_segment(parts, tk, ta, 9, 7, body_fill, shadow_fill)
    parts.append(_circle_svg(*tk, 10, joint_fill))
    # Foot
    parts.append(
        f'<polygon points="{ta[0]:.1f},{ta[1]:.1f} '
        f'{tft[0]:.1f},{tft[1]:.1f} '
        f'{tfh[0]:.1f},{tfh[1]:.1f}" '
        f'fill="{shadow_fill}"/>'
    )

    # ── Lead (stride) leg ─────────────────────────────────────────────────
    lk  = j["lead_knee"]
    la  = j["lead_ankle"]
    lft = j["lead_foot_toe"]
    lfh = j["lead_foot_heel"]
    _draw_limb_segment(parts, lh, lk, 11, 9, body_fill, shadow_fill)
    _draw_limb_segment(parts, lk, la, 9, 7, shadow_fill, body_fill)
    parts.append(_circle_svg(*lk, 10, joint_fill))
    # Foot
    parts.append(
        f'<polygon points="{la[0]:.1f},{la[1]:.1f} '
        f'{lft[0]:.1f},{lft[1]:.1f} '
        f'{lfh[0]:.1f},{lfh[1]:.1f}" '
        f'fill="{body_fill}"/>'
    )

    # ── Shoulder joint caps ──────────────────────────────────────────────
    parts.append(_circle_svg(*ts, 11, joint_fill))
    parts.append(_circle_svg(*ls, 11, joint_fill))

    # Wrist / hand nubs
    parts.append(_ellipse(tw[0], tw[1], 6, 7, joint_fill))
    parts.append(_ellipse(lw[0], lw[1], 6, 7, joint_fill))

    # ── Cap (hat brim) for visual interest ──────────────────────────────────
    brim_y = hy - head_ry + 2
    parts.append(
        f'<ellipse cx="{hx:.1f}" cy="{brim_y:.1f}" '
        f'rx="20" ry="5" fill="#303040" opacity="0.9"/>'
    )
    parts.append(
        f'<rect x="{hx-14:.1f}" y="{brim_y-14:.1f}" '
        f'width="28" height="14" rx="6" fill="#303040" opacity="0.9"/>'
    )

    return "\n".join(parts)


def _draw_limb_segment(
    parts: list[str],
    a: tuple[float, float],
    b: tuple[float, float],
    w_a: float, w_b: float,
    fill_front: str, fill_back: str,
) -> None:
    """Draw a tapered limb segment (quad) between joints a and b."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy) or 1
    nx_unit = -dy / length
    ny_unit =  dx / length

    p1 = (a[0] + nx_unit * w_a, a[1] + ny_unit * w_a)
    p2 = (a[0] - nx_unit * w_a, a[1] - ny_unit * w_a)
    p3 = (b[0] - nx_unit * w_b, b[1] - ny_unit * w_b)
    p4 = (b[0] + nx_unit * w_b, b[1] + ny_unit * w_b)

    def fmt(p: tuple[float, float]) -> str:
        return f"{p[0]:.1f},{p[1]:.1f}"

    # Shadow face
    parts.append(
        f'<polygon points="{fmt(p2)} {fmt(p3)} {fmt(p4)} {fmt(p1)}" '
        f'fill="{fill_back}" opacity="0.7"/>'
    )
    # Front face
    parts.append(
        f'<polygon points="{fmt(p1)} {fmt(p2)} {fmt(p3)} {fmt(p4)}" '
        f'fill="{fill_front}"/>'
    )


# ── Zone band drawing ────────────────────────────────────────────────────────────────────────────

def _arc_band(cx: float, cy: float, rx: float, ry: float,
              start_deg: float, end_deg: float,
              color: str, width: float = 18,
              opacity: float = 0.35, filter_id: str = "") -> str:
    """
    Draw a thick arc band (elliptical) around a joint.
    Rendered as a filled arc sweep with inner/outer radii.
    """
    parts: list[str] = []
    filt = f' filter="url(#{filter_id})"' if filter_id else ""

    def _arc_pt(rx_: float, ry_: float, deg: float) -> tuple[float, float]:
        rad = math.radians(deg)
        return (cx + rx_ * math.cos(rad), cy + ry_ * math.sin(rad))

    inner_rx = max(rx - width/2, rx * 0.55)
    inner_ry = max(ry - width/2, ry * 0.55)
    outer_rx = rx + width/2
    outer_ry = ry + width/2

    steps = max(24, int(abs(end_deg - start_deg) / 4))
    angles = [start_deg + (end_deg - start_deg) * i / steps for i in range(steps + 1)]

    outer_pts = [_arc_pt(outer_rx, outer_ry, a) for a in angles]
    inner_pts = [_arc_pt(inner_rx, inner_ry, a) for a in reversed(angles)]

    all_pts = outer_pts + inner_pts
    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in all_pts)

    parts.append(
        f'<polygon points="{pts_str}" fill="{color}" '
        f'opacity="{opacity:.2f}"{filt}/>'
    )
    # Bright inner stroke for crispness
    outer_d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in outer_pts)
    parts.append(
        f'<path d="{outer_d}" fill="none" stroke="{color}" '
        f'stroke-width="2" opacity="{min(opacity*2.2, 0.85):.2f}"{filt}/>'
    )
    return "\n".join(parts)


def _build_zone_bands(
    joints: dict[str, tuple[float, float]],
    grades: dict[str, str],
    fig_ox: float, fig_oy: float,
) -> str:
    """Draw translucent colored zone bands around each graded body region."""
    parts: list[str] = []

    def jt(name: str) -> tuple[float, float]:
        x, y = joints[name]
        return (x + fig_ox, y + fig_oy)

    def grade(metric: str) -> str:
        return grades.get(metric, "yellow")

    def color(metric: str) -> str:
        return _GRADE_COLOR[grade(metric)]

    def filt(metric: str) -> str:
        return _glow_filter_id(grade(metric))

    # ── Shoulder band ───────────────────────────────────────────────
    ts = jt("throw_shoulder")
    g_sa = grade("shoulder_abduction_fp")
    parts.append(_arc_band(
        ts[0], ts[1], 34, 26,
        start_deg=-210, end_deg=-30,
        color=color("shoulder_abduction_fp"),
        width=22, opacity=0.38,
        filter_id=filt("shoulder_abduction_fp"),
    ))

    # ── Elbow band ───────────────────────────────────────────────────
    te = jt("throw_elbow")
    parts.append(_arc_band(
        te[0], te[1], 26, 20,
        start_deg=-170, end_deg=20,
        color=color("elbow_flexion_fp"),
        width=18, opacity=0.38,
        filter_id=filt("elbow_flexion_fp"),
    ))

    # ── Torso band (wide band across chest area) ─────────────────────
    chest = jt("chest")
    ts_  = jt("throw_shoulder")
    ls_  = jt("lead_shoulder")
    mid_x = (ts_[0] + ls_[0]) / 2
    mid_y = (ts_[1] + ls_[1]) / 2
    parts.append(_arc_band(
        mid_x, chest[1] + 12, 42, 18,
        start_deg=-175, end_deg=5,
        color=color("torso_anterior_tilt_fp"),
        width=28, opacity=0.30,
        filter_id=filt("torso_anterior_tilt_fp"),
    ))

    # ── Hip / separation band ─────────────────────────────────────────
    th = jt("throw_hip")
    lh = jt("lead_hip")
    hip_cx = (th[0] + lh[0]) / 2
    hip_cy = (th[1] + lh[1]) / 2
    parts.append(_arc_band(
        hip_cx, hip_cy, 44, 16,
        start_deg=-180, end_deg=0,
        color=color("hip_shoulder_separation_fp"),
        width=22, opacity=0.36,
        filter_id=filt("hip_shoulder_separation_fp"),
    ))

    # ── Stride leg band (along the lead thigh) ────────────────────────
    lk = jt("lead_knee")
    la = jt("lead_ankle")
    mid_leg = ((lh[0]+lk[0])/2, (lh[1]+lk[1])/2 + 4)
    parts.append(_arc_band(
        mid_leg[0], mid_leg[1], 30, 18,
        start_deg=20, end_deg=180,
        color=color("stride_length_pct_height"),
        width=22, opacity=0.35,
        filter_id=filt("stride_length_pct_height"),
    ))

    # ── Lead knee band ───────────────────────────────────────────────
    parts.append(_arc_band(
        lk[0], lk[1], 28, 22,
        start_deg=10, end_deg=190,
        color=color("lead_knee_angle_fp"),
        width=20, opacity=0.38,
        filter_id=filt("lead_knee_angle_fp"),
    ))

    return "\n".join(parts)


# ── Side labels ─────────────────────────────────────────────────────────────────────────────

def _build_labels(
    joints: dict[str, tuple[float, float]],
    grades: dict[str, str],
    fig_ox: float, fig_oy: float,
) -> str:
    """
    Draw zone labels in two columns (left and right) of the figure.
    Each label has a colored circle + name + one-word grade.
    """
    parts: list[str] = []

    def jt(name: str) -> tuple[float, float]:
        x, y = joints[name]
        return (x + fig_ox, y + fig_oy)

    # Label positions: (metric_key, label_x, label_y, anchor)
    # Left column: shoulder, elbow, stride  → anchor start, left of figure
    # Right column: torso, hip, front leg  → anchor start, right of figure
    # fig_ox=130; figure spans ~x:130-510 (lead foot toe at ~372+130=502)
    left_x  = fig_ox - 12   # right-anchored
    right_x = fig_ox + 380  # left-anchored, well clear of rightmost joint

    # Compute raw Y positions, then de-overlap within each column.
    # Each label needs ~28px vertical space (name + grade + gap).
    MIN_GAP = 28

    raw_left = [
        ("elbow_flexion_fp",         jt("throw_elbow")[1] - 20),
        ("shoulder_abduction_fp",    jt("throw_shoulder")[1] + 10),
        ("stride_length_pct_height", jt("lead_knee")[1] + 30),
    ]
    raw_right = [
        ("torso_anterior_tilt_fp",     jt("chest")[1] + 10),
        ("hip_shoulder_separation_fp", jt("pelvis")[1] + 8),
        ("lead_knee_angle_fp",         jt("lead_knee")[1] + 5),
    ]

    def _deoverlap(specs: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Push labels apart so none overlap within a column."""
        sorted_specs = sorted(specs, key=lambda s: s[1])
        result = []
        prev_y = -999.0
        for metric, y in sorted_specs:
            if y - prev_y < MIN_GAP:
                y = prev_y + MIN_GAP
            result.append((metric, y))
            prev_y = y
        return result

    label_specs = [
        (m, "left", y) for m, y in _deoverlap(raw_left)
    ] + [
        (m, "right", y) for m, y in _deoverlap(raw_right)
    ]

    for metric, side, lbl_y in label_specs:
        g = grades.get(metric, "yellow")
        color = _GRADE_COLOR[g]
        word  = _GRADE_WORD[g]
        name  = ZONE_BANDS[metric]["label"]

        if side == "left":
            dot_x  = left_x - 100
            anchor = "start"
            txt_x  = dot_x + 14
        else:
            dot_x  = right_x
            anchor = "start"
            txt_x  = dot_x + 14

        # Connector line to body (subtle)
        body_x = jt("throw_shoulder")[0] - 15 if side == "left" else right_x - 8
        parts.append(
            f'<line x1="{dot_x+7:.1f}" y1="{lbl_y:.1f}" x2="{body_x:.1f}" y2="{lbl_y:.1f}" '
            f'stroke="{color}" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.4"/>'
        )

        # Colored dot with glow
        fid = _glow_filter_id(g)
        parts.append(
            f'<circle cx="{dot_x+7:.1f}" cy="{lbl_y:.1f}" r="5" '
            f'fill="{color}" opacity="0.9" filter="url(#{fid})"/>'
        )

        # Label name
        parts.append(_text_svg(
            txt_x, lbl_y - 5, name, size=11, fill="#d0d0d0",
            anchor="start", weight="600",
        ))
        # One-word grade
        parts.append(_text_svg(
            txt_x, lbl_y + 8, word, size=10, fill=color,
            anchor="start",
        ))

    return "\n".join(parts)


# ── Donut score gauge ───────────────────────────────────────────────────────────────────────────

def _build_score_gauge(score: int, cx: float, cy: float, r: float = 46) -> str:
    """
    Build a circular donut arc gauge showing the overall PitchZone score.
    The arc sweeps from 225° to 315° (270° total, clockwise), colored by score.
    """
    parts: list[str] = []

    # Score range → color
    if score >= 85:
        arc_color = _GREEN
        filt_id   = "glow-green"
    elif score >= 60:
        arc_color = _YELLOW
        filt_id   = "glow-yellow"
    else:
        arc_color = _RED
        filt_id   = "glow-red"

    ring_w = 10
    bg_r   = r
    inner_r = r - ring_w

    # Background track (full 270° gray arc)
    start_rad = math.radians(135)   # 225° but SVG angles: start at -135 from east
    sweep_deg = 270.0
    start_a   = 135   # degrees from east, clockwise
    end_a     = 135 + 270

    def _arc_pts(radius: float, a_start: float, a_end: float,
                 steps: int = 60) -> list[tuple[float, float]]:
        pts = []
        for i in range(steps + 1):
            t   = i / steps
            deg = a_start + t * (a_end - a_start)
            rad = math.radians(deg)
            pts.append((cx + radius * math.cos(rad), cy + radius * math.sin(rad)))
        return pts

    # Full track (gray)
    outer_bg = _arc_pts(bg_r, start_a, end_a)
    inner_bg = _arc_pts(inner_r, start_a, end_a)
    bg_poly  = outer_bg + list(reversed(inner_bg))
    bg_pts   = " ".join(f"{x:.1f},{y:.1f}" for x, y in bg_poly)
    parts.append(
        f'<polygon points="{bg_pts}" fill="#2a2a3a" opacity="0.8"/>'
    )

    # Score arc
    score_sweep = sweep_deg * score / 100
    outer_sc = _arc_pts(bg_r,    start_a, start_a + score_sweep)
    inner_sc = _arc_pts(inner_r, start_a, start_a + score_sweep)
    sc_poly  = outer_sc + list(reversed(inner_sc))
    sc_pts   = " ".join(f"{x:.1f},{y:.1f}" for x, y in sc_poly)
    parts.append(
        f'<polygon points="{sc_pts}" fill="{arc_color}" '
        f'opacity="0.9" filter="url(#{filt_id})"/>'
    )

    # Center circle (dark)
    parts.append(_circle_svg(cx, cy, inner_r - 2, "#111118"))

    # Score number
    parts.append(_text_svg(cx, cy - 4, str(score), size=26,
                           fill="#f0f0f0", weight="700"))
    parts.append(_text_svg(cx, cy + 16, "PitchZone", size=9,
                           fill="#888888"))

    # End cap dots
    start_pt_o = (cx + bg_r * math.cos(math.radians(start_a)),
                  cy + bg_r * math.sin(math.radians(start_a)))
    end_pt_o   = (cx + bg_r * math.cos(math.radians(start_a + score_sweep)),
                  cy + bg_r * math.sin(math.radians(start_a + score_sweep)))
    parts.append(_circle_svg(*start_pt_o, ring_w/2+1, arc_color, opacity=0.7))
    if score > 2:
        parts.append(_circle_svg(*end_pt_o, ring_w/2+1, arc_color, opacity=0.9,
                                 filter=f"url(#{filt_id})"))

    return "\n".join(parts)


# ── Title / header area ──────────────────────────────────────────────────────────────────────────

def _build_header(score: int, throws: str, title: str = "PitchZone") -> str:
    parts: list[str] = []

    # Section title left
    parts.append(_text_svg(16, 30, title, size=20, fill="#e8e8f0",
                           anchor="start", weight="700"))
    hand_label = "RHP" if throws == "R" else "LHP"
    parts.append(_text_svg(16, 50, hand_label, size=11, fill="#666688",
                           anchor="start"))

    # Overall score label  (right-aligned)
    if score >= 85:
        sc_color = _GREEN
    elif score >= 60:
        sc_color = _YELLOW
    else:
        sc_color = _RED

    # Gauge positioned top-right
    gauge_cx = _VW - 72
    gauge_cy = 62
    parts.append(_build_score_gauge(score, gauge_cx, gauge_cy, r=46))

    return "\n".join(parts)


# ── Main public function ───────────────────────────────────────────────────────────────────────────

def generate_pitchzone_svg(
    grades: dict[str, str],
    metrics: Optional[dict] = None,
    throws: str = "R",
    title: str = "PitchZone",
    width: int = 700,
    height: int = 600,
) -> str:
    """Generate a SwingAI-inspired SVG of a pitcher at foot plant with
    colored zone bands around each body region.

    Args:
        grades:  dict mapping metric_name → "green" / "yellow" / "red".
                 Any missing metric defaults to "yellow".
        metrics: dict mapping metric_name → actual float value (unused in SVG,
                 reserved for future tooltip use).
        throws:  "R" for RHP (default) or "L" for LHP (mirrors the figure).
        title:   Label shown in the SVG header.
        width:   SVG width attribute in pixels.
        height:  SVG height attribute in pixels.

    Returns:
        Complete inline SVG string (no XML declaration, suitable for HTML).
    """
    # Normalise grades to lowercase, fill missing with "yellow"
    norm_grades: dict[str, str] = {}
    for metric in ZONE_BANDS:
        raw = grades.get(metric, "yellow")
        if isinstance(raw, str):
            raw = raw.lower()
        if raw not in ("green", "yellow", "red"):
            raw = "yellow"
        norm_grades[metric] = raw

    score  = calculate_pitchzone_score(norm_grades)
    joints = _RHP if throws == "R" else _mirror(_RHP, cx=200.0)

    # Figure offset within canvas:
    # The figure sub-canvas is 400×430 px; we place it slightly left of centre
    # to leave room for right-column labels.
    fig_ox = 130.0   # left edge offset
    fig_oy =  85.0   # top offset (below header)

    parts: list[str] = []

    # ── SVG open ───────────────────────────────────────────────────
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {_VW} {_VH}" '
        f'style="background:#0a0a0a;border-radius:10px;display:block;">'
    )

    # Filters
    parts.append(_build_filters())

    # Background
    parts.append(f'<rect width="{_VW}" height="{_VH}" fill="#0a0a0a"/>')

    # Subtle radial vignette
    parts.append(
        f'<radialGradient id="vignette" cx="50%" cy="50%" r="70%">'
        f'<stop offset="0%" stop-color="#141420" stop-opacity="0"/>'
        f'<stop offset="100%" stop-color="#000000" stop-opacity="0.7"/>'
        f'</radialGradient>'
        f'<rect width="{_VW}" height="{_VH}" fill="url(#vignette)"/>'
    )

    # Perspective floor grid (bottom third)
    floor_y = fig_oy + 420   # approximate ground line
    floor_y = min(floor_y, int(_VH * 0.78))
    parts.append(_build_floor_grid(ox=350, oy=floor_y, vp_x=_VW*0.38, vp_y=floor_y-100))

    # Header / score gauge
    parts.append(_build_header(score, throws, title))

    # Divider line under header
    parts.append(
        f'<line x1="0" y1="72" x2="{_VW}" y2="72" '
        f'stroke="#1e1e2e" stroke-width="1"/>'
    )

    # Zone bands (drawn BEHIND the figure)
    parts.append(_build_zone_bands(joints, norm_grades, fig_ox, fig_oy))

    # Mannequin figure
    parts.append(_build_mannequin(joints, fig_ox, fig_oy))

    # Zone labels
    parts.append(_build_labels(joints, norm_grades, fig_ox, fig_oy))

    # ── SVG close ───────────────────────────────────────────────────
    parts.append("</svg>")

    return "\n".join(parts)
