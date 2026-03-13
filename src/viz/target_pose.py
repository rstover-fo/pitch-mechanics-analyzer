"""SVG target pose diagram of ideal foot-plant position.

Generates a self-contained inline SVG showing a stick-figure pitcher at foot
plant with labeled ASMI target angles.  Dark background (#0a0a0a), figure in
light gray, angle arcs in accent blue (#4A90D9).  Mirrored for LHP.
"""

from typing import Optional


# ── Ideal ASMI targets ──────────────────────────────────────────────────
_TARGETS = {
    "elbow_flexion":     "~90°",
    "shoulder_abduction": "~90°",
    "trunk_tilt":        "~30°",
    "hip_shoulder_sep":  "~30°",
    "stride_length":     "75-85% height",
    "lead_knee":         "~160°",
}

# ── Joint positions for a canonical RHP at foot-plant ───────────────────
# Coordinates in a 400×500 viewBox.  Y increases downward.
_RHP_JOINTS = {
    "head":             (200, 60),
    "neck":             (200, 90),
    "throw_shoulder":   (170, 115),
    "throw_elbow":      (140, 95),
    "throw_wrist":      (135, 65),
    "glove_shoulder":   (230, 115),
    "glove_elbow":      (260, 150),
    "glove_wrist":      (270, 180),
    "torso_mid":        (200, 175),
    "throw_hip":        (185, 230),
    "lead_hip":         (215, 230),
    "throw_knee":       (185, 310),
    "throw_ankle":      (185, 390),
    "lead_knee":        (280, 300),
    "lead_ankle":       (340, 380),
}


def _mirror_x(joints: dict[str, tuple[int, int]], cx: int = 200) -> dict[str, tuple[int, int]]:
    """Mirror joint positions horizontally around cx for LHP."""
    return {k: (2 * cx - x, y) for k, (x, y) in joints.items()}


def _line(x1: int, y1: int, x2: int, y2: int, **kw) -> str:
    stroke = kw.get("stroke", "#d0d0d0")
    width = kw.get("width", 3)
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>'


def _circle(cx: int, cy: int, r: int = 5, fill: str = "#d0d0d0") -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>'


def _text(x: int, y: int, label: str, font_size: int = 11, color: str = "#4A90D9") -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{font_size}" '
        f'font-family="sans-serif" text-anchor="middle">{label}</text>'
    )


def _arc_path(cx: int, cy: int, r: int, start_deg: float, end_deg: float) -> str:
    """SVG arc path for an angle annotation."""
    import math
    s = math.radians(start_deg)
    e = math.radians(end_deg)
    sx, sy = cx + r * math.cos(s), cy + r * math.sin(s)
    ex, ey = cx + r * math.cos(e), cy + r * math.sin(e)
    sweep = end_deg - start_deg
    large = 1 if abs(sweep) > 180 else 0
    return (
        f'<path d="M {sx:.1f} {sy:.1f} A {r} {r} 0 {large} 1 {ex:.1f} {ey:.1f}" '
        f'fill="none" stroke="#4A90D9" stroke-width="1.5" stroke-dasharray="4 2"/>'
    )


# ── Skeleton bone connections ───────────────────────────────────────────
_BONES = [
    ("head", "neck"),
    ("neck", "throw_shoulder"),
    ("neck", "glove_shoulder"),
    ("throw_shoulder", "throw_elbow"),
    ("throw_elbow", "throw_wrist"),
    ("glove_shoulder", "glove_elbow"),
    ("glove_elbow", "glove_wrist"),
    ("neck", "torso_mid"),
    ("torso_mid", "throw_hip"),
    ("torso_mid", "lead_hip"),
    ("throw_hip", "throw_knee"),
    ("throw_knee", "throw_ankle"),
    ("lead_hip", "lead_knee"),
    ("lead_knee", "lead_ankle"),
    ("throw_hip", "lead_hip"),
]


def build_target_pose_svg(throws: str = "R", width: int = 400, height: int = 500) -> str:
    """Generate an inline SVG of the ideal foot-plant pose.

    Args:
        throws: "R" or "L" — mirrors the figure for left-handers.
        width: SVG width attribute.
        height: SVG height attribute.

    Returns:
        Complete SVG string (no XML declaration, suitable for HTML embedding).
    """
    joints = _RHP_JOINTS if throws == "R" else _mirror_x(_RHP_JOINTS)
    parts: list[str] = []

    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 400 500" style="background:#0a0a0a;border-radius:8px;">'
    )

    # Title
    hand_label = "RHP" if throws == "R" else "LHP"
    parts.append(_text(200, 30, f"Target Foot-Plant Position ({hand_label})", 14, "#e0e0e0"))

    # Draw bones
    for a, b in _BONES:
        ax, ay = joints[a]
        bx, by = joints[b]
        parts.append(_line(ax, ay, bx, by))

    # Draw joints
    for name, (x, y) in joints.items():
        r = 8 if name == "head" else 4
        parts.append(_circle(x, y, r))

    # ── Angle annotations ───────────────────────────────────────────
    te = joints["throw_elbow"]
    ts = joints["throw_shoulder"]

    # Elbow flexion arc
    parts.append(_arc_path(te[0], te[1], 25, -120, -30))
    parts.append(_text(te[0] - 45, te[1] - 15, _TARGETS["elbow_flexion"], 10))

    # Shoulder abduction
    parts.append(_text(ts[0] - 50, ts[1] - 8, f'Abd {_TARGETS["shoulder_abduction"]}', 10))

    # Trunk tilt annotation
    tm = joints["torso_mid"]
    parts.append(_arc_path(tm[0], tm[1], 30, -90, -60))
    parts.append(_text(tm[0] + 50, tm[1] - 10, f'Trunk {_TARGETS["trunk_tilt"]}', 10))

    # Hip-shoulder separation
    th = joints["throw_hip"]
    lh = joints["lead_hip"]
    mid_hip = ((th[0] + lh[0]) // 2, (th[1] + lh[1]) // 2)
    parts.append(_text(mid_hip[0], mid_hip[1] + 25, f'H-S Sep {_TARGETS["hip_shoulder_sep"]}', 10))

    # Stride length annotation
    ta = joints["throw_ankle"]
    la = joints["lead_ankle"]
    mid_stride_x = (ta[0] + la[0]) // 2
    parts.append(
        f'<line x1="{ta[0]}" y1="{ta[1]+15}" x2="{la[0]}" y2="{la[1]+15}" '
        f'stroke="#4A90D9" stroke-width="1" stroke-dasharray="4 2"/>'
    )
    parts.append(_text(mid_stride_x, ta[1] + 35, f'Stride {_TARGETS["stride_length"]}', 10))

    # Lead knee angle
    lk = joints["lead_knee"]
    parts.append(_arc_path(lk[0], lk[1], 22, -160, -110))
    parts.append(_text(lk[0] + 40, lk[1] - 10, f'Knee {_TARGETS["lead_knee"]}', 10))

    # Footer
    parts.append(_text(200, 480, "ASMI Target Positions", 10, "#666"))

    parts.append("</svg>")
    return "\n".join(parts)
