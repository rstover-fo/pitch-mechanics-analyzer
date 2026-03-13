"""HTML report generator for pitch mechanics validation.

Assembles pose estimation outputs — annotated video, trajectory plots,
key frame images, and extracted metrics — into a single self-contained
HTML diagnostic report. Coaching-first layout: coaching narrative and
percentile charts appear above technical diagnostics.
"""

from html import escape
from typing import Optional


_PLOTLY_CDN = "https://cdn.plot.ly/plotly-latest.min.js"

_CSS = """
body {
    background: #0a0a0a;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 2rem 1rem;
}
.container {
    max-width: 960px;
    margin: 0 auto;
}
h1, h2, h3 {
    color: #f0f0f0;
    border-bottom: 1px solid #333;
    padding-bottom: 0.4rem;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 2rem;
}
th, td {
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #222;
}
th {
    color: #aaa;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
tr.warning td {
    color: #f5a623;
}
video {
    width: 100%;
    max-width: 640px;
    border-radius: 4px;
    margin-bottom: 1.5rem;
}
.plot-container {
    margin-bottom: 2rem;
}
.key-frames-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.key-frame {
    text-align: center;
}
.key-frame img {
    width: 100%;
    border-radius: 4px;
    border: 1px solid #333;
}
.key-frame h3 {
    border: none;
    font-size: 0.95rem;
    margin: 0.5rem 0 0.25rem;
}
.coaching-section {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    font-size: 1.05rem;
    line-height: 1.7;
}
.coaching-section h2 {
    color: #7eb8da;
    border-bottom: 1px solid #2a2a4e;
}
.coaching-section h3 {
    color: #a0c4e0;
    border: none;
    font-size: 1rem;
    margin-top: 1.2rem;
}
.coaching-section p {
    margin: 0.4rem 0;
}
.coaching-section li {
    margin: 0.2rem 0;
}
.profile-card {
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    background: #141428;
    border-radius: 6px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
}
.profile-card div {
    min-width: 80px;
}
.profile-card .label {
    color: #888;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.profile-card .value {
    color: #ddd;
    font-weight: 600;
    font-size: 1.1rem;
}
"""


def _coaching_text_to_html(text: str) -> str:
    """Convert coaching report text to simple HTML.

    Handles markdown-ish formatting from Claude API output and
    plain text from the offline report generator.
    """
    lines = text.strip().split("\n")
    html_parts: list[str] = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        # Section headers (## or lines of = or -)
        if stripped.startswith("## ") or stripped.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{escape(stripped.lstrip('#').strip())}</h3>")
        elif stripped.startswith("===") or stripped.startswith("---"):
            continue
        # Numbered items that look like section headers (e.g., "1. STRENGTHS")
        elif len(stripped) > 3 and stripped[0].isdigit() and stripped[1] in ".)" and stripped[2] == " " and stripped[3:].strip().isupper():
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{escape(stripped)}</h3>")
        # Bullet points
        elif stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("  -") or stripped.startswith("  *"):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            bullet_text = stripped.lstrip("-* ").strip()
            html_parts.append(f"<li>{escape(bullet_text)}</li>")
        # Indented continuation or note lines
        elif stripped.startswith("  ") and in_list:
            html_parts.append(f"<li>{escape(stripped.strip())}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p>{escape(stripped)}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(html_parts)


def _format_height_imperial(inches: float) -> str:
    """Convert inches to feet'inches\" display format."""
    feet = int(inches // 12)
    remaining = int(inches % 12)
    return f"{feet}'{remaining}\""


def build_report_html(
    video_filename: str,
    video_rel_path: Optional[str],
    fps: float,
    frame_count: int,
    backend: str,
    pitcher_throws: str,
    trajectory_plots_html: list[str],
    key_frame_images: dict[str, str],
    metrics_rows: list[dict],
    diagnostics: dict,
    coaching_html: str = "",
    percentile_charts_html: Optional[list[str]] = None,
    pitcher_profile: Optional[dict] = None,
) -> str:
    """Build a complete HTML report with coaching-first layout.

    Args:
        video_filename: Original video file name.
        video_rel_path: Relative path to annotated MP4 for the video tag.
        fps: Video frames per second.
        frame_count: Total frame count.
        backend: Pose estimation backend ("yolov8" or "mediapipe").
        pitcher_throws: "R" or "L".
        trajectory_plots_html: Plotly chart HTML fragments.
        key_frame_images: Mapping of event_name to base64-encoded PNG string.
        metrics_rows: Dicts with keys: metric, value, unit, obp_median, percentile, status.
        diagnostics: Arbitrary key-value diagnostic info.
        coaching_html: Coaching report text (plain text or markdown-ish).
        percentile_charts_html: Plotly chart fragments for radar/gauges.
        pitcher_profile: Youth profile dict with age, height_in, weight_lbs, developmental_stage.

    Returns:
        Complete HTML string.
    """
    duration = frame_count / fps if fps > 0 else 0.0
    parts: list[str] = []

    # --- Document open ---
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pitch Mechanics Report — {escape(video_filename)}</title>
<script src="{_PLOTLY_CDN}"></script>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
""")

    # --- 1. Header / video info table ---
    parts.append(f"""<h1>Pitch Mechanics Report</h1>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Video</td><td>{escape(video_filename)}</td></tr>
<tr><td>FPS</td><td>{fps}</td></tr>
<tr><td>Frames</td><td>{frame_count}</td></tr>
<tr><td>Duration</td><td>{duration:.1f}s</td></tr>
<tr><td>Backend</td><td>{escape(backend)}</td></tr>
<tr><td>Throws</td><td>{escape(pitcher_throws)}</td></tr>
</table>
""")

    # --- Pitcher profile card (if youth profile provided) ---
    if pitcher_profile:
        age = pitcher_profile.get("age", "")
        height_in = pitcher_profile.get("height_in")
        weight_lbs = pitcher_profile.get("weight_lbs", "")
        stage = pitcher_profile.get("developmental_stage", "")
        height_display = _format_height_imperial(height_in) if height_in else ""
        stage_display = stage.replace("_", " ").title() if stage else ""

        parts.append(f"""<div class="profile-card">
<div><span class="label">Age</span><br><span class="value">{escape(str(age))}</span></div>
<div><span class="label">Height</span><br><span class="value">{escape(height_display)}</span></div>
<div><span class="label">Weight</span><br><span class="value">{escape(str(weight_lbs))} lbs</span></div>
<div><span class="label">Stage</span><br><span class="value">{escape(stage_display)}</span></div>
</div>
""")

    # --- 2. Coaching Narrative ---
    if coaching_html and coaching_html.strip():
        coaching_content = _coaching_text_to_html(coaching_html)
        parts.append(f"""<div class="coaching-section">
<h2>Coaching Report</h2>
{coaching_content}
</div>
""")

    # --- 3. Percentile Charts ---
    if percentile_charts_html:
        parts.append("<h2>Percentile Comparison</h2>\n")
        for chart_html in percentile_charts_html:
            parts.append(f'<div class="plot-container">{chart_html}</div>\n')

    # --- 4. Key frames ---
    if key_frame_images:
        parts.append("<h2>Key Frames</h2>\n")
        parts.append('<div class="key-frames-grid">\n')
        for event_name, b64_png in key_frame_images.items():
            parts.append(f"""<div class="key-frame">
<h3>{escape(event_name)}</h3>
<img src="data:image/png;base64,{b64_png}" alt="{escape(event_name)}">
</div>
""")
        parts.append("</div>\n")

    # --- 5. Metrics table (with percentile column) ---
    if metrics_rows:
        has_percentile = any(row.get("percentile") and row.get("percentile") != "--" for row in metrics_rows)
        parts.append("""<h2>Extracted Metrics</h2>
<table>
<thead>
<tr><th>Metric</th><th>Value</th><th>Unit</th>""")
        if has_percentile:
            parts.append("<th>Percentile</th>")
        parts.append("<th>OBP Median</th><th>Status</th></tr>\n</thead>\n<tbody>\n")

        for row in metrics_rows:
            status = row.get("status", "")
            row_class = ' class="warning"' if status == "missing" else ""
            value_str = escape(str(row.get("value", "")))
            parts.append(
                f"<tr{row_class}>"
                f"<td>{escape(str(row.get('metric', '')))}</td>"
                f"<td>{value_str}</td>"
                f"<td>{escape(str(row.get('unit', '')))}</td>"
            )
            if has_percentile:
                pct = row.get("percentile", "--")
                parts.append(f"<td>{escape(str(pct))}</td>")
            parts.append(
                f"<td>{escape(str(row.get('obp_median', '')))}</td>"
                f"<td>{escape(str(status))}</td>"
                f"</tr>\n"
            )
        parts.append("</tbody>\n</table>\n")

    # --- 6. Trajectory plots ---
    if trajectory_plots_html:
        parts.append("<h2>Trajectory Plots</h2>\n")
        for plot_html in trajectory_plots_html:
            parts.append(f'<div class="plot-container">{plot_html}</div>\n')

    # --- 7. Annotated Video ---
    if video_rel_path:
        parts.append(f"""<h2>Annotated Video</h2>
<video controls>
<source src="{escape(video_rel_path)}" type="video/mp4">
Your browser does not support the video tag.
</video>
""")

    # --- 8. Diagnostics ---
    if diagnostics:
        parts.append("""<h2>Diagnostics</h2>
<table>
<tr><th>Key</th><th>Value</th></tr>
""")
        for key, value in diagnostics.items():
            parts.append(
                f"<tr><td>{escape(str(key))}</td>"
                f"<td>{escape(str(value))}</td></tr>\n"
            )
        parts.append("</table>\n")

    # --- Document close ---
    parts.append("</div>\n</body>\n</html>")

    return "".join(parts)
