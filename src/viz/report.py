"""HTML report generator for pitch mechanics validation.

Assembles pose estimation outputs — annotated video, trajectory plots,
key frame images, and extracted metrics — into a single self-contained
HTML diagnostic report.
"""

from html import escape


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
"""


def build_report_html(
    video_filename: str,
    video_rel_path: str,
    fps: float,
    frame_count: int,
    backend: str,
    pitcher_throws: str,
    trajectory_plots_html: list[str],
    key_frame_images: dict[str, str],
    metrics_rows: list[dict],
    diagnostics: dict,
) -> str:
    """Build a complete HTML diagnostic report.

    Args:
        video_filename: Original video file name.
        video_rel_path: Relative path to annotated MP4 for the video tag.
        fps: Video frames per second.
        frame_count: Total frame count.
        backend: Pose estimation backend ("yolov8" or "mediapipe").
        pitcher_throws: "R" or "L".
        trajectory_plots_html: Plotly chart HTML fragments.
        key_frame_images: Mapping of event_name to base64-encoded PNG string.
        metrics_rows: Dicts with keys: metric, value, unit, obp_median, status.
        diagnostics: Arbitrary key-value diagnostic info.

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

    # --- Header / video info table ---
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

    # --- Video section ---
    parts.append(f"""<h2>Annotated Video</h2>
<video controls>
<source src="{escape(video_rel_path)}" type="video/mp4">
Your browser does not support the video tag.
</video>
""")

    # --- Trajectory plots ---
    if trajectory_plots_html:
        parts.append("<h2>Trajectory Plots</h2>\n")
        for plot_html in trajectory_plots_html:
            parts.append(f'<div class="plot-container">{plot_html}</div>\n')

    # --- Key frames ---
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

    # --- Metrics table ---
    if metrics_rows:
        parts.append("""<h2>Extracted Metrics</h2>
<table>
<thead>
<tr><th>Metric</th><th>Value</th><th>Unit</th><th>OBP Median</th><th>Status</th></tr>
</thead>
<tbody>
""")
        for row in metrics_rows:
            status = row.get("status", "")
            row_class = ' class="warning"' if status == "missing" else ""
            value_str = escape(str(row.get("value", "")))
            parts.append(
                f"<tr{row_class}>"
                f"<td>{escape(str(row.get('metric', '')))}</td>"
                f"<td>{value_str}</td>"
                f"<td>{escape(str(row.get('unit', '')))}</td>"
                f"<td>{escape(str(row.get('obp_median', '')))}</td>"
                f"<td>{escape(str(status))}</td>"
                f"</tr>\n"
            )
        parts.append("</tbody>\n</table>\n")

    # --- Diagnostics ---
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
