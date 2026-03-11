#!/usr/bin/env python3
"""Generate an HTML frame reviewer for ground truth event labeling.

Extracts every frame from an annotated video and generates a self-contained
HTML page with arrow-key navigation, event markers, and labeling buttons.

Usage:
    python scripts/label_events.py --video data/outputs/validate_IMG_3108/annotated_video.mp4 \
        --results data/outputs/validate_IMG_3108/results.json \
        --output data/ground_truth/label_IMG_3108.html
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2


def extract_frames_as_base64(video_path: Path, max_dim: int = 800) -> list[str]:
    """Extract all frames from video, resize, encode as base64 JPEG."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(base64.b64encode(buf).decode("utf-8"))
    cap.release()
    return frames


def build_reviewer_html(
    frames_b64: list[str],
    fps: float,
    detected_events: dict,
    video_name: str,
) -> str:
    """Build self-contained HTML frame reviewer."""
    frames_json = json.dumps(frames_b64)
    events_json = json.dumps(detected_events)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Frame Reviewer — {video_name}</title>
<style>
body {{ background: #111; color: #e0e0e0; font-family: system-ui; margin: 0; padding: 1rem; }}
.viewer {{ max-width: 900px; margin: 0 auto; }}
h1 {{ font-size: 1.2rem; margin-bottom: 0.5rem; }}
#frame-img {{ width: 100%; border: 3px solid #333; border-radius: 4px; }}
.controls {{ display: flex; align-items: center; gap: 1rem; margin: 0.8rem 0; flex-wrap: wrap; }}
.controls button {{ background: #222; color: #e0e0e0; border: 1px solid #444; padding: 0.4rem 0.8rem;
    border-radius: 4px; cursor: pointer; font-size: 0.85rem; }}
.controls button:hover {{ background: #333; }}
.controls button.active {{ background: #c44; border-color: #c44; }}
#frame-info {{ font-size: 1rem; font-weight: bold; }}
.event-badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 3px;
    font-size: 0.8rem; font-weight: bold; margin-left: 0.5rem; }}
.event-badge.detected {{ background: #2a5; color: #fff; }}
.event-badge.labeled {{ background: #c80; color: #fff; }}
.label-row {{ display: flex; gap: 0.5rem; margin: 0.5rem 0; flex-wrap: wrap; }}
.label-row button {{ padding: 0.5rem 1rem; font-size: 0.9rem; }}
.label-row button.set {{ background: #c80; border-color: #c80; color: #fff; }}
#slider {{ flex: 1; min-width: 200px; }}
.help {{ font-size: 0.75rem; color: #888; margin-top: 0.5rem; }}
#labels-output {{ background: #1a1a1a; padding: 1rem; border-radius: 4px;
    margin-top: 1rem; font-family: monospace; font-size: 0.8rem; white-space: pre-wrap; }}
textarea {{ width: 100%; background: #1a1a1a; color: #e0e0e0; border: 1px solid #444;
    border-radius: 4px; padding: 0.5rem; font-family: monospace; margin-top: 0.5rem; }}
</style>
</head>
<body>
<div class="viewer">
<h1>Frame Reviewer: {video_name}</h1>
<img id="frame-img" src="" alt="frame">
<div class="controls">
    <button onclick="step(-10)">-10</button>
    <button onclick="step(-1)">&larr; Prev</button>
    <span id="frame-info">Frame 0</span>
    <button onclick="step(1)">Next &rarr;</button>
    <button onclick="step(10)">+10</button>
    <input type="range" id="slider" min="0" max="0" value="0" oninput="goTo(+this.value)">
</div>
<div class="label-row">
    <button id="btn-leg_lift" onclick="setLabel('leg_lift')">Mark Leg Lift</button>
    <button id="btn-foot_plant" onclick="setLabel('foot_plant')">Mark Foot Plant</button>
    <button id="btn-max_er" onclick="setLabel('max_er')">Mark Max ER</button>
    <button id="btn-ball_release" onclick="setLabel('ball_release')">Mark Ball Release</button>
    <button onclick="clearLabels()" style="margin-left:auto;background:#511;">Clear All</button>
</div>
<p class="help">Arrow keys: prev/next frame. Shift+Arrow: ±10 frames. Click buttons or press 1-4 to label current frame.</p>
<label>Notes: <textarea id="notes" rows="2" placeholder="Optional notes about this clip..."></textarea></label>
<div id="labels-output"></div>
<button onclick="copyJSON()" style="margin-top:0.5rem;">Copy JSON to Clipboard</button>
<button onclick="downloadJSON()" style="margin-top:0.5rem;">Download JSON</button>
</div>
<script>
const frames = {frames_json};
const detected = {events_json};
const fps = {fps};
const videoName = "{video_name}";
let current = 0;
let labels = {{}};

const img = document.getElementById('frame-img');
const info = document.getElementById('frame-info');
const slider = document.getElementById('slider');
slider.max = frames.length - 1;

function render() {{
    img.src = 'data:image/jpeg;base64,' + frames[current];
    let txt = 'Frame ' + current + ' / ' + (frames.length-1) + '  (' + (current/fps).toFixed(3) + 's)';
    // Check detected events
    for (const [evt, fr] of Object.entries(detected)) {{
        if (fr === current) txt += '<span class="event-badge detected">DETECTED: ' + evt + '</span>';
    }}
    // Check labels
    for (const [evt, fr] of Object.entries(labels)) {{
        if (fr === current) txt += '<span class="event-badge labeled">LABELED: ' + evt + '</span>';
    }}
    info.innerHTML = txt;
    slider.value = current;
    // Update button highlights
    for (const evt of ['leg_lift','foot_plant','max_er','ball_release']) {{
        const btn = document.getElementById('btn-' + evt);
        btn.classList.toggle('set', labels[evt] === current);
        btn.textContent = labels[evt] !== undefined
            ? evt.replace('_',' ') + ' [' + labels[evt] + ']'
            : 'Mark ' + evt.replace('_',' ');
    }}
    renderJSON();
}}

function step(n) {{ goTo(current + n); }}
function goTo(n) {{ current = Math.max(0, Math.min(frames.length-1, +n)); render(); }}

function setLabel(evt) {{
    labels[evt] = current;
    render();
}}

function clearLabels() {{ labels = {{}}; render(); }}

function getJSON() {{
    return JSON.stringify({{
        video: videoName,
        labeler: "rob",
        events: labels,
        detected_events: detected,
        notes: document.getElementById('notes').value,
    }}, null, 2);
}}

function renderJSON() {{
    document.getElementById('labels-output').textContent = getJSON();
}}

function copyJSON() {{
    navigator.clipboard.writeText(getJSON());
}}

function downloadJSON() {{
    const blob = new Blob([getJSON()], {{type: 'application/json'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = videoName.replace(/\\.[^.]+$/, '') + '_ground_truth.json';
    a.click();
}}

document.addEventListener('keydown', (e) => {{
    if (e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowRight') step(e.shiftKey ? 10 : 1);
    else if (e.key === 'ArrowLeft') step(e.shiftKey ? -10 : -1);
    else if (e.key === '1') setLabel('leg_lift');
    else if (e.key === '2') setLabel('foot_plant');
    else if (e.key === '3') setLabel('max_er');
    else if (e.key === '4') setLabel('ball_release');
}});

render();
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frame reviewer for event labeling")
    parser.add_argument("--video", type=Path, required=True, help="Annotated video path")
    parser.add_argument("--results", type=Path, required=True, help="Pipeline results.json path")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    parser.add_argument("--max-dim", type=int, default=800, help="Max frame dimension in pixels")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    results = json.loads(args.results.read_text())
    video_name = results.get("video", args.video.stem)
    fps = results.get("fps", 30.0)
    detected = results.get("events", {})

    output_path = args.output or Path(f"data/ground_truth/label_{Path(video_name).stem}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting frames from {args.video}...")
    frames = extract_frames_as_base64(args.video, max_dim=args.max_dim)
    print(f"  {len(frames)} frames extracted")

    print("Building reviewer HTML...")
    html = build_reviewer_html(frames, fps, detected, video_name)
    output_path.write_text(html)
    print(f"  Reviewer saved: {output_path}")
    print(f"\nOpen in browser, label events, then click 'Download JSON'")
    print(f"Save the JSON to data/ground_truth/{Path(video_name).stem}.json")


if __name__ == "__main__":
    main()
