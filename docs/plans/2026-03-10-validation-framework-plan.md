# Validation Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a validation harness that measures event detection accuracy, cross-pitch consistency, and failure modes — starting with a frame reviewer for human ground truth labeling.

**Architecture:** Pipeline saves JSON alongside HTML reports. Frame reviewer extracts annotated frames into a navigable HTML page for labeling. Evaluation scripts compare pipeline output to ground truth labels and print scored summaries.

**Tech Stack:** Python, OpenCV (frame extraction), vanilla HTML/JS (reviewer), JSON (data interchange), pytest (unit tests)

---

### Task 1: Save Pipeline Results as JSON

The evaluation scripts need machine-readable pipeline output. Currently only HTML is saved.

**Files:**
- Modify: `scripts/validate_pose.py`

**Step 1: Add JSON export after metrics extraction**

Add this after the metrics_rows loop (around line 483) and before the Assembly section:

```python
import json

pipeline_output = {
    "video": args.video.name,
    "backend": args.backend,
    "pitcher_throws": args.throws,
    "fps": fps,
    "total_frames": video_info.total_frames,
    "events": {
        "leg_lift": events.leg_lift_apex,
        "foot_plant": events.foot_plant,
        "max_er": events.max_external_rotation,
        "ball_release": events.ball_release,
    },
    "metrics": {
        row["metric"]: row["value"]
        for row in metrics_rows
        if row["status"] == "ok"
    },
    "metrics_raw": {
        k: getattr(metrics, k)
        for k in [
            "elbow_flexion_fp", "torso_anterior_tilt_fp", "lead_knee_angle_fp",
            "max_shoulder_external_rotation", "torso_anterior_tilt_br",
            "arm_slot_angle", "lead_knee_angle_br",
        ]
        if getattr(metrics, k) is not None
    },
    "diagnostics": {
        "frames_with_poses": len(pose_seq.frames),
        "avg_confidence": float(avg_confidence),
    },
}

results_path = output_dir / "results.json"
results_path.write_text(json.dumps(pipeline_output, indent=2))
print(f"  Results saved: {results_path}")
```

**Step 2: Run pipeline to verify JSON is created**

Run: `.venv/bin/python3.11 scripts/validate_pose.py --video data/uploads/IMG_3108.MOV --throws R --no-open`

Expected: `data/outputs/validate_IMG_3108/results.json` created with events and metrics.

**Step 3: Commit**

```bash
git add scripts/validate_pose.py
git commit -m "Save pipeline results as JSON for evaluation scripts"
```

---

### Task 2: Frame Reviewer HTML Generator

**Files:**
- Create: `scripts/label_events.py`

**Step 1: Write the frame extraction and HTML generation script**

```python
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
```

**Step 2: Run it on one clip to verify**

Run: `.venv/bin/python3.11 scripts/label_events.py --video data/outputs/validate_IMG_3108/annotated_video.mp4 --results data/outputs/validate_IMG_3108/results.json`

Expected: `data/ground_truth/label_IMG_3108.html` created, opens in browser, arrow keys navigate frames.

**Step 3: Commit**

```bash
git add scripts/label_events.py
git commit -m "Add frame reviewer for ground truth event labeling"
```

---

### Task 3: Pitcher Registry and Ground Truth Data Structure

**Files:**
- Create: `data/ground_truth/pitchers.json`

**Step 1: Create pitcher registry**

```json
{
  "player_a": {
    "clips": ["IMG_3106", "IMG_3107"],
    "throws": "R",
    "description": "Red shirt, driveway"
  },
  "player_b": {
    "clips": ["IMG_3108", "IMG_3109"],
    "throws": "R",
    "description": "Blue shirt, driveway"
  }
}
```

**Step 2: Commit**

```bash
git add data/ground_truth/pitchers.json
git commit -m "Add pitcher registry for ground truth evaluation"
```

---

### Task 4: Failure Mode Detection Module

**Files:**
- Create: `src/biomechanics/validation.py`
- Create: `tests/test_validation.py`

**Step 1: Write failing tests**

```python
"""Tests for pipeline validation and sanity checks."""

import pytest
from src.biomechanics.events import DeliveryEvents
from src.biomechanics.validation import validate_pipeline_output


class TestEventOrdering:
    def test_correct_ordering_no_warnings(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 70
        events.max_external_rotation = 75
        events.ball_release = 80
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        error_warnings = [w for w in warnings if w["severity"] == "error"]
        assert len(error_warnings) == 0

    def test_out_of_order_events(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 80
        events.foot_plant = 70
        events.max_external_rotation = 75
        events.ball_release = 90
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        codes = [w["code"] for w in warnings]
        assert "events_out_of_order" in codes

    def test_missing_event_flagged(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 70
        events.max_external_rotation = None
        events.ball_release = 80
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        codes = [w["code"] for w in warnings]
        assert "event_not_detected" in codes


class TestPhaseTiming:
    def test_plausible_timing_no_warnings(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 65
        events.max_external_rotation = 69
        events.ball_release = 72
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        timing_warnings = [w for w in warnings if "duration" in w["code"]]
        assert len(timing_warnings) == 0

    def test_implausible_stride_duration(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 10
        events.foot_plant = 100  # 3.0s stride — too long
        events.max_external_rotation = 105
        events.ball_release = 108
        warnings = validate_pipeline_output(events, avg_confidence=0.7)
        codes = [w["code"] for w in warnings]
        assert "stride_duration_implausible" in codes


class TestConfidence:
    def test_low_confidence_flagged(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 70
        events.max_external_rotation = 75
        events.ball_release = 80
        warnings = validate_pipeline_output(events, avg_confidence=0.15)
        codes = [w["code"] for w in warnings]
        assert "low_tracking_confidence" in codes

    def test_good_confidence_no_warning(self):
        events = DeliveryEvents(fps=30.0)
        events.leg_lift_apex = 50
        events.foot_plant = 70
        events.max_external_rotation = 75
        events.ball_release = 80
        warnings = validate_pipeline_output(events, avg_confidence=0.8)
        codes = [w["code"] for w in warnings]
        assert "low_tracking_confidence" not in codes
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3.11 -m pytest tests/test_validation.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'src.biomechanics.validation'`

**Step 3: Write the validation module**

```python
"""Pipeline output validation and sanity checks.

Runs automatic checks on detected events and metrics to flag
implausible results. Not pass/fail gates — warnings for human review.
"""

from typing import Optional

from src.biomechanics.events import DeliveryEvents


def validate_pipeline_output(
    events: DeliveryEvents,
    avg_confidence: float = 1.0,
    metrics: Optional[dict] = None,
) -> list[dict]:
    """Run all sanity checks on pipeline output.

    Args:
        events: Detected delivery events.
        avg_confidence: Mean keypoint confidence across all frames.
        metrics: Optional dict of extracted metric values.

    Returns:
        List of warning dicts with keys: code, severity, message.
    """
    warnings: list[dict] = []
    warnings.extend(_check_event_ordering(events))
    warnings.extend(_check_missing_events(events))
    warnings.extend(_check_phase_timing(events))
    warnings.extend(_check_confidence(avg_confidence))
    if metrics:
        warnings.extend(_check_metric_ranges(metrics))
    return warnings


def _check_event_ordering(events: DeliveryEvents) -> list[dict]:
    ordered = [
        ("leg_lift", events.leg_lift_apex),
        ("foot_plant", events.foot_plant),
        ("max_er", events.max_external_rotation),
        ("ball_release", events.ball_release),
    ]
    present = [(name, frame) for name, frame in ordered if frame is not None]
    for i in range(len(present) - 1):
        if present[i][1] >= present[i + 1][1]:
            return [{
                "code": "events_out_of_order",
                "severity": "error",
                "message": f"{present[i][0]} (frame {present[i][1]}) is not before "
                           f"{present[i+1][0]} (frame {present[i+1][1]})",
            }]
    return []


def _check_missing_events(events: DeliveryEvents) -> list[dict]:
    warnings = []
    for name, frame in [
        ("leg_lift", events.leg_lift_apex),
        ("foot_plant", events.foot_plant),
        ("max_er", events.max_external_rotation),
        ("ball_release", events.ball_release),
    ]:
        if frame is None:
            warnings.append({
                "code": "event_not_detected",
                "severity": "error",
                "message": f"{name} was not detected",
            })
    return warnings


def _check_phase_timing(events: DeliveryEvents) -> list[dict]:
    warnings = []
    fps = events.fps

    phases = [
        ("stride_duration", events.leg_lift_apex, events.foot_plant, 0.1, 1.5),
        ("arm_cocking_duration", events.foot_plant, events.max_external_rotation, 0.01, 0.5),
        ("arm_accel_duration", events.max_external_rotation, events.ball_release, 0.01, 0.5),
    ]

    for name, start, end, min_sec, max_sec in phases:
        if start is None or end is None:
            continue
        duration = (end - start) / fps
        if duration < min_sec or duration > max_sec:
            warnings.append({
                "code": f"{name}_implausible",
                "severity": "warning",
                "message": f"{name.replace('_', ' ')}: {duration:.3f}s "
                           f"(expected {min_sec}-{max_sec}s)",
            })

    return warnings


def _check_confidence(avg_confidence: float) -> list[dict]:
    if avg_confidence < 0.3:
        return [{
            "code": "low_tracking_confidence",
            "severity": "warning",
            "message": f"Mean keypoint confidence {avg_confidence:.3f} is below 0.3",
        }]
    return []


def _check_metric_ranges(metrics: dict) -> list[dict]:
    warnings = []
    ranges = {
        "elbow_flexion_fp": (40, 160, "Elbow flexion @ FP"),
        "max_shoulder_external_rotation": (60, 190, "Peak shoulder ER"),
        "torso_anterior_tilt_fp": (-10, 60, "Trunk tilt @ FP"),
        "lead_knee_angle_fp": (60, 180, "Lead knee @ FP"),
    }
    for key, (lo, hi, label) in ranges.items():
        val = metrics.get(key)
        if val is not None and (val < lo or val > hi):
            warnings.append({
                "code": f"{key}_out_of_range",
                "severity": "warning",
                "message": f"{label}: {val:.1f} deg (expected {lo}-{hi} deg)",
            })
    return warnings
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3.11 -m pytest tests/test_validation.py -v`

Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add src/biomechanics/validation.py tests/test_validation.py
git commit -m "Add pipeline validation module with sanity checks"
```

---

### Task 5: Integrate Validation Into Pipeline

**Files:**
- Modify: `scripts/validate_pose.py`

**Step 1: Add validation call after metrics extraction**

Import at the top of the file:
```python
from src.biomechanics.validation import validate_pipeline_output
```

Add after the metrics loop (before JSON export):
```python
# Run sanity checks
validation_warnings = validate_pipeline_output(
    events, avg_confidence=avg_confidence, metrics=metrics.__dict__,
)
if validation_warnings:
    print("\n  Validation warnings:")
    for w in validation_warnings:
        print(f"    [{w['severity']}] {w['code']}: {w['message']}")
else:
    print("\n  No validation warnings")
```

Add warnings to the pipeline_output JSON dict:
```python
"warnings": validation_warnings,
```

Add warnings to the diagnostics dict passed to the HTML report:
```python
"warnings": "; ".join(w["message"] for w in validation_warnings) if validation_warnings else "none",
```

**Step 2: Run pipeline and verify warnings appear**

Run: `.venv/bin/python3.11 scripts/validate_pose.py --video data/uploads/IMG_3108.MOV --throws R --no-open`

Expected: Validation warnings printed (e.g., arm_accel_duration_implausible for the long MER→BR phase).

**Step 3: Run full test suite**

Run: `.venv/bin/python3.11 -m pytest tests/ -v`

Expected: All tests pass (82 existing + 7 new = 89).

**Step 4: Commit**

```bash
git add scripts/validate_pose.py
git commit -m "Integrate validation warnings into pipeline output"
```

---

### Task 6: Event Detection Accuracy Scorer

**Files:**
- Create: `scripts/eval_events.py`

**Step 1: Write the evaluation script**

```python
#!/usr/bin/env python3
"""Evaluate event detection accuracy against ground truth labels.

Compares pipeline-detected event frames to human-labeled ground truth
and computes MAE, bias, and hit rate per event type.

Usage:
    python scripts/eval_events.py
    python scripts/eval_events.py --ground-truth data/ground_truth/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


EVENT_NAMES = ["leg_lift", "foot_plant", "max_er", "ball_release"]


def load_ground_truth(gt_dir: Path) -> dict[str, dict]:
    """Load all ground truth JSON files from directory."""
    labels = {}
    for f in sorted(gt_dir.glob("*.json")):
        if f.name == "pitchers.json":
            continue
        data = json.loads(f.read_text())
        video_stem = Path(data.get("video", f.stem)).stem
        labels[video_stem] = data
    return labels


def load_pipeline_results(outputs_dir: Path) -> dict[str, dict]:
    """Load all pipeline results.json files."""
    results = {}
    for results_file in sorted(outputs_dir.glob("validate_*/results.json")):
        data = json.loads(results_file.read_text())
        video_stem = Path(data["video"]).stem
        results[video_stem] = data
    return results


def evaluate(gt_labels: dict, pipeline_results: dict) -> None:
    """Compare pipeline results to ground truth and print accuracy report."""
    matched_clips = set(gt_labels.keys()) & set(pipeline_results.keys())
    if not matched_clips:
        print("No matching clips found between ground truth and pipeline results.")
        sys.exit(1)

    print(f"Evaluating {len(matched_clips)} clips: {', '.join(sorted(matched_clips))}")
    print()

    errors: dict[str, list[int]] = {evt: [] for evt in EVENT_NAMES}
    missed: dict[str, int] = {evt: 0 for evt in EVENT_NAMES}
    total: dict[str, int] = {evt: 0 for evt in EVENT_NAMES}

    # Per-clip detail
    for clip in sorted(matched_clips):
        gt = gt_labels[clip].get("events", {})
        det = pipeline_results[clip].get("events", {})
        fps = pipeline_results[clip].get("fps", 30.0)

        print(f"  {clip} (fps={fps}):")
        for evt in EVENT_NAMES:
            gt_frame = gt.get(evt)
            det_frame = det.get(evt)
            if gt_frame is None:
                continue  # Not labeled
            total[evt] += 1
            if det_frame is None:
                missed[evt] += 1
                print(f"    {evt:15s}: MISSED (gt={gt_frame})")
            else:
                err = det_frame - gt_frame
                errors[evt].append(err)
                ms = err / fps * 1000
                print(f"    {evt:15s}: det={det_frame} gt={gt_frame} err={err:+d} frames ({ms:+.0f}ms)")
        print()

    # Summary
    print("=" * 65)
    print(f"{'Event':15s} {'MAE':>8s} {'Bias':>8s} {'MAE(ms)':>8s} {'Bias(ms)':>9s} {'Hit%':>6s}")
    print("-" * 65)
    for evt in EVENT_NAMES:
        errs = errors[evt]
        n = total[evt]
        if n == 0:
            print(f"{evt:15s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>9s} {'--':>6s}")
            continue
        hit_pct = (n - missed[evt]) / n * 100
        if errs:
            mae = np.mean(np.abs(errs))
            bias = np.mean(errs)
            # Assume ~30fps for ms conversion
            fps_avg = 30.0
            print(f"{evt:15s} {mae:8.1f} {bias:+8.1f} {mae/fps_avg*1000:8.0f} {bias/fps_avg*1000:+9.0f} {hit_pct:5.0f}%")
        else:
            print(f"{evt:15s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>9s} {hit_pct:5.0f}%")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate event detection accuracy")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth"),
                        help="Directory with ground truth JSON files")
    parser.add_argument("--outputs", type=Path, default=Path("data/outputs"),
                        help="Directory with pipeline output directories")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    results = load_pipeline_results(args.outputs)
    evaluate(gt, results)


if __name__ == "__main__":
    main()
```

**Step 2: Commit (runs after ground truth is labeled)**

```bash
git add scripts/eval_events.py
git commit -m "Add event detection accuracy evaluation script"
```

---

### Task 7: Cross-Pitch Consistency Scorer

**Files:**
- Create: `scripts/eval_consistency.py`

**Step 1: Write the consistency evaluation script**

```python
#!/usr/bin/env python3
"""Evaluate within-pitcher metric consistency across clips.

Groups pipeline results by pitcher (from pitchers.json), computes
mean, std dev, and coefficient of variation for each metric.

Usage:
    python scripts/eval_consistency.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-pitch metric consistency")
    parser.add_argument("--pitchers", type=Path, default=Path("data/ground_truth/pitchers.json"))
    parser.add_argument("--outputs", type=Path, default=Path("data/outputs"))
    args = parser.parse_args()

    if not args.pitchers.exists():
        print(f"Error: {args.pitchers} not found")
        sys.exit(1)

    pitchers = json.loads(args.pitchers.read_text())

    for pitcher_id, info in pitchers.items():
        clips = info["clips"]
        print(f"\n{'=' * 70}")
        print(f"Pitcher: {pitcher_id} ({info.get('description', '')}) — {len(clips)} clips")
        print(f"{'=' * 70}")

        # Load metrics from each clip
        all_metrics: dict[str, list[float]] = {}
        loaded = 0
        for clip_stem in clips:
            results_path = args.outputs / f"validate_{clip_stem}" / "results.json"
            if not results_path.exists():
                print(f"  Warning: {results_path} not found, skipping")
                continue
            data = json.loads(results_path.read_text())
            loaded += 1
            for key, val in data.get("metrics_raw", {}).items():
                if val is not None:
                    all_metrics.setdefault(key, []).append(val)

        if loaded < 2:
            print(f"  Need at least 2 clips for consistency analysis (found {loaded})")
            continue

        print(f"\n{'Metric':35s} {'Values':>25s} {'Mean':>8s} {'Std':>8s} {'CV%':>6s} {'Status':>10s}")
        print("-" * 100)

        for key, values in sorted(all_metrics.items()):
            if len(values) < 2:
                continue
            mean = np.mean(values)
            std = np.std(values)
            cv = (std / abs(mean) * 100) if abs(mean) > 0.1 else float("inf")
            vals_str = ", ".join(f"{v:.1f}" for v in values)

            if cv > 15:
                status = "UNSTABLE"
            elif cv > 8:
                status = "variable"
            else:
                status = "stable"

            print(f"{key:35s} {vals_str:>25s} {mean:8.1f} {std:8.1f} {cv:5.1f}% {status:>10s}")


if __name__ == "__main__":
    main()
```

**Step 2: Run it with existing pipeline outputs**

Run: `.venv/bin/python3.11 scripts/eval_consistency.py`

Expected: Prints per-pitcher metric table with CV% and stability flags for player_a and player_b.

**Step 3: Commit**

```bash
git add scripts/eval_consistency.py
git commit -m "Add cross-pitch consistency evaluation script"
```

---

### Task 8: Re-run Pipeline on All 4 Clips (Generate results.json)

After Task 1 adds JSON export, re-run the pipeline on all 4 driveway clips so results.json files exist for the evaluation scripts.

**Step 1: Run pipeline on all clips**

```bash
for vid in IMG_3106.MOV IMG_3107.MOV IMG_3108.MOV IMG_3109.MOV; do
  .venv/bin/python3.11 scripts/validate_pose.py --video "data/uploads/$vid" --throws R --no-open
done
```

Expected: `data/outputs/validate_IMG_310{6,7,8,9}/results.json` all created.

**Step 2: Generate frame reviewers for all clips**

```bash
for stem in IMG_3106 IMG_3107 IMG_3108 IMG_3109; do
  .venv/bin/python3.11 scripts/label_events.py \
    --video "data/outputs/validate_${stem}/annotated_video.mp4" \
    --results "data/outputs/validate_${stem}/results.json"
done
```

Expected: 4 HTML frame reviewers in `data/ground_truth/`.

**Step 3: Run consistency evaluation (no ground truth needed)**

Run: `.venv/bin/python3.11 scripts/eval_consistency.py`

Expected: Per-pitcher consistency report printed.

---

## Execution Order

| Task | Depends On | Description |
|------|-----------|-------------|
| 1 | — | JSON output from pipeline |
| 2 | — | Frame reviewer HTML generator |
| 3 | — | Pitcher registry |
| 4 | — | Validation module + tests |
| 5 | 1, 4 | Integrate validation into pipeline |
| 6 | 1 | Event accuracy scorer |
| 7 | 1, 3 | Consistency scorer |
| 8 | 1, 2, 5, 7 | Re-run pipeline + generate reviewers |

Tasks 1-4 are independent and can be implemented in parallel.
Tasks 5-7 depend on Task 1 (JSON output).
Task 8 is the integration run after everything is in place.
