# Pose Validation Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `scripts/validate_pose.py` that runs the full video→pose→events→metrics pipeline and produces an HTML diagnostic report with annotated video, trajectory plots, key frame stills, and a metrics table.

**Architecture:** A linear 4-stage pipeline script. Each stage produces artifacts in an output directory. A final assembly step generates a self-contained HTML report. New visualization helpers go in `src/viz/`. The script uses existing modules (`estimator`, `events`, `features`, `benchmarks`) and adds thin wrappers for drawing and reporting.

**Tech Stack:** OpenCV (video I/O, drawing), Plotly (trajectory charts, inline HTML), NumPy (trajectory math), existing `src/` modules.

---

### Task 1: Skeleton Drawing Utility

**Files:**
- Create: `src/viz/skeleton.py`
- Test: `tests/test_skeleton.py`

**Step 1: Write the failing test**

```python
"""Tests for skeleton drawing utilities."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz.skeleton import (
    SKELETON_CONNECTIONS,
    confidence_color,
    draw_skeleton,
)


class TestConfidenceColor:
    def test_high_confidence_green(self):
        color = confidence_color(0.85)
        assert color == (0, 200, 0)  # BGR green

    def test_medium_confidence_yellow(self):
        color = confidence_color(0.55)
        assert color == (0, 200, 200)  # BGR yellow

    def test_low_confidence_red(self):
        color = confidence_color(0.2)
        assert color == (0, 0, 200)  # BGR red


class TestSkeletonConnections:
    def test_has_throwing_arm(self):
        pairs = [(a, b) for a, b in SKELETON_CONNECTIONS]
        assert ("right_shoulder", "right_elbow") in pairs
        assert ("right_elbow", "right_wrist") in pairs

    def test_has_trunk(self):
        pairs = [(a, b) for a, b in SKELETON_CONNECTIONS]
        assert ("left_shoulder", "right_shoulder") in pairs
        assert ("left_hip", "right_hip") in pairs


class TestDrawSkeleton:
    def test_draws_on_frame_without_error(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = {
            "right_shoulder": np.array([300, 200]),
            "right_elbow": np.array([350, 250]),
            "right_wrist": np.array([380, 300]),
            "left_shoulder": np.array([260, 200]),
            "left_hip": np.array([260, 350]),
            "right_hip": np.array([300, 350]),
        }
        confidence = {k: 0.9 for k in keypoints}
        result = draw_skeleton(frame, keypoints, confidence)
        # Should return a modified frame (not all black)
        assert result.shape == frame.shape
        assert result.sum() > 0

    def test_handles_missing_joints(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints = {"right_shoulder": np.array([300, 200])}
        confidence = {"right_shoulder": 0.9}
        # Should not raise even with missing joints
        result = draw_skeleton(frame, keypoints, confidence)
        assert result.shape == frame.shape
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_skeleton.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.viz.skeleton'`

**Step 3: Write minimal implementation**

```python
"""Skeleton drawing utilities for pose validation visualization."""

from typing import Optional

import cv2
import numpy as np


# Joint connection pairs for skeleton overlay
SKELETON_CONNECTIONS: list[tuple[str, str]] = [
    # Throwing arm (right)
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Glove arm (left)
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # Shoulder line
    ("left_shoulder", "right_shoulder"),
    # Hip line
    ("left_hip", "right_hip"),
    # Trunk (both sides)
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    # Left leg
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # Right leg
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Drawing constants
JOINT_RADIUS = 6
LINE_THICKNESS = 2
BBOX_THICKNESS = 1
BBOX_COLOR = (200, 200, 200)  # Light gray BGR


def confidence_color(conf: float) -> tuple[int, int, int]:
    """Map confidence score to BGR color.

    Green (>0.7), yellow (0.4-0.7), red (<0.4).
    """
    if conf > 0.7:
        return (0, 200, 0)
    elif conf > 0.4:
        return (0, 200, 200)
    else:
        return (0, 0, 200)


def draw_skeleton(
    frame: np.ndarray,
    keypoints: dict[str, np.ndarray],
    confidence: dict[str, float],
    bbox: Optional[np.ndarray] = None,
    min_confidence: float = 0.1,
) -> np.ndarray:
    """Draw skeleton overlay on a video frame.

    Args:
        frame: BGR image (H, W, 3).
        keypoints: Joint name -> (x, y) position.
        confidence: Joint name -> confidence score.
        bbox: Optional bounding box [x1, y1, x2, y2].
        min_confidence: Skip joints below this threshold.

    Returns:
        Frame with skeleton drawn on it.
    """
    out = frame.copy()

    # Draw bounding box
    if bbox is not None:
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

    # Draw connections
    for joint_a, joint_b in SKELETON_CONNECTIONS:
        if joint_a not in keypoints or joint_b not in keypoints:
            continue
        conf_a = confidence.get(joint_a, 0)
        conf_b = confidence.get(joint_b, 0)
        if conf_a < min_confidence or conf_b < min_confidence:
            continue

        pt_a = tuple(keypoints[joint_a].astype(int))
        pt_b = tuple(keypoints[joint_b].astype(int))
        color = confidence_color(min(conf_a, conf_b))
        cv2.line(out, pt_a, pt_b, color, LINE_THICKNESS)

    # Draw joint dots
    for joint, pos in keypoints.items():
        conf = confidence.get(joint, 0)
        if conf < min_confidence:
            continue
        center = tuple(pos.astype(int))
        color = confidence_color(conf)
        cv2.circle(out, center, JOINT_RADIUS, color, -1)
        cv2.circle(out, center, JOINT_RADIUS, (255, 255, 255), 1)  # White outline

    return out


def draw_angle_arc(
    frame: np.ndarray,
    vertex: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
    angle_deg: float,
    label: str,
    color: tuple[int, int, int] = (255, 255, 255),
    radius: int = 30,
) -> np.ndarray:
    """Draw an angle arc at a joint with a degree label.

    Args:
        frame: BGR image.
        vertex: The joint where the angle is measured.
        point_a: One end of the angle.
        point_b: Other end of the angle.
        angle_deg: The computed angle in degrees.
        label: Text label (e.g., "90°").
        color: BGR color for the arc.
        radius: Arc radius in pixels.

    Returns:
        Frame with arc drawn.
    """
    out = frame.copy()
    v = vertex.astype(int)

    # Compute start and end angles for the arc
    vec_a = point_a - vertex
    vec_b = point_b - vertex
    angle_a = np.degrees(np.arctan2(-vec_a[1], vec_a[0]))  # OpenCV y-axis is inverted
    angle_b = np.degrees(np.arctan2(-vec_b[1], vec_b[0]))

    # Draw arc
    cv2.ellipse(out, tuple(v), (radius, radius), 0, -angle_a, -angle_b, color, 2)

    # Draw label offset from vertex
    label_pos = (v[0] + radius + 5, v[1] - 5)
    cv2.putText(out, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return out
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_skeleton.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/viz/skeleton.py tests/test_skeleton.py
git commit -m "Add skeleton drawing utilities for pose validation"
```

---

### Task 2: Trajectory Plot Generators

**Files:**
- Create: `src/viz/trajectories.py`
- Test: `tests/test_trajectories.py`

**Step 1: Write the failing test**

```python
"""Tests for trajectory plot generation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz.trajectories import (
    plot_joint_trajectory,
    plot_wrist_speed,
    plot_confidence_heatmap,
)


class TestPlotJointTrajectory:
    def test_returns_plotly_figure(self):
        y_data = np.sin(np.linspace(0, 2 * np.pi, 60))
        timestamps = np.linspace(0, 1.0, 60)
        fig = plot_joint_trajectory(
            y_data, timestamps, joint_name="Lead Knee Y",
            events={"Foot Plant": 0.5},
        )
        assert fig is not None
        assert hasattr(fig, "to_html")

    def test_handles_no_events(self):
        y_data = np.ones(30)
        timestamps = np.linspace(0, 0.5, 30)
        fig = plot_joint_trajectory(y_data, timestamps, joint_name="Test")
        assert fig is not None


class TestPlotWristSpeed:
    def test_returns_plotly_figure(self):
        speed = np.random.rand(60) * 100
        timestamps = np.linspace(0, 1.0, 60)
        fig = plot_wrist_speed(speed, timestamps, events={"Release": 0.8})
        assert fig is not None


class TestPlotConfidenceHeatmap:
    def test_returns_plotly_figure(self):
        conf_data = {
            "right_shoulder": np.random.rand(60),
            "right_elbow": np.random.rand(60),
        }
        timestamps = np.linspace(0, 1.0, 60)
        fig = plot_confidence_heatmap(conf_data, timestamps)
        assert fig is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trajectories.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Trajectory visualization for pose validation.

Generates Plotly charts showing joint trajectories, wrist speed,
and per-joint confidence over time, with detected events overlaid.
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go


def plot_joint_trajectory(
    y_data: np.ndarray,
    timestamps: np.ndarray,
    joint_name: str,
    events: Optional[dict[str, float]] = None,
    invert_y: bool = False,
) -> go.Figure:
    """Plot a joint's Y-coordinate over time with event markers.

    Args:
        y_data: Y-coordinate values per frame.
        timestamps: Timestamp in seconds per frame.
        joint_name: Label for the chart.
        events: Dict of event_name -> timestamp_seconds.
        invert_y: If True, invert Y axis (screen coords where up = lower y).

    Returns:
        Plotly Figure.
    """
    display_y = -y_data if invert_y else y_data

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=display_y,
        mode="lines",
        line=dict(color="#4A90D9", width=2),
        name=joint_name,
    ))

    if events:
        for name, t in events.items():
            if t is not None:
                fig.add_vline(
                    x=t, line_dash="dash", line_color="red",
                    annotation_text=name, annotation_position="top right",
                )

    fig.update_layout(
        title=joint_name,
        xaxis_title="Time (s)",
        yaxis_title="Position (px)" + (" (inverted)" if invert_y else ""),
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def plot_wrist_speed(
    speed: np.ndarray,
    timestamps: np.ndarray,
    events: Optional[dict[str, float]] = None,
) -> go.Figure:
    """Plot wrist speed over time with event markers.

    Args:
        speed: Speed in pixels/second per frame.
        timestamps: Timestamp in seconds per frame.
        events: Dict of event_name -> timestamp_seconds.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=speed,
        mode="lines",
        line=dict(color="#FF6B35", width=2),
        name="Wrist Speed",
        fill="tozeroy",
        fillcolor="rgba(255, 107, 53, 0.15)",
    ))

    if events:
        for name, t in events.items():
            if t is not None:
                fig.add_vline(
                    x=t, line_dash="dash", line_color="red",
                    annotation_text=name, annotation_position="top right",
                )

    fig.update_layout(
        title="Throwing Wrist Speed",
        xaxis_title="Time (s)",
        yaxis_title="Speed (px/s)",
        height=300,
        margin=dict(l=60, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


def plot_confidence_heatmap(
    confidence_data: dict[str, np.ndarray],
    timestamps: np.ndarray,
) -> go.Figure:
    """Plot per-joint confidence over time as a heatmap.

    Args:
        confidence_data: Dict of joint_name -> confidence array.
        timestamps: Timestamp in seconds per frame.

    Returns:
        Plotly Figure.
    """
    joints = sorted(confidence_data.keys())
    z_data = np.array([confidence_data[j] for j in joints])

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=timestamps,
        y=joints,
        colorscale=[
            [0, "#dc3545"],     # Red = low confidence
            [0.4, "#ffc107"],   # Yellow = medium
            [0.7, "#28a745"],   # Green = high
            [1.0, "#28a745"],
        ],
        zmin=0, zmax=1,
        colorbar=dict(title="Conf"),
    ))

    fig.update_layout(
        title="Per-Joint Confidence Over Time",
        xaxis_title="Time (s)",
        height=max(200, len(joints) * 25 + 100),
        margin=dict(l=150, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trajectories.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/viz/trajectories.py tests/test_trajectories.py
git commit -m "Add trajectory plot generators for pose validation"
```

---

### Task 3: HTML Report Generator

**Files:**
- Create: `src/viz/report.py`
- Test: `tests/test_report.py`

**Step 1: Write the failing test**

```python
"""Tests for HTML report generation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.viz.report import build_report_html


class TestBuildReportHtml:
    def test_returns_html_string(self):
        html = build_report_html(
            video_filename="test_pitch.mp4",
            video_rel_path="annotated_video.mp4",
            fps=60.0,
            frame_count=180,
            backend="yolov8",
            pitcher_throws="R",
            trajectory_plots_html=["<div>plot1</div>", "<div>plot2</div>"],
            key_frame_images={},
            metrics_rows=[],
            diagnostics={"avg_confidence": 0.82, "frames_processed": 180},
        )
        assert "<html" in html
        assert "test_pitch.mp4" in html
        assert "yolov8" in html

    def test_includes_video_tag(self):
        html = build_report_html(
            video_filename="clip.mp4",
            video_rel_path="annotated_video.mp4",
            fps=60.0,
            frame_count=100,
            backend="yolov8",
            pitcher_throws="R",
            trajectory_plots_html=[],
            key_frame_images={},
            metrics_rows=[],
            diagnostics={},
        )
        assert "<video" in html
        assert "annotated_video.mp4" in html

    def test_includes_key_frame_images(self):
        # Base64 encoded 1x1 red PNG (minimal valid PNG)
        fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        html = build_report_html(
            video_filename="clip.mp4",
            video_rel_path="annotated_video.mp4",
            fps=60.0,
            frame_count=100,
            backend="yolov8",
            pitcher_throws="R",
            trajectory_plots_html=[],
            key_frame_images={"Foot Plant": fake_b64},
            metrics_rows=[],
            diagnostics={},
        )
        assert "Foot Plant" in html
        assert "data:image/png;base64," in html

    def test_includes_metrics_table(self):
        html = build_report_html(
            video_filename="clip.mp4",
            video_rel_path="annotated_video.mp4",
            fps=60.0,
            frame_count=100,
            backend="yolov8",
            pitcher_throws="R",
            trajectory_plots_html=[],
            key_frame_images={},
            metrics_rows=[
                {"metric": "Elbow Flexion @ FP", "value": "92.3", "unit": "deg", "obp_median": "95.1", "status": "ok"},
            ],
            diagnostics={},
        )
        assert "Elbow Flexion @ FP" in html
        assert "92.3" in html
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""HTML report generator for pose validation.

Assembles annotated video, trajectory plots, key frames, and metrics
into a single self-contained HTML report.
"""

from typing import Optional


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
    """Build a self-contained HTML validation report.

    Args:
        video_filename: Original video filename.
        video_rel_path: Relative path to annotated MP4 from report location.
        fps: Video frame rate.
        frame_count: Total frames processed.
        backend: Pose estimation backend used.
        pitcher_throws: "R" or "L".
        trajectory_plots_html: List of Plotly chart HTML strings.
        key_frame_images: Dict of event_name -> base64-encoded PNG string.
        metrics_rows: List of dicts with keys: metric, value, unit, obp_median, status.
        diagnostics: Dict of diagnostic info (avg_confidence, warnings, etc.).

    Returns:
        Complete HTML string.
    """
    # Header section
    duration = frame_count / fps if fps > 0 else 0

    header_html = f"""
    <div class="section">
        <h2>Video Info</h2>
        <table class="info-table">
            <tr><td>File</td><td><strong>{video_filename}</strong></td></tr>
            <tr><td>FPS</td><td>{fps:.1f}</td></tr>
            <tr><td>Frames</td><td>{frame_count}</td></tr>
            <tr><td>Duration</td><td>{duration:.2f}s</td></tr>
            <tr><td>Backend</td><td>{backend}</td></tr>
            <tr><td>Throws</td><td>{pitcher_throws}HP</td></tr>
        </table>
    </div>
    """

    # Video section
    video_html = f"""
    <div class="section">
        <h2>Annotated Video</h2>
        <video controls width="100%" style="max-width: 800px;">
            <source src="{video_rel_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p class="hint">Scrub through to verify skeleton tracking quality.</p>
    </div>
    """

    # Trajectory plots section
    plots_content = "\n".join(
        f'<div class="plot-container">{p}</div>' for p in trajectory_plots_html
    )
    plots_html = f"""
    <div class="section">
        <h2>Joint Trajectories &amp; Events</h2>
        <p class="hint">Vertical dashed lines = detected events. Check that they align with the correct delivery phases.</p>
        {plots_content}
    </div>
    """ if trajectory_plots_html else ""

    # Key frames section
    key_frames_content = ""
    for event_name, b64_img in key_frame_images.items():
        key_frames_content += f"""
        <div class="key-frame">
            <h3>{event_name}</h3>
            <img src="data:image/png;base64,{b64_img}" alt="{event_name}" />
        </div>
        """
    key_frames_html = f"""
    <div class="section">
        <h2>Key Frames</h2>
        <div class="key-frames-grid">
            {key_frames_content}
        </div>
    </div>
    """ if key_frame_images else ""

    # Metrics table section
    metrics_table_rows = ""
    for row in metrics_rows:
        status_class = "metric-warn" if row.get("status") == "missing" else ""
        value_display = row.get("value", "N/A")
        obp_display = row.get("obp_median", "—")
        metrics_table_rows += f"""
        <tr class="{status_class}">
            <td>{row['metric']}</td>
            <td>{value_display}</td>
            <td>{row.get('unit', '')}</td>
            <td>{obp_display}</td>
            <td>{row.get('status', '')}</td>
        </tr>
        """
    metrics_html = f"""
    <div class="section">
        <h2>Extracted Metrics</h2>
        <table class="metrics-table">
            <thead>
                <tr><th>Metric</th><th>Value</th><th>Unit</th><th>OBP Median</th><th>Status</th></tr>
            </thead>
            <tbody>
                {metrics_table_rows}
            </tbody>
        </table>
    </div>
    """ if metrics_rows else ""

    # Diagnostics section
    diag_items = ""
    for k, v in diagnostics.items():
        diag_items += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
    diagnostics_html = f"""
    <div class="section">
        <h2>Diagnostics</h2>
        <table class="info-table">
            {diag_items}
        </table>
    </div>
    """ if diagnostics else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pose Validation: {video_filename}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0a; color: #e0e0e0;
            max-width: 960px; margin: 0 auto; padding: 24px;
        }}
        h1 {{ font-size: 1.5rem; margin-bottom: 8px; color: #fff; }}
        h2 {{ font-size: 1.15rem; margin-bottom: 12px; color: #ccc; border-bottom: 1px solid #333; padding-bottom: 6px; }}
        h3 {{ font-size: 0.95rem; margin-bottom: 8px; color: #aaa; }}
        .section {{ margin-bottom: 36px; }}
        .hint {{ font-size: 0.8rem; color: #888; margin-top: 6px; }}
        .info-table {{ border-collapse: collapse; }}
        .info-table td {{ padding: 4px 16px 4px 0; font-size: 0.9rem; }}
        .info-table td:first-child {{ color: #888; }}
        .metrics-table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
        .metrics-table th {{ text-align: left; padding: 8px; border-bottom: 2px solid #333; color: #aaa; }}
        .metrics-table td {{ padding: 8px; border-bottom: 1px solid #222; }}
        .metric-warn {{ color: #ffc107; }}
        .key-frames-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 16px; }}
        .key-frame img {{ width: 100%; border: 1px solid #333; border-radius: 4px; }}
        .plot-container {{ margin-bottom: 16px; }}
        video {{ border: 1px solid #333; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Pose Validation Report</h1>
    {header_html}
    {video_html}
    {plots_html}
    {key_frames_html}
    {metrics_html}
    {diagnostics_html}
</body>
</html>"""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/viz/report.py tests/test_report.py
git commit -m "Add HTML report generator for pose validation"
```

---

### Task 4: Approximate MER from 2D Keypoints

The existing `detect_max_external_rotation` requires a `shoulder_er_angle` column that isn't yet computed. Add a 2D approximation function to `events.py`.

**Files:**
- Modify: `src/biomechanics/events.py`
- Modify: `tests/test_benchmarks.py` (add test for new function)

**Step 1: Write the failing test**

Add to `tests/test_benchmarks.py`:

```python
from src.biomechanics.events import approximate_shoulder_er_2d


class TestApproximateShoulderER:
    def test_arm_behind_head_high_er(self):
        """When wrist is behind and above shoulder, ER should be high."""
        # Shoulder at (300, 200), elbow at (280, 150), wrist at (250, 120)
        # Arm is up and behind — high layback
        shoulder = np.array([300, 200])
        elbow = np.array([280, 150])
        wrist = np.array([250, 120])
        hip_center = np.array([300, 350])
        er = approximate_shoulder_er_2d(shoulder, elbow, wrist, hip_center)
        assert er > 90  # High ER

    def test_arm_forward_low_er(self):
        """When arm is forward and down, ER should be lower."""
        shoulder = np.array([300, 200])
        elbow = np.array([350, 220])
        wrist = np.array([400, 250])
        hip_center = np.array([300, 350])
        er = approximate_shoulder_er_2d(shoulder, elbow, wrist, hip_center)
        assert er < 90
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmarks.py::TestApproximateShoulderER -v`
Expected: FAIL — `ImportError: cannot import name 'approximate_shoulder_er_2d'`

**Step 3: Write minimal implementation**

Add to `src/biomechanics/events.py` (before `detect_events`):

```python
def approximate_shoulder_er_2d(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    hip_center: np.ndarray,
) -> float:
    """Approximate shoulder external rotation from 2D side-view keypoints.

    Measures the angle of the forearm relative to the trunk line.
    From side view, high ER (layback) appears as the forearm angled
    behind and above the shoulder relative to the trunk.

    This is a rough proxy — true ER requires 3D data. But it's sufficient
    for identifying the approximate frame of max ER.

    Args:
        shoulder: Throwing shoulder (x, y).
        elbow: Throwing elbow (x, y).
        wrist: Throwing wrist (x, y).
        hip_center: Midpoint of hips (x, y).

    Returns:
        Approximate ER angle in degrees (0-180).
    """
    # Trunk vector: hip center to shoulder (pointing up)
    trunk = shoulder - hip_center
    # Forearm vector: elbow to wrist
    forearm = wrist - elbow

    # Angle between forearm and trunk
    cos_angle = np.dot(forearm, trunk) / (
        np.linalg.norm(forearm) * np.linalg.norm(trunk) + 1e-8
    )
    angle = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    return angle
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_benchmarks.py::TestApproximateShoulderER -v`
Expected: Both tests PASS

**Step 5: Commit**

```bash
git add src/biomechanics/events.py tests/test_benchmarks.py
git commit -m "Add 2D shoulder external rotation approximation for event detection"
```

---

### Task 5: Main Validation Script

This is the orchestrator that ties stages 1-4 together.

**Files:**
- Create: `scripts/validate_pose.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Validate pose estimation pipeline on a pitching video.

Runs the full pipeline and produces an HTML diagnostic report with:
  1. Annotated video (skeleton overlay on every frame)
  2. Joint trajectory plots with detected events
  3. Key frame stills at delivery events
  4. Extracted metrics table with OBP context

Usage:
    python scripts/validate_pose.py --video path/to/clip.mp4
    python scripts/validate_pose.py --video clip.mp4 --throws L --backend mediapipe
"""

import argparse
import base64
import sys
import webbrowser
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def encode_frame_as_base64(frame: np.ndarray) -> str:
    """Encode a BGR frame as base64 PNG string."""
    _, buf = cv2.imencode(".png", frame)
    return base64.b64encode(buf).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Validate pose estimation on pitching video")
    parser.add_argument("--video", type=Path, required=True, help="Path to pitching video")
    parser.add_argument("--throws", type=str, default="R", choices=["R", "L"])
    parser.add_argument("--backend", type=str, default="yolov8", choices=["yolov8", "mediapipe"])
    parser.add_argument("--model-size", type=str, default="m", help="YOLOv8 model size")
    parser.add_argument("--confidence", type=float, default=0.3, help="Min keypoint confidence")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-open", action="store_true", help="Don't open report in browser")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    # Output directory
    video_stem = args.video.stem
    output_dir = args.output_dir or Path(f"data/outputs/validate_{video_stem}")
    output_dir.mkdir(parents=True, exist_ok=True)
    key_frames_dir = output_dir / "key_frames"
    key_frames_dir.mkdir(exist_ok=True)

    print(f"\nPose Validation: {args.video.name}")
    print(f"Output: {output_dir}")
    print(f"Backend: {args.backend}, Throws: {args.throws}HP")
    print("=" * 50)

    # ── Stage 1: Pose Estimation ──────────────────────────────────────
    print("\n[1/4] Running pose estimation...")
    from src.pose.estimator import extract_poses, load_video

    video_info = load_video(args.video)
    print(f"  Video: {video_info.width}x{video_info.height} @ {video_info.fps:.1f}fps, "
          f"{video_info.total_frames} frames ({video_info.duration_sec:.2f}s)")

    kwargs = {"confidence": args.confidence}
    if args.backend == "yolov8":
        kwargs["model_size"] = args.model_size
    pose_seq = extract_poses(args.video, backend=args.backend, **kwargs)
    print(f"  Extracted keypoints from {len(pose_seq.frames)} frames")

    if len(pose_seq.frames) == 0:
        print("  ERROR: No poses detected. Check video content and confidence threshold.")
        sys.exit(1)

    # Write annotated video
    print("  Writing annotated video...")
    from src.viz.skeleton import draw_skeleton

    annotated_path = output_dir / "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(annotated_path),
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )

    cap = cv2.VideoCapture(str(args.video))
    frame_lookup = {f.frame_idx: f for f in pose_seq.frames}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_lookup:
            pf = frame_lookup[frame_idx]
            frame = draw_skeleton(
                frame, pf.keypoints, pf.confidence,
                bbox=pf.bbox, min_confidence=args.confidence,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Saved: {annotated_path}")

    # ── Stage 2: Event Detection + Trajectory Plots ───────────────────
    print("\n[2/4] Detecting events and generating trajectory plots...")

    lead_side = "left" if args.throws == "R" else "right"
    throw_side = "right" if args.throws == "R" else "left"

    # Extract trajectories
    timestamps = np.array([f.timestamp for f in pose_seq.frames])

    lead_knee_traj = pose_seq.get_joint_trajectory(f"{lead_side}_knee")
    lead_ankle_traj = pose_seq.get_joint_trajectory(f"{lead_side}_ankle")
    throw_wrist_traj = pose_seq.get_joint_trajectory(f"{throw_side}_wrist")

    # Knee and ankle Y (invert: screen coords where lower y = higher position)
    lead_knee_y = -lead_knee_traj[:, 1]  # Invert so higher = up
    lead_ankle_y = -lead_ankle_traj[:, 1]

    # Wrist speed (pixels/second)
    wrist_speed_raw = pose_seq.get_joint_speed(f"{throw_side}_wrist") * video_info.fps
    # Smooth with 3-frame moving average
    kernel = np.ones(3) / 3
    wrist_speed = np.convolve(wrist_speed_raw, kernel, mode="same")

    # Compute approximate shoulder ER over time for MER detection
    from src.biomechanics.events import (
        detect_leg_lift,
        detect_foot_plant_from_keypoints,
        detect_max_external_rotation,
        detect_ball_release,
        approximate_shoulder_er_2d,
        DeliveryEvents,
    )

    shoulder_er_series = np.zeros(len(pose_seq.frames))
    for i, pf in enumerate(pose_seq.frames):
        ts = f"{throw_side}_shoulder"
        te = f"{throw_side}_elbow"
        tw = f"{throw_side}_wrist"
        lh = f"{lead_side}_hip"
        rh = f"{throw_side}_hip"
        if all(k in pf.keypoints for k in [ts, te, tw, lh, rh]):
            hip_center = (pf.keypoints[lh] + pf.keypoints[rh]) / 2
            shoulder_er_series[i] = approximate_shoulder_er_2d(
                pf.keypoints[ts], pf.keypoints[te], pf.keypoints[tw], hip_center,
            )

    # Run event detection
    events = DeliveryEvents(fps=video_info.fps)
    events.leg_lift_apex = detect_leg_lift(lead_knee_y)
    events.foot_plant = detect_foot_plant_from_keypoints(
        lead_ankle_traj[:, 1], fps=video_info.fps,
    )
    events.max_external_rotation = detect_max_external_rotation(
        shoulder_er_series, after_frame=events.foot_plant,
    )
    events.ball_release = detect_ball_release(
        wrist_speed, after_frame=events.max_external_rotation,
    )

    # Convert event frames to timestamps
    def event_time(frame_idx):
        if frame_idx is None:
            return None
        return frame_idx / video_info.fps

    event_times = {
        "Leg Lift": event_time(events.leg_lift_apex),
        "Foot Plant": event_time(events.foot_plant),
        "Max ER": event_time(events.max_external_rotation),
        "Ball Release": event_time(events.ball_release),
    }

    for name, t in event_times.items():
        status = f"{t:.3f}s" if t is not None else "NOT DETECTED"
        print(f"  {name}: {status}")

    # Generate trajectory plots
    from src.viz.trajectories import (
        plot_joint_trajectory,
        plot_wrist_speed,
        plot_confidence_heatmap,
    )

    trajectory_plots = []

    knee_fig = plot_joint_trajectory(
        lead_knee_y, timestamps, f"{lead_side.title()} Knee Y (inverted)",
        events={"Leg Lift": event_times["Leg Lift"], "Foot Plant": event_times["Foot Plant"]},
    )
    trajectory_plots.append(knee_fig.to_html(full_html=False, include_plotlyjs=False))

    ankle_fig = plot_joint_trajectory(
        lead_ankle_y, timestamps, f"{lead_side.title()} Ankle Y (inverted)",
        events={"Foot Plant": event_times["Foot Plant"]},
    )
    trajectory_plots.append(ankle_fig.to_html(full_html=False, include_plotlyjs=False))

    speed_fig = plot_wrist_speed(
        wrist_speed, timestamps,
        events={"Max ER": event_times["Max ER"], "Ball Release": event_times["Ball Release"]},
    )
    trajectory_plots.append(speed_fig.to_html(full_html=False, include_plotlyjs=False))

    # Confidence heatmap
    conf_data = {}
    for joint in [f"{throw_side}_shoulder", f"{throw_side}_elbow", f"{throw_side}_wrist",
                  f"{lead_side}_hip", f"{lead_side}_knee", f"{lead_side}_ankle"]:
        conf_data[joint] = np.array([
            pf.confidence.get(joint, 0) for pf in pose_seq.frames
        ])
    conf_fig = plot_confidence_heatmap(conf_data, timestamps)
    trajectory_plots.append(conf_fig.to_html(full_html=False, include_plotlyjs=False))

    # ── Stage 3: Key Frame Extraction ─────────────────────────────────
    print("\n[3/4] Extracting key frames...")
    from src.viz.skeleton import draw_angle_arc
    from src.biomechanics.features import angle_between_points, compute_trunk_tilt

    key_frame_images = {}
    event_frame_map = {
        "Leg Lift": events.leg_lift_apex,
        "Foot Plant": events.foot_plant,
        "Max ER": events.max_external_rotation,
        "Ball Release": events.ball_release,
    }

    cap = cv2.VideoCapture(str(args.video))
    for event_name, frame_idx in event_frame_map.items():
        if frame_idx is None:
            print(f"  {event_name}: skipped (not detected)")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw skeleton
        if frame_idx in frame_lookup:
            pf = frame_lookup[frame_idx]
            frame = draw_skeleton(frame, pf.keypoints, pf.confidence, bbox=pf.bbox)

            # Draw angle overlays at foot plant and ball release
            ts = f"{throw_side}_shoulder"
            te = f"{throw_side}_elbow"
            tw = f"{throw_side}_wrist"

            if event_name == "Foot Plant" and all(k in pf.keypoints for k in [ts, te, tw]):
                elbow_angle = angle_between_points(
                    pf.keypoints[ts], pf.keypoints[te], pf.keypoints[tw],
                )
                frame = draw_angle_arc(
                    frame, pf.keypoints[te], pf.keypoints[ts], pf.keypoints[tw],
                    elbow_angle, f"{elbow_angle:.0f} deg",
                    color=(0, 255, 255), radius=35,
                )

        # Add event label
        cv2.putText(
            frame, f"{event_name} (frame {frame_idx})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Save as file and encode for HTML
        out_path = key_frames_dir / f"{event_name.lower().replace(' ', '_')}.png"
        cv2.imwrite(str(out_path), frame)
        key_frame_images[event_name] = encode_frame_as_base64(frame)
        print(f"  {event_name}: frame {frame_idx} saved")

    cap.release()

    # ── Stage 4: Metrics Extraction ───────────────────────────────────
    print("\n[4/4] Extracting metrics...")
    from src.biomechanics.features import extract_metrics
    from src.biomechanics.benchmarks import OBPBenchmarks, METRIC_DISPLAY_NAMES

    keypoints_dict = pose_seq.to_keypoints_dict()
    metrics = extract_metrics(keypoints_dict, events, pitcher_throws=args.throws)
    obp_dict = metrics.to_obp_comparison_dict()

    # Load OBP medians for context
    obp_medians = {}
    obp_data_path = Path("data/obp")
    if (obp_data_path / "poi_metrics.csv").exists():
        obp = OBPBenchmarks(obp_data_path=obp_data_path).load()
        for b in obp.compute_benchmarks(metrics=list(obp_dict.keys())):
            obp_medians[b.metric] = b.percentiles[50]

    metrics_rows = []
    all_metric_fields = [
        ("elbow_flexion_fp", "Elbow Flexion @ FP", "deg"),
        ("shoulder_abduction_fp", "Shoulder Abduction @ FP", "deg"),
        ("shoulder_horizontal_abduction_fp", "Shoulder Horiz. Abduction @ FP", "deg"),
        ("torso_anterior_tilt_fp", "Forward Trunk Tilt @ FP", "deg"),
        ("torso_lateral_tilt_fp", "Lateral Trunk Tilt @ FP", "deg"),
        ("lead_knee_angle_fp", "Lead Knee Angle @ FP", "deg"),
        ("hip_shoulder_separation_fp", "Hip-Shoulder Separation @ FP", "deg"),
        ("max_shoulder_external_rotation", "Peak Shoulder ER", "deg"),
        ("torso_anterior_tilt_br", "Forward Trunk Tilt @ Release", "deg"),
        ("arm_slot_angle", "Arm Slot Angle", "deg"),
        ("lead_knee_angle_br", "Lead Knee Angle @ Release", "deg"),
    ]

    for field_name, display_name, unit in all_metric_fields:
        value = getattr(metrics, field_name, None)
        obp_key = field_name
        # Map internal names to OBP names where different
        if field_name == "hip_shoulder_separation_fp":
            obp_key = "rotation_hip_shoulder_separation_fp"

        obp_med = obp_medians.get(obp_key)
        metrics_rows.append({
            "metric": display_name,
            "value": f"{value:.1f}" if value is not None else "—",
            "unit": unit,
            "obp_median": f"{obp_med:.1f}" if obp_med is not None else "—",
            "status": "ok" if value is not None else "missing",
        })

    computed = sum(1 for r in metrics_rows if r["status"] == "ok")
    print(f"  Computed {computed}/{len(metrics_rows)} metrics")

    # ── Assemble Report ───────────────────────────────────────────────
    print("\nAssembling report...")
    from src.viz.report import build_report_html

    # Compute diagnostics
    all_confs = []
    for pf in pose_seq.frames:
        all_confs.extend(pf.confidence.values())
    avg_conf = float(np.mean(all_confs)) if all_confs else 0

    diagnostics = {
        "frames_processed": len(pose_seq.frames),
        "frames_with_poses": sum(1 for f in pose_seq.frames if len(f.keypoints) > 0),
        "avg_confidence": f"{avg_conf:.3f}",
        "events_detected": sum(1 for v in event_frame_map.values() if v is not None),
        "metrics_computed": computed,
    }

    html = build_report_html(
        video_filename=args.video.name,
        video_rel_path="annotated_video.mp4",
        fps=video_info.fps,
        frame_count=len(pose_seq.frames),
        backend=args.backend,
        pitcher_throws=args.throws,
        trajectory_plots_html=trajectory_plots,
        key_frame_images=key_frame_images,
        metrics_rows=metrics_rows,
        diagnostics=diagnostics,
    )

    report_path = output_dir / "report.html"
    report_path.write_text(html)
    print(f"  Saved: {report_path}")

    if not args.no_open:
        webbrowser.open(f"file://{report_path.resolve()}")
        print("  Opened in browser")

    print("\nDone!")


if __name__ == "__main__":
    main()
```

**Step 2: Run a smoke test (no video needed — just verify imports)**

Run: `python scripts/validate_pose.py --help`
Expected: Prints usage without errors

**Step 3: Commit**

```bash
git add scripts/validate_pose.py
git commit -m "Add pose validation script with full pipeline and HTML report"
```

---

### Task 6: Run All Tests and Push

**Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (existing + new)

**Step 2: Run linter**

Run: `ruff check src/ tests/ scripts/`
Fix any issues.

**Step 3: Push**

```bash
git push origin main
```
