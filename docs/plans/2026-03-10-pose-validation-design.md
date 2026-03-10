# Pose Validation Script Design

**Date:** 2026-03-10
**Status:** Approved
**Phase:** 2 (Video → Pose → Validated Keypoints)

## Purpose

Build `scripts/validate_pose.py` that takes a pitching video and produces a layered HTML report for visually verifying each stage of the pipeline: pose estimation quality, event detection accuracy, and extracted metric plausibility.

## Context

- Camera: iPhone at 60fps, side view (perpendicular to pitch direction)
- Pitcher: right-handed youth pitcher (configurable)
- Backend: YOLOv8-pose primary, MediaPipe fallback
- Goal: diagnostic tool for human review, not automated judgment

## Script Interface

```bash
python scripts/validate_pose.py --video path/to/clip.mp4 [options]
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--video` | required | Path to pitching video |
| `--throws` | `R` | Pitcher handedness |
| `--backend` | `yolov8` | Pose backend (`yolov8` or `mediapipe`) |
| `--model-size` | `m` | YOLOv8 variant (accuracy over speed for validation) |
| `--confidence` | `0.3` | Min keypoint confidence (low to see what's missed) |
| `--output-dir` | `data/outputs/validate_<video_stem>/` | Output location |

No target FPS resampling — process every frame for full visibility. Opens HTML report in default browser on completion.

## Output Structure

```
data/outputs/validate_<video_stem>/
  ├── annotated_video.mp4      # Skeleton overlay on every frame
  ├── key_frames/              # Annotated stills at detected events
  │   ├── leg_lift.png
  │   ├── foot_plant.png
  │   ├── max_er.png
  │   └── ball_release.png
  └── report.html              # Everything assembled
```

## Pipeline Stages

### Stage 1: Pose Estimation → Annotated Video

Run pose estimation on every frame. Draw skeleton overlay:
- Joint connections: shoulder-elbow-wrist (both arms), shoulder-shoulder, hip-hip, shoulder-hip (both sides), hip-knee-ankle (both legs)
- Joint dots: 6px radius
- Color by confidence: green (>0.7), yellow (0.4-0.7), red (<0.4)
- Bounding box around detected pitcher
- Export as MP4 via OpenCV VideoWriter

### Stage 2: Event Detection → Trajectory Plots

Extract joint trajectories, run event detection, generate Plotly charts:
- Lead knee Y over time — leg lift apex marked
- Lead ankle Y over time — foot plant marked
- Throwing wrist speed over time — MER and ball release marked
- Per-joint confidence over time — tracking dropout visibility

Events shown as vertical dashed lines with labels on each chart.

### Stage 3: Key Frame Extraction → Annotated Stills

For each detected event, extract the frame and draw:
- Skeleton overlay
- Angle arcs at joints (OpenCV ellipse) with degree labels
- Foot plant: elbow flexion, trunk tilt
- Ball release: arm slot, trunk tilt
- Limit to 2-3 angles per frame for readability
- Skip undetected events, note in report

### Stage 4: Metrics Extraction → Summary Table

Run `extract_metrics()` with detected events. Present table:
- Metric name, value, unit, OBP median for context
- Flag metrics that returned None (unreliable joints)

## HTML Report

Single self-contained file (inline CSS, no external deps):
1. Header: video filename, FPS, frame count, backend, handedness
2. Annotated video: `<video>` tag referencing the MP4
3. Trajectory plots: Plotly charts as inline JSON
4. Key frames: base64-encoded PNGs with annotations
5. Metrics table: extracted values with OBP context
6. Diagnostics: frames processed, avg confidence per joint, warnings

## Implementation Details

### MER Approximation
Current `detect_max_external_rotation` needs `shoulder_er_angle` not yet computed by the pose estimator. Approximate from 2D: angle of forearm relative to trunk line when arm is in layback (wrist-behind-shoulder position). Rough proxy from side view, sufficient for frame marking.

### Wrist Speed
Use `PoseSequence.get_joint_speed()` (pixels/frame), convert to pixels/second (multiply by FPS), smooth with 3-frame moving average. Peak wrist speed approximates ball release.

### Skeleton Connection Map
```
shoulder ─ elbow ─ wrist    (throwing arm)
shoulder ─ elbow ─ wrist    (glove arm)
shoulder ─ shoulder          (shoulder line)
hip ─ hip                    (hip line)
shoulder ─ hip               (trunk, both sides)
hip ─ knee ─ ankle           (both legs)
```
Lines: 2px. Dots: 6px radius.

## Scope Boundaries

This script does NOT:
- Run youth normalization
- Generate coaching reports
- Batch-process multiple videos
- Make automated good/bad judgments

It is purely a diagnostic tool for visual verification of the pipeline.
