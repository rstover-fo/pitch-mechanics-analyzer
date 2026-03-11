# Validation Framework Design

**Date:** 2026-03-10
**Status:** Approved
**Purpose:** Validate and stress test the full video-to-metrics pipeline across all layers: pose estimation, event detection, metrics extraction.

## Overview

The pipeline has zero quantitative validation — 82 tests, but all on synthetic data or module-level units. We need ground truth comparison, consistency measurement, and failure mode detection to know which numbers to trust and where to focus tuning.

## Architecture

```
Ground Truth Labeling          Pipeline Output
  (human frame labels)    →    (detected events/metrics)
         ↓                              ↓
    Event Accuracy Scoring    Cross-Pitch Consistency
         ↓                              ↓
              Failure Mode Detection
                      ↓
              Validation Report
```

## Section 1: Frame Reviewer & Ground Truth Labeling

**Script:** `scripts/label_events.py`

Takes an annotated video + pipeline detected events, extracts every frame as JPEG, generates a self-contained HTML page with:
- Arrow-key frame-by-frame navigation
- Frame number and timestamp display
- Detected event markers (colored highlight when on a detected frame)
- "Mark as X" buttons for each event type (Leg Lift, Foot Plant, MER, Ball Release)
- Export to ground truth JSON

**Ground truth format:** `data/ground_truth/<video_stem>.json`
```json
{
  "video": "IMG_3108.MOV",
  "labeler": "rob",
  "events": {
    "leg_lift": 65,
    "foot_plant": 79,
    "max_er": 84,
    "ball_release": 89
  },
  "calibration_frames": {
    "standing": { "frame": 12, "expected_elbow_flex": 175, "expected_knee_angle": 175 }
  },
  "notes": ""
}
```

**Pitcher registry:** `data/ground_truth/pitchers.json`
```json
{
  "player_a": ["IMG_3106", "IMG_3107"],
  "player_b": ["IMG_3108", "IMG_3109"]
}
```

**Initial labeling set:** 4 driveway clips (2 pitchers × 2 pitches), 4 events each = 32 labeled events.

### Event Identification Guide

| Event | What to look for |
|-------|-----------------|
| Leg Lift | Lead knee at highest point, just before stride leg goes forward/down |
| Foot Plant | Stride foot touches ground, lead foot stops moving down |
| Max ER | Throwing arm at maximum layback — forearm pointed most toward sky/behind body |
| Ball Release | Arm fully extended forward, wrist at most forward point (ball may not be visible at 30fps) |

## Section 2: Event Detection Accuracy Scoring

**Script:** `scripts/eval_events.py`

Compares pipeline detections against ground truth labels.

### Per-event per-clip metrics
- **Frame error** (signed): `detected_frame - ground_truth_frame`
- **Time error**: frame error × (1/fps) in milliseconds
- **Hit rate**: % of events detected (vs None)

### Aggregated metrics
- **MAE** per event type across all clips
- **Bias** per event type (mean signed error — reveals systematic early/late)
- **Per-pitcher grouping** to check if errors are pitcher-dependent

### Output format
Terminal summary table:
```
Event Detection Accuracy (4 clips):
  Leg Lift:     MAE 2.5 frames (83ms)   bias: +1.2 (late)
  Foot Plant:   MAE 1.8 frames (60ms)   bias: -0.5 (early)
  Max ER:       MAE 3.0 frames (100ms)  bias: +2.8 (late)
  Ball Release: MAE 4.2 frames (140ms)  bias: +3.5 (late)
```

## Section 3: Cross-Pitch Consistency Scoring

**Script:** `scripts/eval_consistency.py`

Groups clips by pitcher (from pitchers.json), computes within-pitcher metric stability.

### Per-metric per-pitcher
- Mean, std dev, coefficient of variation (CV = std/mean)
- Flag metrics with CV > 15% as unstable

### Output format
```
Player A (2 pitches):
  Metric                    P1       P2      CV      Status
  Elbow Flex @ FP          174.8°   154.4°   8.7%   variable
  Trunk Tilt @ FP           19.4°    19.5°   0.4%   stable
  Peak Shoulder ER         166.2°   159.6°   2.9%   stable
```

### Limitations
With only 2 pitches per pitcher, CV is noisy. The value improves as more clips are filmed per pitcher. Even with 2 clips, large discrepancies (like elbow flex varying 80°+) are clearly meaningful.

## Section 4: Failure Mode Detection

**Module:** `src/biomechanics/validation.py`

Runs automatically at the end of `validate_pose.py`. Returns warnings with severity levels.

### Sanity checks

| Check | Rule | Severity |
|-------|------|----------|
| Event ordering | LL < FP < MER < BR | error |
| Phase: LL→FP | 0.1s - 1.5s | warning |
| Phase: FP→MER | 0.01s - 0.5s | warning |
| Phase: MER→BR | 0.01s - 0.5s | warning |
| Elbow flex @ FP | 40° - 160° | warning |
| Peak shoulder ER | 60° - 190° | warning |
| Mean keypoint confidence | > 0.3 | warning |
| Confidence at event frames | Key joints > 0.2 | warning |
| Stride dip depth | > 5% of baseline | warning |
| Missing events | Any event is None | error |

Ranges from biomechanical literature (ASMI, Driveline) widened for youth + 2D noise. Not pass/fail gates — flags for human review.

### Output
Warnings list in the report diagnostics section:
```
Warnings:
  [warning] arm_accel_duration_implausible: MER->BR is 0.37s (expected 0.01-0.5s)
  [warning] poor_confidence_at_event: right_wrist confidence 0.18 at Ball Release
```

## Section 5: Metrics Accuracy via Calibration Poses

Uses known poses (standing, optionally T-pose) as calibration to measure angle extraction bias.

### Standing calibration (free — clips already start with standing)
- Expected: elbow flex ≈ 175°, knee angle ≈ 175°, trunk tilt ≈ 0-5°
- Pipeline extracts angles from calibration frame
- Difference = measurement bias for this camera angle

### Added to ground truth JSON
```json
"calibration_frames": {
  "standing": { "frame": 12, "expected_elbow_flex": 175, "expected_knee_angle": 175 }
}
```

### Output
Per-clip calibration error:
```
Calibration (standing frame 12):
  Elbow flex:  extracted 165°  expected 175°  error: -10°
  Knee angle:  extracted 172°  expected 175°  error: -3°
  Trunk tilt:  extracted 2°    expected 0°    error: +2°
```

## Deliverables Summary

| Deliverable | Type | Purpose |
|-------------|------|---------|
| `scripts/label_events.py` | Script → HTML | Frame reviewer + ground truth labeling |
| `scripts/eval_events.py` | Script | Event detection accuracy scoring |
| `scripts/eval_consistency.py` | Script | Cross-pitch metric consistency |
| `src/biomechanics/validation.py` | Module | Automatic sanity checks + warnings |
| `data/ground_truth/*.json` | Data | Human-labeled event frames + calibration |
| `data/ground_truth/pitchers.json` | Data | Clip-to-pitcher mapping |

## Priority Order

1. Frame reviewer + ground truth labeling (enables everything else)
2. Failure mode detection (automatic, no ground truth needed)
3. Event detection accuracy scoring (requires labels from step 1)
4. Cross-pitch consistency (requires pipeline outputs, already have them)
5. Metrics accuracy via calibration (requires calibration frame labels)
