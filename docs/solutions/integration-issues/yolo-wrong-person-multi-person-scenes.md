---
title: "YOLOv8 Tracks Wrong Person in Multi-Person Scenes"
date: 2026-03-10
category: integration-issues
module: src/pose/estimator.py
tags: [yolov8, pose-estimation, person-tracking, roi]
symptoms: [wrong person tracked, nonsensical metrics, no error thrown]
---

# YOLOv8 Tracks Wrong Person in Multi-Person Scenes

## Problem

When multiple people appear in the video frame (e.g., pitcher on the mound with a parent standing behind the backstop), the pose estimation pipeline silently tracks the wrong person. All downstream biomechanical metrics are computed on the wrong body, producing garbage output with no errors or warnings.

## Symptoms

- **Nonsensical metrics**: Elbow flexion, hip-shoulder separation, and other angles returned values consistent with a standing person rather than a pitching motion.
- **No errors thrown**: The pipeline completed successfully. Every stage (pose estimation, event detection, feature extraction) ran without raising an exception.
- **Event detection failed silently**: Delivery events (leg lift, foot plant, MER, ball release) either were not detected or were placed at arbitrary frames, because the tracked body was not performing a pitching motion.
- **Difficult to diagnose**: Without visually inspecting the annotated video output, there was no obvious signal that the wrong person was being tracked.

## Root Cause

`extract_poses_yolo()` in `src/pose/estimator.py` always selected `kpts.data[0]` -- the highest-confidence person detection. YOLOv8's person detection confidence correlates primarily with:

1. **Proximity to the camera** (closer = larger bounding box = higher confidence)
2. **Visibility of keypoints** (standing upright with all joints visible scores higher than a pitcher in a dynamic, partially occluded pose)

In a typical youth pitching video, a parent or coach standing near the backstop (closer to the camera) often produces a higher confidence detection than the pitcher on the mound (farther away, limbs overlapping during the delivery).

The original code had no mechanism to distinguish which detected person was the pitcher:

```python
# Before fix: always took the first (highest confidence) detection
person_kpts = kpts.data[0].cpu().numpy()
```

## Solution

Added ROI (region of interest) based person selection. The user specifies a rectangular region containing the pitcher, and the pipeline selects the detected person whose bounding box center is closest to the ROI center.

### 1. `_select_person_by_roi()` function in `src/pose/estimator.py`

```python
def _select_person_by_roi(
    boxes: np.ndarray,
    roi: tuple[int, int, int, int],
) -> int:
    """Select the detected person whose bbox center is closest to the ROI center.

    Args:
        boxes: (N, 4) array of bounding boxes [x1, y1, x2, y2].
        roi: Region of interest as (x1, y1, x2, y2) in pixels.

    Returns:
        Index of the best-matching person detection.
    """
    roi_cx = (roi[0] + roi[2]) / 2
    roi_cy = (roi[1] + roi[3]) / 2

    best_idx = 0
    best_dist = float("inf")
    for i, box in enumerate(boxes):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        dist = np.sqrt((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    return best_idx
```

### 2. Integration in `extract_poses_yolo()`

The `roi` parameter is threaded through the extraction loop. When `roi` is set and multiple people are detected, `_select_person_by_roi` replaces the default index-0 selection:

```python
# Select which person to track
person_idx = 0
if roi is not None and results[0].boxes is not None:
    all_boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(all_boxes) > 1:
        person_idx = _select_person_by_roi(all_boxes, roi)

person_kpts = kpts.data[person_idx].cpu().numpy()
```

Key details:
- When `roi` is `None` (default), behavior is unchanged -- index 0 (highest confidence) is used.
- When only one person is detected, ROI selection is skipped (no ambiguity to resolve).
- The selected person's bounding box is also stored in the `PoseFrame` for downstream inspection.

### 3. CLI argument in `scripts/validate_pose.py`

```python
parser.add_argument(
    "--roi", type=str, default=None,
    help="Region of interest as x1,y1,x2,y2 (pixels). Selects the person "
         "closest to this region instead of highest confidence.",
)
```

Parsed and forwarded to the pose extraction backend:

```python
roi = None
if args.roi:
    parts = [int(x.strip()) for x in args.roi.split(",")]
    if len(parts) != 4:
        print("Error: --roi must be x1,y1,x2,y2 (4 integers)")
        sys.exit(1)
    roi = tuple(parts)
```

### Usage

```bash
# Open the first frame to identify pixel coordinates of the pitcher
# Then pass the ROI as x1,y1,x2,y2
python scripts/validate_pose.py \
    --video data/uploads/game_clip.mp4 \
    --roi 200,100,500,600
```

The ROI does not need to be precise. It only needs to be closer to the pitcher than to other people in the frame. A rough bounding box around where the pitcher stands on the mound is sufficient.

## Prevention

1. **Always visually verify** the annotated video output (`annotated_video.mp4`) before trusting metrics. The skeleton overlay makes it immediately obvious if the wrong person is being tracked.
2. **Use `--roi` by default** for any video where bystanders, coaches, or catchers might be in frame. This is the common case for youth baseball recordings.
3. **Check bbox consistency**: If the tracked person's bounding box position jumps dramatically between frames, it may indicate the tracker switched between people. Future work could add automatic detection of person-switching via bbox trajectory smoothness.
4. **Consider adding automatic pitcher detection**: A more robust long-term solution would identify the pitcher by motion pattern (the person performing the windup/delivery) rather than requiring manual ROI specification. This is a Phase 2+ enhancement.

## Related

- `src/pose/estimator.py` -- pose extraction pipeline with ROI selection
- `scripts/validate_pose.py` -- CLI entry point with `--roi` argument
- `src/viz/skeleton.py` -- skeleton overlay for visual verification of tracked person
- MediaPipe backend (`extract_poses_mediapipe`) does not have this issue because it tracks a single person by design, but it is less robust for fast pitching motions
