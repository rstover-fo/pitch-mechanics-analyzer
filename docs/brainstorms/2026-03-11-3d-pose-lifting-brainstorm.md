# 3D Pose Lifting with MotionBERT

**Date:** 2026-03-11
**Status:** Brainstorm
**Phase:** 2.5 (enhancement to existing pose pipeline)

## What We're Building

Add a 3D pose lifting stage to the pipeline using MotionBERT. The lifter sits between
YOLOv8 2D pose estimation and event detection, converting (x, y) keypoints to (x, y, z).
This unlocks 4 currently unmeasurable OBP metrics and improves accuracy of the 7 we
already compute by removing 2D projection distortion.

## Why This Approach

- **Drop-in replacement**: PoseFrame keypoints go from (x, y) to (x, y, z). Downstream
  code gains z-awareness incrementally — existing 2D calculations still work as fallback.
- **MotionBERT over alternatives**: Best accuracy on Human3.6M (39mm MPJPE), temporal
  model that uses surrounding frames for smoother joint trajectories, well-maintained.
- **No new heavy deps**: PyTorch already installed via ultralytics/YOLOv8.
- **Works retroactively**: Can re-process all 4 existing driveway clips without re-filming.
- **Graceful degradation**: If MotionBERT weights aren't downloaded, pipeline runs in
  2D-only mode exactly as it does today.

## Key Decisions

1. **Integration style**: Drop-in replacement, not parallel track or separate backend.
   PoseFrame grows a z-coordinate; downstream code checks for z presence.

2. **Model**: MotionBERT. Temporal lifting (uses sequence context) is better for smooth
   pitching motion than single-frame models.

3. **Joint mapping**: MotionBERT uses Human3.6M skeleton (17 joints). YOLOv8 uses COCO
   (17 joints). Need a COCO → H36M mapping layer. 12 of our pitching joints have direct
   equivalents; may need interpolation for hip center.

4. **Output coordinate space**: MotionBERT outputs in camera-relative millimeters.
   Normalize to body-height units (divide by torso length) for scale-invariant metrics.

5. **Backward compatibility**: `PoseFrame.keypoints` values become (x, y, z) ndarrays.
   Existing code that indexes `[:2]` or uses only x/y continues to work. Feature
   extraction checks `keypoints.shape[-1] == 3` to decide 2D vs 3D angle calc.

## New Capabilities (with 3D)

### Currently unmeasurable metrics (unlocked)
- **Shoulder abduction** — arm angle relative to torso in frontal plane
- **Shoulder horizontal abduction** — arm position in transverse plane
- **Torso lateral tilt** — sideways trunk lean
- **Hip-shoulder separation** — differential rotation between pelvis and shoulders

### Existing metrics improved
- **Peak layback (shoulder ER)** — true 3D rotation instead of 2D projection (~85-112 current vs ~170 OBP because 2D can't see the full rotation arc)
- **Forward trunk tilt** — measured in sagittal plane regardless of camera angle
- **Elbow flexion** — true joint angle, not projected angle (removes the 137 vs 93 pitch-to-pitch noise)
- **Arm slot** — currently reads 15-22 (low) due to projection; 3D gives true release angle
- **Stride length** — depth component captured

## Architecture

```
Video
  → YOLOv8 (2D keypoints per frame, 17 COCO joints)
  → Joint mapper (COCO 17 → H36M 17)
  → MotionBERT (sequence of 2D joints → 3D joints in camera coords)
  → Joint mapper (H36M 17 → 12 pitching joints with z)
  → PoseFrame with (x, y, z) keypoints
  → Event detection (unchanged — uses x, y only)
  → Feature extraction (upgraded — 3D angles when z available)
  → OBP comparison (more metrics, better accuracy)
  → Coaching report (richer data for Claude API)
```

### New Files
- `src/pose/lifter.py` — MotionBERT wrapper, model loading, joint mapping, inference
- `src/biomechanics/angles_3d.py` — 3D angle calculations (cross products, rotation matrices)

### Modified Files
- `src/pose/estimator.py` — PoseFrame keypoints shape (2,) → (3,), call lifter after 2D extraction
- `src/biomechanics/features.py` — dispatch to 3D angle functions when z present
- `scripts/validate_pose.py` — no CLI changes needed; auto-detects 3D availability

## Open Questions

1. **MotionBERT model variant**: The repo has multiple checkpoints (pose, mesh, action).
   We need the "pose" variant. Need to verify exact checkpoint file and download process.

2. **Temporal window size**: MotionBERT processes sequences of N frames. Need to determine
   optimal window (full delivery? fixed 243-frame window?) and how to handle videos
   shorter/longer than the window.

3. **COCO → H36M joint mapping accuracy**: Some joints don't map 1:1 (e.g., H36M has
   "spine" and "neck" that COCO doesn't). Need to validate which pitching-relevant
   joints have clean mappings vs. need interpolation.

4. **Inference speed**: MotionBERT on CPU vs GPU. Our videos are ~120-220 frames.
   Need to benchmark — if it's 30+ seconds on CPU, may want to note GPU recommendation.

5. **Validation**: How do we know the 3D output is correct? We don't have ground-truth
   3D data. Could compare lifted metrics against known OBP distributions to see if
   percentiles become more plausible (e.g., layback moving from 0th to a reasonable range).

6. **Coordinate alignment**: MotionBERT outputs in camera coordinates. For pitching
   metrics, do we need to rotate into a pitching-aligned coordinate system (e.g.,
   x = toward home plate, y = up, z = toward 3rd base)?

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MotionBERT accuracy on pitching motion (trained on walking/daily activities) | Medium | High | Validate against OBP percentile plausibility; fall back to 2D if nonsensical |
| Joint mapping gaps between COCO and H36M | Low | Medium | 12 pitching joints mostly overlap; interpolate hip center from L/R hip |
| Slow inference on CPU | Medium | Low | Pipeline already takes ~10s for YOLOv8; add a few seconds acceptable |
| Breaking existing 2D tests | Low | Medium | Feature flag on z-dimension presence; existing tests pass with (x,y) data |

## Success Criteria

- All 113 existing tests continue to pass (2D fallback works)
- 3D lifting runs on all 4 existing clips without errors
- Peak layback percentile moves from 0th to a plausible range (20th-80th)
- At least 2 of the 4 previously unmeasurable metrics become computable
- Pipeline runtime increases by < 30 seconds per clip on CPU
