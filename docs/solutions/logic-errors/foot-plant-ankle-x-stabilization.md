---
title: "Foot Plant Detection Early by 8-13 Frames Due to Y-Recovery Ground Plane Assumption"
date: 2026-03-10
category: logic-errors
module: src/biomechanics/events.py
tags: [event-detection, foot-plant, ankle-tracking, ground-plane, camera-angle]
symptoms: [foot plant too early, inconsistent across pitchers, ankle still moving at detected FP]
---

# Foot Plant Detection Early by 8-13 Frames Due to Y-Recovery Ground Plane Assumption

## Problem

Foot plant detection was 8-13 frames (270-430ms) early on 2 of 4 test clips (player_b's clips). The `detect_foot_plant_from_keypoints` function used ankle Y recovery-to-baseline to determine when the foot touched ground. But it does not know where the ground plane is -- the stride foot lands at a different Y position than the standing baseline, so the 85% recovery threshold triggers while the foot is still airborne.

## Symptoms

- Foot plant detected 12-13 frames early on player_b clips (IMG_3108, IMG_3109).
- Only 2 frames early on player_a clips (where stride happened to recover closer to baseline).
- At the detected FP frame, ankle Y velocity was still +19 to +30 (foot still moving fast).
- Inconsistent error across pitchers suggested the algorithm was view/pitcher dependent.

## Root Cause

The algorithm used ankle Y recovery to a baseline derived from the "standing" portion of the video. Three related problems:

1. **The stride foot lands well forward of the starting position** -- ankle Y at plant can be significantly different from standing baseline. The pitcher strides downhill off the mound, so the landing position in screen coordinates is not the same as the standing position.
2. **The 85% recovery threshold triggered prematurely.** As soon as ankle Y was "close enough" to baseline, the function returned -- but the foot was still descending through the air.
3. **The error was pitcher/angle dependent.** For some pitchers and camera angles, ankle Y at true foot plant happened to be close to baseline, so the threshold worked by coincidence. For others, ankle Y at plant was well below baseline (longer stride, different mound geometry), so the threshold fired during the descent phase before the foot arrived.

This is a category error: the algorithm measured a **position proxy** (ankle Y near a baseline value) instead of the **physical event** (foot stops moving forward).

## Solution

Replaced Y-recovery with ankle X velocity stabilization (`_detect_fp_ankle_x_stop`):

### Step 1: Smooth ankle X coordinates and compute X velocity

```python
ankle_x_smooth = gaussian_filter1d(ankle_x_raw, sigma=2)
ankle_x_velo = np.gradient(ankle_x_smooth)
```

Smoothing removes frame-to-frame jitter so velocity reflects the true stride trajectory rather than keypoint noise.

### Step 2: Find peak forward velocity during stride phase

```python
peak_velo = np.max(np.abs(ankle_x_velo[search_start:search_end]))
```

The stride phase produces a clear peak in ankle X velocity as the foot drives forward toward the landing spot.

### Step 3: Foot plant = first frame where X velocity drops below 30% of peak

```python
threshold = peak_velo * 0.30
for i in range(peak_idx, search_end):
    if np.abs(ankle_x_velo[i]) < threshold:
        return i
```

Foot plant IS the moment the foot stops moving forward. The 30% threshold catches the deceleration at ground contact while avoiding false triggers from momentary velocity dips during the stride.

### Fallback

Falls back to Y-recovery if ankle X data is unavailable (legacy API compatibility).

### Results

| Metric | Before (Y-recovery) | After (X-stabilization) |
|--------|---------------------|-------------------------|
| MAE    | 8.8 frames (292ms)  | 1.5 frames (50ms)       |
| Bias   | -8.8 (always early) | -1.5 (slight early)     |

## Key Insight

Ankle Y recovery detects a **position** -- "the foot is near where it started." Ankle X stabilization detects an **event** -- "the foot stopped moving." The position-based approach requires knowing the ground plane, which varies by pitcher, stride length, mound geometry, and camera angle. The event-based approach is self-referencing: it only needs the ankle's own velocity history, making it camera-angle independent and pitcher independent.

More generally: when the same algorithm works for some subjects but fails for others, the heuristic is likely coupled to a subject-specific or view-specific parameter. Look for a signal that is invariant to those factors. In this case, ankle X velocity at foot plant is always near zero regardless of where the foot lands in screen coordinates.

## Prevention

- **Prefer detecting the physical event over detecting a position.** "Foot stops moving forward" is the event; "ankle Y near baseline" is a position proxy that assumes ground plane knowledge.
- **When an algorithm works for some subjects but not others, the heuristic is likely camera/subject dependent.** Look for a view-invariant signal instead.
- **Test detection algorithms across multiple pitchers/clips, not just one.** The validation framework's `eval_events.py` enables this -- a fix that improves one clip but regresses another is not a fix.
- **Check velocity at detected events as a sanity check.** If ankle velocity is still +19 to +30 at "foot plant," the detection is clearly wrong. Adding velocity-based sanity gates can catch these failures at runtime.

## Related

- `detect_foot_plant_from_keypoints()` in `src/biomechanics/events.py` -- the function this fix modifies. The Y-recovery pattern-matching (dip-then-recovery) from the previous false-trigger fix remains as the fallback path.
- `foot-plant-false-trigger-on-standing.md` -- the prior fix that introduced the dip-then-recovery Y-based approach. That fix solved the standing-pose false trigger problem; this fix addresses the accuracy problem that remained.
- `find_delivery_anchor()` in `src/biomechanics/events.py` -- the MER anchor that defines the search window for foot plant detection.
