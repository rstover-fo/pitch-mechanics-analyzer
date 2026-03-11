---
title: "MER Detection Consistently Early Due to 2D Angle Inversion"
date: 2026-03-10
category: logic-errors
module: src/biomechanics/events.py
tags: [event-detection, MER, shoulder-ER, 2D-approximation, camera-angle]
symptoms: [MER too early, FP-MER gap too small, ball release early]
---

# MER Detection Consistently Early Due to 2D Angle Inversion

## Problem

MER (Max External Rotation) detection was consistently 10-12 frames (350ms) early across all 4 test clips. The `find_delivery_anchor` function in `src/biomechanics/events.py` searched for the shoulder ER angle MAXIMUM before a rapid drop. But the 2D shoulder ER approximation (`approximate_shoulder_er_2d`) is INVERTED when viewed from a front-quarter camera angle -- deeper arm layback makes the forearm-trunk angle DECREASE, not increase.

## Symptoms

1. **MER detected ~10 frames before ground truth on every clip.** The error was systematic, not random -- every clip showed the same early bias, indicating a fundamental signal direction issue rather than noise or tuning.

2. **FP-to-MER gap was only 1 frame.** Biomechanically, the arm cocking phase (foot plant to MER) takes 0.03-0.15s. A 1-frame gap at 30fps (0.033s) is at the extreme lower bound and was implausibly consistent across clips with different pitchers.

3. **Ball release also detected 7 frames early.** Because ball release detection cascades from the MER anchor (searching a 12-frame window after MER), an early MER shifts the entire downstream cascade early.

4. **Validation warnings for arm acceleration duration being too long.** The early MER combined with a less-affected ball release detection produced an artificially stretched MER-to-BR phase.

## Root Cause

`approximate_shoulder_er_2d` measures the angle between the forearm vector (elbow-to-wrist) and the trunk vector (hip-to-shoulder). From a front-quarter camera angle, as the arm lays back further into true external rotation, the forearm progressively aligns WITH the trunk vector in the 2D projection. This means the measured angle DECREASES as real shoulder ER increases.

The `find_delivery_anchor` algorithm was looking for the ER peak (maximum), then detecting the rapid drop after it. But from this camera angle:

- True MER corresponds to the ER signal MINIMUM (forearm most aligned with trunk)
- The "rapid drop" the algorithm found was actually the arm cocking phase (ER increasing in reality, decreasing in the 2D signal)
- The detected "peak before the drop" was ~10 frames before actual MER -- roughly where foot plant occurs

This is a classic 2D projection artifact: a 3D angle's monotonic increase can project as a monotonic decrease depending on the viewing angle. The algorithm's assumptions about signal direction were correct for a pure side view but inverted for front-quarter footage.

## Solution

Replaced the complex derivative-based peak detection with a simpler, more robust approach that does not depend on the signal direction of the ER approximation:

1. Find peak wrist speed (a reliable delivery anchor that is direction-invariant)
2. Find the ER minimum in a 0.5s window before peak wrist speed
3. That minimum = MER

The key insight is that wrist speed is a scalar magnitude -- it is always positive and always peaks during arm acceleration regardless of camera angle. By anchoring to wrist speed first, then searching for the ER extremum in a constrained window, the detection works correctly whether the ER signal is normal or inverted.

```python
# Before: searched for ER MAXIMUM, then detected rapid drop
# This fails when 2D projection inverts the ER signal
smoothed = uniform_filter1d(shoulder_er, size=5)
er_deriv = np.gradient(smoothed)
neg_deriv = -smoothed_deriv
peaks, props = find_peaks(neg_deriv, prominence=1.0, ...)
# ... complex scoring of derivative peaks

# After: find wrist speed peak, then ER minimum in window before it
wrist_peak_frame = int(np.argmax(wrist_speed))
window_start = max(0, wrist_peak_frame - int(fps * 0.5))
er_window = smoothed[window_start:wrist_peak_frame]
mer_frame = int(np.argmin(er_window) + window_start)
```

### Results

| Metric | Before | After |
|--------|--------|-------|
| Mean absolute error (frames) | 10.8 | 1.0 |
| Mean absolute error (ms) | 358 | 33 |
| Bias (frames) | -10.8 (always early) | +0.5 (negligible) |

## Key Insight

When using 2D projections of 3D biomechanical angles, the projected signal can be monotonically inverted depending on the camera viewpoint. An algorithm that assumes a specific signal direction (e.g., "ER rises then drops sharply at MER") will fail systematically when the projection inverts. Two defenses:

1. **Anchor on direction-invariant signals.** Wrist speed magnitude is always positive and always peaks during arm acceleration, regardless of camera angle. Use it as the primary anchor, then search for the ER extremum (not specifically max or min) in a constrained window.

2. **Detect extrema, not specific directions.** If the algorithm must use the ER signal directly, search for both the maximum and minimum in the candidate window and score both. The correct one will have better temporal alignment with other biomechanical signals.

## Prevention

1. **When using 2D angle approximations, always verify the signal direction against ground truth before building detection logic.** Run the approximation on labeled data and plot the signal to confirm it behaves as expected from the target camera angle.

2. **The 2D projection of a 3D angle can be monotonically inverted depending on camera viewpoint.** This is not noise or distortion -- it is a deterministic geometric property. If the algorithm works from one angle but fails from another, signal inversion is the first thing to check.

3. **Always validate event detection against human-labeled ground truth.** The validation framework (`scripts/validate_pose.py`) now supports frame-level comparison against ground truth timestamps, which caught this systematic bias immediately.

4. **Prefer scalar magnitudes (speed, distance) over directional signals (angles) for primary anchoring.** Magnitudes are viewpoint-invariant; angles are not.

## Related

- `src/biomechanics/events.py:find_delivery_anchor()` -- The function where the fix was applied
- `src/biomechanics/events.py:approximate_shoulder_er_2d()` -- The 2D ER approximation that produces inverted signals from front-quarter views
- `scripts/validate_pose.py` -- Validation pipeline that detected the systematic early bias
- `docs/solutions/logic-errors/event-detection-temporal-ordering.md` -- The anchor-based cascade design that this fix builds on
