---
title: "Foot Plant Detection Triggers on Standing Poses"
date: 2026-03-10
category: logic-errors
module: src/biomechanics/events.py
tags: [foot-plant, event-detection, velocity-threshold, pattern-matching]
symptoms: [foot plant during standing, foot plant before leg lift, false trigger at clip start]
---

# Foot Plant Detection Triggers on Standing Poses

## Problem

The foot plant detector fired on stationary/standing poses at the beginning of pitching clips instead of on the actual stride landing. Because foot plant is the temporal anchor for several downstream phases (arm cocking, acceleration), a false early trigger corrupted every subsequent event and made the full pipeline unusable.

## Symptoms

- Foot plant detected during the pre-pitch standing phase (frames 0-20), before the pitcher even begins the windup.
- Foot plant detected *before* leg lift apex -- temporally impossible in a real delivery.
- Every clip exhibited the same failure: the very first frames satisfied the detection threshold.
- Downstream phase durations (foot plant to MER, arm cocking) were wildly inflated because the anchor was 30-60 frames too early.

## Root Cause

The original detector used a **velocity-threshold** approach: when the lead ankle's Y-velocity dropped near zero, it concluded the foot had just planted. The fundamental flaw is that a **stationary foot** (standing still before the windup) has the exact same zero-velocity signature as a **newly-planted foot** (stride landing). The detector had no way to distinguish "the foot was never moving" from "the foot just stopped moving."

Any video that begins with the pitcher standing in the stretch or windup position will have near-zero ankle velocity in the opening frames, immediately satisfying the threshold.

## Solution

Replaced velocity-threshold detection with **pattern-based detection** that requires the biomechanical signature of an actual stride: a dip in ankle Y (foot lifts during stride) followed by a recovery back toward baseline (foot comes back down to the ground).

The implementation in `detect_foot_plant_from_keypoints()` follows a three-step approach:

### Step 1: Establish baseline ankle Y from the early portion of the search window

```python
# Step 1: Find the baseline ankle Y (standing level) from the early part of the window
early_portion = region[: max(5, len(region) // 4)]
baseline = np.median(early_portion)
```

The median of the first quarter of the search window captures the standing height. Using `median` rather than `mean` makes it robust to any initial movement noise.

### Step 2: Find the stride dip and reject standing poses

```python
# Step 2: Find the stride dip — minimum ankle Y (foot at highest point during stride)
dip_idx = int(np.argmin(region))
dip_value = region[dip_idx]
dip_depth = baseline - dip_value

# If the dip is too shallow (< 5% of baseline), no real stride detected
if dip_depth < baseline * 0.05:
    return None
```

This is the critical gate. During standing, ankle Y stays roughly constant -- there is no dip, so `dip_depth` is near zero and the function returns `None`. A real stride produces a clear dip as the foot lifts off the ground and travels forward. The 5% threshold rejects noise while catching even short strides.

### Step 3: Find the recovery point (actual foot plant)

```python
# Step 3: From the dip, look forward for where ankle Y recovers to near-baseline
# Foot plant = first frame after the dip where ankle Y rises to within 15% of baseline
recovery_threshold = baseline - dip_depth * 0.15
for i in range(dip_idx, len(region)):
    if region[i] >= recovery_threshold:
        return int(i + start)

# Fallback: frame closest to MER where ankle is near max Y
return int(np.argmax(region[dip_idx:]) + dip_idx + start)
```

After the dip, the function scans forward for the frame where ankle Y climbs back to within 15% of the original dip depth relative to baseline. This corresponds to the moment the stride foot makes contact with the ground. The fallback handles edge cases where recovery is incomplete (e.g., clip ends early).

## Key Insight

Velocity-based thresholds are **state-ambiguous** -- they detect a condition (low velocity) that can arise from multiple biomechanical states (standing still vs. foot just planted). Pattern-based detection resolves the ambiguity by requiring a specific *trajectory shape* (dip-then-recovery) that only occurs during an actual stride. The standing pose lacks this shape entirely, so it cannot produce a false positive.

More generally: when detecting a transition event (something starts or stops), a threshold on the instantaneous value is almost always insufficient. You need to verify the *transition itself* -- evidence that the signal changed from one state to another.

## Prevention

- **Event detectors should require a preceding state transition**, not just a threshold on the current state. "Foot planted" requires evidence of prior foot movement.
- **Gate detectors on temporal ordering.** Foot plant cannot precede leg lift. If it does, the detection is wrong. The pipeline now uses `find_delivery_anchor()` to locate MER first, then searches backward for foot plant within a bounded 2-second window.
- **Test with standing-start clips.** Any pitching video from the stretch or windup will begin with a stationary pose. Include these in the test suite to catch false-trigger regressions.
- **Prefer shape-matching over single-frame thresholds** for biomechanical event detection. The stride dip-then-recovery pattern, the ER rise-then-drop pattern (`find_delivery_anchor()`), and the wrist speed peak pattern (`detect_ball_release()`) are all examples of this principle applied across the pipeline.

## Related

- `find_delivery_anchor()` in `src/biomechanics/events.py` -- uses the same pattern-matching philosophy (ER rise-then-rapid-drop) to locate MER as the temporal anchor for the entire delivery.
- `detect_leg_lift()` in `src/biomechanics/events.py` -- searches for peak knee height, which should always precede foot plant.
- `DeliveryEvents.phase_durations()` -- downstream consumer that depends on correct foot plant timing for arm cocking and acceleration phase calculations.
