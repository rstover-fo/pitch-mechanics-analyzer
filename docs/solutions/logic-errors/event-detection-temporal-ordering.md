---
title: "Event Detection Produces Temporally Impossible Ordering"
date: 2026-03-10
category: logic-errors
module: src/biomechanics/events.py
tags: [event-detection, temporal-ordering, anchor-based, cascade]
symptoms: [events out of order, leg lift at clip boundary, MER during standing]
---

# Event Detection Produces Temporally Impossible Ordering

## Problem

The pitching delivery event detection pipeline produced events in temporally impossible order. Four delivery events -- leg lift, foot plant, MER (max external rotation), and ball release -- were each detected independently using global argmax on their respective signals. Because each detector searched the entire video clip without knowledge of the others, the resulting event frames frequently violated the physical constraint that these events must occur in strict chronological sequence.

## Symptoms

1. **Events appeared out of temporal order.** Ball release was detected before foot plant, or MER was placed before leg lift. Phase duration calculations returned negative values.

2. **Leg lift detected at clip boundaries.** The first or last few frames of a clip (where the pitcher is walking to or from the mound) often had transient noise in knee Y position that exceeded the actual leg lift peak during the windup.

3. **MER detected during non-delivery movements.** Standing posture or casual arm movements could produce shoulder ER angles comparable to actual layback, especially in 2D approximations from side-view video. The global argmax over the entire clip had no way to distinguish delivery from non-delivery frames.

4. **No validation caught the problem.** There was no temporal ordering check after detection, so downstream code (phase duration computation, metrics extraction) silently consumed impossible orderings.

## Root Cause

Each event detector used a global argmax over the full clip, independently:

- Leg lift = `argmax(knee_y)` over all frames
- Foot plant = velocity threshold crossing over all frames
- MER = `argmax(shoulder_er)` over all frames
- Ball release = `argmax(wrist_speed)` over all frames

This approach has a fundamental flaw: it assumes the delivery signal always dominates the non-delivery signal for each metric across the entire clip. That assumption fails when:

- The clip includes non-delivery movement (walking, standing, warmup arm swings)
- Noise at clip boundaries produces spurious peaks
- The 2D shoulder ER approximation gives comparable values for standing vs. actual layback
- Any single metric's global peak occurs outside the actual delivery window

Since each detector was blind to the others, there was no mechanism to enforce the physical ordering constraint: leg lift < foot plant < MER < ball release.

## Solution

Replace independent global-argmax detection with **anchor-based cascade detection**. The key insight is that MER has the most biomechanically distinctive signal -- a sharp rise in shoulder ER followed by an extremely rapid drop as the arm accelerates forward. This pattern is unique to the pitching delivery and does not appear during walking, standing, or casual movement.

### Step 1: Find the delivery anchor (MER)

The `find_delivery_anchor()` function identifies MER by detecting the shoulder ER rise-then-rapid-drop pattern, confirmed by nearby wrist speed:

```python
def find_delivery_anchor(
    shoulder_er: np.ndarray,
    wrist_speed: np.ndarray,
    fps: float = 30.0,
) -> Optional[int]:
    """Find the MER frame by detecting the shoulder ER rise-then-rapid-drop pattern."""
    # ...
    smoothed = uniform_filter1d(shoulder_er, size=5)
    er_deriv = np.gradient(smoothed)
    smoothed_deriv = uniform_filter1d(er_deriv, size=3)

    # Find the most negative derivative (fastest ER drop = arm acceleration)
    neg_deriv = -smoothed_deriv
    peaks, props = find_peaks(neg_deriv, prominence=1.0, distance=int(fps * 0.3))

    # Score each candidate: ER peak height + nearby wrist speed
    for peak in peaks:
        search_start = max(0, peak - int(fps * 0.3))
        mer_candidate = int(np.argmax(smoothed[search_start:peak + 1]) + search_start)
        er_value = smoothed[mer_candidate]

        wrist_window_end = min(len(wrist_speed), mer_candidate + int(fps * 0.5))
        wrist_peak = np.max(wrist_speed[mer_candidate:wrist_window_end])

        score = er_value * 0.5 + (wrist_peak / (np.max(wrist_speed) + 1e-8)) * 100
        # Keep the highest-scoring candidate
```

The scoring combines two independent signals -- high ER angle at the candidate frame and high wrist speed shortly after -- which together strongly discriminate actual delivery from noise.

### Step 2: Cascade outward from the anchor

Once MER is located, the remaining events are detected within constrained temporal windows relative to it. From `scripts/validate_pose.py`:

```python
# 1. MER (anchor) via delivery-signature detection
events.max_external_rotation = find_delivery_anchor(
    shoulder_er_series, wrist_speed, fps=fps,
)

if events.max_external_rotation is not None:
    # 2. Ball release = peak wrist speed in 12-frame window AFTER MER
    events.ball_release = detect_ball_release(
        wrist_speed, after_frame=events.max_external_rotation,
    )

    # 3. Foot plant = ankle dip-then-recovery pattern BEFORE MER
    events.foot_plant = detect_foot_plant_from_keypoints(
        lead_ankle_y_raw, fps=fps,
        before_frame=events.max_external_rotation,
    )

    # 4. Leg lift = peak knee height in ~1.5s window BEFORE foot plant
    events.leg_lift_apex = detect_leg_lift(
        lead_knee_y_inverted,
        before_frame=events.foot_plant,
    )
```

Each detector accepts a `before_frame` or `after_frame` parameter that constrains its search window:

- **Ball release** searches only the 12 frames after MER (`after_frame=events.max_external_rotation`), because release always follows MER within ~0.4s.
- **Foot plant** searches only the 2 seconds before MER (`before_frame=events.max_external_rotation`), looking for the ankle dip-then-recovery pattern.
- **Leg lift** searches only the 1.5 seconds before foot plant (`before_frame=events.foot_plant`), because the windup always precedes stride.

The individual detector functions enforce these constraints. For example, `detect_leg_lift`:

```python
def detect_leg_lift(
    lead_knee_y: np.ndarray,
    before_frame: Optional[int] = None,
) -> Optional[int]:
    # ...
    if before_frame is not None:
        window_start = max(0, before_frame - 45)  # ~1.5s at 30fps
        region = smoothed[window_start:before_frame]
        if len(region) == 0:
            return None
        return int(np.argmax(region) + window_start)

    return int(np.argmax(smoothed))
```

And `detect_ball_release`:

```python
def detect_ball_release(
    wrist_velo: np.ndarray,
    after_frame: Optional[int] = None,
    window_frames: int = 12,
) -> Optional[int]:
    # ...
    if after_frame is not None:
        end = min(len(wrist_velo), after_frame + window_frames)
        search = wrist_velo[after_frame:end]
        offset = after_frame
```

### Step 3: Graceful fallback

If `find_delivery_anchor()` cannot identify MER (returns `None`), the pipeline falls back to the original independent detection with a warning:

```python
else:
    print("  WARNING: Could not find delivery anchor (MER). "
          "Falling back to independent detection.")
    events.leg_lift_apex = detect_leg_lift(lead_knee_y_inverted)
    events.foot_plant = detect_foot_plant_from_keypoints(lead_ankle_y_raw, fps=fps)
    events.max_external_rotation = detect_max_external_rotation(
        shoulder_er_series, after_frame=events.foot_plant,
    )
    events.ball_release = detect_ball_release(
        wrist_speed, after_frame=events.max_external_rotation,
    )
```

Even the fallback path applies partial ordering constraints (`after_frame=events.foot_plant` for MER, `after_frame=events.max_external_rotation` for ball release), so it is still better than fully independent detection.

## Key Insight

When detecting a sequence of events that must obey a temporal ordering, start with the most biomechanically distinctive event and cascade outward. MER works as the anchor because it combines two features that are unique to the delivery: (1) a high shoulder ER peak and (2) an extremely rapid ER drop immediately after, coinciding with high wrist speed. Neither of these patterns appears during non-delivery movements, making MER far more robust as an anchor than leg lift (which is just "high knee") or foot plant (which is just "ankle stops moving").

The general principle: **constrained local search from a reliable anchor beats unconstrained global search for each event independently.** The anchor converts a set of independent optimization problems (each prone to false peaks) into a single anchor-finding problem followed by constrained searches that can only produce temporally valid results.

## Prevention

1. **Design event detection as a cascade, not independent detectors.** Any time multiple events must obey an ordering constraint, bake that constraint into the detection flow by using earlier detections to bound later searches.

2. **Score candidates with multiple independent signals.** The `find_delivery_anchor()` scoring combines shoulder ER peak height with nearby wrist speed. A false positive must fool both signals simultaneously, which is far less likely than fooling either one alone.

3. **Validate ordering after detection.** Even with cascade detection, add a post-hoc check that the detected events satisfy `leg_lift < foot_plant < MER < ball_release`. If they don't, flag the result rather than passing impossible orderings downstream.

4. **Window-bound all searches.** Never search the entire clip when biomechanical knowledge constrains the valid window. Arm cocking (foot plant to MER) takes 0.03-0.15s. Arm acceleration (MER to release) takes 0.03-0.05s. These durations directly translate to frame-count search windows.

## Related

- `src/biomechanics/events.py` -- All detector functions with `before_frame`/`after_frame` parameters
- `scripts/validate_pose.py` -- Anchor-based cascade orchestration (lines 235-269)
- `src/biomechanics/events.py:find_delivery_anchor()` -- MER anchor detection with dual-signal scoring
- `src/biomechanics/events.py:DeliveryEvents.phase_durations()` -- Downstream consumer that produces negative values with impossible ordering
