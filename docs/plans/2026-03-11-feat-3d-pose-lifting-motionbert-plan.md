---
title: "feat: Add 3D pose lifting with MotionBERT"
type: feat
date: 2026-03-11
brainstorm: docs/brainstorms/2026-03-11-3d-pose-lifting-brainstorm.md
reviewed: true
reviewers: DHH-style, Kieran-Python, Code-Simplicity
---

# feat: Add 3D Pose Lifting with MotionBERT

## Overview

Add a MotionBERT-based 3D pose lifting stage between YOLOv8 2D pose estimation and
event detection. This converts (x, y) pixel keypoints to (x, y, z) 3D coordinates,
unlocking 4 currently unmeasurable OBP metrics and improving accuracy of the 7 existing
ones by removing 2D projection distortion.

## Problem Statement

The current pipeline computes biomechanical metrics from 2D keypoints projected onto the
camera plane. This causes:

1. **4 metrics unmeasurable**: shoulder abduction, horizontal abduction, torso lateral
   tilt, and hip-shoulder separation all require depth information
2. **Projection distortion**: peak layback reads 85-112° (0th percentile vs OBP) when
   true values should be ~150-170°. Trunk tilt reads 4-8° when real values are ~30-40°.
3. **Pitch-to-pitch noise**: elbow flexion swings 44° between pitches from the same
   pitcher (137° vs 93°) partly due to slight arm rotation changes in the depth axis
4. **Misleading coaching**: Claude API reports over-emphasize trunk tilt as a priority
   because it reads artificially low, and undervalue layback because it reads artificially low

All OBP benchmark data comes from 3D motion capture. Comparing 2D-projected metrics
against 3D benchmarks is apples-to-oranges.

## Proposed Solution

Vendor ~5 core MotionBERT files (~600 lines) into the project. After YOLOv8 extracts
2D COCO keypoints, map them to H36M format, normalize, run MotionBERT inference, and
map results back to pitching joints with z-coordinates. Feature extraction dispatches
to 3D angle functions when z is present, falling back to existing 2D calculations.

**Graceful degradation**: If MotionBERT checkpoint is not downloaded, the pipeline runs
in 2D-only mode exactly as it does today. All 113 existing tests continue to pass.

## Technical Approach

### Architecture

```
Video
  → YOLOv8 (2D COCO 17 keypoints per frame)
  → [NEW] COCO→H36M joint mapping (17→17, synthesize root/spine/neck/head)
  → [NEW] Normalize to [-1, 1] (crop_scale per MotionBERT convention)
  → [NEW] MotionBERT-Lite inference (sequence → 3D positions)
  → [NEW] H36M→pitching joint mapping (17→12 pitching joints + spine/neck/root with z)
  → PoseFrame with (x, y, z) keypoints
  → Event detection (UNCHANGED — uses x, y slices only)
  → Feature extraction (UPGRADED — 3D angles when z present, 2D fallback)
  → OBP comparison (more metrics, better accuracy)
  → Coaching report (richer data, one-line 3D context note)
```

### Joint Mapping Tables

**COCO 17 → H36M 17** (from MotionBERT's own `coco2h36m()`):

| H36M Index | H36M Joint | Source |
|---|---|---|
| 0 | Root (pelvis) | avg(COCO left_hip[11], right_hip[12]) |
| 1 | Right Hip | COCO right_hip[12] |
| 2 | Right Knee | COCO right_knee[14] |
| 3 | Right Ankle | COCO right_ankle[16] |
| 4 | Left Hip | COCO left_hip[11] |
| 5 | Left Knee | COCO left_knee[13] |
| 6 | Left Ankle | COCO left_ankle[15] |
| 7 | Belly/Spine | avg(Root, Neck) — synthesized |
| 8 | Neck | avg(COCO left_shoulder[5], right_shoulder[6]) |
| 9 | Nose | COCO nose[0] |
| 10 | Head | avg(COCO left_eye[1], right_eye[2]) |
| 11 | Left Shoulder | COCO left_shoulder[5] |
| 12 | Left Elbow | COCO left_elbow[7] |
| 13 | Left Wrist | COCO left_wrist[9] |
| 14 | Right Shoulder | COCO right_shoulder[6] |
| 15 | Right Elbow | COCO right_elbow[8] |
| 16 | Right Wrist | COCO right_wrist[10] |

**H36M 17 → Pitching Joints** (reverse mapping after 3D lift):

| Pitching Joint | H36M Index | Notes |
|---|---|---|
| left_shoulder | 11 | |
| right_shoulder | 14 | |
| left_elbow | 12 | |
| right_elbow | 15 | |
| left_wrist | 13 | |
| right_wrist | 16 | |
| left_hip | 4 | |
| right_hip | 1 | |
| left_knee | 5 | |
| right_knee | 2 | |
| left_ankle | 6 | |
| right_ankle | 3 | |
| spine | 7 | Added to `keypoints` dict (3D only) |
| neck | 8 | Added to `keypoints` dict (3D only) |
| root | 0 | Added to `keypoints` dict (3D only) |

**Decision (from review):** Extra joints (spine, neck, root) go directly into the
existing `keypoints` dict — NOT a separate `auxiliary_joints_3d` field. The dict is
already name-keyed and variable-length. `to_keypoints_dict()` filters to
`PITCHING_JOINTS` so downstream code won't see them unless it asks.

### Coordinate System Strategy

**Problem**: MotionBERT outputs in camera coordinates. Pitching metrics require anatomical
planes (sagittal, coronal, transverse) that depend on the pitcher's orientation.

**Solution**: Body-centered reference frame derived from the 3D skeleton:

1. **Vertical axis (Y_body)**: Camera Y is "up" (gravity direction). This holds for
   standard handheld/tripod video. No rotation needed.
2. **Lateral axis (Z_body)**: The shoulder line (left_shoulder → right_shoulder),
   smoothed over a 5-frame window around foot plant, defines the coronal plane direction.
3. **Forward axis (X_body)**: Cross product of Y_body × Z_body gives the pitching
   direction (toward home plate).

This body-centered frame is computed once at foot plant and applied to all frames.
It handles arbitrary camera angles because it's derived from the skeleton, not the camera.

**Implementation note (from review):** Inline the shoulder-line cross-product math
directly into each 3D angle function rather than building a general-purpose
`compute_body_frame()` → rotation matrix → `project_to_plane()` utility chain. Each
angle function is self-contained and easier to debug. The math is 3-4 lines of numpy.

**Sign convention note (from review):** The cross product direction depends on which
shoulder is "left" vs "right" AND whether the pitcher faces toward or away from the
camera. Resolve using the existing `pitcher_throws` parameter. Add an assertion that
the derived forward direction is consistent with the known camera perspective.

### Scale Recovery

MotionBERT output is in normalized coordinates (proportional, not mm). For metrics that
need real-world scale (stride length as % of body height), we use the pitcher's known
height from the existing `--height` flag (provided as part of the `--age --height --weight`
triplet for youth normalization).

**If height not provided**: stride length falls back to 2D pixel-based calculation
(current behavior). Angular metrics are scale-invariant and always benefit from 3D.

**Decision (from review):** Keep the existing all-or-nothing CLI validation for
`--age/--height/--weight`. Do NOT add a `--height`-alone code path — it only benefits
one metric (stride length) and creates a partial-flag state that complicates the code.

### PoseFrame Dimension Handling

**Decision (from review):** Make dimensionality explicit, not inferred from array shapes.

Add an `is_3d` property to `PoseFrame`:

```python
@property
def is_3d(self) -> bool:
    """Whether keypoints include depth coordinates."""
    if not self.keypoints:
        return False
    first = next(iter(self.keypoints.values()))
    return first.shape[-1] == 3
```

Fix every downstream consumer:

- `get_joint_trajectory()`: detect dimension from first valid frame, NaN fill matches
  (currently hardcoded `np.array([np.nan, np.nan])` — will break in 3D)
- `to_dataframe()`: add `{joint}_z` columns when 3D (currently silently drops Z)
- `to_keypoints_dict()`: update docstring from `(N_frames, 2)` to `(N_frames, D)`
- `extract_metrics()`: accept explicit `use_3d: bool` parameter from caller, not
  shape-sniffing scattered through the function

### Files Changed / Created

**New files:**
- `vendor/motionbert/` — vendored model code (pinned to specific commit hash):
  - `VERSION` — records the commit hash vendored from
  - `DSTformer.py` — model architecture (~300 lines)
  - `drop.py` — DropPath utility
  - `utils_data.py` — `crop_scale()`, `flip_data()` normalization helpers
- `src/pose/lifter.py` — pipeline integration wrapper:
  - `load_motionbert(checkpoint_path, device)` → model (uses `SimpleNamespace` for
    config, NOT `easydict`)
  - `coco_to_h36m(kpts_coco)` → H36M format
  - `h36m_to_pitching_joints(kpts_h36m_3d)` → pitching joint dict with z
  - `lift_to_3d(pose_sequence, model)` → updated PoseSequence with (x,y,z)
  - `is_3d_available()` → bool (checks for checkpoint file)
- `src/biomechanics/angles_3d.py` — 3D angle calculations:
  - `compute_hip_shoulder_separation_3d(l_hip, r_hip, l_sho, r_sho)` — inline body frame
  - `compute_shoulder_abduction_3d(shoulder, elbow, hip_center, shoulder_center)`
  - `compute_shoulder_horizontal_abduction_3d(shoulder, elbow, l_sho, r_sho)`
  - `compute_torso_lateral_tilt_3d(hip_center, shoulder_center, l_sho, r_sho)`
- `tests/test_lifter.py` — joint mapping, inference mock, graceful degradation
- `tests/test_angles_3d.py` — golden-value tests with known geometry
- `data/models/.gitkeep` — checkpoint directory (64MB .bin file gitignored)

**Modified files:**
- `src/pose/estimator.py`:
  - `PoseFrame.is_3d` property (new)
  - `PoseFrame.keypoints` accepts shape (2,) or (3,); extra joints (spine/neck/root)
    added to same dict when 3D
  - `get_joint_trajectory()`: dynamic NaN fill based on first valid keypoint dimension
  - `to_dataframe()`: add `{joint}_z` columns when 3D
  - `to_keypoints_dict()`: updated docstring for (N, D) where D is 2 or 3
- `src/biomechanics/features.py`:
  - `compute_trunk_tilt()`: dimension-aware vertical reference `[0, -1]` vs `[0, -1, 0]`
  - `compute_arm_slot()`: dimension-aware horizontal reference
  - `extract_metrics()`: accept `use_3d: bool` param; compute 4 new metrics when True
- `scripts/validate_pose.py`:
  - Insert 3D lifting stage between pose estimation and event detection
  - Add `--no-3d` flag to force 2D mode
  - Add `pose_mode` to results.json
  - Append one-line 3D context to coaching additional_context when 3D active
- `.gitignore`:
  - Add `data/models/*.bin` and `vendor/` patterns
- `requirements.txt`:
  - Add `einops>=0.6` only if vendored code audit confirms it's needed
  - NO `easydict` — use `SimpleNamespace` instead

**Removed from original plan (per review):**
- ~~`scripts/download_motionbert.py`~~ → curl one-liner in CLAUDE.md
- ~~`auxiliary_joints_3d` field on PoseFrame~~ → use existing `keypoints` dict
- ~~`easydict` dependency~~ → `SimpleNamespace`
- ~~`--height` alone code path~~ → keep all-or-nothing CLI validation
- ~~Mode-aware coaching prompt branching~~ → one-line context string
- ~~`compute_body_frame()` / `project_to_plane()` utilities~~ → inline math per function
- ~~Chunking with overlap for >243 frames~~ → raise clear error (YAGNI, clips are 120-220)

### Implementation Phases

#### Phase 0: Validation Prototype — Does MotionBERT Work on Pitching? (GO/NO-GO)

**Goal**: Answer the single most important question before writing any integration code:
does MotionBERT produce reasonable 3D positions during the extreme arm cocking and
acceleration phases of a pitching delivery?

**This is a go/no-go gate.** If MotionBERT hallucinates arm positions at MER (where the
forearm is behind the head at ~170° of layback — a pose unlike anything in its training
data), the entire plan is dead and we explore alternatives.

**Tasks:**

- [ ] Download MotionBERT-Lite checkpoint:
  ```bash
  mkdir -p data/models
  curl -L https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin -o data/models/motionbert_lite.bin
  ```
- [ ] Clone MotionBERT repo to a temp directory for reference
- [ ] Write `scripts/prototype_3d_lift.py` — standalone throwaway script:
  - Load YOLOv8 2D keypoints from IMG_3106 (existing pipeline)
  - Map COCO → H36M, normalize, run MotionBERT inference
  - Print 3D positions for shoulder/elbow/wrist at key frames (foot plant, MER, release)
  - Generate a simple Plotly 3D scatter of the skeleton at MER
- [ ] Visually inspect: does the arm at MER show plausible layback (forearm behind head)?
- [ ] Check bone length consistency: are upper arm / forearm lengths stable across frames?
- [ ] If plausible → proceed to Phase A
- [ ] If garbage → investigate MotionAGFormer or fine-tuned sports models before continuing

**Acceptance criteria:**
- 3D skeleton at MER visually shows arm cocked behind head (not collapsed or flipped)
- Bone lengths (shoulder-elbow, elbow-wrist) are consistent within 15% across frames
- Shoulder ER angle from 3D positions is in a plausible range (120-180°)

**Time box: 2-3 hours.** If it takes longer, the vendoring is harder than expected.

#### Phase A: Full Integration (Sprint 1)

**Goal**: Wire 3D lifting into the pipeline end-to-end. All 7 existing metrics improve,
4 new metrics become available, coaching reports get richer data.

**Tasks:**

*Vendoring & Lifter:*
- [ ] Audit vendored code imports; identify dependencies beyond torch/numpy
- [ ] Vendor the minimum files into `vendor/motionbert/`; record commit hash in `VERSION`
- [ ] Strip unnecessary imports (smplx, chumpy, etc.) from vendored code
- [ ] Replace any `easydict` usage with `SimpleNamespace` in lifter wrapper
- [ ] Add `data/models/*.bin` and `vendor/` to `.gitignore`
- [ ] Implement `coco_to_h36m()` in `src/pose/lifter.py`
- [ ] Implement `h36m_to_pitching_joints()` (12 pitching + 3 extra joints)
- [ ] Implement `load_motionbert()` with device detection (CPU/MPS)
- [ ] Implement `infer_3d()` with test-time flip augmentation
- [ ] Implement `lift_to_3d(pose_sequence, model)` — full pipeline function
- [ ] Implement `is_3d_available()` — checks checkpoint file exists

*PoseFrame & PoseSequence updates:*
- [ ] Add `is_3d` property to `PoseFrame`
- [ ] Fix `get_joint_trajectory()` NaN fill to be dimension-aware
- [ ] Fix `to_dataframe()` to add `{joint}_z` columns when 3D
- [ ] Update `to_keypoints_dict()` docstring

*3D Angle Calculations:*
- [ ] Implement `src/biomechanics/angles_3d.py` — 4 new metric functions with inline
      body-frame math (no general-purpose utilities)
- [ ] Update `compute_trunk_tilt()` and `compute_arm_slot()` for dimension-aware refs
- [ ] Update `extract_metrics()` with `use_3d: bool` parameter; dispatch to 3D functions

*Pipeline Wiring:*
- [ ] Wire into `validate_pose.py`: 3D lifting as Stage 1.5
- [ ] Add `--no-3d` flag
- [ ] Add `pose_mode: "3d"` to results.json
- [ ] Append one-line 3D context to coaching prompt when 3D active
- [ ] Raise clear error for sequences > 243 frames (not chunking — YAGNI)

*Tests (write mapping tests FIRST, before model code):*
- [ ] `tests/test_lifter.py`:
  - COCO→H36M mapping correctness (verify all 17 joints land in correct positions)
  - H36M→pitching reverse mapping (12 joints + 3 extra, no data loss)
  - `is_3d_available()` returns False when checkpoint missing
  - `lift_to_3d` with missing checkpoint returns input unchanged
  - PoseFrame `is_3d` property, `get_joint_trajectory` with 3D, `to_dataframe` with Z
- [ ] `tests/test_angles_3d.py`:
  - Golden-value tests: T-pose → 90° shoulder abduction, known hip-shoulder sep config
  - Perpendicular vectors → 90°, parallel → 0° (at least 3 configs per function)
  - Sign convention: RHP with known arm position produces expected positive/negative
- [ ] Regression guard: `extract_metrics` with 2D keypoints produces identical results
- [ ] Verify event detection passes unchanged with 3D PoseSequence

*Validation on real clips:*
- [ ] Run all 4 clips with 3D lifting
- [ ] Peak layback should move from 0th percentile toward plausible range
- [ ] Trunk tilt should increase from 4-8° toward 30-40°
- [ ] At least 2 of 4 new metrics produce non-None values
- [ ] Benchmark CPU inference time (target: < 30s added per clip)

**Acceptance criteria:**
- Pipeline runs end-to-end with 3D on all 4 clips, no crashes
- `--no-3d` forces 2D mode; missing checkpoint falls back silently
- 7 existing metrics computed with improved accuracy
- At least 2 of 4 new metrics produce valid values
- Peak layback percentile moves from 0th to plausible range (20th-80th)
- All 113 existing tests pass; new tests pass
- results.json includes `pose_mode` field
- CPU inference adds < 30s per clip

#### Phase B: Polish (Optional, do incrementally)

**Goal**: Nice-to-haves that improve the experience but aren't needed for the core
3D lifting to work.

- [ ] Update HTML report header to show "3D Lifting: MotionBERT-Lite" or "2D Only"
- [ ] Add new metrics to report radar/gauge charts
- [ ] Update `data/prompts/measurement_caveats.md` with 3D confidence section
- [ ] Add curl download command to CLAUDE.md "Running" section
- [ ] Consider 3D skeleton visualization (Plotly 3D scatter at key frames)
- [ ] Consider MPS (Apple Metal) benchmarking for faster inference

## Alternative Approaches Considered

| Approach | Why Rejected |
|---|---|
| **MMPose wrapper** | Massive dependency chain (mmengine, mmcv, mmdet). Overkill for one model call. |
| **MediaPipe 3D (re-enable z)** | Already discarded for good reason — accuracy significantly worse than dedicated lifters |
| **VideoPose3D (Meta)** | Older (2019), lower accuracy (46mm vs 39mm MPJPE), same vendoring complexity |
| **Multi-camera triangulation** | Requires filming with 2 phones simultaneously. Higher barrier. |
| **Parallel pipeline (keep 2D separate)** | More code duplication. Drop-in replacement is simpler and avoids maintaining two paths. |
| **MotionAGFormer** | Newer but less documented. Fallback option if Phase 0 shows MotionBERT fails on pitching. |

## Acceptance Criteria

### Functional Requirements
- [ ] Phase 0 go/no-go passes (MotionBERT produces plausible 3D on pitching video)
- [ ] MotionBERT-Lite runs inference on all 4 existing video clips
- [ ] 3D keypoints flow through PoseFrame → events → features without errors
- [ ] 7 existing metrics have improved accuracy (layback, trunk tilt especially)
- [ ] At least 2 of 4 new metrics produce valid values
- [ ] `--no-3d` flag forces 2D mode
- [ ] Missing checkpoint → graceful 2D fallback with no errors

### Non-Functional Requirements
- [ ] Pipeline runtime increases by < 30 seconds per clip on CPU
- [ ] All 113 existing tests continue to pass
- [ ] New test coverage for: joint mapping, 3D angles, graceful degradation, PoseFrame 3D

### Quality Gates
- [ ] Peak layback percentile moves from 0th to plausible range (20th-80th)
- [ ] Trunk tilt at release increases from 4-8° toward 30-40°
- [ ] Elbow flexion pitch-to-pitch variance decreases for same pitcher
- [ ] Run all 4 clips; no crashes, no NaN metrics, no validation warnings

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MotionBERT fails on pitching (out-of-distribution) | Medium | **Critical** | **Phase 0 go/no-go gate** — validate before any integration work |
| Coordinate alignment wrong for anatomical planes | Medium | High | Inline body frame from shoulder line; assert sign convention with `pitcher_throws` |
| Vendored code breaks on Python 3.11 / PyTorch 2.10 | Low | Medium | Test imports in Phase 0 prototype; pin commit hash |
| COCO→H36M joint mapping bugs | Medium | High | Write mapping tests FIRST before model code; verify invertibility |
| Scale ambiguity makes stride length wrong | Low | Medium | Angular metrics don't need scale; stride falls back to 2D if no height |
| > 243 frame clips | Low | Low | Raise clear error (YAGNI — clips are 120-220 frames) |
| Frame rate mismatch (30fps vs 50fps training) | Low | Low | Monitor empirically; MotionBERT Lite reported robust across frame rates |

## Dependencies & Prerequisites

- PyTorch 2.10.0 (already installed via ultralytics)
- `einops>=0.6` — **only if audit of vendored code confirms it's needed**
- NO `easydict` — use `SimpleNamespace` instead
- 64MB disk space for MotionBERT-Lite checkpoint (downloaded via curl)

## Review Feedback Applied

Changes made based on parallel review by DHH-style, Kieran-Python, and Code-Simplicity
reviewers:

| Reviewer | Feedback | Action Taken |
|---|---|---|
| All three | Add Phase 0 go/no-go validation | Added as explicit gate before integration |
| All three | Collapse 4 phases → 2 | Phases 0 → A → B (optional) |
| All three | Kill `auxiliary_joints_3d` dict | Extra joints go in existing `keypoints` dict |
| All three | Kill `easydict` dependency | Use `SimpleNamespace` |
| All three | Kill download script | Curl one-liner in docs |
| Kieran | Fix NaN fill in `get_joint_trajectory` | Documented as explicit task in Phase A |
| Kieran | Fix `to_dataframe` silent Z drop | Documented as explicit task |
| Kieran | Add `is_3d` property to PoseFrame | Added to Phase A |
| Kieran | Explicit `use_3d: bool` in `extract_metrics` | Changed from shape-sniffing |
| Kieran | Write mapping tests FIRST | Reordered in Phase A task list |
| Kieran | Smooth shoulder line over 5 frames | Added to coordinate system strategy |
| Kieran | Assert sign convention with `pitcher_throws` | Added to coordinate system notes |
| Kieran | Vendor code in `vendor/` not `src/` | Changed directory |
| DHH | Pin vendor commit hash | Added VERSION file |
| DHH | No separate `angles_3d.py` module | **Kept** — matches project structure split (pose/ vs biomechanics/), 2/3 reviewers agreed |
| Simplicity | Inline body frame math per function | Changed from general-purpose utilities |
| Simplicity | Kill `--height`-alone code path | Kept all-or-nothing CLI validation |
| Simplicity | Replace mode-aware coaching with one-liner | One-line context string |
| Simplicity | YAGNI on >243 frame chunking | Raise error instead |

## References

### Internal
- Brainstorm: `docs/brainstorms/2026-03-11-3d-pose-lifting-brainstorm.md`
- Pose estimator: `src/pose/estimator.py` (PoseFrame, PoseSequence)
- Feature extraction: `src/biomechanics/features.py` (angle functions, extract_metrics)
- Event detection: `src/biomechanics/events.py` (anchor-based cascade)
- MER angle inversion doc: `docs/solutions/logic-errors/mer-detection-2d-angle-inversion.md`
- Validation: `src/biomechanics/validation.py`

### External
- [MotionBERT GitHub](https://github.com/Walter0807/MotionBERT) — Apache 2.0 license
- [MotionBERT HuggingFace checkpoints](https://huggingface.co/walterzhu/MotionBERT)
- [MotionBERT COCO-to-H36M mapping](https://github.com/Walter0807/MotionBERT/blob/main/lib/data/dataset_action.py)
- [Human3.6M skeleton definition](http://vision.imar.ro/human3.6m/)
- MotionBERT Issue #121 — output units/scale ambiguity
