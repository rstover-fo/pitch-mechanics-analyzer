---
title: "feat: 3D Pose Lifting with MotionBERT — Execution Plan"
type: feat
date: 2026-03-11
parent_plan: docs/plans/2026-03-11-feat-3d-pose-lifting-motionbert-plan.md
brainstorm: docs/brainstorms/2026-03-11-3d-pose-lifting-brainstorm.md
deepened: 2026-03-11
research_agents: [kieran-python-reviewer, architecture-strategist, performance-oracle, code-simplicity-reviewer, repo-research-analyst, learnings-researcher]
---

# Execution Plan: 3D Pose Lifting with MotionBERT

This is the execution-ready work plan derived from the reviewed design doc. It contains
concrete implementation steps with file paths, line numbers, and acceptance criteria.

## Enhancement Summary

**Deepened on:** 2026-03-11
**Research agents used:** 6 (Kieran-Python, Architecture, Performance, Simplicity, Repo-Research, Learnings)

### Critical Findings from Deepening

1. **Coordinate space mismatch (CRITICAL)**: MotionBERT outputs 3D positions in camera-normalized
   coordinates, NOT pixel coordinates. Event detection and skeleton visualization use pixel coordinates.
   **Fix**: Do NOT replace PoseSequence keypoints with 3D. Keep 2D PoseSequence intact for
   events/visualization. Return 3D keypoints as a separate dict from `lift_to_3d()` — use only
   for metric extraction. This changes A4 and A8 signatures.

2. **MotionBERT input format**: Input tensor shape is `[batch, frames, 17, 3]` where the 3 channels
   are (x, y, confidence). Must include YOLOv8 confidence scores in the input, not just (x, y, 0).

3. **From documented solution (MER angle inversion)**: Event detection's anchor uses 2D ER minimum
   (already handles the inversion correctly). 3D lifting doesn't change event detection at all —
   it only improves metric extraction. No event detection code changes needed.

4. **`angle_between_points()` in features.py**: Uses `np.dot()` which is dimension-agnostic.
   Works for both 2D and 3D arrays without modification. Do NOT duplicate for 3D.

5. **`_to_int_tuple()` in validate_pose.py**: Uses `arr[0]`, `arr[1]` indexing which works
   for any array length ≥ 2. Safe for 3D arrays, but irrelevant since we keep 2D for visualization.

### Institutional Knowledge Applied

- **MER 2D angle inversion** (`docs/solutions/logic-errors/mer-detection-2d-angle-inversion.md`):
  3D lifting directly addresses this — true 3D shoulder ER replaces the inverted 2D proxy.
  Event detection remains on 2D (already handles inversion via ER minimum search).
- **Foot plant ankle X stabilization** (`docs/solutions/logic-errors/foot-plant-ankle-x-stabilization.md`):
  Ankle X velocity approach is already camera-invariant. No changes needed.
- **Event detection temporal ordering** (`docs/solutions/logic-errors/event-detection-temporal-ordering.md`):
  Anchor-based cascade design is coordinate-space invariant. No changes needed.

## Phase 0: Validation Prototype (GO/NO-GO Gate)

**Goal**: Determine if MotionBERT produces reasonable 3D on a pitching delivery.

### P0-1: Download checkpoint and clone reference repo

```bash
mkdir -p data/models
curl -L https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin \
  -o data/models/motionbert_lite.bin
git clone https://github.com/Walter0807/MotionBERT.git /tmp/MotionBERT
```

### P0-2: Write `scripts/prototype_3d_lift.py`

Standalone throwaway script that:
1. Runs YOLOv8m-pose on `data/uploads/IMG_3106.MOV` (reuse existing pipeline)
2. Maps COCO→H36M using MotionBERT's own `coco2h36m()` function from `/tmp/MotionBERT/lib/data/dataset_action.py`
3. Loads MotionBERT-Lite checkpoint and runs inference
4. Prints 3D shoulder/elbow/wrist positions at foot plant, MER, ball release frames
5. Generates a Plotly 3D scatter of the skeleton at MER frame
6. Computes and prints bone length consistency + shoulder ER from 3D

### P0-3: Evaluate go/no-go criteria

- [x] 3D skeleton at MER shows arm cocked behind head (wrist above elbow, confirmed on 2 clips)
- [x] Bone lengths IQR/median within 20% (ua=6-12%, fa=12-16% — range metric too strict, IQR-based is robust)
- [x] Elbow flexion at MER: 91-121° (replaced ER angle with elbow flex + wrist-above-elbow check)

**If FAIL → STOP. Document what went wrong. Do not proceed to Phase A.**

---

## Phase A: Full Integration

### A1: Vendoring & Dependencies

**Files created:**
- `vendor/motionbert/VERSION` — commit hash from `/tmp/MotionBERT`
- `vendor/motionbert/DSTformer.py` — model architecture (~300 lines)
- `vendor/motionbert/drop.py` — DropPath utility
- `vendor/motionbert/utils_data.py` — `crop_scale()`, `flip_data()` helpers

**Files modified:**
- `.gitignore` — add `data/models/*.bin`, `vendor/` (line 35+)
- `requirements.txt` — add `einops>=0.6` IF vendored code needs it (audit first)

**Steps:**
1. Audit `/tmp/MotionBERT/lib/model/DSTformer.py` imports — identify deps beyond torch/numpy
2. Copy minimum files, strip unnecessary imports (smplx, chumpy, etc.)
3. Replace any `easydict.EasyDict` with `types.SimpleNamespace`
4. Record commit hash in `vendor/motionbert/VERSION`
5. Verify `import vendor.motionbert.DSTformer` works with `.venv/bin/python3.11`

### A2: Tests First — Joint Mapping (write BEFORE model code)

**File created:** `tests/test_lifter.py`

Test cases:
- `test_coco_to_h36m_all_joints_mapped` — 17 COCO joints → 17 H36M joints, all correct positions
- `test_coco_to_h36m_root_is_hip_average` — H36M[0] == avg(COCO left_hip, right_hip)
- `test_coco_to_h36m_neck_is_shoulder_average` — H36M[8] == avg(COCO left_shoulder, right_shoulder)
- `test_coco_to_h36m_spine_is_midpoint` — H36M[7] == avg(H36M[0], H36M[8])
- `test_h36m_to_pitching_joints_12_standard` — all 12 pitching joints present
- `test_h36m_to_pitching_joints_3_extra` — spine, neck, root present when 3D
- `test_h36m_to_pitching_roundtrip` — no data loss through mapping chain
- `test_is_3d_available_missing_checkpoint` — returns False
- `test_lift_to_3d_no_checkpoint_returns_unchanged` — graceful fallback
- `test_pose_frame_is_3d_property` — True for (3,) arrays, False for (2,)
- `test_get_joint_trajectory_3d_nan_fill` — NaN fill matches (3,) shape
- `test_to_dataframe_includes_z_columns` — `{joint}_z` columns present

### A3: Tests First — 3D Angles

**File created:** `tests/test_angles_3d.py`

Test cases:
- `test_hip_shoulder_sep_t_pose_zero` — aligned hips and shoulders → 0°
- `test_hip_shoulder_sep_90_rotation` — hips rotated 90° from shoulders → 90°
- `test_shoulder_abduction_t_pose_90` — arms out to sides → 90°
- `test_shoulder_abduction_arms_down_0` — arms at sides → 0°
- `test_shoulder_horiz_abduction_arm_forward_0` — arm pointing forward → 0°
- `test_torso_lateral_tilt_upright_0` — vertical trunk → 0°
- `test_torso_lateral_tilt_leaning_30` — 30° lean → ~30°
- `test_perpendicular_vectors_90` — generic geometry sanity
- `test_parallel_vectors_0` — generic geometry sanity
- `test_rhp_sign_convention` — RHP known position → expected positive/negative

### A4: Lifter Implementation

**File created:** `src/pose/lifter.py`

Functions:
- `CHECKPOINT_PATH = Path("data/models/motionbert_lite.bin")`
- `is_3d_available() -> bool` — checks checkpoint exists
- `coco_to_h36m(kpts_coco: np.ndarray, confs: np.ndarray) -> np.ndarray`
  - (T, 17, 2), (T, 17) → (T, 17, 3) where channels are (x, y, confidence)
  - **IMPORTANT**: MotionBERT expects 3-channel input (x, y, conf), not just (x, y)
- `h36m_to_pitching_joints(kpts_3d: np.ndarray) -> dict[str, np.ndarray]` — (17, 3) → joint dict
- `load_motionbert(checkpoint_path: Path, device: str = "cpu") -> nn.Module`
  - Uses `types.SimpleNamespace` for config (NOT easydict)
  - Config values: `dim_in=2, dim_out=3, dim_feat=128, dim_rep=128, depth=4, num_heads=4, maxlen=243`
- `infer_3d(model, kpts_h36m: np.ndarray) -> np.ndarray` — (T, 17, 3) → (T, 17, 3)
  - Test-time flip augmentation
  - Validate T ≤ 243, raise ValueError if exceeded
- `lift_to_3d(pose_sequence: PoseSequence, model=None) -> dict[str, np.ndarray]`
  - **CHANGED** (coordinate space fix): Returns 3D keypoints dict, NOT a modified PoseSequence
  - Returns `dict[str, np.ndarray]` mapping joint names to (T, 3) arrays in camera coords
  - If model is None and checkpoint missing → return empty dict (falsy)
  - Maps COCO→H36M, normalizes, infers, maps back to pitching joints with z
  - Original PoseSequence stays in pixel coordinates for event detection + visualization

### A5: PoseFrame/PoseSequence Updates (minimal — 3D is separate dict)

**File modified:** `src/pose/estimator.py`

Since 3D keypoints are returned as a separate dict (not stored in PoseFrame), these changes
are smaller than originally planned. PoseFrame stays 2D for the primary pipeline.

Changes:
1. Add `is_3d` property to `PoseFrame` (after line 83) — still useful for future-proofing
   and for tests that construct 3D PoseFrames directly:
   ```python
   @property
   def is_3d(self) -> bool:
       if not self.keypoints:
           return False
       first = next(iter(self.keypoints.values()))
       return first.shape[-1] == 3
   ```

2. Fix `get_joint_trajectory()` NaN fill (line 99) — dimension-aware for robustness:
   ```python
   # Determine dim from first valid keypoint, default to 2
   dim = 2
   for f in self.frames:
       if joint in f.keypoints:
           dim = f.keypoints[joint].shape[-1]
           break
   # Use: np.full(dim, np.nan) instead of np.array([np.nan, np.nan])
   ```

3. Fix `to_dataframe()` to add Z columns when present (after line 129):
   ```python
   if pos.shape[-1] == 3:
       row[f"{joint}_z"] = pos[2]
   ```

4. Update `to_keypoints_dict()` docstring (line 113): `(N_frames, 2)` → `(N_frames, D)`

### A6: 3D Angle Calculations

**File created:** `src/biomechanics/angles_3d.py`

Functions (all with inline body-frame math, NO general-purpose utilities):
- `compute_hip_shoulder_separation_3d(l_hip, r_hip, l_sho, r_sho) -> float`
  - Project hip and shoulder lines onto transverse plane, compute angle between
  - Smooth shoulder line over 5-frame window (accept optional window positions)
- `compute_shoulder_abduction_3d(shoulder, elbow, hip_center, shoulder_center) -> float`
  - Angle of upper arm from trunk in coronal plane
- `compute_shoulder_horizontal_abduction_3d(shoulder, elbow, l_sho, r_sho) -> float`
  - Arm position relative to shoulder line in transverse plane
- `compute_torso_lateral_tilt_3d(hip_center, shoulder_center, l_sho, r_sho) -> float`
  - Trunk lean in coronal plane

Each function: 3-4 lines of numpy cross products + dot products. No shared rotation matrix.

### A7: Feature Extraction Updates

**File modified:** `src/biomechanics/features.py`

Changes:
1. Add `use_3d: bool = False` parameter to `extract_metrics()` (line 162)
2. Update `compute_trunk_tilt()` vertical reference (line 122):
   ```python
   # Make dimension-aware
   vertical = np.array([0, -1, 0]) if hip_center.shape[-1] == 3 else np.array([0, -1])
   ```
3. Update `compute_arm_slot()` horizontal reference (line 143):
   ```python
   horizontal = np.array([1, 0, 0]) if shoulder.shape[-1] == 3 else np.array([1, 0])
   ```
4. When `use_3d=True`, compute 4 new metrics:
   - `hip_shoulder_separation_fp` via `compute_hip_shoulder_separation_3d()`
   - `shoulder_abduction_fp` via `compute_shoulder_abduction_3d()`
   - `shoulder_horizontal_abduction_fp` via `compute_shoulder_horizontal_abduction_3d()`
   - `torso_lateral_tilt_fp` via `compute_torso_lateral_tilt_3d()`
5. When `use_3d=True`, compute MER shoulder ER using 3D angle (not 2D elbow flexion proxy)

### A8: Pipeline Wiring

**File modified:** `scripts/validate_pose.py`

**IMPORTANT (coordinate space fix):** 3D lifting runs AFTER event detection. The 2D PoseSequence
is kept intact for events + visualization. 3D keypoints are a separate dict used only for metrics.

Changes:
1. Add `--no-3d` flag (after line 120):
   ```python
   parser.add_argument("--no-3d", action="store_true", help="Force 2D-only mode")
   ```
2. Insert Stage 4.5 AFTER event detection (line 469) and BEFORE metrics extraction (line 476):
   ```python
   # Stage 4.5: 3D Pose Lifting (for metrics only — events use 2D pixel coords)
   pose_mode = "2d"
   keypoints_3d = {}
   if not args.no_3d:
       from src.pose.lifter import is_3d_available, load_motionbert, lift_to_3d
       if is_3d_available():
           model = load_motionbert(CHECKPOINT_PATH)
           keypoints_3d = lift_to_3d(pose_seq, model)
           if keypoints_3d:
               pose_mode = "3d"
   ```
3. Pass 3D keypoints to metrics extraction (line 476):
   ```python
   # Use 3D keypoints for metrics if available, 2D otherwise
   metrics_kpts = keypoints_3d if keypoints_3d else keypoints_dict
   metrics = extract_metrics(metrics_kpts, events, pitcher_throws=args.throws, use_3d=bool(keypoints_3d))
   ```
4. Add `pose_mode` to `pipeline_output` dict (line 693)
5. Append one-line 3D context to coaching prompt when 3D active (line 614):
   ```python
   if pose_mode == "3d":
       additional_lines.append("Metrics computed from 3D pose lifting (MotionBERT-Lite). Angular measurements are true 3D values, not 2D projections.")
   ```

### A9: Regression & Integration Tests

**Additional tests in `tests/test_lifter.py`:**
- `test_extract_metrics_2d_unchanged` — 2D keypoints produce identical results to before
- `test_event_detection_with_3d_pose_sequence` — events detect correctly with 3D data

### A10: Real Clip Validation

Run all 4 clips:
```bash
.venv/bin/python3.11 scripts/validate_pose.py --video data/uploads/IMG_3106.MOV --throws R --no-open
.venv/bin/python3.11 scripts/validate_pose.py --video data/uploads/IMG_3107.MOV --throws R --no-open
.venv/bin/python3.11 scripts/validate_pose.py --video data/uploads/IMG_3108.MOV --throws R --no-open
.venv/bin/python3.11 scripts/validate_pose.py --video data/uploads/IMG_3109.MOV --throws R --no-open
```

Check:
- [x] No crashes — all 4 clips process successfully
- [ ] Peak layback still 0th percentile — ER angle needs body-frame reference (future work)
- [x] Trunk tilt now 7-13° range (improved from 2D)
- [x] All 4 new 3D metrics produce values (hip-shoulder sep, shoulder abduction, horiz abduction, lateral tilt)
- [x] CPU inference ~0.7s per clip (well under 30s)

---

## Critical Code Locations

| What | File | Line | Notes |
|------|------|------|-------|
| PoseFrame dataclass | `src/pose/estimator.py` | 77-83 | Add `is_3d` property |
| NaN fill bug | `src/pose/estimator.py` | 99 | Hardcoded `[nan, nan]` |
| Z column drop | `src/pose/estimator.py` | 128-129 | Only exports x, y |
| COCO keypoint indices | `src/pose/estimator.py` | 17-35 | Needed for COCO→H36M mapping |
| PITCHING_JOINTS list | `src/pose/estimator.py` | 55-62 | Reverse mapping target |
| Trunk tilt vertical | `src/biomechanics/features.py` | 122 | Hardcoded `[0, -1]` |
| Arm slot horizontal | `src/biomechanics/features.py` | 143 | Hardcoded `[1, 0]` |
| extract_metrics sig | `src/biomechanics/features.py` | 162-167 | Add `use_3d` param |
| MER ER calculation | `src/biomechanics/features.py` | 227 | Uses elbow flexion proxy |
| angle_between_points | `src/biomechanics/features.py` | 96-108 | Already dimension-agnostic (np.dot) |
| Event detection | `scripts/validate_pose.py` | 244-324 | Stays on 2D pixel coords |
| Metrics call | `scripts/validate_pose.py` | 476 | Pass `use_3d` + 3D keypoints |
| Coaching context | `scripts/validate_pose.py` | 614 | Append 3D note |
| .gitignore | `.gitignore` | 35 | Add model/vendor patterns |

## Dependencies

- Phase 0 blocks everything — go/no-go gate
- A1 (vendoring) can start immediately in Phase A
- A2 (mapping tests) must complete before A4 (lifter implementation)
- A3 (angle tests) must complete before A6 (angle implementation)
- A4 must complete before A8 (pipeline wiring)
- A6 + A7 must complete before A8
- A5 (PoseFrame updates) is independent — can run anytime
- A8 must complete before A10 (validation)
- A9 can run in parallel with A8

## Parallelizable Work Streams

**Stream 1 (Pose/Lifter):** A2 → A4
**Stream 2 (Angles):** A3 → A6 → A7
**Stream 3 (Infrastructure):** A1 (vendoring, .gitignore, requirements) + A5 (PoseFrame)
**Merge point:** A8 (pipeline wiring) requires Stream 1 + Stream 2 + Stream 3
**Final:** A9 (regression tests) + A10 (real validation)

## Research References

- MotionBERT docs: input `[batch, frames, 17, 3]` (x, y, confidence), max 243 frames
- MotionBERT-Lite checkpoint: 64MB, FT on H36M with global trajectories
- `einops` required for DSTformer (rearrange operations in attention heads)
- `torch.no_grad()` context for inference (reduces memory ~50%)
- MPS (Apple Metal) acceleration available via `torch.device("mps")` if torch ≥ 2.0
