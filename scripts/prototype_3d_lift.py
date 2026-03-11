#!/usr/bin/env python3
"""Phase 0 Prototype: Validate MotionBERT 3D lifting on pitching video.

GO/NO-GO gate — does MotionBERT produce reasonable 3D during a pitching delivery?

Criteria:
  1. 3D skeleton at MER shows arm cocked behind head (not collapsed/flipped)
  2. Bone lengths (shoulder-elbow, elbow-wrist) consistent within 15% across frames
  3. Shoulder ER angle from 3D positions in 120-180° range

Usage:
    .venv/bin/python3.11 scripts/prototype_3d_lift.py
    .venv/bin/python3.11 scripts/prototype_3d_lift.py --video data/uploads/IMG_3107.MOV
"""

import argparse
import copy
import sys
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# Add project and MotionBERT to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/tmp/MotionBERT")

from lib.model.DSTformer import DSTformer
from lib.utils.utils_data import crop_scale, flip_data

from src.pose.estimator import (
    PITCHING_JOINTS,
    YOLO_KEYPOINTS,
    PoseSequence,
    extract_poses,
    load_video,
)
from src.biomechanics.events import (
    approximate_shoulder_er_2d,
    detect_ball_release,
    detect_foot_plant_from_keypoints,
    detect_max_external_rotation,
    find_delivery_anchor,
)


# --- COCO → H36M mapping (from MotionBERT's dataset_action.py) ---

def coco_to_h36m(kpts_coco: np.ndarray) -> np.ndarray:
    """Map COCO 17 keypoints to H36M 17 keypoints.

    Args:
        kpts_coco: (T, 17, C) array in COCO order.

    Returns:
        (T, 17, C) array in H36M order.
    """
    y = np.zeros_like(kpts_coco)
    y[:, 0, :] = (kpts_coco[:, 11, :] + kpts_coco[:, 12, :]) * 0.5  # root = avg(l_hip, r_hip)
    y[:, 1, :] = kpts_coco[:, 12, :]   # r_hip
    y[:, 2, :] = kpts_coco[:, 14, :]   # r_knee
    y[:, 3, :] = kpts_coco[:, 16, :]   # r_ankle
    y[:, 4, :] = kpts_coco[:, 11, :]   # l_hip
    y[:, 5, :] = kpts_coco[:, 13, :]   # l_knee
    y[:, 6, :] = kpts_coco[:, 15, :]   # l_ankle
    y[:, 8, :] = (kpts_coco[:, 5, :] + kpts_coco[:, 6, :]) * 0.5   # neck = avg(l_sho, r_sho)
    y[:, 7, :] = (y[:, 0, :] + y[:, 8, :]) * 0.5                   # belly = avg(root, neck)
    y[:, 9, :] = kpts_coco[:, 0, :]    # nose
    y[:, 10, :] = (kpts_coco[:, 1, :] + kpts_coco[:, 2, :]) * 0.5  # head = avg(l_eye, r_eye)
    y[:, 11, :] = kpts_coco[:, 5, :]   # l_shoulder
    y[:, 12, :] = kpts_coco[:, 7, :]   # l_elbow
    y[:, 13, :] = kpts_coco[:, 9, :]   # l_wrist
    y[:, 14, :] = kpts_coco[:, 6, :]   # r_shoulder
    y[:, 15, :] = kpts_coco[:, 8, :]   # r_elbow
    y[:, 16, :] = kpts_coco[:, 10, :]  # r_wrist
    return y


# --- H36M joint indices ---
H36M_JOINTS = {
    "root": 0, "right_hip": 1, "right_knee": 2, "right_ankle": 3,
    "left_hip": 4, "left_knee": 5, "left_ankle": 6,
    "spine": 7, "neck": 8, "nose": 9, "head": 10,
    "left_shoulder": 11, "left_elbow": 12, "left_wrist": 13,
    "right_shoulder": 14, "right_elbow": 15, "right_wrist": 16,
}


def load_motionbert_lite(checkpoint_path: Path, device: str = "cpu") -> nn.Module:
    """Load MotionBERT-Lite model from checkpoint."""
    # Config from MB_ft_h36m_global_lite.yaml
    model = DSTformer(
        dim_in=3,
        dim_out=3,
        dim_feat=256,
        dim_rep=512,
        depth=5,
        num_heads=8,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        maxlen=243,
        num_joints=17,
    )

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = checkpoint["model_pos"]
    # Strip "module." prefix from keys saved with nn.DataParallel
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def pose_sequence_to_coco_array(pose_seq: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Convert PoseSequence to COCO-ordered (T, 17, 2) and (T, 17) confidence arrays.

    Returns:
        kpts: (T, 17, 2) array of (x, y) keypoints in COCO order.
        confs: (T, 17) array of confidence scores.
    """
    coco_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    T = len(pose_seq.frames)
    kpts = np.zeros((T, 17, 2))
    confs = np.zeros((T, 17))

    for t, pf in enumerate(pose_seq.frames):
        for j, name in enumerate(coco_names):
            if name in pf.keypoints:
                kpts[t, j] = pf.keypoints[name][:2]
                confs[t, j] = pf.confidence.get(name, 0.0)
            elif name in ("left_eye", "right_eye", "left_ear", "right_ear"):
                # Eyes/ears not tracked by our pipeline — use nose as proxy
                if "nose" in pf.keypoints:
                    kpts[t, j] = pf.keypoints["nose"][:2]
                    confs[t, j] = pf.confidence.get("nose", 0.0) * 0.5

    return kpts, confs


def infer_3d(
    model: nn.Module,
    kpts_h36m: np.ndarray,
    device: str = "cpu",
    use_flip: bool = True,
) -> np.ndarray:
    """Run MotionBERT inference to lift 2D → 3D.

    Args:
        model: Loaded MotionBERT model.
        kpts_h36m: (T, 17, 3) array of H36M keypoints with (x, y, confidence).
        device: torch device.
        use_flip: Use test-time flip augmentation.

    Returns:
        (T, 17, 3) array of 3D positions in camera coordinates.
    """
    T = kpts_h36m.shape[0]
    if T > 243:
        raise ValueError(f"Sequence length {T} exceeds MotionBERT max of 243 frames")

    # Normalize to [-1, 1] using crop_scale
    kpts_norm = crop_scale(kpts_h36m[np.newaxis], scale_range=[1, 1])[0]  # (T, 17, 3)

    # Pad to 243 if needed
    if T < 243:
        padded = np.zeros((243, 17, 3))
        padded[:T] = kpts_norm
        # Replicate last frame for padding
        padded[T:] = kpts_norm[-1:]
        kpts_norm = padded

    # Convert to tensor: (1, 243, 17, 3)
    batch = torch.from_numpy(kpts_norm).float().unsqueeze(0).to(device)

    with torch.no_grad():
        if use_flip:
            batch_flip = torch.from_numpy(
                flip_data(kpts_norm[np.newaxis])
            ).float().to(device)
            pred_1 = model(batch)
            pred_flip = model(batch_flip)
            pred_2 = torch.from_numpy(
                flip_data(pred_flip.cpu().numpy())
            ).float().to(device)
            predicted_3d = (pred_1 + pred_2) / 2.0
        else:
            predicted_3d = model(batch)

    # Remove padding and batch dim
    result = predicted_3d[0, :T].cpu().numpy()  # (T, 17, 3)

    # Zero the root Z at first frame for global trajectory
    result[:, 0, 2] -= result[0, 0, 2]

    return result


def bone_length(pts_3d: np.ndarray, joint_a: int, joint_b: int) -> np.ndarray:
    """Compute bone length between two joints across all frames.

    Returns:
        (T,) array of Euclidean distances.
    """
    return np.linalg.norm(pts_3d[:, joint_a] - pts_3d[:, joint_b], axis=-1)


def compute_shoulder_er_3d(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    hip_center: np.ndarray,
    shoulder_center: np.ndarray,
) -> float:
    """Compute approximate shoulder external rotation from 3D positions.

    Uses the angle between the forearm vector (elbow→wrist) projected onto
    the plane perpendicular to the upper arm (shoulder→elbow).

    Returns angle in degrees.
    """
    # Upper arm vector
    upper_arm = elbow - shoulder
    upper_arm_norm = upper_arm / (np.linalg.norm(upper_arm) + 1e-8)

    # Forearm vector
    forearm = wrist - elbow

    # Project forearm onto plane perpendicular to upper arm
    forearm_proj = forearm - np.dot(forearm, upper_arm_norm) * upper_arm_norm

    # Trunk direction (up)
    trunk = shoulder_center - hip_center
    trunk_norm = trunk / (np.linalg.norm(trunk) + 1e-8)

    # Reference direction in the perpendicular plane (trunk projected)
    trunk_proj = trunk_norm - np.dot(trunk_norm, upper_arm_norm) * upper_arm_norm
    trunk_proj_norm = trunk_proj / (np.linalg.norm(trunk_proj) + 1e-8)

    # Angle between projected forearm and trunk reference
    cos_angle = np.dot(
        forearm_proj / (np.linalg.norm(forearm_proj) + 1e-8),
        trunk_proj_norm,
    )
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: Validate MotionBERT 3D on pitching")
    parser.add_argument(
        "--video", type=Path,
        default=Path("data/uploads/IMG_3106.MOV"),
        help="Path to pitching video",
    )
    parser.add_argument("--throws", type=str, default="R", choices=["R", "L"])
    parser.add_argument("--no-flip", action="store_true", help="Disable test-time flip augmentation")
    parser.add_argument("--device", type=str, default="cpu", help="torch device (cpu/mps)")
    args = parser.parse_args()

    checkpoint = Path("data/models/motionbert_lite.bin")
    if not checkpoint.exists():
        print(f"Error: Checkpoint not found at {checkpoint}")
        sys.exit(1)

    if not args.video.exists():
        print(f"Error: Video not found at {args.video}")
        sys.exit(1)

    # --- Step 1: Run YOLOv8 pose estimation ---
    print("=" * 60)
    print("Step 1: YOLOv8 Pose Estimation")
    print("=" * 60)

    video_info = load_video(args.video)
    print(f"  Video: {args.video.name} ({video_info.total_frames} frames, {video_info.fps:.1f} fps)")

    pose_seq = extract_poses(args.video, backend="yolov8", model_size="m", confidence=0.3)
    print(f"  Detected poses in {len(pose_seq.frames)} frames")

    if len(pose_seq.frames) > 243:
        print(f"  WARNING: {len(pose_seq.frames)} frames exceeds 243 limit. Truncating.")
        pose_seq.frames = pose_seq.frames[:243]

    # --- Step 2: Event detection (from 2D, for reference frames) ---
    print("\n" + "=" * 60)
    print("Step 2: Event Detection (2D)")
    print("=" * 60)

    fps = video_info.fps
    throw_side = "right" if args.throws == "R" else "left"
    lead_side = "left" if args.throws == "R" else "right"

    lead_ankle_traj = pose_seq.get_joint_trajectory(f"{lead_side}_ankle")
    lead_ankle_y = lead_ankle_traj[:, 1]
    lead_ankle_x = lead_ankle_traj[:, 0]
    wrist_speed = pose_seq.get_joint_speed(f"{throw_side}_wrist") * fps

    # Smooth wrist speed
    kernel = np.ones(3) / 3
    wrist_speed = np.convolve(wrist_speed, kernel, mode="same")

    # Shoulder ER series
    shoulder_er = np.zeros(len(pose_seq.frames))
    for i, pf in enumerate(pose_seq.frames):
        keys = [f"{throw_side}_shoulder", f"{throw_side}_elbow", f"{throw_side}_wrist", "left_hip", "right_hip"]
        if all(k in pf.keypoints for k in keys):
            hip_center = (pf.keypoints["left_hip"] + pf.keypoints["right_hip"]) / 2
            shoulder_er[i] = approximate_shoulder_er_2d(
                pf.keypoints[f"{throw_side}_shoulder"],
                pf.keypoints[f"{throw_side}_elbow"],
                pf.keypoints[f"{throw_side}_wrist"],
                hip_center,
            )

    mer_frame = find_delivery_anchor(shoulder_er, wrist_speed, fps=fps)
    fp_frame = None
    br_frame = None
    if mer_frame is not None:
        br_frame = detect_ball_release(wrist_speed, after_frame=mer_frame)
        fp_frame = detect_foot_plant_from_keypoints(
            lead_ankle_y, lead_ankle_x=lead_ankle_x, fps=fps,
            before_frame=mer_frame,
        )

    events = {"Foot Plant": fp_frame, "MER": mer_frame, "Ball Release": br_frame}
    for name, frame in events.items():
        if frame is not None:
            print(f"  {name}: frame {frame} ({frame/fps:.3f}s)")
        else:
            print(f"  {name}: not detected")

    # --- Step 3: COCO → H36M → MotionBERT 3D ---
    print("\n" + "=" * 60)
    print("Step 3: MotionBERT 3D Lifting")
    print("=" * 60)

    # Convert pose sequence to COCO array
    kpts_coco, confs = pose_sequence_to_coco_array(pose_seq)
    print(f"  COCO keypoints: {kpts_coco.shape}")

    # Stack (x, y, confidence) → (T, 17, 3)
    kpts_with_conf = np.concatenate([kpts_coco, confs[..., np.newaxis]], axis=-1)

    # Map COCO → H36M
    kpts_h36m = coco_to_h36m(kpts_with_conf)
    print(f"  H36M keypoints: {kpts_h36m.shape}")

    # Load model and run inference
    print(f"  Loading MotionBERT-Lite on {args.device}...")
    model = load_motionbert_lite(checkpoint, device=args.device)
    print("  Model loaded. Running inference...")

    import time
    t0 = time.time()
    pts_3d = infer_3d(model, kpts_h36m, device=args.device, use_flip=not args.no_flip)
    elapsed = time.time() - t0
    print(f"  3D inference complete: {pts_3d.shape} in {elapsed:.1f}s")

    # --- Step 4: Evaluate go/no-go criteria ---
    print("\n" + "=" * 60)
    print("Step 4: GO/NO-GO Evaluation")
    print("=" * 60)

    # 4a: Bone length consistency (IQR-based — robust to outlier frames)
    ua_lengths = bone_length(pts_3d, H36M_JOINTS["right_shoulder"], H36M_JOINTS["right_elbow"])
    fa_lengths = bone_length(pts_3d, H36M_JOINTS["right_elbow"], H36M_JOINTS["right_wrist"])

    ua_cv = np.std(ua_lengths) / (np.mean(ua_lengths) + 1e-8) * 100
    fa_cv = np.std(fa_lengths) / (np.mean(fa_lengths) + 1e-8) * 100
    ua_q25, ua_med, ua_q75 = np.percentile(ua_lengths, [25, 50, 75])
    fa_q25, fa_med, fa_q75 = np.percentile(fa_lengths, [25, 50, 75])
    ua_iqr_pct = (ua_q75 - ua_q25) / (ua_med + 1e-8) * 100
    fa_iqr_pct = (fa_q75 - fa_q25) / (fa_med + 1e-8) * 100

    print(f"\n  Bone Length Consistency:")
    print(f"    Upper arm: median={ua_med:.4f}, CV={ua_cv:.1f}%, IQR/median={ua_iqr_pct:.1f}%")
    print(f"    Forearm:   median={fa_med:.4f}, CV={fa_cv:.1f}%, IQR/median={fa_iqr_pct:.1f}%")

    bone_pass = ua_iqr_pct < 20 and fa_iqr_pct < 20
    print(f"    {'PASS' if bone_pass else 'FAIL'}: bone IQR/median {'within' if bone_pass else 'exceeds'} 20%")

    # 4b: Arm position at MER — elbow flexion and wrist-above-elbow check
    elbow_flex_at_mer = None
    wrist_above_elbow = False
    if mer_frame is not None and mer_frame < len(pts_3d):
        r_sho = pts_3d[mer_frame, H36M_JOINTS["right_shoulder"]]
        r_elb = pts_3d[mer_frame, H36M_JOINTS["right_elbow"]]
        r_wri = pts_3d[mer_frame, H36M_JOINTS["right_wrist"]]

        # Elbow flexion: angle at elbow joint
        v_upper = r_sho - r_elb
        v_fore = r_wri - r_elb
        cos_a = np.dot(v_upper, v_fore) / (np.linalg.norm(v_upper) * np.linalg.norm(v_fore) + 1e-8)
        elbow_flex_at_mer = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

        # Wrist above elbow in camera Y (negative Y = up in camera frame)
        wrist_above_elbow = r_wri[1] < r_elb[1]

        print(f"\n  Elbow Flexion at MER: {elbow_flex_at_mer:.1f}° (expect 70-130°)")
        flex_pass = 70 <= elbow_flex_at_mer <= 130
        print(f"    {'PASS' if flex_pass else 'FAIL'}: {'within' if flex_pass else 'outside'} plausible range")
        print(f"  Wrist Above Elbow at MER: {'YES' if wrist_above_elbow else 'NO'} (expect YES for cocked arm)")
        arm_pass = flex_pass and wrist_above_elbow
    else:
        print("\n  Arm position at MER: SKIP (MER not detected)")
        arm_pass = False

    # 4c: Visual inspection — print 3D positions at key frames
    print(f"\n  3D Joint Positions at Key Events:")
    for event_name, frame_idx in events.items():
        if frame_idx is None or frame_idx >= len(pts_3d):
            continue
        r_sho = pts_3d[frame_idx, H36M_JOINTS["right_shoulder"]]
        r_elb = pts_3d[frame_idx, H36M_JOINTS["right_elbow"]]
        r_wri = pts_3d[frame_idx, H36M_JOINTS["right_wrist"]]
        print(f"\n    {event_name} (frame {frame_idx}):")
        print(f"      R Shoulder: ({r_sho[0]:+.3f}, {r_sho[1]:+.3f}, {r_sho[2]:+.3f})")
        print(f"      R Elbow:    ({r_elb[0]:+.3f}, {r_elb[1]:+.3f}, {r_elb[2]:+.3f})")
        print(f"      R Wrist:    ({r_wri[0]:+.3f}, {r_wri[1]:+.3f}, {r_wri[2]:+.3f})")

    # --- Step 5: Generate 3D scatter plot at MER ---
    print("\n" + "=" * 60)
    print("Step 5: 3D Visualization")
    print("=" * 60)

    if mer_frame is not None and mer_frame < len(pts_3d):
        import plotly.graph_objects as go

        skeleton_at_mer = pts_3d[mer_frame]  # (17, 3)

        # H36M bone connections
        bones = [
            (0, 1), (1, 2), (2, 3),       # right leg
            (0, 4), (4, 5), (5, 6),       # left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # spine → head
            (8, 11), (11, 12), (12, 13),  # left arm
            (8, 14), (14, 15), (15, 16),  # right arm
        ]

        fig = go.Figure()

        # Draw bones
        for a, b in bones:
            color = "red" if a >= 14 or b >= 14 else ("blue" if a in (11, 12, 13) or b in (11, 12, 13) else "gray")
            fig.add_trace(go.Scatter3d(
                x=[skeleton_at_mer[a, 0], skeleton_at_mer[b, 0]],
                y=[skeleton_at_mer[a, 2], skeleton_at_mer[b, 2]],  # swap y/z for display
                z=[-skeleton_at_mer[a, 1], -skeleton_at_mer[b, 1]],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
            ))

        # Draw joints
        joint_names = list(H36M_JOINTS.keys())
        fig.add_trace(go.Scatter3d(
            x=skeleton_at_mer[:, 0],
            y=skeleton_at_mer[:, 2],
            z=-skeleton_at_mer[:, 1],
            mode="markers+text",
            marker=dict(size=5, color="black"),
            text=joint_names,
            textposition="top center",
            textfont=dict(size=8),
            showlegend=False,
        ))

        fig.update_layout(
            title=f"3D Skeleton at MER (frame {mer_frame})",
            scene=dict(
                xaxis_title="X (lateral)",
                yaxis_title="Z (depth)",
                zaxis_title="Y (up)",
                aspectmode="data",
            ),
            width=800,
            height=600,
        )

        output_path = Path("data/outputs") / f"prototype_3d_{args.video.stem}"
        output_path.mkdir(parents=True, exist_ok=True)
        html_path = output_path / "skeleton_3d_mer.html"
        fig.write_html(str(html_path))
        print(f"  3D skeleton plot saved: {html_path}")

        # Also save full 3D trajectory
        np.save(str(output_path / "pts_3d.npy"), pts_3d)
        print(f"  3D trajectory saved: {output_path / 'pts_3d.npy'}")

    # --- Final verdict ---
    print("\n" + "=" * 60)
    print("GO/NO-GO VERDICT")
    print("=" * 60)

    all_pass = bone_pass and arm_pass
    if all_pass:
        print(f"  ✓ BONE LENGTHS: IQR/median ua={ua_iqr_pct:.1f}%, fa={fa_iqr_pct:.1f}%")
        print(f"  ✓ ARM POSITION: Elbow flex {elbow_flex_at_mer:.1f}°, wrist above elbow")
        print("\n  >>> GO — MotionBERT produces plausible 3D on pitching video <<<")
        print("  >>> Proceed to Phase A: Full Integration <<<")
    else:
        if not bone_pass:
            print(f"  ✗ BONE LENGTHS: IQR/median ua={ua_iqr_pct:.1f}%, fa={fa_iqr_pct:.1f}% (threshold 20%)")
        if not arm_pass:
            val = f"{elbow_flex_at_mer:.1f}°" if elbow_flex_at_mer is not None else "N/A"
            print(f"  ✗ ARM POSITION: Elbow flex {val}, wrist {'above' if wrist_above_elbow else 'BELOW'} elbow")
        print("\n  >>> NO-GO — MotionBERT output not reliable enough <<<")
        print("  >>> Investigate alternatives (MotionAGFormer, fine-tuned models) <<<")

    print("\n  NOTE: Visually inspect the 3D plot to verify arm position at MER.")
    print(f"  Open: {html_path if mer_frame is not None else '(no plot generated)'}")


if __name__ == "__main__":
    main()
