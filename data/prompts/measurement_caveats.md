# 2D Video Analysis Measurement Caveats

## Camera Angle Effects

- **Arm slot reads low from front-quarter view**: When the camera is positioned at a front-quarter angle (between 1st/3rd base and home plate), the projected arm slot angle consistently reads 15-22 degrees lower than the true value. A pitcher who appears to have a 55-degree arm slot may actually be closer to 70-75 degrees.
- **Rotation depth is compressed**: Hip rotation and trunk rotation appear smaller in 2D projection. Actual rotation angles are larger than what the camera captures from any single viewpoint.
- **Optimal camera position**: True side view (perpendicular to the pitching direction, along the 1st or 3rd base line) minimizes projection errors. For a right-handed pitcher, the 3rd base side is preferred.

## 2D Projection Limitations

- **Depth information is lost**: Any movement toward or away from the camera cannot be measured. This affects metrics that depend on the depth axis: shoulder abduction, horizontal abduction, lateral trunk tilt, and hip-shoulder separation.
- **Four metrics are unmeasurable from a single 2D view**:
  1. Shoulder abduction (arm angle relative to torso in the frontal plane)
  2. Shoulder horizontal abduction (arm position behind the body)
  3. Trunk lateral tilt (side bend toward glove side)
  4. Hip-shoulder separation (requires measuring rotation of both hips and shoulders in the transverse plane)
- **Trunk tilt at ball release is underestimated**: 2D analysis typically reads 4-24 degrees of forward trunk tilt at release, while motion capture labs measure 30-40 degrees for similar mechanics. The camera angle compresses the apparent tilt.

## Spatial Resolution

- **Lower precision than marker-based motion capture**: Lab-grade systems place reflective markers directly on anatomical landmarks and capture at 200-500 Hz. Video-based pose estimation (YOLOv8, MediaPipe) infers joint positions from pixel data at 30-60 Hz, introducing 2-5 degrees of joint angle uncertainty.
- **Keypoint drift under occlusion**: When a joint is partially hidden (e.g., the knee during high leg lift, the wrist behind the head during arm cocking), the pose model may place the keypoint at mid-limb rather than the true joint center. This causes momentary angle spikes or dips.
- **Joint angle accuracy depends on keypoint confidence**: Always check the confidence diagnostics. Angles computed from low-confidence keypoints (below 0.5) should be treated as rough estimates, not precise measurements.

## Missing Data Modalities

- **No force plate data**: Ground reaction forces (GRF) and joint moments (elbow varus, shoulder internal rotation) cannot be measured from video alone. Any moment or force values in the report are estimated from kinematic proxies, not direct measurement.
- **No EMG data**: Muscle activation timing and magnitude are not available. Kinetic chain sequencing is inferred from joint angle velocities, not muscle firing patterns.
- **No ball tracking**: Ball velocity, spin rate, and spin axis require separate tracking hardware (Rapsodo, TrackMan). Video analysis covers mechanics only.

## Interpreting Results

- Treat 2D metrics as **directional indicators**, not precise measurements. They show whether a pitcher is in the right ballpark, not exact degree values.
- **Trends matter more than absolutes**: Comparing the same pitcher across sessions (same camera setup) reveals improvement or regression reliably, even if the absolute values have 2D projection bias.
- **Cross-reference with the eye test**: If the metrics say one thing but the video shows something different, trust the video. The metrics are a starting point for analysis, not the final word.