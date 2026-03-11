"""Pipeline output validation and sanity checks.

Runs automatic checks on detected events and metrics to flag
implausible results. Not pass/fail gates — warnings for human review.
"""

from typing import Optional

from src.biomechanics.events import DeliveryEvents


def validate_pipeline_output(
    events: DeliveryEvents,
    avg_confidence: float = 1.0,
    metrics: Optional[dict] = None,
) -> list[dict]:
    """Run all sanity checks on pipeline output.

    Args:
        events: Detected delivery events.
        avg_confidence: Mean keypoint confidence across all frames.
        metrics: Optional dict of extracted metric values.

    Returns:
        List of warning dicts with keys: code, severity, message.
    """
    warnings: list[dict] = []
    warnings.extend(_check_event_ordering(events))
    warnings.extend(_check_missing_events(events))
    warnings.extend(_check_phase_timing(events))
    warnings.extend(_check_confidence(avg_confidence))
    if metrics:
        warnings.extend(_check_metric_ranges(metrics))
    return warnings


def _check_event_ordering(events: DeliveryEvents) -> list[dict]:
    ordered = [
        ("leg_lift", events.leg_lift_apex),
        ("foot_plant", events.foot_plant),
        ("max_er", events.max_external_rotation),
        ("ball_release", events.ball_release),
    ]
    present = [(name, frame) for name, frame in ordered if frame is not None]
    for i in range(len(present) - 1):
        if present[i][1] >= present[i + 1][1]:
            return [{
                "code": "events_out_of_order",
                "severity": "error",
                "message": f"{present[i][0]} (frame {present[i][1]}) is not before "
                           f"{present[i+1][0]} (frame {present[i+1][1]})",
            }]
    return []


def _check_missing_events(events: DeliveryEvents) -> list[dict]:
    warnings = []
    for name, frame in [
        ("leg_lift", events.leg_lift_apex),
        ("foot_plant", events.foot_plant),
        ("max_er", events.max_external_rotation),
        ("ball_release", events.ball_release),
    ]:
        if frame is None:
            warnings.append({
                "code": "event_not_detected",
                "severity": "error",
                "message": f"{name} was not detected",
            })
    return warnings


def _check_phase_timing(events: DeliveryEvents) -> list[dict]:
    warnings = []
    fps = events.fps

    phases = [
        ("stride_duration", events.leg_lift_apex, events.foot_plant, 0.1, 1.5),
        ("arm_cocking_duration", events.foot_plant, events.max_external_rotation, 0.01, 0.5),
        ("arm_accel_duration", events.max_external_rotation, events.ball_release, 0.01, 0.5),
    ]

    for name, start, end, min_sec, max_sec in phases:
        if start is None or end is None:
            continue
        duration = (end - start) / fps
        if duration < min_sec or duration > max_sec:
            warnings.append({
                "code": f"{name}_implausible",
                "severity": "warning",
                "message": f"{name.replace('_', ' ')}: {duration:.3f}s "
                           f"(expected {min_sec}-{max_sec}s)",
            })

    return warnings


def _check_confidence(avg_confidence: float) -> list[dict]:
    if avg_confidence < 0.3:
        return [{
            "code": "low_tracking_confidence",
            "severity": "warning",
            "message": f"Mean keypoint confidence {avg_confidence:.3f} is below 0.3",
        }]
    return []


def _check_metric_ranges(metrics: dict) -> list[dict]:
    warnings = []
    ranges = {
        "elbow_flexion_fp": (40, 160, "Elbow flexion @ FP"),
        "max_shoulder_external_rotation": (60, 190, "Peak shoulder ER"),
        "torso_anterior_tilt_fp": (-10, 60, "Trunk tilt @ FP"),
        "lead_knee_angle_fp": (60, 180, "Lead knee @ FP"),
    }
    for key, (lo, hi, label) in ranges.items():
        val = metrics.get(key)
        if val is not None and (val < lo or val > hi):
            warnings.append({
                "code": f"{key}_out_of_range",
                "severity": "warning",
                "message": f"{label}: {val:.1f} deg (expected {lo}-{hi} deg)",
            })
    return warnings
