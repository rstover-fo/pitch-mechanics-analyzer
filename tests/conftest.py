"""Shared test fixtures for pitch mechanics analyzer tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biomechanics.events import DeliveryEvents


@pytest.fixture
def fps():
    return 30.0


@pytest.fixture
def n_frames():
    return 60


@pytest.fixture
def delivery_events():
    """Realistic delivery events for a ~2-second clip at 30 fps."""
    return DeliveryEvents(
        leg_lift_apex=10,
        foot_plant=25,
        max_external_rotation=35,
        ball_release=40,
        max_internal_rotation=48,
        fps=30.0,
    )


@pytest.fixture
def synthetic_keypoints(n_frames):
    """Synthetic keypoints dict for a RHP with 60 frames.

    Produces plausible (x, y) positions in screen coordinates (y-down)
    for all 12 joints.  Joints drift slightly frame-to-frame so that
    velocity-based detections have something to work with.
    """
    rng = np.random.RandomState(42)
    n = n_frames

    def make_joint(base_x, base_y, dx=0.0, dy=0.0, noise=1.0):
        """Create (N, 2) array: linear drift + small noise."""
        xs = np.linspace(base_x, base_x + dx * n, n) + rng.randn(n) * noise
        ys = np.linspace(base_y, base_y + dy * n, n) + rng.randn(n) * noise
        return np.column_stack([xs, ys])

    kp = {
        # Right side (throwing arm for RHP)
        "right_shoulder": make_joint(320, 200, dx=0.1),
        "right_elbow": make_joint(360, 220, dx=0.3),
        "right_wrist": make_joint(400, 240, dx=0.8, noise=2.0),
        "right_hip": make_joint(310, 350),
        "right_knee": make_joint(310, 450),
        "right_ankle": make_joint(310, 540),
        # Left side (lead leg for RHP)
        "left_shoulder": make_joint(280, 200, dx=0.1),
        "left_elbow": make_joint(240, 220, dx=0.2),
        "left_wrist": make_joint(200, 240, dx=0.3),
        "left_hip": make_joint(290, 350, dx=0.2),
        "left_knee": make_joint(270, 450, dx=0.3),
        "left_ankle": make_joint(250, 540, dx=0.4),
    }

    # Give the right wrist a clear speed burst around frame 38-42
    # to simulate arm acceleration / ball release
    for i in range(38, min(43, n)):
        kp["right_wrist"][i] += np.array([15.0 * (i - 37), -8.0 * (i - 37)])

    # Give hips and shoulders different rotations mid-delivery to
    # create hip-shoulder separation
    for i in range(15, 30):
        offset = (i - 15) * 0.6
        kp["left_hip"][i, 0] -= offset
        kp["right_hip"][i, 0] += offset
        # Shoulders lag behind hips
    for i in range(20, 35):
        offset = (i - 20) * 0.5
        kp["left_shoulder"][i, 0] -= offset
        kp["right_shoulder"][i, 0] += offset

    return kp


@pytest.fixture
def synthetic_ankle_x_stride():
    """Ankle X data with clear forward motion then stop (foot plant pattern)."""
    n = 60
    # Forward motion accelerates then decelerates to zero
    vx = np.concatenate([
        np.linspace(0, 15, 20),   # acceleration
        np.linspace(15, 0, 15),   # deceleration to stop
        np.zeros(25),             # stationary after plant
    ])
    x = np.cumsum(vx) + 100
    return x


@pytest.fixture
def synthetic_ankle_y_dip():
    """Ankle Y data with dip-then-recovery pattern (foot plant fallback)."""
    n = 60
    baseline = 500.0
    y = np.full(n, baseline)
    # Dip during stride (frames 10-25)
    for i in range(10, 25):
        y[i] = baseline - 40 * np.sin(np.pi * (i - 10) / 15)
    # Recovery back to baseline
    for i in range(25, 35):
        frac = (i - 25) / 10
        y[i] = baseline - 40 * np.sin(np.pi) * (1 - frac)
    return y


@pytest.fixture
def synthetic_wrist_speed():
    """Wrist speed signal with clear peak around frame 40."""
    n = 60
    speed = np.zeros(n)
    # Gradual build-up
    for i in range(20, 40):
        speed[i] = 200 * ((i - 20) / 20) ** 2
    # Sharp peak
    speed[39] = 350
    speed[40] = 500
    speed[41] = 450
    # Deceleration
    for i in range(42, 55):
        speed[i] = 450 * np.exp(-(i - 41) * 0.3)
    return speed


@pytest.fixture
def synthetic_shoulder_er():
    """Shoulder ER signal with trough (2D convention) before wrist speed peak."""
    n = 60
    er = np.full(n, 120.0)
    # Decrease toward MER trough (layback) around frame 33-36
    for i in range(25, 40):
        if i <= 34:
            er[i] = 120 - 40 * ((i - 25) / 9)
        else:
            er[i] = 80 + 30 * ((i - 34) / 6)
    return er
