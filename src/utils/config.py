"""Configuration dataclasses for the pitch mechanics analyzer."""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Paths:
    """File system paths."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    obp_data: Path = field(default=None)
    uploads: Path = field(default=None)
    outputs: Path = field(default=None)

    def __post_init__(self):
        obp_env = os.getenv("OBP_DATA_PATH")
        self.obp_data = Path(obp_env) if obp_env else self.project_root / "data" / "obp"
        self.uploads = self.project_root / "data" / "uploads"
        self.outputs = self.project_root / "data" / "outputs"
        for p in [self.uploads, self.outputs]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class PoseConfig:
    """Pose estimation settings."""
    backend: str = "yolov8"  # "yolov8" or "mediapipe"
    model_size: str = "n"    # YOLOv8 model size: n, s, m, l, x
    confidence_threshold: float = 0.5
    target_fps: int = 30     # Resample video to this FPS for processing


@dataclass
class CameraConfig:
    """Camera setup assumptions."""
    view: str = "side"       # "side" (perpendicular to pitch direction) or "behind"
    pitcher_throws: str = "R"  # "R" or "L"


@dataclass
class CoachingConfig:
    """Claude API coaching settings."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2000
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    pitcher_context: str = "youth developing pitcher"


@dataclass
class AppConfig:
    """Top-level application configuration."""
    paths: Paths = field(default_factory=Paths)
    pose: PoseConfig = field(default_factory=PoseConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    coaching: CoachingConfig = field(default_factory=CoachingConfig)


# Singleton for convenience
config = AppConfig()
