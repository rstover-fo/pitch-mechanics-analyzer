"""Load and analyze Driveline OpenBiomechanics Project (OBP) benchmark data.

Provides percentile distributions of pitching biomechanics metrics
segmented by playing level, used as comparison benchmarks for user analysis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Metrics grouped by coaching relevance
TIMING_SEQUENCE_METRICS = [
    "max_pelvis_rotational_velo",
    "max_torso_rotational_velo",
    "max_shoulder_internal_rotational_velo",
    "max_elbow_extension_velo",
]

ARM_MECHANICS_METRICS = [
    "max_shoulder_external_rotation",         # Peak layback
    "shoulder_horizontal_abduction_fp",       # Arm position at foot plant
    "shoulder_abduction_fp",                  # Arm abduction at foot plant
    "shoulder_external_rotation_fp",          # Layback at foot plant
    "elbow_flexion_fp",                       # Elbow angle at foot plant
    "elbow_flexion_mer",                      # Elbow angle at max external rotation
    "max_elbow_flexion",                      # Peak elbow flexion
]

TRUNK_MECHANICS_METRICS = [
    "torso_anterior_tilt_fp",                 # Forward lean at foot plant
    "torso_lateral_tilt_fp",                  # Side bend at foot plant
    "torso_rotation_fp",                      # Trunk rotation at foot plant
    "torso_anterior_tilt_br",                 # Forward lean at ball release
    "torso_lateral_tilt_br",                  # Side bend at ball release
]

HIP_SHOULDER_SEPARATION_METRICS = [
    "rotation_hip_shoulder_separation_fp",    # Separation at foot plant
    "max_rotation_hip_shoulder_separation",   # Peak separation
]

LEAD_LEG_METRICS = [
    "lead_knee_extension_angular_velo_fp",    # Lead leg at foot plant
    "lead_knee_extension_angular_velo_br",    # Lead leg at ball release (block)
    "lead_knee_extension_angular_velo_max",   # Peak lead leg extension velo
]

KINETIC_METRICS = [
    "elbow_varus_moment",                     # UCL stress indicator
    "shoulder_internal_rotation_moment",      # Shoulder loading
]

GROUND_REACTION_METRICS = [
    "rear_grf_mag_max",                       # Peak rear leg force
    "lead_grf_mag_max",                       # Peak lead leg force
    "rear_grf_angle_at_max",                  # Rear leg force direction
    "lead_grf_angle_at_max",                  # Lead leg force direction
]

ENERGY_FLOW_METRICS = [
    "shoulder_transfer_fp_br",
    "shoulder_generation_fp_br",
    "elbow_transfer_fp_br",
    "lead_knee_transfer_fp_br",
    "lead_knee_generation_fp_br",
]

# All coaching-relevant metrics combined
ALL_COACHING_METRICS = (
    TIMING_SEQUENCE_METRICS
    + ARM_MECHANICS_METRICS
    + TRUNK_MECHANICS_METRICS
    + HIP_SHOULDER_SEPARATION_METRICS
    + LEAD_LEG_METRICS
    + KINETIC_METRICS
    + GROUND_REACTION_METRICS
    + ENERGY_FLOW_METRICS
)

# Human-readable names for display
METRIC_DISPLAY_NAMES = {
    "max_shoulder_external_rotation": "Peak Layback (Shoulder ER)",
    "max_rotation_hip_shoulder_separation": "Peak Hip-Shoulder Separation",
    "rotation_hip_shoulder_separation_fp": "Hip-Shoulder Separation @ Foot Plant",
    "max_torso_rotational_velo": "Peak Trunk Rotation Velocity",
    "max_shoulder_internal_rotational_velo": "Peak Arm Speed (Shoulder IR Velo)",
    "max_elbow_extension_velo": "Peak Elbow Extension Velocity",
    "max_pelvis_rotational_velo": "Peak Pelvis Rotation Velocity",
    "elbow_flexion_fp": "Elbow Flexion @ Foot Plant",
    "elbow_flexion_mer": "Elbow Flexion @ Max ER",
    "shoulder_horizontal_abduction_fp": "Shoulder Horiz. Abduction @ FP",
    "shoulder_abduction_fp": "Shoulder Abduction @ FP",
    "shoulder_external_rotation_fp": "Shoulder External Rotation @ FP",
    "torso_anterior_tilt_fp": "Forward Trunk Tilt @ FP",
    "torso_lateral_tilt_fp": "Lateral Trunk Tilt @ FP",
    "torso_anterior_tilt_br": "Forward Trunk Tilt @ Ball Release",
    "torso_lateral_tilt_br": "Lateral Trunk Tilt @ Ball Release",
    "lead_knee_extension_angular_velo_fp": "Lead Knee Extension Velo @ FP",
    "lead_knee_extension_angular_velo_br": "Lead Knee Extension Velo @ Release",
    "lead_knee_extension_angular_velo_max": "Peak Lead Knee Extension Velo",
    "elbow_varus_moment": "Elbow Varus Moment (UCL Stress)",
    "shoulder_internal_rotation_moment": "Shoulder IR Moment",
    "rear_grf_mag_max": "Peak Rear Leg GRF",
    "lead_grf_mag_max": "Peak Lead Leg GRF",
    "pitch_speed_mph": "Pitch Speed (mph)",
}

PERCENTILE_LEVELS = [10, 25, 50, 75, 90]


@dataclass
class BenchmarkResult:
    """Percentile distributions for a single metric."""
    metric: str
    display_name: str
    unit: str
    percentiles: dict[int, float]       # {10: val, 25: val, 50: val, 75: val, 90: val}
    mean: float
    std: float
    n_samples: int
    playing_level: str


class OBPBenchmarks:
    """Load and query Driveline OBP pitching biomechanics benchmarks."""

    def __init__(self, obp_data_path: Optional[Path] = None):
        if obp_data_path is None:
            from src.utils.config import config
            obp_data_path = config.paths.obp_data

        self.data_path = Path(obp_data_path)
        self.poi_df: Optional[pd.DataFrame] = None
        self.metadata_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None

    def load(self) -> "OBPBenchmarks":
        """Load POI metrics and metadata, merge them."""
        poi_path = self.data_path / "poi_metrics.csv"
        meta_path = self.data_path / "metadata.csv"

        if not poi_path.exists():
            raise FileNotFoundError(
                f"POI metrics not found at {poi_path}. "
                "Copy from openbiomechanics/baseball_pitching/data/poi/poi_metrics.csv"
            )

        self.poi_df = pd.read_csv(poi_path)
        print(f"Loaded {len(self.poi_df)} pitches with {len(self.poi_df.columns)} metrics")

        if meta_path.exists():
            self.metadata_df = pd.read_csv(meta_path)
            # Merge on session (metadata has session, POI has session)
            self.merged_df = self.poi_df.merge(
                self.metadata_df[["session", "session_pitch", "age_yrs", "playing_level",
                                  "session_height_m", "session_mass_kg"]],
                on=["session"],
                suffixes=("", "_meta"),
                how="left",
            )
            # Drop duplicate session_pitch column if created
            if "session_pitch_meta" in self.merged_df.columns:
                self.merged_df.drop(columns=["session_pitch_meta"], inplace=True)
            print(f"Merged with metadata: {self.merged_df['playing_level'].value_counts().to_dict()}")
        else:
            self.merged_df = self.poi_df.copy()
            print("No metadata found — using POI data only (no playing level segmentation)")

        return self

    def get_available_metrics(self) -> list[str]:
        """Return list of numeric metrics available in the dataset."""
        if self.poi_df is None:
            raise RuntimeError("Call .load() first")
        return [c for c in self.poi_df.columns if self.poi_df[c].dtype in ["float64", "int64"]]

    def compute_benchmarks(
        self,
        metrics: Optional[list[str]] = None,
        playing_level: Optional[str] = None,
    ) -> list[BenchmarkResult]:
        """Compute percentile distributions for specified metrics.

        Args:
            metrics: List of metric column names. Defaults to ALL_COACHING_METRICS.
            playing_level: Filter to specific level ('college', 'high_school', 'milb', 'independent').
                          None = all levels combined.

        Returns:
            List of BenchmarkResult objects with percentile distributions.
        """
        if self.merged_df is None:
            raise RuntimeError("Call .load() first")

        if metrics is None:
            metrics = [m for m in ALL_COACHING_METRICS if m in self.merged_df.columns]

        df = self.merged_df.copy()
        level_label = "all"

        if playing_level and "playing_level" in df.columns:
            df = df[df["playing_level"] == playing_level]
            level_label = playing_level

        results = []
        for metric in metrics:
            if metric not in df.columns:
                continue

            values = df[metric].dropna()
            if len(values) < 5:
                continue

            pcts = {p: float(np.percentile(values, p)) for p in PERCENTILE_LEVELS}

            results.append(BenchmarkResult(
                metric=metric,
                display_name=METRIC_DISPLAY_NAMES.get(metric, metric),
                unit="deg" if "rotation" in metric or "tilt" in metric or "flexion" in metric
                     or "abduction" in metric or "separation" in metric
                     else "deg/s" if "velo" in metric
                     else "Nm" if "moment" in metric
                     else "N" if "grf" in metric
                     else "W" if "transfer" in metric or "generation" in metric or "absorption" in metric
                     else "mph" if "speed" in metric
                     else "",
                percentiles=pcts,
                mean=float(values.mean()),
                std=float(values.std()),
                n_samples=len(values),
                playing_level=level_label,
            ))

        return results

    def compare_to_benchmarks(
        self,
        pitcher_metrics: dict[str, float],
        playing_level: Optional[str] = None,
    ) -> list[dict]:
        """Compare a pitcher's metrics to OBP benchmarks.

        Args:
            pitcher_metrics: Dict mapping metric name to measured value.
            playing_level: OBP subgroup to compare against.

        Returns:
            List of comparison dicts with metric, value, percentile, and coaching flag.
        """
        benchmarks = self.compute_benchmarks(
            metrics=list(pitcher_metrics.keys()),
            playing_level=playing_level,
        )
        bench_map = {b.metric: b for b in benchmarks}

        comparisons = []
        for metric, value in pitcher_metrics.items():
            if metric not in bench_map:
                continue

            bench = bench_map[metric]

            # Compute percentile rank within the OBP distribution
            if self.merged_df is not None:
                df = self.merged_df
                if playing_level and "playing_level" in df.columns:
                    df = df[df["playing_level"] == playing_level]
                col = df[metric].dropna()
                percentile_rank = float((col < value).sum() / len(col) * 100)
            else:
                percentile_rank = None

            # Flag metrics outside normal range for coaching attention
            flag = None
            if percentile_rank is not None:
                if percentile_rank < 15:
                    flag = "well_below_average"
                elif percentile_rank < 30:
                    flag = "below_average"
                elif percentile_rank > 85:
                    flag = "well_above_average"
                elif percentile_rank > 70:
                    flag = "above_average"

            comparisons.append({
                "metric": metric,
                "display_name": bench.display_name,
                "value": value,
                "unit": bench.unit,
                "percentile_rank": percentile_rank,
                "benchmark_median": bench.percentiles[50],
                "benchmark_p25": bench.percentiles[25],
                "benchmark_p75": bench.percentiles[75],
                "flag": flag,
                "playing_level": bench.playing_level,
                "n_samples": bench.n_samples,
            })

        return sorted(comparisons, key=lambda x: abs(50 - (x["percentile_rank"] or 50)), reverse=True)

    def summary_table(
        self,
        playing_level: Optional[str] = None,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Generate a summary DataFrame of benchmark distributions.

        Returns a DataFrame with columns: metric, display_name, n, mean, std, p10, p25, p50, p75, p90.
        """
        benchmarks = self.compute_benchmarks(metrics=metrics, playing_level=playing_level)

        rows = []
        for b in benchmarks:
            rows.append({
                "metric": b.metric,
                "display_name": b.display_name,
                "unit": b.unit,
                "n": b.n_samples,
                "mean": round(b.mean, 1),
                "std": round(b.std, 1),
                "p10": round(b.percentiles[10], 1),
                "p25": round(b.percentiles[25], 1),
                "p50": round(b.percentiles[50], 1),
                "p75": round(b.percentiles[75], 1),
                "p90": round(b.percentiles[90], 1),
                "playing_level": b.playing_level,
            })

        return pd.DataFrame(rows)
