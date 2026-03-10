"""Tests for OBP benchmark loading and analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biomechanics.benchmarks import (
    OBPBenchmarks,
    ALL_COACHING_METRICS,
    METRIC_DISPLAY_NAMES,
    BenchmarkResult,
)
from src.biomechanics.features import angle_between_points, compute_trunk_tilt
from src.biomechanics.events import detect_leg_lift, DeliveryEvents


class TestOBPBenchmarks:
    """Tests for OBP data loading and benchmark computation."""

    @pytest.fixture
    def obp(self):
        """Load OBP benchmarks if data is available."""
        data_path = Path(__file__).parent.parent / "data" / "obp"
        if not (data_path / "poi_metrics.csv").exists():
            pytest.skip("OBP data not available — run setup first")
        return OBPBenchmarks(obp_data_path=data_path).load()

    def test_load_poi_data(self, obp):
        assert obp.poi_df is not None
        assert len(obp.poi_df) > 100

    def test_load_metadata(self, obp):
        assert obp.metadata_df is not None
        assert "playing_level" in obp.merged_df.columns

    def test_compute_benchmarks_all(self, obp):
        benchmarks = obp.compute_benchmarks()
        assert len(benchmarks) > 10
        for b in benchmarks:
            assert isinstance(b, BenchmarkResult)
            assert b.n_samples > 0
            assert b.percentiles[25] <= b.percentiles[50] <= b.percentiles[75]

    def test_compute_benchmarks_by_level(self, obp):
        college = obp.compute_benchmarks(playing_level="college")
        assert all(b.playing_level == "college" for b in college)

    def test_summary_table(self, obp):
        df = obp.summary_table()
        assert isinstance(df, pd.DataFrame)
        assert "p50" in df.columns
        assert len(df) > 0

    def test_compare_to_benchmarks(self, obp):
        # Use median values — should land near 50th percentile
        benchmarks = obp.compute_benchmarks()
        test_metrics = {b.metric: b.percentiles[50] for b in benchmarks[:5]}

        comparisons = obp.compare_to_benchmarks(test_metrics)
        assert len(comparisons) == 5
        for c in comparisons:
            assert 30 <= c["percentile_rank"] <= 70  # Near median


class TestGeometry:
    """Tests for angle computation helpers."""

    def test_straight_angle(self):
        a = np.array([0, 0])
        b = np.array([1, 0])
        c = np.array([2, 0])
        assert abs(angle_between_points(a, b, c) - 180.0) < 1.0

    def test_right_angle(self):
        a = np.array([0, 0])
        b = np.array([0, 1])
        c = np.array([1, 1])
        assert abs(angle_between_points(a, b, c) - 90.0) < 1.0

    def test_trunk_tilt_upright(self):
        hip = np.array([100, 300])    # Lower
        shoulder = np.array([100, 100])  # Higher (lower y = higher in screen coords)
        tilt = compute_trunk_tilt(hip, shoulder, vertical=np.array([0, -1]))
        assert tilt < 5  # Nearly upright


class TestEventDetection:
    """Tests for delivery event detection."""

    def test_detect_leg_lift(self):
        # Simulate a knee that goes up then down
        y = np.concatenate([
            np.linspace(300, 100, 15),   # Going up (lower y = higher)
            np.linspace(100, 300, 15),   # Coming down
        ])
        # Since we detect MAX, and lower y = higher... we need to invert
        # In our convention, higher values = higher position
        y_inverted = -y
        apex = detect_leg_lift(y_inverted)
        assert apex is not None
        assert 10 <= apex <= 20  # Should be near the middle

    def test_phase_durations(self):
        events = DeliveryEvents(
            leg_lift_apex=10,
            foot_plant=25,
            max_external_rotation=35,
            ball_release=40,
            fps=30.0,
        )
        durations = events.phase_durations()
        assert durations["windup_to_foot_plant"] == pytest.approx(0.5, abs=0.01)
        assert durations["arm_acceleration"] == pytest.approx(5 / 30, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
