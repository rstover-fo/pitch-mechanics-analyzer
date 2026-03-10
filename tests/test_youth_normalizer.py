"""Tests for the youth pitcher normalization framework."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.biomechanics.youth_normalizer import (
    YouthPitcherProfile,
    YouthNormalizer,
    DevStage,
    OBP_REF,
    classify_dev_stage,
    get_scale_factor,
    METRIC_TIER_MAP,
    MetricTier,
    COACHING_PRIORITIES,
    CDC_BOYS_50TH,
)


class TestDevStageClassification:
    """Test developmental stage classification."""

    def test_pre_pubescent(self):
        assert classify_dev_stage(10) == DevStage.PRE_PUBESCENT
        assert classify_dev_stage(11.5) == DevStage.PRE_PUBESCENT

    def test_early_adolescent(self):
        assert classify_dev_stage(12) == DevStage.EARLY_ADOLESCENT
        assert classify_dev_stage(13.9) == DevStage.EARLY_ADOLESCENT

    def test_mid_adolescent(self):
        assert classify_dev_stage(14) == DevStage.MID_ADOLESCENT
        assert classify_dev_stage(15.5) == DevStage.MID_ADOLESCENT

    def test_late_adolescent(self):
        assert classify_dev_stage(16) == DevStage.LATE_ADOLESCENT
        assert classify_dev_stage(17) == DevStage.LATE_ADOLESCENT

    def test_adult(self):
        assert classify_dev_stage(18) == DevStage.ADULT
        assert classify_dev_stage(22) == DevStage.ADULT


class TestYouthPitcherProfile:
    """Test the youth pitcher profile calculations."""

    def test_basic_properties(self):
        p = YouthPitcherProfile(age=12, height_cm=152.0, weight_kg=41.0)
        assert p.height_m == pytest.approx(1.52, abs=0.01)
        assert p.bmi == pytest.approx(41.0 / 1.52**2, abs=0.1)
        assert p.dev_stage == DevStage.EARLY_ADOLESCENT

    def test_size_relative_to_cdc(self):
        # A 12-year-old at exactly CDC 50th percentile height (149 cm)
        p = YouthPitcherProfile(age=12, height_cm=149.0, weight_kg=40.0)
        assert p.size_relative_to_cdc_50th == pytest.approx(1.0, abs=0.01)

    def test_tall_for_age_maturity_offset(self):
        # A 12-year-old who is 10% taller than average → ~1 year advanced
        cdc_h = CDC_BOYS_50TH[12][0]  # 149 cm
        tall_height = cdc_h * 1.10     # 163.9 cm
        p = YouthPitcherProfile(age=12, height_cm=tall_height, weight_kg=45.0)
        assert p.maturity_offset_estimate > 0.5
        assert p.effective_dev_age > 12.5

    def test_small_for_age_maturity_offset(self):
        # A 12-year-old who is 10% shorter → ~1 year behind
        cdc_h = CDC_BOYS_50TH[12][0]
        short_height = cdc_h * 0.90
        p = YouthPitcherProfile(age=12, height_cm=short_height, weight_kg=35.0)
        assert p.maturity_offset_estimate < -0.5
        assert p.effective_dev_age < 11.5

    def test_arm_length_estimate(self):
        p = YouthPitcherProfile(age=12, height_cm=149.0, weight_kg=40.0)
        arm = p.estimated_arm_length_m
        # Should be roughly 30% of height
        assert 0.4 < arm < 0.55

    def test_bw_height_product(self):
        p = YouthPitcherProfile(age=12, height_cm=149.0, weight_kg=40.0)
        assert p.bw_height_product == pytest.approx(40.0 * 1.49, abs=0.1)


class TestScaleFactors:
    """Test allometric scaling calculations."""

    def test_angle_metrics_not_scaled(self):
        p = YouthPitcherProfile(age=12, height_cm=149.0, weight_kg=40.0)
        sf = get_scale_factor(p, "elbow_flexion_fp")
        assert sf == 1.0

    def test_velocity_scales_with_arm_length(self):
        p = YouthPitcherProfile(age=12, height_cm=149.0, weight_kg=40.0)
        sf = get_scale_factor(p, "max_shoulder_internal_rotational_velo")
        # Youth arm shorter than OBP reference → scale factor < 1
        assert 0.5 < sf < 0.9

    def test_moment_scales_with_bw_height(self):
        p = YouthPitcherProfile(age=12, height_cm=149.0, weight_kg=40.0)
        sf = get_scale_factor(p, "elbow_varus_moment")
        expected = (40.0 * 1.49) / OBP_REF.bw_height_product
        assert sf == pytest.approx(expected, abs=0.01)
        assert sf < 0.5  # Youth should be well below adult

    def test_force_scales_with_bodyweight(self):
        p = YouthPitcherProfile(age=12, height_cm=149.0, weight_kg=40.0)
        sf = get_scale_factor(p, "lead_grf_mag_max")
        assert sf == pytest.approx(40.0 / OBP_REF.mass_kg, abs=0.01)

    def test_adult_scale_factor_near_one(self):
        # An adult-sized pitcher should have scale factors near 1.0
        p = YouthPitcherProfile(age=20, height_cm=188.0, weight_kg=90.0)
        sf_vel = get_scale_factor(p, "max_shoulder_internal_rotational_velo")
        sf_mom = get_scale_factor(p, "elbow_varus_moment")
        assert 0.9 < sf_vel < 1.1
        assert 0.9 < sf_mom < 1.1


class TestYouthNormalizer:
    """Test the full normalization pipeline."""

    @pytest.fixture
    def obp(self):
        data_path = Path(__file__).parent.parent / "data" / "obp"
        if not (data_path / "poi_metrics.csv").exists():
            pytest.skip("OBP data not available")
        from src.biomechanics.benchmarks import OBPBenchmarks
        return OBPBenchmarks(obp_data_path=data_path).load()

    @pytest.fixture
    def youth_12(self):
        return YouthPitcherProfile(age=12, height_cm=152.0, weight_kg=41.0)

    @pytest.fixture
    def youth_10(self):
        return YouthPitcherProfile(age=10, height_cm=138.0, weight_kg=32.0)

    @pytest.fixture
    def youth_15(self):
        return YouthPitcherProfile(age=15, height_cm=173.0, weight_kg=61.0)

    def test_adjusted_benchmarks_produced(self, obp, youth_12):
        norm = YouthNormalizer(obp, youth_12)
        adjusted = norm.get_adjusted_benchmarks()
        assert len(adjusted) > 10

    def test_angle_benchmarks_not_scaled(self, obp, youth_12):
        norm = YouthNormalizer(obp, youth_12)
        adjusted = {b.metric: b for b in norm.get_adjusted_benchmarks()}
        # Hip-shoulder separation is a Tier 1 angle — scale should be 1.0
        hss = adjusted.get("rotation_hip_shoulder_separation_fp")
        if hss:
            assert hss.scale_factor == 1.0
            # But IQR should be wider than adult
            assert hss.variability_multiplier > 1.0

    def test_moment_benchmarks_scaled_down(self, obp, youth_12):
        norm = YouthNormalizer(obp, youth_12)
        adjusted = {b.metric: b for b in norm.get_adjusted_benchmarks()}
        evm = adjusted.get("elbow_varus_moment")
        if evm:
            # Scaled-down median should be much lower than adult
            assert evm.youth_p50 < evm.obp_p50 * 0.5

    def test_pre_pubescent_coaching_relevance(self, obp, youth_10):
        norm = YouthNormalizer(obp, youth_10)
        adjusted = norm.get_adjusted_benchmarks()
        # Moment metrics should NOT be coaching-relevant for a 10-year-old
        for b in adjusted:
            if b.tier == MetricTier.TIER_2_MOMENT:
                assert not b.coaching_relevant

    def test_mid_adolescent_all_relevant(self, obp, youth_15):
        norm = YouthNormalizer(obp, youth_15)
        adjusted = norm.get_adjusted_benchmarks()
        # All metrics should be coaching-relevant by age 15
        for b in adjusted:
            assert b.coaching_relevant

    def test_compare_good_mechanics(self, obp, youth_12):
        norm = YouthNormalizer(obp, youth_12)
        # Feed in ASMI-ideal values for a 12-year-old
        metrics = {
            "elbow_flexion_fp": 90.0,                    # Perfect ASMI target
            "shoulder_abduction_fp": 88.0,               # Near target
            "rotation_hip_shoulder_separation_fp": 28.0,  # Close to target
        }
        results = norm.compare(metrics)
        assert len(results) == 3
        # Good mechanics should get positive flags
        for r in results:
            assert r.flag in ("excellent", "on_track")

    def test_compare_poor_mechanics(self, obp, youth_12):
        norm = YouthNormalizer(obp, youth_12)
        metrics = {
            "elbow_flexion_fp": 50.0,                     # Way too low
            "shoulder_external_rotation_fp": 110.0,       # Excessive ER at FP
        }
        results = norm.compare(metrics)
        # Should flag these as concerns
        has_concern = any(r.flag in ("needs_attention", "concern") for r in results)
        assert has_concern

    def test_coaching_priorities(self, obp, youth_12):
        norm = YouthNormalizer(obp, youth_12)
        priorities = norm.get_coaching_priorities()
        assert "focus" in priorities
        assert len(priorities["emphasis"]) > 0

    def test_report_context(self, obp, youth_12):
        norm = YouthNormalizer(obp, youth_12)
        ctx = norm.generate_youth_report_context()
        assert ctx["pitcher_age"] == 12
        assert ctx["dev_stage"] == "early_adolescent"
        assert "coaching_focus" in ctx
        assert "injury_watch" in ctx

    def test_comparison_sorted_by_priority(self, obp, youth_12):
        norm = YouthNormalizer(obp, youth_12)
        metrics = {
            "elbow_flexion_fp": 90.0,
            "rotation_hip_shoulder_separation_fp": 5.0,   # Very low — should be flagged
        }
        results = norm.compare(metrics)
        # Coaching-relevant items should come first
        assert all(r.coaching_relevant for r in results)
        # More concerning items should come before less concerning
        if len(results) >= 2:
            flag_order = {"concern": 0, "needs_attention": 1, "on_track": 2, "excellent": 3}
            flags = [flag_order.get(r.flag, 4) for r in results]
            assert flags == sorted(flags)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
