"""Youth pitcher normalization framework.

Adapts Driveline OBP benchmarks (trained on college/pro athletes) for use with
developing youth pitchers aged 10-18. Implements a three-tier normalization
approach grounded in ASMI's longitudinal research (Fleisig et al. 2018).

Key research findings that inform this module:
  - Fleisig 7-year longitudinal study: kinematics change most between ages 9-13,
    kinetics change most between ages 13-15 (when normalized by BW × height)
  - Movement PATTERNS (angles, positions, sequence) are similar across ages;
    FORCES and SPEEDS scale with physical maturity
  - Body composition (BMI) strongly predicts joint kinetics in youth
    (shoulder IR torque R²=0.93, elbow varus torque R²=0.57 in 9-10 yr olds)
  - Variability of kinematic parameters decreases with development level
  - Youth pitchers use similar motions to older pitchers but generate lower
    kinetics and angular velocities

Three normalization tiers:
  Tier 1: Body-size-invariant metrics (angles) — compare directly, widen expected range
  Tier 2: Allometric scaling (velocities, forces, moments) — normalize by body dimensions
  Tier 3: Developmental stage interpolation — growth-adjusted expected ranges

References:
  Fleisig GS et al. (2018) Changes in Youth Baseball Pitching Biomechanics:
    A 7-Year Longitudinal Study. Am J Sports Med, 46(1):44-51.
  Fleisig GS et al. (1999) Kinematic and kinetic comparison of baseball
    pitching among various levels of development. J Biomech, 32:1371-1375.
  Darke JD et al. (2018) Effects of game pitch count and body mass index
    on pitching biomechanics in 9-10 year old baseball athletes.
    Orthop J Sports Med, 6(4).
  ASMI Clinician's Guide to Baseball Pitching Biomechanics (2023).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# CDC Growth Reference Data (Boys, 50th percentile)
# Source: CDC 2000 Growth Charts, stature-for-age and weight-for-age
# ---------------------------------------------------------------------------

# fmt: off
CDC_BOYS_50TH = {
    # age: (height_cm, weight_kg)
    8:  (128.0, 25.8),
    9:  (133.5, 28.6),
    10: (138.5, 31.9),
    11: (143.5, 35.6),
    12: (149.0, 40.0),
    13: (156.0, 45.3),
    14: (163.5, 51.0),
    15: (170.0, 56.5),
    16: (173.5, 61.5),
    17: (175.5, 64.7),
    18: (176.0, 67.2),
    19: (176.5, 68.9),
    20: (177.0, 70.3),
}

# Approximate arm length (shoulder-to-wrist) as fraction of height
# Based on anthropometric proportions; refines with age during puberty
ARM_LENGTH_FRACTION_OF_HEIGHT = {
    8:  0.295, 9:  0.297, 10: 0.300, 11: 0.303,
    12: 0.307, 13: 0.312, 14: 0.318, 15: 0.323,
    16: 0.327, 17: 0.330, 18: 0.332, 19: 0.333, 20: 0.333,
}
# fmt: on


# ---------------------------------------------------------------------------
# OBP Reference Athlete Profile
# Median values from the OBP dataset for scaling denominators
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OBPReferenceAthlete:
    """Represents the 'typical' OBP pitcher used as the scaling baseline."""
    age_yrs: float = 20.5
    height_m: float = 1.88          # ~6'2" — OBP median
    mass_kg: float = 90.0           # ~198 lbs — OBP median
    arm_length_m: float = 0.625     # ~height × 0.333

    @property
    def bw_height_product(self) -> float:
        """Bodyweight × height normalizer for kinetics (Nm → Nm/(kg·m))."""
        return self.mass_kg * self.height_m

    @property
    def bmi(self) -> float:
        return self.mass_kg / (self.height_m ** 2)


OBP_REF = OBPReferenceAthlete()


# ---------------------------------------------------------------------------
# Developmental Stage Classification
# ---------------------------------------------------------------------------

class DevStage(Enum):
    """Developmental stage based on ASMI longitudinal research windows."""
    PRE_PUBESCENT = "pre_pubescent"       # ~8-11: focus on movement quality
    EARLY_ADOLESCENT = "early_adolescent"  # ~12-13: kinematic refinement window
    MID_ADOLESCENT = "mid_adolescent"      # ~14-15: kinetics start increasing
    LATE_ADOLESCENT = "late_adolescent"    # ~16-17: approaching adult patterns
    ADULT = "adult"                        # 18+: OBP norms apply directly


def classify_dev_stage(age: float) -> DevStage:
    """Classify developmental stage from age.

    These windows are approximate and based on ASMI's longitudinal findings.
    A more precise classification would also consider Tanner stage or
    skeletal maturity, but age is a reasonable proxy for a personal app.
    """
    if age < 12:
        return DevStage.PRE_PUBESCENT
    elif age < 14:
        return DevStage.EARLY_ADOLESCENT
    elif age < 16:
        return DevStage.MID_ADOLESCENT
    elif age < 18:
        return DevStage.LATE_ADOLESCENT
    else:
        return DevStage.ADULT


# ---------------------------------------------------------------------------
# Youth Pitcher Profile
# ---------------------------------------------------------------------------

@dataclass
class YouthPitcherProfile:
    """Physical profile of a youth pitcher for normalization."""
    age: float                       # Years (e.g. 12.5)
    height_cm: float                 # Measured height in cm
    weight_kg: float                 # Measured weight in kg
    throws: str = "R"                # "R" or "L"

    @property
    def height_m(self) -> float:
        return self.height_cm / 100.0

    @property
    def bmi(self) -> float:
        return self.weight_kg / (self.height_m ** 2)

    @property
    def bw_height_product(self) -> float:
        return self.weight_kg * self.height_m

    @property
    def dev_stage(self) -> DevStage:
        return classify_dev_stage(self.age)

    @property
    def estimated_arm_length_m(self) -> float:
        """Estimate arm length from height using age-adjusted proportions."""
        age_floor = max(8, min(20, int(self.age)))
        frac = ARM_LENGTH_FRACTION_OF_HEIGHT.get(age_floor, 0.333)
        return self.height_m * frac

    @property
    def size_relative_to_cdc_50th(self) -> float:
        """How this pitcher's height compares to CDC 50th percentile for age.

        Returns ratio: >1.0 means taller than average for age.
        Used in Tier 3 developmental interpolation.
        """
        age_floor = max(8, min(20, int(self.age)))
        cdc_height_cm, _ = CDC_BOYS_50TH.get(age_floor, (170.0, 60.0))
        return self.height_cm / cdc_height_cm

    @property
    def maturity_offset_estimate(self) -> float:
        """Rough estimate of physical maturity relative to age peers.

        Based on height-for-age ratio. A tall-for-age 12-year-old is
        biomechanically closer to a typical 13-14 year old.

        Returns an "effective age" adjustment in years.
        """
        size_ratio = self.size_relative_to_cdc_50th
        # Each 5% above/below average height ≈ 0.5 year maturity offset
        return (size_ratio - 1.0) * 10.0  # ±10% height → ±1 year

    @property
    def effective_dev_age(self) -> float:
        """Age adjusted for physical maturity (size-for-age)."""
        return self.age + self.maturity_offset_estimate


# ---------------------------------------------------------------------------
# Metric Classification
# ---------------------------------------------------------------------------

class MetricTier(Enum):
    """Which normalization tier a metric belongs to."""
    TIER_1_ANGLE = "angle"              # Body-size invariant, compare directly
    TIER_2_VELOCITY = "velocity"        # Scale by limb length
    TIER_2_MOMENT = "moment"            # Scale by BW × height
    TIER_2_FORCE = "force"              # Scale by bodyweight
    TIER_2_POWER = "power"              # Scale by BW × height
    TIER_1_RATIO = "ratio"             # Already normalized (e.g. stride % height)


# Map every OBP metric to its normalization tier
METRIC_TIER_MAP: dict[str, MetricTier] = {
    # --- Tier 1: Angles (degrees) — body-size invariant ---
    "elbow_flexion_fp": MetricTier.TIER_1_ANGLE,
    "elbow_pronation_fp": MetricTier.TIER_1_ANGLE,
    "rotation_hip_shoulder_separation_fp": MetricTier.TIER_1_ANGLE,
    "shoulder_horizontal_abduction_fp": MetricTier.TIER_1_ANGLE,
    "shoulder_abduction_fp": MetricTier.TIER_1_ANGLE,
    "shoulder_external_rotation_fp": MetricTier.TIER_1_ANGLE,
    "torso_anterior_tilt_fp": MetricTier.TIER_1_ANGLE,
    "torso_lateral_tilt_fp": MetricTier.TIER_1_ANGLE,
    "torso_rotation_fp": MetricTier.TIER_1_ANGLE,
    "pelvis_anterior_tilt_fp": MetricTier.TIER_1_ANGLE,
    "pelvis_lateral_tilt_fp": MetricTier.TIER_1_ANGLE,
    "pelvis_rotation_fp": MetricTier.TIER_1_ANGLE,
    "max_shoulder_external_rotation": MetricTier.TIER_1_ANGLE,
    "max_rotation_hip_shoulder_separation": MetricTier.TIER_1_ANGLE,
    "max_elbow_flexion": MetricTier.TIER_1_ANGLE,
    "elbow_flexion_mer": MetricTier.TIER_1_ANGLE,
    "torso_anterior_tilt_mer": MetricTier.TIER_1_ANGLE,
    "torso_lateral_tilt_mer": MetricTier.TIER_1_ANGLE,
    "torso_rotation_mer": MetricTier.TIER_1_ANGLE,
    "torso_anterior_tilt_br": MetricTier.TIER_1_ANGLE,
    "torso_lateral_tilt_br": MetricTier.TIER_1_ANGLE,
    "torso_rotation_min": MetricTier.TIER_1_ANGLE,
    "glove_shoulder_horizontal_abduction_fp": MetricTier.TIER_1_ANGLE,
    "glove_shoulder_abduction_fp": MetricTier.TIER_1_ANGLE,
    "glove_shoulder_external_rotation_fp": MetricTier.TIER_1_ANGLE,
    "glove_shoulder_abduction_mer": MetricTier.TIER_1_ANGLE,

    # --- Tier 2: Angular velocities (deg/s) — scale with limb length ---
    "max_shoulder_internal_rotational_velo": MetricTier.TIER_2_VELOCITY,
    "max_elbow_extension_velo": MetricTier.TIER_2_VELOCITY,
    "max_torso_rotational_velo": MetricTier.TIER_2_VELOCITY,
    "max_pelvis_rotational_velo": MetricTier.TIER_2_VELOCITY,
    "lead_knee_extension_angular_velo_fp": MetricTier.TIER_2_VELOCITY,
    "lead_knee_extension_angular_velo_br": MetricTier.TIER_2_VELOCITY,
    "lead_knee_extension_angular_velo_max": MetricTier.TIER_2_VELOCITY,

    # --- Tier 2: Joint moments (Nm) — scale by BW × height ---
    "elbow_varus_moment": MetricTier.TIER_2_MOMENT,
    "shoulder_internal_rotation_moment": MetricTier.TIER_2_MOMENT,

    # --- Tier 2: Ground reaction forces (N) — scale by bodyweight ---
    "rear_grf_x_max": MetricTier.TIER_2_FORCE,
    "rear_grf_y_max": MetricTier.TIER_2_FORCE,
    "rear_grf_z_max": MetricTier.TIER_2_FORCE,
    "rear_grf_mag_max": MetricTier.TIER_2_FORCE,
    "lead_grf_x_max": MetricTier.TIER_2_FORCE,
    "lead_grf_y_max": MetricTier.TIER_2_FORCE,
    "lead_grf_z_max": MetricTier.TIER_2_FORCE,
    "lead_grf_mag_max": MetricTier.TIER_2_FORCE,
    "rear_grf_angle_at_max": MetricTier.TIER_1_ANGLE,  # Angle, not force
    "lead_grf_angle_at_max": MetricTier.TIER_1_ANGLE,
    "peak_rfd_rear": MetricTier.TIER_2_FORCE,
    "peak_rfd_lead": MetricTier.TIER_2_FORCE,

    # --- Tier 2: Energy flow (W) — scale by BW × height ---
    "shoulder_transfer_fp_br": MetricTier.TIER_2_POWER,
    "shoulder_generation_fp_br": MetricTier.TIER_2_POWER,
    "shoulder_absorption_fp_br": MetricTier.TIER_2_POWER,
    "elbow_transfer_fp_br": MetricTier.TIER_2_POWER,
    "elbow_generation_fp_br": MetricTier.TIER_2_POWER,
    "elbow_absorption_fp_br": MetricTier.TIER_2_POWER,
    "lead_hip_transfer_fp_br": MetricTier.TIER_2_POWER,
    "lead_hip_generation_fp_br": MetricTier.TIER_2_POWER,
    "lead_hip_absorption_fp_br": MetricTier.TIER_2_POWER,
    "lead_knee_transfer_fp_br": MetricTier.TIER_2_POWER,
    "lead_knee_generation_fp_br": MetricTier.TIER_2_POWER,
    "lead_knee_absorption_fp_br": MetricTier.TIER_2_POWER,
    "rear_hip_transfer_pkh_fp": MetricTier.TIER_2_POWER,
    "rear_hip_generation_pkh_fp": MetricTier.TIER_2_POWER,
    "rear_hip_absorption_pkh_fp": MetricTier.TIER_2_POWER,
    "rear_knee_transfer_pkh_fp": MetricTier.TIER_2_POWER,
    "rear_knee_generation_pkh_fp": MetricTier.TIER_2_POWER,
    "rear_knee_absorption_pkh_fp": MetricTier.TIER_2_POWER,
    "pelvis_lumbar_transfer_fp_br": MetricTier.TIER_2_POWER,
    "thorax_distal_transfer_fp_br": MetricTier.TIER_2_POWER,

    # --- Special: linear velocity (m/s) — scale with height ---
    "max_cog_velo_x": MetricTier.TIER_2_VELOCITY,

    # --- Pitch speed (mph) — partially scales but also technique-driven ---
    "pitch_speed_mph": MetricTier.TIER_2_VELOCITY,
}


# ---------------------------------------------------------------------------
# Tier 1: Variability Widening
# ---------------------------------------------------------------------------

# Youth pitchers have wider variance in kinematic metrics.
# Fleisig et al. showed variability decreases with development level.
# These multipliers widen the "normal range" (IQR) for each dev stage.
VARIABILITY_MULTIPLIER: dict[DevStage, float] = {
    DevStage.PRE_PUBESCENT: 2.0,      # Double the IQR — very wide range is normal
    DevStage.EARLY_ADOLESCENT: 1.6,   # Still quite variable
    DevStage.MID_ADOLESCENT: 1.3,     # Tightening up
    DevStage.LATE_ADOLESCENT: 1.1,    # Near adult consistency
    DevStage.ADULT: 1.0,              # OBP norms as-is
}


# ---------------------------------------------------------------------------
# Tier 2: Allometric Scaling Functions
# ---------------------------------------------------------------------------

def compute_velocity_scale_factor(
    youth: YouthPitcherProfile,
    metric: str,
) -> float:
    """Compute scaling factor for angular velocity metrics.

    Angular velocity for a given 'quality' of motion scales inversely with
    limb length (shorter arm → less moment of inertia → could spin faster
    per unit torque, but youth produce less torque).

    The net effect: youth velocities are lower primarily due to strength,
    not limb length. We scale by (youth_arm / ref_arm) to account for
    the geometric component, leaving the strength-driven difference
    as the meaningful comparison.

    For trunk/pelvis rotation, scale by height ratio instead of arm length.
    """
    if "torso" in metric or "pelvis" in metric or "cog" in metric:
        return youth.height_m / OBP_REF.height_m
    elif "knee" in metric:
        # Leg length scales similarly to height
        return youth.height_m / OBP_REF.height_m
    elif "pitch_speed" in metric:
        # Pitch speed scales with arm length AND strength
        # Use arm length ratio as the geometric component
        return youth.estimated_arm_length_m / OBP_REF.arm_length_m
    else:
        # Shoulder, elbow velocities — scale by arm length
        return youth.estimated_arm_length_m / OBP_REF.arm_length_m


def compute_moment_scale_factor(youth: YouthPitcherProfile) -> float:
    """Compute scaling factor for joint moment metrics.

    Following ASMI convention: normalize by bodyweight × height.
    A 12-year-old producing 40 Nm at 40 kg × 1.49 m = 0.67 Nm/(kg·m)
    An OBP pitcher producing 110 Nm at 90 kg × 1.88 m = 0.65 Nm/(kg·m)
    → similar NORMALIZED loading despite very different absolute values.
    """
    return youth.bw_height_product / OBP_REF.bw_height_product


def compute_force_scale_factor(youth: YouthPitcherProfile) -> float:
    """Compute scaling factor for ground reaction force metrics.

    Normalize by bodyweight (N = kg × 9.81).
    """
    return youth.weight_kg / OBP_REF.mass_kg


def compute_power_scale_factor(youth: YouthPitcherProfile) -> float:
    """Compute scaling factor for energy flow / power metrics.

    Normalize by BW × height, same as moments.
    """
    return youth.bw_height_product / OBP_REF.bw_height_product


def get_scale_factor(
    youth: YouthPitcherProfile,
    metric: str,
) -> float:
    """Get the appropriate allometric scale factor for a metric.

    Returns a multiplier to apply to OBP benchmark values to create
    youth-appropriate expected ranges.

    For Tier 1 metrics, returns 1.0 (no scaling needed).
    """
    tier = METRIC_TIER_MAP.get(metric, MetricTier.TIER_1_ANGLE)

    if tier == MetricTier.TIER_1_ANGLE or tier == MetricTier.TIER_1_RATIO:
        return 1.0
    elif tier == MetricTier.TIER_2_VELOCITY:
        return compute_velocity_scale_factor(youth, metric)
    elif tier == MetricTier.TIER_2_MOMENT:
        return compute_moment_scale_factor(youth)
    elif tier == MetricTier.TIER_2_FORCE:
        return compute_force_scale_factor(youth)
    elif tier == MetricTier.TIER_2_POWER:
        return compute_power_scale_factor(youth)
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Tier 3: Developmental Stage Expected Ranges
# ---------------------------------------------------------------------------

# ASMI Clinician's Guide "proper mechanics" targets (all ages):
# These are POSITIONS (angles) that are considered biomechanically sound
# regardless of age. They serve as the base targets.
ASMI_IDEAL_TARGETS: dict[str, dict] = {
    "elbow_flexion_fp": {
        "target": 90.0,
        "description": "Elbow should be flexed ~90° at foot contact",
        "tolerance": 15.0,  # degrees of acceptable deviation
    },
    "shoulder_abduction_fp": {
        "target": 90.0,
        "description": "Shoulder abduction ~90° at foot contact",
        "tolerance": 15.0,
    },
    "shoulder_horizontal_abduction_fp": {
        "target": 20.0,
        "description": "Shoulder should be ~20° horizontally abducted at FC",
        "tolerance": 20.0,
    },
    "shoulder_external_rotation_fp": {
        "target": 45.0,
        "description": "Shoulder ER ~45° at foot contact",
        "tolerance": 20.0,
    },
    "rotation_hip_shoulder_separation_fp": {
        "target": 30.0,
        "description": "Hip-shoulder separation ~30° at foot plant",
        "tolerance": 15.0,
    },
    "torso_anterior_tilt_br": {
        "target": 35.0,
        "description": "Trunk forward tilt ~35° at ball release",
        "tolerance": 15.0,
    },
}


# Developmental coaching priorities — what to focus on at each stage
# Based on Fleisig's finding that kinematics improve ages 9-13,
# kinetics increase after age 13
COACHING_PRIORITIES: dict[DevStage, dict] = {
    DevStage.PRE_PUBESCENT: {
        "focus": "Movement quality and arm path",
        "emphasis": [
            "Proper arm positions at foot contact (elbow ~90°, shoulder ~90° abduction)",
            "Hip-shoulder separation timing (hips open before shoulders)",
            "Consistent arm slot and release point",
            "Full follow-through without deceleration braking",
            "Having fun and building love of the game",
        ],
        "de_emphasize": [
            "Velocity/speed numbers — not meaningful at this stage",
            "Joint forces and moments — dominated by body size, not technique",
            "Comparison to older pitcher benchmarks",
        ],
        "injury_watch": [
            "Excessive shoulder external rotation at foot contact (arm too far back too early)",
            "Elbow below shoulder at foot contact (low elbow / inverted W)",
            "Excessive contralateral trunk tilt (leaning away from glove side)",
        ],
    },
    DevStage.EARLY_ADOLESCENT: {
        "focus": "Kinematic refinement and consistency",
        "emphasis": [
            "Stride length approaching 75-85% of height",
            "Consistent hip-shoulder separation timing",
            "Lead leg block development (stiffening at release)",
            "Trunk rotation sequencing (hips → trunk → arm)",
            "Reducing pitch-to-pitch variability in mechanics",
        ],
        "de_emphasize": [
            "Absolute velocity numbers (still mostly physical maturity)",
            "Force production metrics",
        ],
        "injury_watch": [
            "Growth-plate vulnerability increasing — monitor volume carefully",
            "Normalized elbow varus torque starts increasing significantly",
            "Any persistent elbow or shoulder pain is a red flag",
        ],
    },
    DevStage.MID_ADOLESCENT: {
        "focus": "Strength integration and kinetic chain efficiency",
        "emphasis": [
            "Kinetic chain sequencing (proximal-to-distal firing order)",
            "Ground reaction force production (drive leg push)",
            "Lead leg block timing and stiffness at release",
            "Trunk rotation velocity development",
            "Pitch speed should be improving with physical growth",
        ],
        "de_emphasize": [
            "Comparing to college/pro norms on absolute metrics",
        ],
        "injury_watch": [
            "Peak window for normalized kinetic increases (elbow varus stress)",
            "Overuse monitoring is critical during growth spurts",
            "Shoulder and hip ROM should be monitored",
        ],
    },
    DevStage.LATE_ADOLESCENT: {
        "focus": "Approaching adult-level mechanical efficiency",
        "emphasis": [
            "OBP benchmarks becoming increasingly applicable",
            "Fine-tuning release point consistency",
            "Velocity development through mechanical efficiency",
            "Pitch-to-pitch repeatability",
        ],
        "de_emphasize": [],
        "injury_watch": [
            "Absolute joint loads approaching adult levels",
            "UCL stress monitoring",
        ],
    },
    DevStage.ADULT: {
        "focus": "Performance optimization",
        "emphasis": ["Full OBP benchmark comparisons apply"],
        "de_emphasize": [],
        "injury_watch": ["Standard injury prevention monitoring"],
    },
}


# ---------------------------------------------------------------------------
# Main Normalization Engine
# ---------------------------------------------------------------------------

@dataclass
class YouthAdjustedBenchmark:
    """A single metric benchmark adjusted for a youth pitcher's profile."""
    metric: str
    display_name: str
    unit: str
    tier: MetricTier

    # Original OBP benchmark values (adult)
    obp_p25: float
    obp_p50: float
    obp_p75: float

    # Youth-adjusted values
    youth_p25: float
    youth_p50: float
    youth_p75: float
    scale_factor: float
    variability_multiplier: float

    # Whether this metric is recommended for coaching at this dev stage
    coaching_relevant: bool
    coaching_note: Optional[str] = None


@dataclass
class YouthNormalizedComparison:
    """Comparison of a youth pitcher's metric against adjusted benchmarks."""
    metric: str
    display_name: str
    unit: str
    tier: MetricTier

    # The pitcher's measured value
    measured_value: float

    # Youth-adjusted benchmark range
    youth_p25: float
    youth_p50: float
    youth_p75: float

    # Where this pitcher falls in the adjusted range
    youth_percentile_rank: Optional[float]

    # ASMI ideal target (if applicable)
    asmi_target: Optional[float]
    asmi_tolerance: Optional[float]
    deviation_from_asmi: Optional[float]

    # Coaching flags
    flag: Optional[str]              # 'excellent', 'on_track', 'needs_attention', 'concern'
    coaching_relevant: bool
    coaching_note: Optional[str]

    @property
    def flag_emoji(self) -> str:
        return {
            "excellent": "🌟",
            "on_track": "✅",
            "needs_attention": "📋",
            "concern": "⚠️",
            None: "➖",
        }.get(self.flag, "➖")


class YouthNormalizer:
    """Normalize OBP benchmarks for youth pitcher comparison.

    Usage:
        from src.biomechanics.benchmarks import OBPBenchmarks

        obp = OBPBenchmarks().load()
        profile = YouthPitcherProfile(age=12, height_cm=152, weight_kg=41)
        normalizer = YouthNormalizer(obp, profile)

        # Get adjusted benchmarks
        adjusted = normalizer.get_adjusted_benchmarks()

        # Compare a pitcher's metrics
        pitcher_metrics = {"elbow_flexion_fp": 95.0, "rotation_hip_shoulder_separation_fp": 22.0}
        comparisons = normalizer.compare(pitcher_metrics)

        # Get coaching priorities
        priorities = normalizer.get_coaching_priorities()
    """

    def __init__(
        self,
        obp_benchmarks,   # OBPBenchmarks instance (already loaded)
        pitcher: YouthPitcherProfile,
    ):
        self.obp = obp_benchmarks
        self.pitcher = pitcher
        self.dev_stage = pitcher.dev_stage
        self.var_mult = VARIABILITY_MULTIPLIER[self.dev_stage]

        # Cache OBP benchmark results
        self._obp_benchmarks = None

    def _get_obp_benchmarks(self):
        """Lazily compute and cache OBP benchmarks."""
        if self._obp_benchmarks is None:
            from src.biomechanics.benchmarks import ALL_COACHING_METRICS
            self._obp_benchmarks = {
                b.metric: b
                for b in self.obp.compute_benchmarks(metrics=ALL_COACHING_METRICS)
            }
        return self._obp_benchmarks

    def _is_coaching_relevant(self, metric: str) -> bool:
        """Determine if a metric should be emphasized for coaching at this dev stage."""
        tier = METRIC_TIER_MAP.get(metric, MetricTier.TIER_1_ANGLE)

        if self.dev_stage == DevStage.PRE_PUBESCENT:
            # Only focus on body-position angles for young pitchers
            return tier in (MetricTier.TIER_1_ANGLE, MetricTier.TIER_1_RATIO)

        elif self.dev_stage == DevStage.EARLY_ADOLESCENT:
            # Add velocities but still skip raw kinetics
            return tier in (MetricTier.TIER_1_ANGLE, MetricTier.TIER_1_RATIO,
                           MetricTier.TIER_2_VELOCITY)

        else:
            # Mid-adolescent and up: all metrics are relevant
            return True

    def _adjust_benchmark(
        self,
        metric: str,
        obp_p25: float,
        obp_p50: float,
        obp_p75: float,
    ) -> tuple[float, float, float, float]:
        """Apply Tier 2 scaling and Tier 1 variability widening.

        Returns (adjusted_p25, adjusted_p50, adjusted_p75, scale_factor).
        """
        scale = get_scale_factor(self.pitcher, metric)

        # Scale the central tendency
        adj_p50 = obp_p50 * scale
        adj_p25 = obp_p25 * scale
        adj_p75 = obp_p75 * scale

        # Widen the IQR by the developmental variability multiplier
        iqr = adj_p75 - adj_p25
        iqr_expansion = (iqr * self.var_mult - iqr) / 2
        adj_p25 -= iqr_expansion
        adj_p75 += iqr_expansion

        return adj_p25, adj_p50, adj_p75, scale

    def get_adjusted_benchmarks(
        self,
        metrics: Optional[list[str]] = None,
    ) -> list[YouthAdjustedBenchmark]:
        """Compute youth-adjusted benchmarks for all (or specified) metrics.

        Args:
            metrics: List of metric names. Defaults to all coaching metrics.

        Returns:
            List of YouthAdjustedBenchmark objects.
        """
        from src.biomechanics.benchmarks import METRIC_DISPLAY_NAMES
        obp_map = self._get_obp_benchmarks()

        if metrics is None:
            metrics = list(obp_map.keys())

        results = []
        for metric in metrics:
            if metric not in obp_map:
                continue

            b = obp_map[metric]
            tier = METRIC_TIER_MAP.get(metric, MetricTier.TIER_1_ANGLE)

            y25, y50, y75, scale = self._adjust_benchmark(
                metric, b.percentiles[25], b.percentiles[50], b.percentiles[75],
            )

            coaching_relevant = self._is_coaching_relevant(metric)
            note = None
            if not coaching_relevant:
                if tier in (MetricTier.TIER_2_MOMENT, MetricTier.TIER_2_FORCE):
                    note = "Kinetic metric — dominated by body size at this age, not technique"
                elif tier == MetricTier.TIER_2_VELOCITY:
                    note = "Velocity metric — will increase naturally with physical development"

            results.append(YouthAdjustedBenchmark(
                metric=metric,
                display_name=METRIC_DISPLAY_NAMES.get(metric, metric),
                unit=b.unit,
                tier=tier,
                obp_p25=b.percentiles[25],
                obp_p50=b.percentiles[50],
                obp_p75=b.percentiles[75],
                youth_p25=round(y25, 1),
                youth_p50=round(y50, 1),
                youth_p75=round(y75, 1),
                scale_factor=round(scale, 3),
                variability_multiplier=self.var_mult,
                coaching_relevant=coaching_relevant,
                coaching_note=note,
            ))

        return results

    def compare(
        self,
        pitcher_metrics: dict[str, float],
    ) -> list[YouthNormalizedComparison]:
        """Compare a youth pitcher's metrics against adjusted benchmarks.

        This is the main entry point for generating coaching-ready comparisons.

        Args:
            pitcher_metrics: Dict mapping metric name to measured value.

        Returns:
            List of YouthNormalizedComparison sorted by coaching priority.
        """
        from src.biomechanics.benchmarks import METRIC_DISPLAY_NAMES
        adjusted = {b.metric: b for b in self.get_adjusted_benchmarks(list(pitcher_metrics.keys()))}

        comparisons = []
        for metric, value in pitcher_metrics.items():
            if metric not in adjusted:
                continue

            bench = adjusted[metric]

            # Compute where this value falls in the adjusted distribution
            # Simple linear interpolation within the IQR
            if bench.youth_p75 != bench.youth_p25:
                normalized_pos = (value - bench.youth_p25) / (bench.youth_p75 - bench.youth_p25)
                # Map to percentile: p25 → 25th, p75 → 75th
                youth_pct = 25 + normalized_pos * 50
                youth_pct = max(1, min(99, youth_pct))
            else:
                youth_pct = 50.0

            # Check against ASMI ideal targets
            asmi = ASMI_IDEAL_TARGETS.get(metric)
            asmi_target = asmi["target"] if asmi else None
            asmi_tol = asmi["tolerance"] if asmi else None
            deviation = abs(value - asmi_target) if asmi_target is not None else None

            # Generate coaching flag
            flag = self._classify_flag(metric, value, bench, asmi_target, asmi_tol)

            coaching_note = bench.coaching_note
            if asmi and deviation is not None and deviation > (asmi_tol or 15):
                coaching_note = asmi["description"]

            comparisons.append(YouthNormalizedComparison(
                metric=metric,
                display_name=bench.display_name,
                unit=bench.unit,
                tier=bench.tier,
                measured_value=value,
                youth_p25=bench.youth_p25,
                youth_p50=bench.youth_p50,
                youth_p75=bench.youth_p75,
                youth_percentile_rank=round(youth_pct, 1),
                asmi_target=asmi_target,
                asmi_tolerance=asmi_tol,
                deviation_from_asmi=round(deviation, 1) if deviation is not None else None,
                flag=flag,
                coaching_relevant=bench.coaching_relevant,
                coaching_note=coaching_note,
            ))

        # Sort: coaching-relevant items first, then by flag severity
        flag_order = {"concern": 0, "needs_attention": 1, "on_track": 2, "excellent": 3, None: 4}
        comparisons.sort(key=lambda c: (
            0 if c.coaching_relevant else 1,
            flag_order.get(c.flag, 4),
        ))

        return comparisons

    def _classify_flag(
        self,
        metric: str,
        value: float,
        bench: YouthAdjustedBenchmark,
        asmi_target: Optional[float],
        asmi_tolerance: Optional[float],
    ) -> Optional[str]:
        """Classify a metric value into a coaching flag.

        For youth, we prioritize ASMI positional targets over percentile rank.
        Being "below average" in velocity is expected and normal.
        Being outside ASMI positional targets IS coachable.
        """
        tier = bench.tier

        # For Tier 1 angles: use ASMI targets when available
        if tier == MetricTier.TIER_1_ANGLE and asmi_target is not None:
            deviation = abs(value - asmi_target)
            if deviation <= asmi_tolerance * 0.5:
                return "excellent"
            elif deviation <= asmi_tolerance:
                return "on_track"
            elif deviation <= asmi_tolerance * 1.5:
                return "needs_attention"
            else:
                return "concern"

        # For Tier 1 angles without ASMI targets: use adjusted percentile
        if tier == MetricTier.TIER_1_ANGLE:
            if bench.youth_p25 <= value <= bench.youth_p75:
                return "on_track"
            elif value < bench.youth_p25:
                return "needs_attention"
            else:
                return "on_track"  # High angles aren't necessarily bad

        # For Tier 2 velocity/force metrics: don't flag low values in youth
        if tier in (MetricTier.TIER_2_VELOCITY, MetricTier.TIER_2_FORCE,
                    MetricTier.TIER_2_POWER):
            if not bench.coaching_relevant:
                return None  # Don't flag what we're not coaching
            # Only flag if extremely high (potential overload concern)
            if value > bench.youth_p75 * 1.3:
                return "concern"  # Unusually high force for their size
            return "on_track"

        # For moments: flag high values (injury risk)
        if tier == MetricTier.TIER_2_MOMENT:
            if value > bench.youth_p75:
                return "concern"  # High joint loading for body size
            return "on_track"

        return None

    def get_coaching_priorities(self) -> dict:
        """Get developmental-stage-appropriate coaching priorities."""
        return COACHING_PRIORITIES[self.dev_stage]

    def generate_youth_report_context(self) -> dict:
        """Generate context dict for coaching report generation.

        Provides all the metadata needed by the coaching insight module
        to generate age-appropriate feedback.
        """
        priorities = self.get_coaching_priorities()
        cdc_age = int(max(8, min(20, self.pitcher.age)))
        cdc_h, cdc_w = CDC_BOYS_50TH.get(cdc_age, (170, 60))

        return {
            "pitcher_age": self.pitcher.age,
            "pitcher_height_cm": self.pitcher.height_cm,
            "pitcher_weight_kg": self.pitcher.weight_kg,
            "pitcher_throws": self.pitcher.throws,
            "dev_stage": self.dev_stage.value,
            "effective_dev_age": round(self.pitcher.effective_dev_age, 1),
            "size_vs_peers": f"{self.pitcher.size_relative_to_cdc_50th:.0%}",
            "cdc_50th_height_cm": cdc_h,
            "cdc_50th_weight_kg": cdc_w,
            "coaching_focus": priorities["focus"],
            "coaching_emphasis": priorities["emphasis"],
            "coaching_de_emphasize": priorities["de_emphasize"],
            "injury_watch": priorities["injury_watch"],
            "variability_note": (
                "Youth pitchers naturally show more pitch-to-pitch variability than adults. "
                f"Expected variability at this stage is ~{self.var_mult:.0%} of adult range."
            ),
        }
