"""Generate coaching insights from biomechanical analysis using Claude API.

Takes extracted pitcher metrics, OBP benchmark comparisons, and detected
events to produce natural-language coaching feedback appropriate for
youth/developing pitchers.
"""

import json
from typing import Optional

SYSTEM_PROMPT = """You are a pitching mechanics coach with deep knowledge of biomechanics.
You're analyzing a developing pitcher's mechanics from video analysis and comparing
them to elite-level benchmarks from the Driveline OpenBiomechanics Project.

Your coaching style:
- Positive-first: always lead with what the pitcher does well
- Developmentally appropriate: these are youth/amateur pitchers, not pros
- Focus on 2-3 key areas, not everything at once
- Use simple language a player or parent can understand
- Connect mechanics to outcomes (velocity, control, injury prevention)
- Acknowledge camera-based limitations vs. lab-grade motion capture
- Never recommend extreme changes — small, incremental adjustments
- Emphasize that percentile comparisons are against college/pro norms,
  so being below average is expected and normal for developing pitchers

Key biomechanical concepts to reference:
- Hip-shoulder separation: the engine of velocity
- Lead leg block: energy transfer from legs to trunk to arm
- Arm slot consistency: control and deception
- Trunk tilt at release: affects pitch plane and arm stress
- Timing: the kinetic chain should fire proximal to distal (hips → trunk → arm)
"""

YOUTH_SYSTEM_PROMPT = """You are a pitching mechanics coach specializing in youth baseball
player development (ages 10-14). You have deep knowledge of biomechanics AND
child motor development, and you understand that youth pitchers are fundamentally
different from college/pro athletes.

CRITICAL CONTEXT from ASMI research (Fleisig et al., 7-year longitudinal study):
- Ages 9-13: kinematics (movement patterns) change the most — this is THE window
  to build proper arm path, timing, and body positions
- Ages 13-15: kinetics (forces) start increasing significantly as the body matures
- Movement PATTERNS are similar across ages; FORCES and SPEEDS scale with physical maturity
- Youth pitchers naturally show much wider pitch-to-pitch variability than adults
- Variability DECREASES with development — inconsistency IS the stage, not a flaw

YOUR APPROACH FOR YOUTH PITCHERS:
1. NEVER compare raw velocity/force numbers to adult norms — always use
   body-size-normalized values or focus on movement quality metrics
2. CELEBRATE movement quality (angles, positions, sequence) over output (speed, force)
3. Use ASMI positional targets as the gold standard for youth:
   - Elbow flexion ~90° at foot contact
   - Shoulder abduction ~90° at foot contact
   - Shoulder ~20° horizontal abduction at foot contact
   - Shoulder ER ~45° at foot contact
   - Hip-shoulder separation ~30° at foot plant
   - Stride length 75-85% of body height
4. Frame everything through the lens of LONG-TERM DEVELOPMENT:
   - "Building the foundation" not "fixing flaws"
   - "Your body is still growing and these numbers will improve naturally"
   - "The goal right now is teaching your body the right movement patterns"
5. INJURY PREVENTION is priority #1:
   - Growth plates are vulnerable during rapid growth
   - Watch for excessive arm stress patterns regardless of force magnitude
   - Pitch count adherence and rest are more important than any mechanic
6. PARENT-FRIENDLY language — assume a parent will read this report
7. Keep it to 2-3 focus areas maximum — don't overwhelm a developing kid

Biomechanical concepts appropriate for youth coaching:
- "Loading" (hip-shoulder separation) — how the body stores energy
- "Staying tall" (trunk tilt) — posture through the delivery
- "Getting the arm up" (shoulder positions) — safe arm path
- "Using your legs" (lead leg block) — ground force connection
- "Finishing the pitch" (follow-through) — deceleration and arm health
"""


def build_analysis_prompt(
    comparisons: list[dict],
    pitcher_context: str = "youth developing pitcher",
    additional_context: Optional[str] = None,
) -> str:
    """Build the analysis prompt from benchmark comparison results.

    Args:
        comparisons: Output from OBPBenchmarks.compare_to_benchmarks().
        pitcher_context: Description of the pitcher being analyzed.
        additional_context: Any additional notes (camera angle, pitch type, etc.).

    Returns:
        Formatted prompt string for Claude API.
    """
    # Format the comparison data
    metrics_text = []
    for c in comparisons:
        flag_label = ""
        if c["flag"] == "well_below_average":
            flag_label = " ⚠️ WELL BELOW AVG"
        elif c["flag"] == "below_average":
            flag_label = " ↓ Below avg"
        elif c["flag"] == "above_average":
            flag_label = " ↑ Above avg"
        elif c["flag"] == "well_above_average":
            flag_label = " ✅ WELL ABOVE AVG"

        metrics_text.append(
            f"- {c['display_name']}: {c['value']:.1f} {c['unit']} "
            f"(percentile: {c['percentile_rank']:.0f}th, "
            f"benchmark median: {c['benchmark_median']:.1f}){flag_label}"
        )

    prompt = f"""Analyze this {pitcher_context}'s mechanics based on the following measurements
compared to Driveline OBP benchmarks ({comparisons[0]['playing_level'] if comparisons else 'all'} level, 
n={comparisons[0]['n_samples'] if comparisons else 'N/A'} pitches):

METRICS:
{chr(10).join(metrics_text)}

CAMERA SETUP: Side view, 2D analysis (some metrics are approximations)

Please provide:
1. STRENGTHS (2-3 things this pitcher does well relative to their level)
2. KEY AREAS TO WORK ON (2-3 highest-priority mechanical adjustments)
3. DRILLS & CUES (specific, actionable drills or verbal cues for each area)
4. CONTEXT (what these numbers mean for a developing pitcher)
"""

    if additional_context:
        prompt += f"\nADDITIONAL NOTES: {additional_context}"

    return prompt


def generate_coaching_report(
    comparisons: list[dict],
    pitcher_context: str = "youth developing pitcher",
    additional_context: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Generate a coaching report using Claude API.

    Args:
        comparisons: Output from OBPBenchmarks.compare_to_benchmarks().
        pitcher_context: Description of the pitcher.
        additional_context: Optional additional notes.
        api_key: Anthropic API key. Falls back to env var.
        model: Claude model to use.

    Returns:
        Natural-language coaching report.
    """
    import anthropic

    if api_key is None:
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set it as an environment variable "
                "or pass api_key directly."
            )

    client = anthropic.Anthropic(api_key=api_key)

    prompt = build_analysis_prompt(
        comparisons=comparisons,
        pitcher_context=pitcher_context,
        additional_context=additional_context,
    )

    message = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def build_youth_analysis_prompt(
    comparisons: list,
    youth_context: dict,
    additional_context: Optional[str] = None,
) -> str:
    """Build an analysis prompt using youth-normalized comparisons.

    Args:
        comparisons: Output from YouthNormalizer.compare().
        youth_context: Output from YouthNormalizer.generate_youth_report_context().
        additional_context: Optional additional notes.

    Returns:
        Formatted prompt string for Claude API.
    """
    # Group by coaching relevance and flag
    coaching_items = [c for c in comparisons if c.coaching_relevant]
    info_only = [c for c in comparisons if not c.coaching_relevant]

    metrics_text = []
    for c in coaching_items:
        line = f"- {c.flag_emoji} {c.display_name}: {c.measured_value:.1f} {c.unit}"
        if c.asmi_target is not None:
            line += f" (ASMI target: {c.asmi_target:.0f}°, deviation: {c.deviation_from_asmi:.1f}°)"
        line += f" — {c.flag or 'no flag'}"
        if c.coaching_note:
            line += f" [{c.coaching_note}]"
        metrics_text.append(line)

    if info_only:
        metrics_text.append("\n(The following metrics are informational only at this stage — "
                          "they're dominated by physical growth, not technique):")
        for c in info_only[:5]:
            metrics_text.append(f"  ➖ {c.display_name}: {c.measured_value:.1f} {c.unit}")

    ctx = youth_context
    prompt = f"""Analyze this pitcher's mechanics and provide a parent-friendly coaching report.

PITCHER PROFILE:
  Age: {ctx['pitcher_age']} years old
  Height: {ctx['pitcher_height_cm']:.0f} cm ({ctx['size_vs_peers']} of CDC 50th percentile for age)
  Weight: {ctx['pitcher_weight_kg']:.0f} kg
  Throws: {ctx['pitcher_throws']}HP
  Developmental stage: {ctx['dev_stage'].replace('_', ' ').title()}
  Effective developmental age: {ctx['effective_dev_age']} (adjusted for body size)

DEVELOPMENTAL CONTEXT:
  Current coaching focus: {ctx['coaching_focus']}
  Key emphasis areas: {'; '.join(ctx['coaching_emphasis'][:3])}
  De-emphasize at this stage: {'; '.join(ctx['coaching_de_emphasize'][:2]) if ctx['coaching_de_emphasize'] else 'N/A'}

MECHANICS ANALYSIS (benchmarks adjusted for age/size):
{chr(10).join(metrics_text)}

INJURY WATCH ITEMS FOR THIS AGE:
{chr(10).join('  - ' + item for item in ctx['injury_watch'])}

Please provide:
1. WHAT THIS PITCHER DOES WELL (2-3 positives — celebrate what's working)
2. ONE OR TWO THINGS TO WORK ON (highest priority, most coachable)
3. SIMPLE DRILLS OR CUES (age-appropriate, one drill per focus area)
4. PARENT CONTEXT (what to expect at this developmental stage, why patience matters)
5. HEALTH NOTE (any injury-prevention concerns based on the mechanics)
"""

    if additional_context:
        prompt += f"\nADDITIONAL NOTES: {additional_context}"

    return prompt


def generate_youth_coaching_report(
    comparisons: list,
    youth_context: dict,
    additional_context: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Generate a coaching report for a youth pitcher using Claude API.

    Args:
        comparisons: Output from YouthNormalizer.compare().
        youth_context: Output from YouthNormalizer.generate_youth_report_context().
        additional_context: Optional additional notes.
        api_key: Anthropic API key.
        model: Claude model to use.

    Returns:
        Natural-language coaching report tailored for youth development.
    """
    import anthropic

    if api_key is None:
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = build_youth_analysis_prompt(
        comparisons=comparisons,
        youth_context=youth_context,
        additional_context=additional_context,
    )

    message = client.messages.create(
        model=model,
        max_tokens=2500,
        system=YOUTH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def generate_report_offline(comparisons: list[dict]) -> str:
    """Generate a basic coaching report without Claude API (rule-based fallback).

    Useful for testing or when API key isn't available.
    """
    strengths = []
    areas = []

    for c in comparisons:
        name = c["display_name"]
        pct = c["percentile_rank"]

        if pct is not None:
            if pct >= 70:
                strengths.append(f"✅ {name}: {c['value']:.1f} {c['unit']} ({pct:.0f}th percentile)")
            elif pct <= 25:
                areas.append(f"📋 {name}: {c['value']:.1f} {c['unit']} ({pct:.0f}th percentile, median: {c['benchmark_median']:.1f})")

    report = "PITCHING MECHANICS ANALYSIS\n" + "=" * 40 + "\n\n"

    if strengths:
        report += "STRENGTHS:\n" + "\n".join(strengths[:3]) + "\n\n"
    else:
        report += "STRENGTHS: Metrics are within normal range — solid foundation!\n\n"

    if areas:
        report += "AREAS FOR DEVELOPMENT:\n" + "\n".join(areas[:3]) + "\n\n"
    else:
        report += "AREAS FOR DEVELOPMENT: No major outliers detected. Continue building consistency.\n\n"

    report += (
        "NOTE: These comparisons are against Driveline's OBP dataset of primarily\n"
        "college-level pitchers. Being below the median is normal and expected for\n"
        "developing pitchers. Focus on trends over time, not absolute numbers.\n"
    )

    return report


def generate_youth_report_offline(
    comparisons: list,
    youth_context: dict,
) -> str:
    """Generate a youth coaching report without Claude API (rule-based fallback)."""
    ctx = youth_context

    report = f"""YOUTH PITCHING MECHANICS ANALYSIS
{'=' * 50}

PITCHER: {ctx['pitcher_age']:.0f} years old, {ctx['pitcher_height_cm']:.0f}cm, {ctx['pitcher_weight_kg']:.0f}kg, {ctx['pitcher_throws']}HP
STAGE: {ctx['dev_stage'].replace('_', ' ').title()}
SIZE vs PEERS: {ctx['size_vs_peers']} of average for age

CURRENT DEVELOPMENTAL FOCUS: {ctx['coaching_focus']}
{'-' * 50}

"""
    # Sort coaching-relevant items
    relevant = [c for c in comparisons if c.coaching_relevant]

    excellent = [c for c in relevant if c.flag == "excellent"]
    on_track = [c for c in relevant if c.flag == "on_track"]
    needs_attn = [c for c in relevant if c.flag == "needs_attention"]
    concern = [c for c in relevant if c.flag == "concern"]

    if excellent or on_track:
        report += "WHAT'S WORKING WELL:\n"
        for c in (excellent + on_track)[:3]:
            report += f"  {c.flag_emoji} {c.display_name}: {c.measured_value:.1f}{c.unit}"
            if c.asmi_target:
                report += f" (target: {c.asmi_target:.0f}°)"
            report += "\n"
        report += "\n"

    if needs_attn or concern:
        report += "AREAS TO DEVELOP:\n"
        for c in (concern + needs_attn)[:2]:
            report += f"  {c.flag_emoji} {c.display_name}: {c.measured_value:.1f}{c.unit}"
            if c.asmi_target:
                report += f" → work toward {c.asmi_target:.0f}°"
            if c.coaching_note:
                report += f"\n     Note: {c.coaching_note}"
            report += "\n"
        report += "\n"
    else:
        report += "AREAS TO DEVELOP: No major concerns — keep building consistency!\n\n"

    report += f"""DEVELOPMENTAL CONTEXT:
  {ctx['variability_note']}

  Key emphasis at this stage:
"""
    for item in ctx["coaching_emphasis"][:3]:
        report += f"    • {item}\n"

    if ctx["coaching_de_emphasize"]:
        report += "\n  Don't worry about:\n"
        for item in ctx["coaching_de_emphasize"][:2]:
            report += f"    • {item}\n"

    report += "\n  Injury watch:\n"
    for item in ctx["injury_watch"][:2]:
        report += f"    ⚠️ {item}\n"

    return report
