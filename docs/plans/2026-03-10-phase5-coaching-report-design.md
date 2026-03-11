# Phase 5 Design: End-to-End Coaching Report Generation

**Date:** 2026-03-10
**Status:** Approved
**Author:** Rob Stover

## What We're Building

Upgrade `validate_pose.py` to produce a single enhanced HTML report that goes from
video to coaching insights in one command. The report leads with coaching (what it
means) and follows with technical diagnostics (what happened).

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Deliverable | Single enhanced HTML report | One file to open, share with coaches/parents |
| Default path | Youth-normalized (when profile provided) | Primary use case is youth pitchers |
| Input units | Imperial (inches, pounds) | US youth baseball context; convert to metric internally |
| API behavior | Auto-detect key, offline fallback | Always get coaching output, no flag needed |
| Script target | Upgrade `validate_pose.py` | Has the working anchor-based pipeline; `analyze_video.py` is stale |
| Prompt management | External files in `data/prompts/` | Tune coaching quality without code changes |
| Report order | Coaching first, diagnostics second | Coaches read top-down; developers debug bottom-up |

## Pipeline Flow

```
Video -> Pose Estimation -> Event Detection -> Feature Extraction
  -> OBP Benchmark Comparison (percentile ranks, flags)
  -> Youth Normalization (if --age/--height/--weight provided)
  -> Claude API Coaching Narrative (auto if API key set, offline fallback)
  -> Enhanced HTML Report (coaching-first layout)
  -> Enriched results.json
```

## CLI Interface

```bash
# Youth pitcher (primary use case)
python scripts/validate_pose.py --video clip.mp4 --age 12 --height 60 --weight 90

# Adult comparison fallback (no youth profile)
python scripts/validate_pose.py --video clip.mp4

# Existing flags still work
python scripts/validate_pose.py --video clip.mp4 --throws L --backend mediapipe --model-size l
```

- `--age INT` — pitcher age in years (required for youth path)
- `--height FLOAT` — height in inches (required for youth path)
- `--weight FLOAT` — weight in pounds (required for youth path)
- If any of the three are provided, all three are required.
- Heights/weights converted internally: inches * 2.54 = cm, pounds * 0.4536 = kg.

## HTML Report Layout

Top-to-bottom order, coaching-first:

1. **Header** — video info, pitcher profile card (if youth: age, height, weight, developmental stage)
2. **Coaching Narrative** — Claude-generated (or offline) report:
   - Strengths (2-3 things the pitcher does well)
   - Areas to work on (2-3 highest-priority items)
   - Drills and cues (specific, age-appropriate)
   - Developmental context (what to expect at this stage)
   - Health notes (injury prevention items)
3. **Percentile Charts** — Plotly radar chart + percentile gauges (embedded)
4. **Key Frame Grid** — annotated stills at leg lift, foot plant, MER, release
5. **Metrics Table** — with percentile ranks and OBP medians added
6. **Trajectory Plots** — wrist speed, joint trajectories, confidence heatmap
7. **Annotated Video** — skeleton overlay video
8. **Diagnostics** — frame count, confidence, warnings

Coaching section uses visually distinct styling (lighter background, larger text)
to separate it from technical content below.

## Coaching Intelligence Architecture

### Prompt Files (data/prompts/)

System prompts extracted from Python into editable markdown files:

- `youth_coaching_persona.md` — voice, tone, developmental philosophy
- `adult_coaching_persona.md` — voice, tone for adult/general comparison
- `coaching_knowledge.md` — drill prescriptions, red flag patterns, age-appropriate progressions
- `measurement_caveats.md` — camera angle limitations, 2D vs 3D notes, metric-specific caveats

### Prompt Assembly

The coaching prompt is built from parts:
1. System prompt = persona file (youth or adult)
2. User prompt = structured pitcher data + metrics + additional context
3. Additional context includes: non-OBP metrics with caveats, camera setup notes

### Non-OBP Metrics

7 of 11 extracted metrics map to OBP benchmark names. The remaining 4:
- `arm_slot_angle` — passed as additional context with camera angle caveat
- `lead_knee_angle_fp` / `lead_knee_angle_br` — raw angles, useful for coaching
- `stride_length_pct` — stride as % of height (ASMI target: 75-85%)

These are included in the prompt as supplementary observations, not percentile comparisons.

### Coach's Eye Patterns (coaching_knowledge.md)

Qualitative movement patterns that experienced coaches identify, mapped to
detectable metric signals. Based on feedback from a former MLB pitcher reviewing
ground truth video:

| Coach Observation | Metric Signal | Detectable? |
|---|---|---|
| Back knee collapsed | Back knee angle during drive phase | Yes (knee keypoint) |
| Back foot not rotating | Back ankle position during hip rotation | Partial (ankle trajectory) |
| Hip leaking (early opening) | Low hip-shoulder sep at foot plant | Yes (existing metric) |
| Elbow too low at key positions | Arm slot / shoulder abduction | Yes (with camera caveat) |

These patterns inform the coaching knowledge base — teaching Claude to flag
combinations of metrics that suggest these qualitative issues, and to use
coach-friendly language ("hip is leaking" vs "low hip-shoulder separation").

### Tuning Loop

```
Run pipeline -> Read coaching report -> Edit prompt files -> Re-run
```

- Every coaching report saved in results.json for review
- Prompt file edits don't require code changes
- Coach feedback (like the MLB pitcher notes) becomes ground truth for prompt quality
- Over time, coaching_knowledge.md grows with drill prescriptions and pattern rules

## Data Flow Details

### New Steps in validate_pose.py

After existing metrics extraction:

```python
# Step 5: OBP Benchmark Comparison (NEW)
obp = OBPBenchmarks().load()
obp_comparison_dict = metrics.to_obp_comparison_dict()
comparisons = obp.compare_to_benchmarks(obp_comparison_dict, playing_level=level)

# Step 6: Youth Normalization (NEW, conditional)
if youth_profile_provided:
    profile = YouthPitcherProfile(age=age, height_cm=h_cm, weight_kg=w_kg, throws=throws)
    normalizer = YouthNormalizer(obp, profile)
    youth_comparisons = normalizer.compare(obp_comparison_dict)
    youth_context = normalizer.generate_youth_report_context()

# Step 7: Coaching Report (NEW)
if ANTHROPIC_API_KEY:
    if youth_profile_provided:
        report_text = generate_youth_coaching_report(youth_comparisons, youth_context, ...)
    else:
        report_text = generate_coaching_report(comparisons, ...)
else:
    # offline fallback
    ...
```

### Enriched results.json

Adds to existing structure:
```json
{
  "obp_comparisons": [...],
  "coaching_report": "...",
  "youth_profile": {
    "age": 12,
    "height_in": 60,
    "weight_lbs": 90,
    "developmental_stage": "early_adolescent"
  }
}
```

### build_report_html Changes

New parameters:
- `coaching_html: str` — rendered coaching narrative (pre-formatted HTML)
- `percentile_charts_html: list[str]` — Plotly chart fragments for radar/gauges
- `pitcher_profile: dict | None` — youth profile for header card

Report template reordered to coaching-first layout.

## Out of Scope (Phase 5)

- Streamlit UI (Phase 6)
- New metric extraction (back knee collapse, back foot rotation)
- Multi-pitch session aggregation
- Video comparison (side-by-side two pitches)
- Updating analyze_video.py (leave as-is or deprecate later)
