---
title: "feat: Phase 5 End-to-End Coaching Report Generation"
type: feat
date: 2026-03-10
design_doc: docs/plans/2026-03-10-phase5-coaching-report-design.md
---

# Phase 5: End-to-End Coaching Report Generation

Wire the existing coaching module into the working video pipeline so that
`validate_pose.py` produces a single HTML report with coaching insights,
benchmark comparisons, and percentile visualizations — all in one command.

## Overview

The pipeline today stops at metrics extraction (Stage 4). This plan adds three
new stages: OBP benchmark comparison, youth normalization (conditional), and
coaching report generation (API or offline). The HTML report is reordered to
lead with coaching content. All existing modules are already built — this is
integration work.

## Proposed Solution

Seven ordered tasks, grouped into three parallelizable waves:

```
Wave 1 (independent):
  Task 1: Extract prompts to data/prompts/
  Task 2: Add youth profile CLI flags + validation
  Task 3: New test file scaffolding

Wave 2 (depends on Wave 1):
  Task 4: Wire OBP benchmark comparison into pipeline
  Task 5: Wire youth normalization into pipeline
  Task 6: Wire coaching report generation into pipeline

Wave 3 (depends on Wave 2):
  Task 7: Reorder HTML report to coaching-first layout + enrich results.json
```

## Technical Approach

### Task 1: Extract Coaching Prompts to `data/prompts/`

**Files:** `src/coaching/insights.py`, `data/prompts/*.md` (4 new files)

**What to do:**
1. Create `data/prompts/` directory
2. Create 4 markdown prompt files extracted from the hardcoded strings in `insights.py`:
   - `youth_coaching_persona.md` — extract from `YOUTH_SYSTEM_PROMPT` (lines 34-75)
   - `adult_coaching_persona.md` — extract from `SYSTEM_PROMPT` (lines 11-32)
   - `coaching_knowledge.md` — NEW content: drill prescriptions, red flag patterns,
     Coach's Eye mappings (from design doc table), age-appropriate progressions
   - `measurement_caveats.md` — NEW content: 2D vs 3D limitations, camera angle notes,
     per-metric confidence notes (arm_slot is camera-dependent, etc.)
3. Add a `load_prompt(name: str) -> str` function to `insights.py` that reads from
   `data/prompts/{name}.md` with a fallback to hardcoded defaults if file not found
4. Update `generate_coaching_report()` and `generate_youth_coaching_report()` to
   build system prompts from loaded files:
   - Youth: `youth_coaching_persona.md` + `coaching_knowledge.md`
   - Adult: `adult_coaching_persona.md` + `coaching_knowledge.md`
5. Keep `SYSTEM_PROMPT` and `YOUTH_SYSTEM_PROMPT` as fallback constants

**Key decisions:**
- `load_prompt()` resolves paths relative to project root via `Path(__file__).parent.parent.parent / "data" / "prompts"`
- Files are plain markdown — no frontmatter, no templating. Concatenated into system prompt.
- Fallback to hardcoded strings ensures the module works without the files (e.g., in tests)

**Acceptance:**
- [ ] `data/prompts/youth_coaching_persona.md` exists with YOUTH_SYSTEM_PROMPT content
- [ ] `data/prompts/adult_coaching_persona.md` exists with SYSTEM_PROMPT content
- [ ] `data/prompts/coaching_knowledge.md` exists with drill/pattern content
- [ ] `data/prompts/measurement_caveats.md` exists with 2D limitation notes
- [ ] `load_prompt("youth_coaching_persona")` returns file content
- [ ] `load_prompt("nonexistent")` returns empty string (graceful fallback)
- [ ] Existing `generate_coaching_report` and `generate_youth_coaching_report` still work

---

### Task 2: Add Youth Profile CLI Flags to `validate_pose.py`

**Files:** `scripts/validate_pose.py`

**What to do:**
1. Add three argparse arguments after existing `--no-open`:
   ```python
   parser.add_argument("--age", type=int, default=None, help="Pitcher age in years")
   parser.add_argument("--height", type=float, default=None, help="Pitcher height in inches")
   parser.add_argument("--weight", type=float, default=None, help="Pitcher weight in pounds")
   ```
2. Add validation block immediately after `args = parser.parse_args()`:
   - If any of `(age, height, weight)` is provided, all three must be provided
   - If not, print error and `sys.exit(1)` with message:
     `"Error: --age, --height, and --weight must all be provided together."`
   - Validate ranges: age 6-25, height 36-84 inches, weight 40-350 lbs
3. Convert to metric:
   ```python
   height_cm = args.height * 2.54
   weight_kg = args.weight * 0.4536
   ```
4. Store as a dict for later use:
   ```python
   youth_profile = None
   if args.age is not None:
       youth_profile = {
           "age": args.age,
           "height_in": args.height,
           "weight_lbs": args.weight,
           "height_cm": height_cm,
           "weight_kg": weight_kg,
       }
   ```

**Edge cases:**
- `--age 12` alone → error (missing --height and --weight)
- `--age 3` → error (out of range)
- `--height 0` → error (out of range)
- No youth flags → `youth_profile = None`, pipeline uses adult comparison path

**Acceptance:**
- [ ] `--age 12 --height 60 --weight 90` works, creates youth_profile dict
- [ ] `--age 12` alone prints error and exits
- [ ] `--age 3` prints range error
- [ ] No youth flags → youth_profile is None

---

### Task 3: Test Scaffolding

**Files:** `tests/test_coaching_integration.py` (NEW)

**What to do:**
Create test file with these test classes/methods:

```python
class TestImperialToMetric:
    def test_inches_to_cm(self):
        assert 60 * 2.54 == 152.4

    def test_pounds_to_kg(self):
        assert abs(90 * 0.4536 - 40.824) < 0.01

class TestPromptLoading:
    def test_load_existing_prompt(self):
        # load_prompt("youth_coaching_persona") returns non-empty string
        ...

    def test_load_nonexistent_prompt_returns_empty(self):
        # load_prompt("does_not_exist") returns ""
        ...

    def test_all_prompt_files_exist(self):
        # Check all 4 data/prompts/*.md files exist
        ...

class TestCliValidation:
    def test_partial_youth_flags_error(self):
        # Providing only --age should fail validation
        ...

    def test_age_range_validation(self):
        # --age 3 should fail
        ...

class TestOfflineCoachingReport:
    def test_offline_report_with_comparisons(self):
        # generate_report_offline() returns string with STRENGTHS section
        ...

    def test_offline_youth_report(self):
        # generate_youth_report_offline() returns string with PITCHER section
        ...

class TestReportCoachingSection:
    def test_report_includes_coaching_section(self):
        # build_report_html(coaching_html="...") includes coaching content
        ...

    def test_report_without_coaching(self):
        # build_report_html(coaching_html="") has no coaching section
        ...

    def test_report_includes_percentile_charts(self):
        # build_report_html(percentile_charts_html=[...]) includes charts
        ...

    def test_report_includes_pitcher_profile(self):
        # build_report_html(pitcher_profile={...}) shows profile card
        ...
```

**Acceptance:**
- [ ] Test file exists with all test stubs
- [ ] Tests for prompt loading, CLI validation, offline reports, HTML coaching section
- [ ] All tests pass after corresponding tasks are complete

---

### Task 4: Wire OBP Benchmark Comparison into Pipeline

**Files:** `scripts/validate_pose.py`

**What to do:**
1. After Stage 4 (Metrics Extraction), add Stage 5:
   ```python
   # =========================================================================
   # Stage 5: OBP Benchmark Comparison
   # =========================================================================
   ```
2. Replace the existing simple median-lookup code (lines 429-438) with full comparison:
   ```python
   obp_comparisons = []
   if poi_path.exists():
       obp = OBPBenchmarks().load()
       obp_dict = metrics.to_obp_comparison_dict()
       if obp_dict:
           obp_comparisons = obp.compare_to_benchmarks(obp_dict)
   ```
3. Build metrics_rows with percentile data from `obp_comparisons`:
   - Add `percentile` key to each metrics_row dict
   - Map comparison results back to internal metric names
4. Generate percentile chart HTML fragments:
   ```python
   from src.viz.plots import plot_pitcher_comparison, plot_percentile_gauges

   percentile_charts_html = []
   if obp_comparisons:
       radar_fig = plot_pitcher_comparison(obp_comparisons)
       percentile_charts_html.append(radar_fig.to_html(full_html=False, include_plotlyjs=False))
       gauges_fig = plot_percentile_gauges(obp_comparisons)
       percentile_charts_html.append(gauges_fig.to_html(full_html=False, include_plotlyjs=False))
   ```

**Integration note:** `compare_to_benchmarks` returns `list[dict]` with keys:
`metric`, `display_name`, `value`, `unit`, `percentile_rank`, `flag`,
`benchmark_median`, `benchmark_p25`, `benchmark_p75`, `playing_level`, `n_samples`

**Edge cases:**
- No OBP data file → skip comparison, empty lists, no charts
- All metrics are None (pose failed) → `obp_dict` is empty → skip comparison
- Only 1-2 metrics available → radar chart with few spokes (Plotly handles this)

**Acceptance:**
- [ ] `obp_comparisons` populated with percentile ranks
- [ ] Radar chart and gauge charts generated as HTML fragments
- [ ] Metrics table rows include percentile column
- [ ] Pipeline works when OBP data missing (graceful skip)

---

### Task 5: Wire Youth Normalization into Pipeline

**Files:** `scripts/validate_pose.py`

**What to do:**
1. Add new imports at top:
   ```python
   from src.biomechanics.youth_normalizer import YouthPitcherProfile, YouthNormalizer
   ```
2. After Stage 5, add Stage 6 (conditional on `youth_profile`):
   ```python
   # =========================================================================
   # Stage 6: Youth Normalization (conditional)
   # =========================================================================
   youth_comparisons = None
   youth_context = None
   if youth_profile and obp_comparisons:
       profile = YouthPitcherProfile(
           age=youth_profile["age"],
           height_cm=youth_profile["height_cm"],
           weight_kg=youth_profile["weight_kg"],
           throws=args.throws,
       )
       normalizer = YouthNormalizer(obp, profile)
       youth_comparisons = normalizer.compare(metrics.to_obp_comparison_dict())
       youth_context = normalizer.generate_youth_report_context()
   ```
3. If youth path taken, add `developmental_stage` to youth_profile dict:
   ```python
   youth_profile["developmental_stage"] = normalizer.dev_stage.value
   ```

**Key detail:** `YouthNormalizer.compare()` returns `list[YouthNormalizedComparison]`
(dataclass with `.coaching_relevant`, `.flag`, `.flag_emoji`, `.asmi_target`, etc.)
This is NOT a dict — the existing `build_youth_analysis_prompt()` and
`generate_youth_report_offline()` already handle these objects correctly.

**Acceptance:**
- [ ] Youth path activated when all three CLI flags provided
- [ ] `youth_comparisons` is list of `YouthNormalizedComparison` objects
- [ ] `youth_context` dict populated with developmental stage, coaching priorities
- [ ] Without youth flags, this stage is skipped entirely

---

### Task 6: Wire Coaching Report Generation into Pipeline

**Files:** `scripts/validate_pose.py`

**What to do:**
1. Add new imports:
   ```python
   import os
   from src.coaching.insights import (
       generate_coaching_report,
       generate_youth_coaching_report,
       generate_report_offline,
       generate_youth_report_offline,
       load_prompt,
   )
   ```
2. After Stage 6, add Stage 7:
   ```python
   # =========================================================================
   # Stage 7: Coaching Report
   # =========================================================================
   coaching_text = ""
   api_key = os.getenv("ANTHROPIC_API_KEY")

   # Build additional context string for non-OBP metrics
   additional_lines = []
   if metrics.arm_slot_angle is not None:
       additional_lines.append(
           f"Arm slot angle: {metrics.arm_slot_angle:.1f}° "
           "(camera angle dependent — from front-quarter view, reads ~15-22° low)"
       )
   if metrics.lead_knee_angle_fp is not None:
       additional_lines.append(f"Lead knee angle at foot plant: {metrics.lead_knee_angle_fp:.1f}°")
   if metrics.lead_knee_angle_br is not None:
       additional_lines.append(f"Lead knee angle at ball release: {metrics.lead_knee_angle_br:.1f}°")
   if metrics.stride_length_pct_height is not None:
       additional_lines.append(
           f"Stride length: {metrics.stride_length_pct_height:.0f}% of height "
           "(ASMI target: 75-85%)"
       )
   additional_context = "\n".join(additional_lines) if additional_lines else None

   # Append measurement caveats
   caveats = load_prompt("measurement_caveats")
   if caveats and additional_context:
       additional_context += f"\n\nMEASUREMENT CAVEATS:\n{caveats}"
   elif caveats:
       additional_context = f"MEASUREMENT CAVEATS:\n{caveats}"

   if youth_comparisons and youth_context:
       # Youth path
       if api_key:
           coaching_text = generate_youth_coaching_report(
               youth_comparisons, youth_context,
               additional_context=additional_context,
           )
       else:
           coaching_text = generate_youth_report_offline(youth_comparisons, youth_context)
   elif obp_comparisons:
       # Adult path
       if api_key:
           coaching_text = generate_coaching_report(
               obp_comparisons,
               additional_context=additional_context,
           )
       else:
           coaching_text = generate_report_offline(obp_comparisons)
   ```

**Edge cases:**
- No API key + no comparisons → empty coaching_text (report still generates)
- API call fails → catch exception, fall back to offline, print warning
- No metrics at all → skip coaching entirely

**Acceptance:**
- [ ] With API key: Claude-generated coaching report
- [ ] Without API key: offline rule-based report
- [ ] Youth path uses `generate_youth_coaching_report()`
- [ ] Adult path uses `generate_coaching_report()`
- [ ] Non-OBP metrics passed as additional context
- [ ] Measurement caveats loaded from `data/prompts/measurement_caveats.md`

---

### Task 7: Coaching-First HTML Report + Enriched results.json

**Files:** `src/viz/report.py`, `scripts/validate_pose.py`

#### 7a: Update `build_report_html()` signature

Add 3 new optional parameters (backward compatible):

```python
def build_report_html(
    video_filename: str,
    video_rel_path: str,
    fps: float,
    frame_count: int,
    backend: str,
    pitcher_throws: str,
    trajectory_plots_html: list[str],
    key_frame_images: dict[str, str],
    metrics_rows: list[dict],
    diagnostics: dict,
    # NEW params
    coaching_html: str = "",
    percentile_charts_html: list[str] | None = None,
    pitcher_profile: dict | None = None,
) -> str:
```

#### 7b: Reorder report sections

New order in `build_report_html()`:
1. **Header** — existing video info table + NEW pitcher profile card
2. **Coaching Narrative** — new section with distinct styling
3. **Percentile Charts** — new section embedding radar + gauges
4. **Key Frames** — existing (moved down)
5. **Metrics Table** — existing + add percentile column
6. **Trajectory Plots** — existing (moved down)
7. **Annotated Video** — existing (moved down)
8. **Diagnostics** — existing (stays last)

#### 7c: Coaching section styling

Add to `_CSS`:
```css
.coaching-section {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    font-size: 1.05rem;
    line-height: 1.7;
}
.coaching-section h2 {
    color: #7eb8da;
    border-bottom: 1px solid #2a2a4e;
}
.profile-card {
    display: flex;
    gap: 2rem;
    background: #141428;
    border-radius: 6px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
}
.profile-card .label { color: #888; }
.profile-card .value { color: #ddd; font-weight: 600; }
```

#### 7d: Pitcher profile card in header

If `pitcher_profile` is provided, render after the video info table:
```html
<div class="profile-card">
  <div><span class="label">Age</span><br><span class="value">12</span></div>
  <div><span class="label">Height</span><br><span class="value">5'0"</span></div>
  <div><span class="label">Weight</span><br><span class="value">90 lbs</span></div>
  <div><span class="label">Stage</span><br><span class="value">Early Adolescent</span></div>
</div>
```

#### 7e: Coaching section rendering

Convert coaching_text (plain text or markdown-ish) to HTML. Since the Claude API
returns formatted text with headers and bullet points:
```python
import re

def _coaching_text_to_html(text: str) -> str:
    """Convert coaching report text to HTML paragraphs."""
    lines = text.strip().split("\n")
    html_parts = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# ") or line.startswith("## "):
            html_parts.append(f"<h3>{escape(line.lstrip('#').strip())}</h3>")
        elif line.startswith("- ") or line.startswith("* "):
            html_parts.append(f"<li>{escape(line[2:])}</li>")
        elif line.startswith(("1.", "2.", "3.", "4.", "5.")):
            # Numbered section headers from Claude output
            html_parts.append(f"<h3>{escape(line)}</h3>")
        else:
            html_parts.append(f"<p>{escape(line)}</p>")
    return "\n".join(html_parts)
```

Wrap in coaching section div:
```html
<div class="coaching-section">
  <h2>Coaching Report</h2>
  {coaching_html_content}
</div>
```

Only render section if `coaching_html` is non-empty.

#### 7f: Metrics table percentile column

Add `Percentile` column to the metrics table between Value and OBP Median.

#### 7g: Update validate_pose.py report assembly

Pass new params to `build_report_html()`:
```python
# Convert coaching text to simple HTML
coaching_html = coaching_text  # pass raw text, report.py converts it

report_html = build_report_html(
    # ... existing params ...
    coaching_html=coaching_text,
    percentile_charts_html=percentile_charts_html,
    pitcher_profile=youth_profile,
)
```

#### 7h: Enrich results.json

Add to `pipeline_output` dict:
```python
if obp_comparisons:
    pipeline_output["obp_comparisons"] = [
        {k: v for k, v in c.items() if k != "values"}  # serializable subset
        for c in obp_comparisons
    ]
if coaching_text:
    pipeline_output["coaching_report"] = coaching_text
if youth_profile:
    pipeline_output["youth_profile"] = {
        "age": youth_profile["age"],
        "height_in": youth_profile["height_in"],
        "weight_lbs": youth_profile["weight_lbs"],
        "developmental_stage": youth_profile.get("developmental_stage"),
    }
```

**Acceptance:**
- [ ] Report sections appear in coaching-first order
- [ ] Coaching section has visually distinct dark blue styling
- [ ] Profile card shows age/height/weight/stage when youth profile provided
- [ ] Percentile column in metrics table
- [ ] Radar chart and gauges embedded in report
- [ ] results.json includes obp_comparisons, coaching_report, youth_profile
- [ ] Report works with all new params as None/empty (backward compatible)
- [ ] Existing test_report.py tests still pass (no breaking signature change)

---

## Edge Cases Identified (SpecFlow Analysis)

| Scenario | Expected Behavior |
|----------|-------------------|
| No OBP data file on disk | Skip comparison, no charts, no coaching — report still generates with metrics only |
| All pose metrics are None | `to_obp_comparison_dict()` returns `{}` → skip comparison |
| Only 1-2 OBP metrics available | Radar chart has few spokes (Plotly handles gracefully) |
| Prompt files missing | `load_prompt()` returns `""`, falls back to hardcoded constants |
| API key set but API call fails | Catch `anthropic.APIError`, fall back to offline report, print warning |
| `--age 12` without --height/--weight | Error message + `sys.exit(1)` |
| `--age 3` (out of range) | Error message with valid range |
| `--throws L` with youth profile | Left-handed path — `throw_side = "left"`, all metric extraction adapts |
| No events detected (no MER) | Metrics extraction produces mostly None → thin comparison, minimal coaching |
| Video with 0 poses detected | Existing error exit at line 149 → never reaches new stages |

## Dependencies & Risks

| Risk | Mitigation |
|------|------------|
| `anthropic` import fails if not installed | It's in requirements.txt; import is inside the function, not top-level |
| Claude API rate limits during testing | Offline fallback always available; tests use offline path |
| Plotly chart HTML too large for report | Charts are already fragment-mode (not full_html); CDN loaded once |
| Youth normalizer requires OBP data loaded | Stage 6 gated on `obp_comparisons` being non-empty |

## Quality Gates

- [x] All 89 existing tests pass (`pytest tests/ -v`) — 113 total now
- [x] New tests in `test_coaching_integration.py` pass — 24 tests
- [x] Lint passes (syntax verified, ruff not installed)
- [x] Pipeline runs end-to-end on IMG_3106.MOV with youth flags
- [x] Pipeline runs end-to-end without youth flags (adult path)
- [x] Pipeline runs without ANTHROPIC_API_KEY (offline fallback)
- [x] HTML report opens in browser with coaching section visible
- [x] results.json contains obp_comparisons and coaching_report keys

## References

### Internal
- Design doc: `docs/plans/2026-03-10-phase5-coaching-report-design.md`
- Pipeline script: `scripts/validate_pose.py`
- Coaching module: `src/coaching/insights.py`
- Report builder: `src/viz/report.py`
- Charts: `src/viz/plots.py` — `plot_pitcher_comparison()`, `plot_percentile_gauges()`
- Benchmarks: `src/biomechanics/benchmarks.py` — `compare_to_benchmarks()`
- Youth normalizer: `src/biomechanics/youth_normalizer.py` — `YouthNormalizer`, `YouthPitcherProfile`
- Features: `src/biomechanics/features.py` — `PitcherMetrics.to_obp_comparison_dict()`
- Existing tests: `tests/test_report.py`, `tests/test_benchmarks.py`, `tests/test_youth_normalizer.py`

### Key Data Structures
- `compare_to_benchmarks()` → `list[dict]` with: metric, display_name, value, unit, percentile_rank, flag, benchmark_median, playing_level, n_samples
- `YouthNormalizer.compare()` → `list[YouthNormalizedComparison]` (dataclass, NOT dict)
- `generate_youth_report_context()` → `dict` with: pitcher_age, pitcher_height_cm, pitcher_weight_kg, pitcher_throws, dev_stage, effective_dev_age, size_vs_peers, coaching_focus, coaching_emphasis, coaching_de_emphasize, injury_watch, variability_note
- `PitcherMetrics.to_obp_comparison_dict()` → `dict[str, float]` mapping OBP metric names to values
