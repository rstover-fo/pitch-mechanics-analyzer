# Pitch Mechanics Analyzer

A personal-use application for analyzing youth/developing pitcher mechanics from video, powered by computer vision and benchmarked against the [Driveline OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics).

## What It Does

Upload a video of a pitcher → get biomechanical analysis comparing their mechanics to elite-level benchmarks → receive AI-generated coaching insights.

**Pipeline:**
```
Video → Pose Estimation → Event Detection → Feature Extraction → Benchmark Comparison → Coaching Report
         (YOLOv8-pose)    (foot plant,      (81 OBP-aligned     (percentile ranks     (Claude API
          or MediaPipe)     MER, release)     metrics)            by playing level)     natural language)
```

## Quick Start

```bash
# Clone and install
git clone <this-repo>
cd pitch-mechanics-analyzer
pip install -e .

# Set up OBP benchmark data (one-time)
# Copy POI metrics from Driveline's repo into data/obp/
git clone https://github.com/drivelineresearch/openbiomechanics /tmp/obp
cp /tmp/obp/baseball_pitching/data/poi/poi_metrics.csv data/obp/
cp /tmp/obp/baseball_pitching/data/metadata.csv data/obp/

# Analyze a pitching video
python scripts/validate_pose.py --video path/to/pitch.mp4 --throws R

# With youth profile (age-adjusted benchmarks)
python scripts/validate_pose.py --video path/to/pitch.mp4 --throws R \
    --age 12 --height 60 --weight 95

# Explore benchmark distributions
python scripts/analyze_benchmarks.py
python scripts/analyze_benchmarks.py --level high_school --output data/outputs/hs_benchmarks.html

# Run tests
pytest tests/ -v
```

## Project Structure

```
src/
  pipeline.py               ← PitchAnalysisPipeline: composable end-to-end pipeline
  pose/estimator.py          ← Video → keypoints (YOLOv8 / MediaPipe)
  biomechanics/
    events.py                ← Delivery event detection (leg lift, foot plant, MER, release)
    features.py              ← Biomechanical metric extraction from keypoints
    benchmarks.py            ← OBP data loading, percentile distributions
    validation.py            ← Pipeline output sanity checks
    youth_normalizer.py      ← Age/size-adjusted benchmark comparisons
  coaching/insights.py       ← Claude API & offline coaching report generation
  viz/
    plots.py                 ← Plotly visualizations (radar, gauges)
    report.py                ← HTML report builder
    skeleton.py              ← Skeleton overlay drawing
    trajectories.py          ← Joint trajectory plots
  utils/config.py            ← Configuration dataclasses
scripts/
  validate_pose.py           ← CLI: full diagnostic report generation
  analyze_benchmarks.py      ← CLI: explore OBP benchmark distributions
  eval_events.py             ← Accuracy evaluation vs ground truth
  eval_consistency.py        ← Event detection consistency tests
  label_events.py            ← Frame-by-frame event labeler
  youth_demo.py              ← Youth normalizer demo
tests/                       ← pytest suite
data/
  obp/                       ← Driveline OBP reference data
  prompts/                   ← Coaching prompt templates
  ground_truth/              ← Labeled event data for evaluation
  outputs/                   ← Generated reports and charts
```

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | OBP benchmark loader + distribution analysis |
| 2 | ✅ Complete | Video → pose estimation → keypoint extraction (YOLOv8/MediaPipe) |
| 3 | ✅ Complete | Automated event detection from keypoints (anchor-based approach) |
| 4 | ✅ Complete | Feature extraction aligned to OBP metrics + percentile comparison |
| 5 | ✅ Complete | Claude API coaching report generation with youth normalization |
| 6 | 🔲 Planned | Desktop App for upload → report workflow |

## Data Source

[Driveline OpenBiomechanics Project](https://www.openbiomechanics.org/) — the largest open-source elite-level baseball motion capture dataset. Free for personal and research use under CC BY-NC-SA 4.0.

## Key Metrics Tracked

- **Timing sequence**: pelvis → trunk → shoulder → elbow rotation velocities
- **Arm mechanics**: shoulder external rotation (layback), elbow flexion, arm slot
- **Trunk mechanics**: forward tilt, lateral tilt at foot plant and release
- **Hip-shoulder separation**: the velocity engine
- **Lead leg block**: energy transfer efficiency
- **Joint loading**: elbow varus moment (UCL stress), shoulder IR moment

## License

Personal use only. OBP data is licensed CC BY-NC-SA 4.0 (non-commercial).
