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
pip install -r requirements.txt

# Set up OBP benchmark data (one-time)
# Copy POI metrics from Driveline's repo into data/obp/
git clone https://github.com/drivelineresearch/openbiomechanics /tmp/obp
cp /tmp/obp/baseball_pitching/data/poi/poi_metrics.csv data/obp/
cp /tmp/obp/baseball_pitching/data/metadata.csv data/obp/

# Run benchmark analysis
python scripts/analyze_benchmarks.py
python scripts/analyze_benchmarks.py --level high_school --output data/outputs/hs_benchmarks.html

# Analyze a video (Phase 2+)
python scripts/analyze_video.py --video path/to/pitch.mp4 --throws R

# Run tests
pytest tests/ -v
```

## Project Structure

```
CLAUDE.md               ← Claude Code project context
src/
  pose/estimator.py     ← Video → keypoints (YOLOv8 / MediaPipe)
  biomechanics/
    benchmarks.py       ← OBP data loading, percentile distributions
    events.py           ← Delivery event detection (foot plant, MER, release)
    features.py         ← Biomechanical metric extraction from keypoints
  coaching/insights.py  ← Claude API coaching report generation
  viz/plots.py          ← Plotly visualizations (radar, gauges, distributions)
  utils/config.py       ← Configuration dataclasses
scripts/
  analyze_benchmarks.py ← CLI: explore OBP benchmark distributions
  analyze_video.py      ← CLI: full video analysis pipeline
tests/
  test_benchmarks.py    ← pytest suite
data/
  obp/                  ← Driveline OBP reference data
  uploads/              ← User videos
  outputs/              ← Generated reports and charts
```

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | OBP benchmark loader + distribution analysis |
| 2 | 🔲 Next | Video → pose estimation → keypoint extraction |
| 3 | 🔲 | Automated event detection from keypoints |
| 4 | 🔲 | Feature extraction aligned to OBP metrics |
| 5 | 🔲 | Claude API coaching report generation |
| 6 | 🔲 | Streamlit UI for upload → report workflow |

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
