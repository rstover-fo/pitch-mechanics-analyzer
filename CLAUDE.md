# Pitch Mechanics Analyzer

## Project Overview
A personal-use application for analyzing youth/developing pitcher mechanics from uploaded video.
The pipeline: video upload → pose estimation → biomechanical feature extraction → comparison against
Driveline OpenBiomechanics Project (OBP) benchmarks → AI-generated coaching insights via Claude API.

## Architecture
```
Video Upload → Pose Estimation → Event Detection → Feature Extraction → Benchmark Comparison → Coaching Report
                (YOLOv8-pose)    (foot plant,     (81 OBP-aligned     (OBP distributions    (Claude API
                 or MediaPipe)    MER, release)     metrics)            by playing level)      natural language)
```

## Tech Stack
- **Python 3.11+** — core runtime
- **YOLOv8-pose** (ultralytics) — primary pose estimation from video
- **MediaPipe** — fallback/lightweight pose estimation
- **OpenCV** — video frame extraction and processing
- **NumPy / Pandas / SciPy** — biomechanical computation
- **Anthropic Python SDK** — coaching insight generation
- **Streamlit** — web UI for upload/review (phase 2)
- **Plotly** — interactive visualization of mechanics

## Project Structure
```
src/
  pose/           — Video ingestion, pose estimation, keypoint extraction
  biomechanics/   — Event detection, feature extraction, angle/velocity computation
  coaching/       — Claude API integration for natural-language feedback
  viz/            — Plotting and visualization (joint angles, comparisons)
  utils/          — Shared helpers (video, math, config)
data/
  obp/            — Driveline OpenBiomechanics reference data (POI metrics, metadata)
  uploads/        — User-uploaded videos
  outputs/        — Generated reports and visualizations
scripts/          — CLI entry points and data prep scripts
tests/            — pytest test suite
docs/             — Documentation and biomechanical references
```

## Key Data Source
Driveline OpenBiomechanics Project (OBP): github.com/drivelineresearch/openbiomechanics
- License: Free for personal/research use, NOT commercial
- POI metrics CSV: 81 biomechanical metrics across 411 pitches from 100 pitchers
- Playing levels: college (314), independent (42), high_school (32), milb (23)
- Pitch speeds: 69.5 - 94.4 mph

## OBP POI Metric Categories (key ones for coaching)
### At Foot Plant (_fp suffix)
- rotation_hip_shoulder_separation_fp — hip-shoulder separation angle
- shoulder_horizontal_abduction_fp — arm position relative to torso
- shoulder_external_rotation_fp — layback at foot plant
- elbow_flexion_fp — elbow angle
- torso_anterior_tilt_fp, torso_lateral_tilt_fp — trunk position
- lead_knee_extension_angular_velo_fp — lead leg firmness

### At Ball Release (_br suffix)
- lead_knee_extension_angular_velo_br — lead leg block
- torso_anterior_tilt_br, torso_lateral_tilt_br — trunk at release

### Peak Values (max_ prefix)
- max_shoulder_external_rotation — peak layback
- max_rotation_hip_shoulder_separation — peak hip-shoulder sep
- max_torso_rotational_velo — trunk rotation speed
- max_shoulder_internal_rotational_velo — arm speed
- max_elbow_extension_velo — elbow extension speed

### Kinetics (injury-relevant)
- elbow_varus_moment — UCL stress indicator
- shoulder_internal_rotation_moment — shoulder load

## Development Conventions
- Use type hints everywhere
- Docstrings in Google style
- Config via dataclasses in src/utils/config.py
- All biomechanical angles in degrees, velocities in deg/s
- Coordinate system: Driveline convention (see baseball_pitching/README.md)
- Tests with pytest; name files test_*.py

## Phase Roadmap
1. **Phase 1 (complete)**: OBP benchmark loader + distribution analysis + static comparison
1b. **Phase 1b (complete)**: Youth normalization framework (3-tier: angles/scaling/developmental)
2. **Phase 2**: Video → pose estimation → keypoint extraction pipeline
3. **Phase 3**: Automated event detection (foot plant, MER, ball release)
4. **Phase 4**: Feature extraction aligned to OBP metrics
5. **Phase 5**: Claude API coaching report generation (youth-specific prompting done)
6. **Phase 6**: Streamlit UI for upload → report workflow

## Youth Normalization (key module: src/biomechanics/youth_normalizer.py)
Three-tier framework for adapting OBP adult benchmarks to youth pitchers (ages 10-18):

### Tier 1: Body-Position Angles (direct comparison)
Angles like hip-shoulder separation, elbow flexion, trunk tilt are body-size invariant.
A 12-year-old with 30° hip-shoulder sep is comparable to a college pitcher with 30°.
The IQR is WIDENED by a developmental variability multiplier (2.0x for pre-pubescent,
1.6x for early adolescent) because youth naturally show more pitch-to-pitch variation.

### Tier 2: Allometric Scaling
- Angular velocities: scaled by arm length ratio (youth / OBP reference)
- Joint moments: scaled by bodyweight × height (Nm/(kg·m)) — ASMI convention
- Ground reaction forces: scaled by bodyweight
- Power/energy: scaled by bodyweight × height

### Tier 3: Developmental Stage Interpolation
- CDC growth chart data maps age + height/weight to physical maturity estimate
- "Effective developmental age" adjusts for tall-for-age or small-for-age kids
- Coaching priorities shift by developmental stage:
  - Pre-pubescent (8-11): movement quality ONLY, ignore velocities/forces
  - Early adolescent (12-13): kinematic refinement + consistency
  - Mid-adolescent (14-15): add strength integration, kinetic chain efficiency
  - Late adolescent (16-17): approaching adult-level comparisons

### ASMI Positional Targets (used for youth coaching flags)
- Elbow flexion ~90° at foot contact
- Shoulder abduction ~90° at foot contact
- Shoulder horizontal abduction ~20° at foot contact
- Shoulder ER ~45° at foot contact
- Hip-shoulder separation ~30° at foot plant
- Stride length 75-85% of body height

## Running
```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark analysis on OBP data
python scripts/analyze_benchmarks.py
python scripts/analyze_benchmarks.py --level high_school

# Run youth normalization demo (12-year-old default)
python scripts/youth_demo.py
python scripts/youth_demo.py --age 10 --height 138 --weight 32
python scripts/youth_demo.py --age 14 --height 170 --weight 55

# Process a video (phase 2+)
python scripts/analyze_video.py --video path/to/video.mp4

# Run tests
pytest tests/ -v
```

## Environment Variables
- `ANTHROPIC_API_KEY` — for coaching insight generation (phase 5)
- `OBP_DATA_PATH` — path to OBP data directory (defaults to data/obp/)
