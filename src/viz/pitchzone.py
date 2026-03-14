"""
pitchzone_v3.py
---------------
Generates a self-contained HTML string with an inline Three.js 3D scene
for pitcher mechanics visualization (PitchZone v3).

Public API
----------
ZONE_BANDS : dict
_GRADE_COLOR : dict
_GREEN, _YELLOW, _RED : str
calculate_pitchzone_score(grades) -> int
generate_pitchzone_svg(grades, metrics, throws, title, width, height) -> str
    NOTE: Returns HTML (not SVG) — backward-compatible name kept for report_parent.py
generate_pitchzone_html(grades, metrics, throws, title, width, height) -> str
"""

import json
import os
from typing import Optional

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

_GREEN  = "#22c55e"
_YELLOW = "#eab308"
_RED    = "#ef4444"

_GRADE_COLOR: dict[str, str] = {
    "green":  _GREEN,
    "yellow": _YELLOW,
    "red":    _RED,
}

ZONE_BANDS: dict[str, dict] = {
    "shoulder_abduction_fp": {
        "label": "Arm Height",
        "region": "shoulder",
        "excellent": "Arm at perfect height",
        "good": "Nearly there",
        "focus": "Needs work",
    },
    "elbow_flexion_fp": {
        "label": "Elbow Bend",
        "region": "elbow",
        "excellent": "Perfect 'L' shape",
        "good": "Almost right",
        "focus": "Needs attention",
    },
    "torso_anterior_tilt_fp": {
        "label": "Posture",
        "region": "torso",
        "excellent": "Staying tall",
        "good": "Slight lean",
        "focus": "Too much lean",
    },
    "hip_shoulder_separation_fp": {
        "label": "Hip Lead",
        "region": "hip",
        "excellent": "Great separation",
        "good": "Getting there",
        "focus": "Opening together",
    },
    "stride_length_pct_height": {
        "label": "Stride",
        "region": "stride_leg",
        "excellent": "Great reach",
        "good": "Almost enough",
        "focus": "Need more reach",
    },
    "lead_knee_angle_fp": {
        "label": "Front Leg",
        "region": "lead_knee",
        "excellent": "Firm brace",
        "good": "Mostly firm",
        "focus": "Too soft",
    },
}

# Grade word → numeric value for score
_GRADE_VALUES = {"green": 100, "yellow": 65, "red": 30}

# ---------------------------------------------------------------------------
# Ideal ranges per metric: (ideal_value, tolerance)
# Green  if |value − ideal| ≤ tolerance
# Yellow if |value − ideal| ≤ 2 × tolerance
# Red    otherwise
# These match src/viz/overlay.py GRADE_RULES exactly.
# ---------------------------------------------------------------------------
GRADE_RULES: dict[str, tuple[float, float]] = {
    "elbow_flexion_fp":           (90.0,  15.0),
    "shoulder_abduction_fp":      (90.0,  15.0),
    "torso_anterior_tilt_fp":     (30.0,  10.0),
    "hip_shoulder_separation_fp": (30.0,  10.0),
    "stride_length_pct_height":   (80.0,  10.0),
    "lead_knee_angle_fp":         (160.0, 15.0),
}

# ---------------------------------------------------------------------------
# Lazy-loaded Three.js content
# ---------------------------------------------------------------------------

_THREE_JS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "three.min.js")
_THREE_JS_CONTENT: Optional[str] = None


def _load_threejs() -> str:
    global _THREE_JS_CONTENT
    if _THREE_JS_CONTENT is None:
        with open(_THREE_JS_PATH, "r", encoding="utf-8") as fh:
            _THREE_JS_CONTENT = fh.read()
    return _THREE_JS_CONTENT


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

def calculate_pitchzone_score(grades: dict[str, str]) -> int:
    """
    Average grade values: green=100, yellow=65, red=30.
    Unknown or missing keys treated as yellow (65).
    Returns 65 for empty input.
    """
    if not grades:
        return 65
    total = sum(_GRADE_VALUES.get(v, 65) for v in grades.values())
    return round(total / len(grades))


# ---------------------------------------------------------------------------
# HTML / JS generation helpers
# ---------------------------------------------------------------------------

def _grade_word(grade: str, metric_key: str) -> str:
    """Return the display word for a grade."""
    band = ZONE_BANDS.get(metric_key, {})
    mapping = {"green": band.get("excellent", "Excellent"),
               "yellow": band.get("good", "Good"),
               "red": band.get("focus", "Focus")}
    return mapping.get(grade, "—")


def _build_overlay_html(grades: dict[str, str], throws: str, score: int) -> str:
    """Build the HTML overlay: title, labels, score gauge."""

    # Score color
    if score >= 85:
        score_color = _GREEN
    elif score >= 60:
        score_color = _YELLOW
    else:
        score_color = _RED

    # Donut gauge using SVG conic-gradient approach via SVG arc
    r = 28
    circ = 2 * 3.14159 * r
    filled = circ * score / 100
    gauge_svg = f"""
<svg width="80" height="80" viewBox="0 0 80 80" style="transform:rotate(-90deg)">
  <circle cx="40" cy="40" r="{r}" fill="none" stroke="#1e1e2e" stroke-width="10"/>
  <circle cx="40" cy="40" r="{r}" fill="none" stroke="{score_color}" stroke-width="10"
          stroke-dasharray="{filled:.1f} {circ:.1f}" stroke-linecap="round"/>
</svg>"""

    # Score gauge widget (top-right)
    gauge_html = f"""
<div id="score-gauge" style="
  position:absolute; top:12px; right:12px; width:80px;
  display:flex; flex-direction:column; align-items:center; z-index:10;
">
  <div style="position:relative; width:80px; height:80px;">
    {gauge_svg}
    <div style="
      position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
      font-family:system-ui,sans-serif; font-weight:700;
      font-size:18px; color:{score_color};
    ">{score}</div>
  </div>
  <div style="
    font-family:system-ui,sans-serif; font-size:9px; color:#888;
    margin-top:-4px; letter-spacing:0.08em; text-transform:uppercase;
  ">PitchZone</div>
</div>"""

    # Title (top-left)
    hand_label = "RHP" if throws.upper() == "R" else "LHP"
    title_html = f"""
<div id="title-block" style="
  position:absolute; top:14px; left:14px; z-index:10;
  font-family:system-ui,sans-serif;
">
  <div style="font-size:15px; font-weight:700; color:#ffffff; letter-spacing:0.04em;">PitchZone</div>
  <div style="font-size:11px; color:#666; margin-top:1px;">{hand_label}</div>
</div>"""

    # Metric labels — left column and right column
    left_keys  = ["elbow_flexion_fp", "shoulder_abduction_fp", "stride_length_pct_height"]
    right_keys = ["torso_anterior_tilt_fp", "hip_shoulder_separation_fp", "lead_knee_angle_fp"]

    def label_item(key: str, align: str) -> str:
        band  = ZONE_BANDS.get(key, {})
        label = band.get("label", key)
        grade = grades.get(key, "yellow")
        color = _GRADE_COLOR.get(grade, _YELLOW)
        word  = _grade_word(grade, key)
        dot_margin = "margin-right:6px;" if align == "left" else "margin-left:6px;"
        flex_dir   = "row" if align == "left" else "row-reverse"
        text_align = "left" if align == "left" else "right"
        return f"""
<div style="
  display:flex; flex-direction:{flex_dir}; align-items:center;
  margin-bottom:9px;
">
  <div style="
    width:8px; height:8px; border-radius:50%;
    background:{color}; flex-shrink:0; {dot_margin}
  "></div>
  <div style="text-align:{text_align};">
    <div style="font-size:11px; font-weight:600; color:#e0e0e0;">{label}</div>
    <div style="font-size:10px; color:{color}; margin-top:1px;">{word}</div>
  </div>
</div>"""

    left_items  = "".join(label_item(k, "left")  for k in left_keys)
    right_items = "".join(label_item(k, "right") for k in right_keys)

    labels_html = f"""
<div id="labels-left" style="
  position:absolute; left:12px; top:50%; transform:translateY(-50%);
  z-index:10; font-family:system-ui,sans-serif;
">{left_items}</div>
<div id="labels-right" style="
  position:absolute; right:12px; top:50%; transform:translateY(-50%);
  z-index:10; font-family:system-ui,sans-serif;
">{right_items}</div>"""

    return gauge_html + title_html + labels_html


def _build_scene_js(grades: dict[str, str], throws: str, score: int,
                    width: int, height: int,
                    metrics: Optional[dict] = None) -> str:
    """
    Build the Three.js scene JavaScript.
    All geometry is procedural — no external files.
    """
    grades_json = json.dumps(grades)
    rules_json = json.dumps(GRADE_RULES)
    metrics_json = json.dumps(metrics or {})
    mirror = -1 if throws.upper() == "L" else 1  # LHP mirrors X

    return f"""
(function() {{
  // -------------------------------------------------------------------------
  // Data
  // -------------------------------------------------------------------------
  const GRADES = {grades_json};
  const THROWS = "{throws.upper()}";
  const SCORE  = {score};
  const MIRROR = {mirror};  // +1 RHP, -1 LHP
  const GRADE_RULES = {rules_json};  // metric -> [ideal, tolerance]
  const METRICS = {metrics_json};     // metric -> actual value (may be empty)

  // -------------------------------------------------------------------------
  // Renderer
  // -------------------------------------------------------------------------
  const canvas = document.getElementById('pz-canvas');
  const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
  renderer.setSize({width}, {height});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputEncoding = THREE.sRGBEncoding;

  // -------------------------------------------------------------------------
  // Scene & Camera
  // -------------------------------------------------------------------------
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);
  scene.fog = new THREE.FogExp2(0x0a0a0a, 0.18);

  const camera = new THREE.PerspectiveCamera(42, {width}/{height}, 0.1, 50);
  camera.position.set(MIRROR * 1.6, 1.55, 2.6);
  camera.lookAt(0, 0.85, 0);

  // -------------------------------------------------------------------------
  // Lights
  // -------------------------------------------------------------------------
  scene.add(new THREE.AmbientLight(0x333355, 0.9));

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(-2, 3, 2);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(1024, 1024);
  dirLight.shadow.camera.near = 0.1;
  dirLight.shadow.camera.far  = 12;
  scene.add(dirLight);

  const rimLight = new THREE.DirectionalLight(0x8888ff, 0.4);
  rimLight.position.set(0, 1, -3);
  scene.add(rimLight);

  const fillLight = new THREE.PointLight(0x224466, 0.6, 5);
  fillLight.position.set(1.5, 2, 1);
  scene.add(fillLight);

  // -------------------------------------------------------------------------
  // Floor grid
  // -------------------------------------------------------------------------
  const gridHelper = new THREE.GridHelper(6, 24, 0x1a1a2e, 0x1a1a2e);
  gridHelper.position.y = 0;
  scene.add(gridHelper);

  const floorGeo = new THREE.PlaneGeometry(6, 6);
  const floorMat = new THREE.MeshStandardMaterial({{
    color: 0x0d0d1a, roughness: 0.95, metalness: 0.05
  }});
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  // -------------------------------------------------------------------------
  // Materials
  // -------------------------------------------------------------------------
  const bodyMat = new THREE.MeshStandardMaterial({{
    color: 0xb0b0b8, metalness: 0.15, roughness: 0.65
  }});
  const jointMat = new THREE.MeshStandardMaterial({{
    color: 0xc8c8d8, metalness: 0.2, roughness: 0.5
  }});
  const uniformMat = new THREE.MeshStandardMaterial({{
    color: 0x1a3a6a, metalness: 0.1, roughness: 0.7
  }});
  const pantsMat = new THREE.MeshStandardMaterial({{
    color: 0xd8d8e8, metalness: 0.05, roughness: 0.8
  }});

  // -------------------------------------------------------------------------
  // Helper: cylinder between two points
  // -------------------------------------------------------------------------
  function cylinderBetween(p1, p2, radius, mat) {{
    const dir = new THREE.Vector3().subVectors(p2, p1);
    const len = dir.length();
    const geo = new THREE.CylinderGeometry(radius, radius * 0.9, len, 10, 1);
    const mesh = new THREE.Mesh(geo, mat);
    const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    mesh.position.copy(mid);
    mesh.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      dir.clone().normalize()
    );
    mesh.castShadow = true;
    return mesh;
  }}

  function addJoint(pos, radius, mat) {{
    const geo = new THREE.SphereGeometry(radius, 10, 8);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(pos);
    mesh.castShadow = true;
    return mesh;
  }}

  // -------------------------------------------------------------------------
  // Key positions (RHP; MIRROR flips X for LHP)
  // -------------------------------------------------------------------------
  const M = MIRROR;  // shorthand

  // Head & neck
  const headPos     = new THREE.Vector3(0, 1.72, 0);
  const neckBot     = new THREE.Vector3(0, 1.55, 0);

  // Torso
  const shoulderL   = new THREE.Vector3( 0.25 * M, 1.42, 0);   // glove shoulder
  const shoulderR   = new THREE.Vector3(-0.25 * M, 1.42, 0);   // throw shoulder
  const pelvisPos   = new THREE.Vector3(0.04 * M, 0.88, 0.0);

  // Throwing arm — cocked at foot plant
  const throwShoulder = shoulderR.clone();
  const throwElbow    = new THREE.Vector3(-0.45 * M, 1.32, -0.18);
  const throwWrist    = new THREE.Vector3(-0.38 * M, 1.52, -0.06);
  const throwHand     = new THREE.Vector3(-0.33 * M, 1.62,  0.02);

  // Glove arm — extended forward-down
  const gloveShoulder = shoulderL.clone();
  const gloveElbow    = new THREE.Vector3( 0.42 * M, 1.15,  0.22);
  const gloveWrist    = new THREE.Vector3( 0.50 * M, 1.02,  0.35);
  const gloveHand     = new THREE.Vector3( 0.52 * M, 0.96,  0.44);

  // Legs
  const hipR      = new THREE.Vector3(-0.14 * M, 0.90, 0);    // pivot hip (throw side)
  const hipL      = new THREE.Vector3( 0.14 * M, 0.90, 0);    // stride hip (glove side)
  const kneeR     = new THREE.Vector3(-0.18 * M, 0.50, -0.08); // pivot knee (slightly bent)
  const kneeL     = new THREE.Vector3( 0.20 * M, 0.52,  0.45); // lead knee
  const ankleR    = new THREE.Vector3(-0.16 * M, 0.10, -0.06); // pivot ankle
  const ankleL    = new THREE.Vector3( 0.18 * M, 0.08,  0.62); // lead ankle

  scene.add(group);

  // -------------------------------------------------------------------------
  // Render loop
  // -------------------------------------------------------------------------
  let animRunning = true;
  function animate() {{
    if (!animRunning) return;
    requestAnimationFrame(animate);
    group.rotation.y += 0.003;
    renderer.render(scene, camera);
  }}
  animate();
}})();
"""