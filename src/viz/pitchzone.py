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
  const GRADES = {grades_json};
  const THROWS = "{throws.upper()}";
  const SCORE  = {score};
  const MIRROR = {mirror};
  const GRADE_RULES = {rules_json};
  const METRICS = {metrics_json};

  const canvas = document.getElementById('pz-canvas');
  const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
  renderer.setSize({width}, {height});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputEncoding = THREE.sRGBEncoding;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);
  scene.fog = new THREE.FogExp2(0x0a0a0a, 0.18);

  const camera = new THREE.PerspectiveCamera(42, {width}/{height}, 0.1, 50);
  camera.position.set(MIRROR * 1.6, 1.55, 2.6);
  camera.lookAt(0, 0.85, 0);

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

  const gridHelper = new THREE.GridHelper(6, 24, 0x1a1a2e, 0x1a1a2e);
  gridHelper.position.y = 0;
  scene.add(gridHelper);

  const floorGeo = new THREE.PlaneGeometry(6, 6);
  const floorMat = new THREE.MeshStandardMaterial({{ color: 0x0d0d1a, roughness: 0.95, metalness: 0.05 }});
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const bodyMat  = new THREE.MeshStandardMaterial({{ color: 0xb0b0b8, metalness: 0.15, roughness: 0.65 }});
  const jointMat = new THREE.MeshStandardMaterial({{ color: 0xc8c8d8, metalness: 0.2,  roughness: 0.5  }});
  const uniformMat = new THREE.MeshStandardMaterial({{ color: 0x1a3a6a, metalness: 0.1, roughness: 0.7  }});
  const pantsMat = new THREE.MeshStandardMaterial({{ color: 0xd8d8e8, metalness: 0.05, roughness: 0.8  }});

  function cylinderBetween(p1, p2, radius, mat) {{
    const dir = new THREE.Vector3().subVectors(p2, p1);
    const len = dir.length();
    const geo = new THREE.CylinderGeometry(radius, radius * 0.9, len, 10, 1);
    const mesh = new THREE.Mesh(geo, mat);
    const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    mesh.position.copy(mid);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
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

  const M = MIRROR;
  const headPos     = new THREE.Vector3(0, 1.72, 0);
  const neckBot     = new THREE.Vector3(0, 1.55, 0);
  const shoulderL   = new THREE.Vector3( 0.25 * M, 1.42, 0);
  const shoulderR   = new THREE.Vector3(-0.25 * M, 1.42, 0);
  const pelvisPos   = new THREE.Vector3(0.04 * M, 0.88, 0.0);
  const throwShoulder = shoulderR.clone();
  const throwElbow    = new THREE.Vector3(-0.45 * M, 1.32, -0.18);
  const throwWrist    = new THREE.Vector3(-0.38 * M, 1.52, -0.06);
  const throwHand     = new THREE.Vector3(-0.33 * M, 1.62,  0.02);
  const gloveShoulder = shoulderL.clone();
  const gloveElbow    = new THREE.Vector3( 0.42 * M, 1.15,  0.22);
  const gloveWrist    = new THREE.Vector3( 0.50 * M, 1.02,  0.35);
  const gloveHand     = new THREE.Vector3( 0.52 * M, 0.96,  0.44);
  const hipR      = new THREE.Vector3(-0.14 * M, 0.90, 0);
  const hipL      = new THREE.Vector3( 0.14 * M, 0.90, 0);
  const kneeR     = new THREE.Vector3(-0.18 * M, 0.50, -0.08);
  const kneeL     = new THREE.Vector3( 0.20 * M, 0.52,  0.45);
  const ankleR    = new THREE.Vector3(-0.16 * M, 0.10, -0.06);
  const ankleL    = new THREE.Vector3( 0.18 * M, 0.08,  0.62);

  const group = new THREE.Group();

  const headGeo  = new THREE.SphereGeometry(0.115, 14, 12);
  const headMesh = new THREE.Mesh(headGeo, bodyMat);
  headMesh.position.copy(headPos);
  headMesh.scale.set(1, 1.08, 0.92);
  headMesh.castShadow = true;
  group.add(headMesh);

  group.add(cylinderBetween(neckBot, headPos, 0.055, bodyMat));

  const torsoGeo = new THREE.CylinderGeometry(0.14, 0.12, 0.52, 10);
  const torsoMesh = new THREE.Mesh(torsoGeo, uniformMat);
  torsoMesh.position.set(0, 1.18, 0);
  torsoMesh.castShadow = true;
  group.add(torsoMesh);

  const chestGeo = new THREE.SphereGeometry(0.18, 12, 8);
  const chestMesh = new THREE.Mesh(chestGeo, uniformMat);
  chestMesh.position.set(0, 1.38, 0);
  chestMesh.scale.set(1.3, 0.55, 0.85);
  chestMesh.castShadow = true;
  group.add(chestMesh);

  const hipsGeo  = new THREE.CylinderGeometry(0.13, 0.12, 0.22, 10);
  const hipsMesh = new THREE.Mesh(hipsGeo, pantsMat);
  hipsMesh.position.set(0.02 * M, 0.94, 0.0);
  hipsMesh.castShadow = true;
  group.add(hipsMesh);

  group.add(cylinderBetween(throwShoulder, throwElbow, 0.052, bodyMat));
  group.add(addJoint(throwElbow, 0.058, jointMat));
  group.add(cylinderBetween(throwElbow, throwWrist, 0.044, bodyMat));
  group.add(addJoint(throwWrist, 0.046, jointMat));
  group.add(cylinderBetween(throwWrist, throwHand, 0.036, bodyMat));

  group.add(cylinderBetween(gloveShoulder, gloveElbow, 0.052, bodyMat));
  group.add(addJoint(gloveElbow, 0.056, jointMat));
  group.add(cylinderBetween(gloveElbow, gloveWrist, 0.044, bodyMat));
  group.add(addJoint(gloveWrist, 0.044, jointMat));
  group.add(cylinderBetween(gloveWrist, gloveHand, 0.036, bodyMat));

  const gloveGeo = new THREE.BoxGeometry(0.12, 0.10, 0.06);
  const gloveMesh = new THREE.Mesh(gloveGeo, new THREE.MeshStandardMaterial({{ color: 0x6b3a1f, roughness: 0.9 }}));
  gloveMesh.position.copy(gloveHand);
  gloveMesh.rotation.z = 0.3 * M;
  gloveMesh.castShadow = true;
  group.add(gloveMesh);

  group.add(addJoint(throwShoulder, 0.07, jointMat));
  group.add(addJoint(gloveShoulder, 0.065, jointMat));

  group.add(cylinderBetween(hipR, kneeR, 0.072, pantsMat));
  group.add(addJoint(kneeR, 0.072, jointMat));
  group.add(cylinderBetween(kneeR, ankleR, 0.060, pantsMat));
  group.add(addJoint(ankleR, 0.060, jointMat));

  group.add(cylinderBetween(hipL, kneeL, 0.072, pantsMat));
  group.add(addJoint(kneeL, 0.075, jointMat));
  group.add(cylinderBetween(kneeL, ankleL, 0.062, pantsMat));
  group.add(addJoint(ankleL, 0.062, jointMat));

  function addFoot(ankle, forward) {{
    const geo  = new THREE.BoxGeometry(0.08, 0.06, 0.20);
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({{ color: 0x202030, roughness: 0.85 }}));
    mesh.position.set(ankle.x, ankle.y - 0.04, ankle.z + (forward ? 0.05 : -0.02));
    mesh.castShadow = true;
    return mesh;
  }}
  group.add(addFoot(ankleR, false));
  group.add(addFoot(ankleL, true));
  group.add(addJoint(hipR, 0.072, jointMat));
  group.add(addJoint(hipL, 0.068, jointMat));
  scene.add(group);

  function colorHex(grade) {{
    if (grade === 'green')  return 0x22c55e;
    if (grade === 'yellow') return 0xeab308;
    return 0xef4444;
  }}

  function createRingSectorGeo(startAngle, endAngle, innerR, outerR, segments) {{
    segments = segments || 20;
    const geo = new THREE.BufferGeometry();
    const verts = [];
    const step = (endAngle - startAngle) / segments;
    for (let i = 0; i < segments; i++) {{
      const a0 = startAngle + step * i;
      const a1 = startAngle + step * (i + 1);
      const ci0 = Math.cos(a0), si0 = Math.sin(a0);
      const ci1 = Math.cos(a1), si1 = Math.sin(a1);
      verts.push(ci0*innerR, si0*innerR, 0, ci0*outerR, si0*outerR, 0, ci1*outerR, si1*outerR, 0);
      verts.push(ci0*innerR, si0*innerR, 0, ci1*outerR, si1*outerR, 0, ci1*innerR, si1*innerR, 0);
    }}
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    geo.computeVertexNormals();
    return geo;
  }}

  function orientQuat(refDir, normal) {{
    const xAxis = refDir.clone().normalize();
    const zAxis = normal.clone().normalize();
    const yAxis = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();
    const m = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
    return new THREE.Quaternion().setFromRotationMatrix(m);
  }}

  function addRangeBands(origin, refDir, normal, metricKey, rayLen, arcSpan, rangeMin, rangeMax) {{
    const rule = GRADE_RULES[metricKey];
    if (!rule) return;
    const ideal = rule[0], tol = rule[1];
    const grade = GRADES[metricKey] || 'yellow';
    const actualVal = METRICS[metricKey];
    const degRange = rangeMax - rangeMin;
    function degToArc(deg) {{ return ((deg - rangeMin) / degRange) * arcSpan; }}
    function clampArc(a) {{ return Math.max(0, Math.min(arcSpan, a)); }}
    const greenStart = clampArc(degToArc(ideal - tol));
    const greenEnd   = clampArc(degToArc(ideal + tol));
    const yellowStart = clampArc(degToArc(ideal - 2*tol));
    const yellowEnd   = clampArc(degToArc(ideal + 2*tol));
    const quat = orientQuat(refDir, normal);
    function addBand(aStart, aEnd, color, opacity, emI, inner, outer) {{
      if (aEnd - aStart < 0.01) return;
      const geo = createRingSectorGeo(aStart, aEnd, inner, outer, 20);
      const mat = new THREE.MeshStandardMaterial({{ color: color, transparent: true, opacity: opacity, side: THREE.DoubleSide, depthWrite: false, emissive: color, emissiveIntensity: emI }});
      const mesh = new THREE.Mesh(geo, mat);
      mesh.quaternion.copy(quat);
      mesh.position.copy(origin);
      scene.add(mesh);
    }}
    addBand(0, arcSpan, 0x333344, 0.08, 0.05, rayLen * 0.05, rayLen);
    if (yellowStart > 0.01) addBand(0, yellowStart, 0xef4444, 0.22, 0.18, rayLen * 0.1, rayLen * 0.92);
    if (arcSpan - yellowEnd > 0.01) addBand(yellowEnd, arcSpan, 0xef4444, 0.22, 0.18, rayLen * 0.1, rayLen * 0.92);
    if (greenStart - yellowStart > 0.01) addBand(yellowStart, greenStart, 0xeab308, 0.28, 0.22, rayLen * 0.1, rayLen * 0.95);
    if (yellowEnd - greenEnd > 0.01) addBand(greenEnd, yellowEnd, 0xeab308, 0.28, 0.22, rayLen * 0.1, rayLen * 0.95);
    addBand(greenStart, greenEnd, 0x22c55e, 0.38, 0.40, rayLen * 0.1, rayLen);
    const idealArc = clampArc(degToArc(ideal));
    const idealDir = new THREE.Vector3(Math.cos(idealArc), Math.sin(idealArc), 0).applyQuaternion(quat);
    const idealPts = [origin.clone(), origin.clone().addScaledVector(idealDir, rayLen * 1.05)];
    const idealLineMat = new THREE.LineDashedMaterial({{ color: 0x22c55e, transparent: true, opacity: 0.5, dashSize: 0.02, gapSize: 0.015 }});
    const idealLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(idealPts), idealLineMat);
    idealLine.computeLineDistances();
    scene.add(idealLine);
    if (actualVal !== undefined && actualVal !== null) {{
      const actArc = clampArc(degToArc(actualVal));
      const actDir = new THREE.Vector3(Math.cos(actArc), Math.sin(actArc), 0).applyQuaternion(quat);
      const actColor = colorHex(grade);
      const actPts = [origin.clone(), origin.clone().addScaledVector(actDir, rayLen * 1.12)];
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(actPts), new THREE.LineBasicMaterial({{ color: actColor, transparent: true, opacity: 0.9 }})));
      const diamGeo = new THREE.OctahedronGeometry(0.028);
      const diam = new THREE.Mesh(diamGeo, new THREE.MeshBasicMaterial({{ color: actColor, transparent: true, opacity: 0.95 }}));
      diam.position.copy(origin.clone().addScaledVector(actDir, rayLen * 1.12));
      scene.add(diam);
      const glow = new THREE.Mesh(new THREE.SphereGeometry(0.04, 8, 6), new THREE.MeshBasicMaterial({{ color: actColor, transparent: true, opacity: 0.3 }}));
      glow.position.copy(diam.position);
      scene.add(glow);
    }}
    const edgeMat = new THREE.LineBasicMaterial({{ color: 0x555566, transparent: true, opacity: 0.3 }});
    const edgeDir0 = new THREE.Vector3(1, 0, 0).applyQuaternion(quat);
    const edgeDirN = new THREE.Vector3(Math.cos(arcSpan), Math.sin(arcSpan), 0).applyQuaternion(quat);
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin.clone(), origin.clone().addScaledVector(edgeDir0, rayLen * 0.9)]), edgeMat));
    scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin.clone(), origin.clone().addScaledVector(edgeDirN, rayLen * 0.9)]), edgeMat));
  }}

  // Zone 1: Arm Height
  {{ const origin = throwShoulder.clone(); const refDir = new THREE.Vector3(-M * 0.85, 0.50, -0.1).normalize(); const normal = new THREE.Vector3(0, 0, M * 1).normalize(); addRangeBands(origin, refDir, normal, 'shoulder_abduction_fp', 0.32, Math.PI * 0.55, 40, 140); }}
  // Zone 2: Elbow Bend
  {{ const origin = throwElbow.clone(); const upperArmDir = new THREE.Vector3().subVectors(throwShoulder, throwElbow).normalize(); const forearmDir = new THREE.Vector3().subVectors(throwWrist, throwElbow).normalize(); const normal = new THREE.Vector3().crossVectors(upperArmDir, forearmDir).normalize(); if (normal.length() < 0.1) normal.set(0, 0, 1); addRangeBands(origin, forearmDir, normal, 'elbow_flexion_fp', 0.24, Math.PI * 0.55, 40, 140); }}
  // Zone 3: Posture
  {{ const torsoMid = new THREE.Vector3(0, 1.18, 0); const vertical = new THREE.Vector3(0, 1, 0); const leanDir = new THREE.Vector3(0, 0.5, 1).normalize(); const normal = new THREE.Vector3().crossVectors(vertical, leanDir).normalize(); if (normal.length() < 0.1) normal.set(1, 0, 0); addRangeBands(torsoMid, vertical, normal, 'torso_anterior_tilt_fp', 0.32, Math.PI * 0.33, 0, 60); }}
  // Zone 4: Hip-Shoulder Separation
  {{
    const shoulderLine = new THREE.Vector3().subVectors(shoulderR, shoulderL); shoulderLine.y = 0; shoulderLine.normalize();
    const shoulderCenter = new THREE.Vector3().addVectors(shoulderL, shoulderR).multiplyScalar(0.5); shoulderCenter.y = 1.42;
    const hipLine = new THREE.Vector3().subVectors(hipR, hipL); hipLine.y = 0; hipLine.normalize();
    const hipCenter = new THREE.Vector3().addVectors(hipR, hipL).multiplyScalar(0.5); hipCenter.y = 0.90;
    const upAxis = new THREE.Vector3(0, 1, 0);
    const shoulderNormal = new THREE.Vector3().crossVectors(shoulderLine, upAxis).normalize();
    const hipNormal = new THREE.Vector3().crossVectors(hipLine, upAxis).normalize();
    addRangeBands(shoulderCenter, shoulderLine, shoulderNormal, 'hip_shoulder_separation_fp', 0.30, Math.PI * 0.44, 0, 60);
    addRangeBands(hipCenter, hipLine, hipNormal, 'hip_shoulder_separation_fp', 0.28, Math.PI * 0.44, 0, 60);
  }}
  // Zone 5: Stride
  {{ const strideMid = new THREE.Vector3().addVectors(ankleR, ankleL).multiplyScalar(0.5); strideMid.y = 0.09; const strideDir = new THREE.Vector3().subVectors(ankleL, ankleR).normalize(); const up = new THREE.Vector3(0, 1, 0); const normal = new THREE.Vector3().crossVectors(strideDir, up).normalize(); addRangeBands(strideMid, strideDir, normal, 'stride_length_pct_height', 0.28, Math.PI * 0.44, 40, 120); }}
  // Zone 6: Lead Knee
  {{ const origin = kneeL.clone(); const thighDir = new THREE.Vector3().subVectors(kneeL, hipL).normalize(); const shinDir = new THREE.Vector3().subVectors(ankleL, kneeL).normalize(); const normal = new THREE.Vector3().crossVectors(thighDir, shinDir).normalize(); if (normal.length() < 0.1) normal.set(M, 0, 0); addRangeBands(origin, shinDir, normal, 'lead_knee_angle_fp', 0.22, Math.PI * 0.55, 100, 180); }}

  let animRunning = true;
  function animate() {{
    if (!animRunning) return;
    requestAnimationFrame(animate);
    group.rotation.y += 0.003;
    renderer.render(scene, camera);
  }}
  animate();
}})();
"""  # noqa: E501


def generate_pitchzone_html(
    grades: dict[str, str],
    metrics: Optional[dict] = None,
    throws: str = "R",
    title: str = "PitchZone",
    width: int = 600,
    height: int = 520,
) -> str:
    """Generate a standalone HTML page with the PitchZone Three.js 3D visualization.

    Args:
        grades:  mapping metric_key -> grade string ("green", "yellow", "red").
        metrics: mapping metric_key -> actual float value (optional).
        throws:  "R" (right-handed pitcher) or "L" (left-handed).
        title:   page title text.
        width:   canvas pixel width.
        height:  canvas pixel height.

    Returns:
        Complete self-contained HTML string (DOCTYPE included).
    """
    threejs = _load_threejs()
    score = calculate_pitchzone_score(grades)
    overlay_html = _build_overlay_html(grades, throws, score)
    scene_js = _build_scene_js(grades, throws, score, width, height, metrics)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0a0a; overflow:hidden; }}
  #pz-wrapper {{ position:relative; width:{width}px; height:{height}px; }}
  canvas {{ display:block; }}
</style>
</head>
<body>
<div id="pz-wrapper">
  <canvas id="pz-canvas" width="{width}" height="{height}"></canvas>
  {overlay_html}
</div>
<script>{threejs}</script>
<script>{scene_js}</script>
</body>
</html>"""


def generate_pitchzone_svg(
    grades: dict[str, str],
    metrics: Optional[dict] = None,
    throws: str = "R",
    title: str = "PitchZone",
    width: int = 600,
    height: int = 520,
) -> str:
    """Backward-compatible wrapper: returns an <iframe srcdoc=...> wrapping the HTML.

    This preserves the old call signature used by report_parent.py.
    """
    from html import escape as _escape
    html = generate_pitchzone_html(grades, metrics=metrics, throws=throws,
                                   title=title, width=width, height=height)
    safe = _escape(html, quote=True)
    return (f'<iframe srcdoc="{safe}" '
            f'width="{width}" height="{height}" '
            f'frameborder="0" scrolling="no" '
            f'sandbox="allow-scripts"></iframe>')
