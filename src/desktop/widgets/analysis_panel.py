"""New Analysis tab: video selection, configuration, session/pitch management, and analysis execution."""

import json
from datetime import date
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QDate

from src.desktop.models import (
    Database,
    Pitch,
    PitchEvent,
    PitchMetric,
    Player,
    Session,
)
from src.pipeline import PipelineConfig, PipelineResult, PitchAnalysisPipeline


class AnalysisWorker(QThread):
    """Runs the analysis pipeline in a background thread."""

    progress = pyqtSignal(str, float)   # stage_name, progress_pct
    finished = pyqtSignal(object)       # PipelineResult
    error = pyqtSignal(str)             # error message

    def __init__(self, config: PipelineConfig, video_path: Path):
        super().__init__()
        self.config = config
        self.video_path = video_path

    def run(self):
        try:
            pipeline = PitchAnalysisPipeline(self.config)
            result = pipeline.run(
                self.video_path,
                progress_callback=lambda stage, pct: self.progress.emit(stage, pct),
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# Maps for the combo box display names → pipeline values
_BACKEND_MAP = {"YOLOv8": "yolov8", "MediaPipe": "mediapipe"}
_MODEL_SIZE_MAP = {
    "Nano": "n", "Small": "s", "Medium": "m", "Large": "l", "XLarge": "x",
}
_PITCH_TYPES = ["fastball", "curveball", "changeup", "slider", "cutter", "sinker", "other"]
_SESSION_TYPES = ["bullpen", "game", "lesson", "warmup", "other"]

_STAGE_LABELS = {
    "pose_estimation": "Pose Estimation",
    "event_detection": "Event Detection",
    "feature_extraction": "Feature Extraction",
    "benchmarking": "Benchmarking",
    "coaching": "Coaching Report",
    "report_generation": "Report Generation",
}


# ---------------------------------------------------------------------------
# New Session dialog
# ---------------------------------------------------------------------------

class NewSessionDialog(QDialog):
    """Dialog to create a new analysis session."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Session")
        self.setMinimumWidth(320)

        layout = QFormLayout(self)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        layout.addRow("Date:", self.date_edit)

        self.location_edit = QLineEdit()
        layout.addRow("Location:", self.location_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(_SESSION_TYPES)
        layout.addRow("Type:", self.type_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    @property
    def session_date(self) -> str:
        return self.date_edit.date().toString("yyyy-MM-dd")

    @property
    def location(self) -> str:
        return self.location_edit.text().strip()

    @property
    def session_type(self) -> str:
        return self.type_combo.currentText()


# ---------------------------------------------------------------------------
# Analysis panel
# ---------------------------------------------------------------------------

class AnalysisPanel(QWidget):
    """Tab for configuring and running a new analysis."""

    analysis_completed = pyqtSignal(int)  # pitch_id

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self._player: Player | None = None
        self._video_path: Path | None = None
        self._worker: AnalysisWorker | None = None
        self._pitch_id: int | None = None
        self._current_session: Session | None = None
        self._setup_ui()
        self._update_run_enabled()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # -- Session management --
        session_group = QGroupBox("Session")
        sg_layout = QVBoxLayout(session_group)

        session_row = QHBoxLayout()
        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self._on_session_changed)
        session_row.addWidget(self.session_combo, stretch=1)

        self.new_session_btn = QPushButton("New Session...")
        self.new_session_btn.clicked.connect(self._on_new_session)
        session_row.addWidget(self.new_session_btn)
        sg_layout.addLayout(session_row)

        # Snapshot info label
        self.snapshot_label = QLabel("")
        self.snapshot_label.setStyleSheet("color: #a0a0b0; font-size: 11px;")
        sg_layout.addWidget(self.snapshot_label)

        layout.addWidget(session_group)

        # -- Video selection --
        video_group = QGroupBox("Video")
        vg_layout = QVBoxLayout(video_group)

        btn_row = QHBoxLayout()
        self.select_btn = QPushButton("Select Video...")
        self.select_btn.clicked.connect(self._on_select_video)
        btn_row.addWidget(self.select_btn)

        self.video_label = QLabel("No video selected")
        self.video_label.setStyleSheet("color: #a0a0b0;")
        btn_row.addWidget(self.video_label, stretch=1)
        vg_layout.addLayout(btn_row)

        # Thumbnail preview
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedHeight(180)
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setStyleSheet(
            "background-color: #16213e; border: 1px solid #0f3460; border-radius: 4px;"
        )
        vg_layout.addWidget(self.thumbnail_label)

        layout.addWidget(video_group)

        # -- Configuration --
        config_group = QGroupBox("Configuration")
        cfg_layout = QFormLayout(config_group)

        self.pitch_type_combo = QComboBox()
        self.pitch_type_combo.addItems(_PITCH_TYPES)
        cfg_layout.addRow("Pitch Type:", self.pitch_type_combo)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(list(_BACKEND_MAP.keys()))
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        cfg_layout.addRow("Backend:", self.backend_combo)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(_MODEL_SIZE_MAP.keys()))
        self.model_combo.setCurrentText("Medium")
        cfg_layout.addRow("Model Size:", self.model_combo)

        # Confidence slider
        slider_row = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(30)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        slider_row.addWidget(self.conf_slider, stretch=1)

        self.conf_label = QLabel("0.30")
        self.conf_label.setFixedWidth(40)
        slider_row.addWidget(self.conf_label)
        cfg_layout.addRow("Confidence:", slider_row)

        layout.addWidget(config_group)

        # -- Run button --
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setFixedHeight(40)
        self.run_btn.setStyleSheet("font-size: 15px;")
        self.run_btn.clicked.connect(self._on_run)
        layout.addWidget(self.run_btn)

        # -- Progress section --
        self.progress_group = QGroupBox("Progress")
        pg_layout = QVBoxLayout(self.progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        pg_layout.addWidget(self.progress_bar)

        self.stage_label = QLabel("Ready")
        self.stage_label.setStyleSheet("color: #a0a0b0;")
        pg_layout.addWidget(self.stage_label)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("dangerButton")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setVisible(False)
        pg_layout.addWidget(self.cancel_btn)

        layout.addWidget(self.progress_group)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Player / session management
    # ------------------------------------------------------------------

    def set_player(self, player: Player | None):
        self._player = player
        self._current_session = None
        self._refresh_sessions()
        self._update_run_enabled()

    def _refresh_sessions(self):
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        if self._player is not None:
            sessions = self.db.get_sessions_for_player(self._player.id)
            self.session_combo.addItem("-- Select session --", None)
            for s in sessions:
                label = f"{s.session_date} — {s.session_type} ({s.location})" if s.location else f"{s.session_date} — {s.session_type}"
                self.session_combo.addItem(label, s.id)
        self.session_combo.blockSignals(False)
        self._on_session_changed(0)

    def _on_session_changed(self, index: int):
        session_id = self.session_combo.currentData()
        if session_id is not None:
            self._current_session = self.db.get_session(session_id)
        else:
            self._current_session = None
        self._update_snapshot_label()
        self._update_run_enabled()

    def _update_snapshot_label(self):
        if self._current_session and self._current_session.physical_snapshot_id:
            snap = self.db.get_snapshot(self._current_session.physical_snapshot_id)
            if snap:
                self.snapshot_label.setText(
                    f"Using measurements from {snap.measured_date}: "
                    f"age {snap.age_years:.0f}, {snap.height_inches:.1f}in, {snap.weight_lbs:.0f}lbs"
                )
                return
        if self._current_session and self._player:
            snap = self.db.get_closest_snapshot(self._player.id, self._current_session.session_date)
            if snap:
                self.snapshot_label.setText(
                    f"Closest measurements ({snap.measured_date}): "
                    f"age {snap.age_years:.0f}, {snap.height_inches:.1f}in, {snap.weight_lbs:.0f}lbs"
                )
                return
        self.snapshot_label.setText("No physical measurements on file")

    def _on_new_session(self):
        if self._player is None:
            return
        dlg = NewSessionDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Auto-select closest snapshot
        snap = self.db.get_closest_snapshot(self._player.id, dlg.session_date)
        session = Session(
            player_id=self._player.id,
            session_date=dlg.session_date,
            location=dlg.location,
            session_type=dlg.session_type,
            physical_snapshot_id=snap.id if snap else None,
        )
        session_id = self.db.add_session(session)
        self._refresh_sessions()
        # Select the new session
        for i in range(self.session_combo.count()):
            if self.session_combo.itemData(i) == session_id:
                self.session_combo.setCurrentIndex(i)
                break

    # ------------------------------------------------------------------
    # Video selection
    # ------------------------------------------------------------------

    def _on_select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)",
        )
        if not path:
            return
        self._video_path = Path(path)
        self.video_label.setText(self._video_path.name)
        self._show_thumbnail(self._video_path)
        self._update_run_enabled()

    def _show_thumbnail(self, video_path: Path):
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaledToHeight(
                self.thumbnail_label.height(), Qt.TransformationMode.SmoothTransformation,
            )
            self.thumbnail_label.setPixmap(scaled)
        except Exception:
            self.thumbnail_label.setText("Could not load preview")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _on_backend_changed(self, text: str):
        is_yolo = text == "YOLOv8"
        self.model_combo.setEnabled(is_yolo)

    def _on_conf_changed(self, value: int):
        self.conf_label.setText(f"{value / 100:.2f}")

    def _update_run_enabled(self):
        has_session = self._current_session is not None
        has_video = self._video_path is not None
        has_player = self._player is not None
        self.run_btn.setEnabled(has_player and has_session and has_video and self._worker is None)

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------

    def _on_run(self):
        if self._player is None or self._video_path is None or self._current_session is None:
            return

        backend = _BACKEND_MAP[self.backend_combo.currentText()]
        model_size = _MODEL_SIZE_MAP[self.model_combo.currentText()]
        confidence = self.conf_slider.value() / 100.0

        # Get physical snapshot for youth mode
        snap = None
        if self._current_session.physical_snapshot_id:
            snap = self.db.get_snapshot(self._current_session.physical_snapshot_id)
        elif self._player:
            snap = self.db.get_closest_snapshot(self._player.id, self._current_session.session_date)

        config = PipelineConfig(
            backend=backend,
            model_size=model_size,
            confidence_threshold=confidence,
            throws=self._player.throws,
            age=int(snap.age_years) if snap and snap.age_years else None,
            height_inches=snap.height_inches if snap else None,
            weight_lbs=snap.weight_lbs if snap else None,
        )

        # Create pitch record
        pitch_number = self.db.get_next_pitch_number(self._current_session.id)
        pitch = Pitch(
            session_id=self._current_session.id,
            pitch_number=pitch_number,
            pitch_type=self.pitch_type_combo.currentText(),
            video_path=str(self._video_path),
            video_filename=self._video_path.name,
            backend=backend,
            model_size=model_size,
            confidence_threshold=confidence,
            status="running",
        )
        self._pitch_id = self.db.add_pitch(pitch)
        self.db.update_pitch_running(self._pitch_id)

        # Update UI
        self.run_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Starting...")
        self.stage_label.setStyleSheet("color: #a0a0b0;")

        # Launch worker
        self._worker = AnalysisWorker(config, self._video_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, stage: str, pct: float):
        label = _STAGE_LABELS.get(stage, stage)
        self.stage_label.setText(f"{label}...")
        self.progress_bar.setValue(int(pct * 100))

    def _on_finished(self, result: PipelineResult):
        pitch_id = self._pitch_id

        # Store report on the pitch record
        self.db.update_pitch_completed(
            pitch_id,
            report_html=result.report_html,
            coaching_report=result.coaching_report,
            output_dir=str(result.output_dir) if result.output_dir else "",
        )

        # Parse events → pitch_events
        events = result.events
        durations = events.phase_durations()
        pe = PitchEvent(
            pitch_id=pitch_id,
            leg_lift_frame=events.leg_lift_apex,
            foot_plant_frame=events.foot_plant,
            max_er_frame=events.max_external_rotation,
            ball_release_frame=events.ball_release,
            max_ir_frame=events.max_internal_rotation,
            windup_to_fp=durations.get("windup_to_foot_plant"),
            fp_to_mer=durations.get("foot_plant_to_mer"),
            mer_to_release=durations.get("mer_to_release"),
            arm_cocking=durations.get("arm_cocking"),
            arm_acceleration=durations.get("arm_acceleration"),
            arm_deceleration=durations.get("arm_deceleration"),
        )
        self.db.add_pitch_event(pe)

        # Parse metrics + benchmarks → pitch_metrics
        # Build a lookup from benchmark_comparisons
        bench_lookup: dict[str, dict] = {}
        for bc in (result.benchmark_comparisons or []):
            bench_lookup[bc.get("metric", "")] = bc

        youth_lookup: dict[str, dict] = {}
        for yc in (result.youth_comparisons or []):
            if isinstance(yc, dict):
                youth_lookup[yc.get("metric", "")] = yc

        # Iterate over all metric attributes on result.metrics
        for attr in vars(result.metrics):
            val = getattr(result.metrics, attr)
            if val is None or attr.startswith("_") or not isinstance(val, (int, float)):
                continue

            bc = bench_lookup.get(attr, {})
            yc = youth_lookup.get(attr, {})

            pm = PitchMetric(
                pitch_id=pitch_id,
                metric_name=attr,
                display_name=bc.get("display_name", attr.replace("_", " ").title()),
                value=float(val),
                unit=bc.get("unit", "deg"),
                obp_median=bc.get("benchmark_median"),
                obp_percentile=bc.get("percentile_rank"),
                obp_flag=bc.get("flag"),
                youth_p25=yc.get("p25"),
                youth_p50=yc.get("p50"),
                youth_p75=yc.get("p75"),
                youth_percentile=yc.get("percentile"),
                youth_flag=yc.get("flag"),
            )
            self.db.add_pitch_metric(pm)

        self.progress_bar.setValue(100)
        self.stage_label.setText("Complete!")
        self.stage_label.setStyleSheet("color: #27ae60;")
        self.cancel_btn.setVisible(False)
        self._worker = None
        self._update_run_enabled()
        self.analysis_completed.emit(pitch_id)

    def _on_error(self, message: str):
        if self._pitch_id is not None:
            self.db.update_pitch_failed(self._pitch_id, message)
        self.stage_label.setText(f"Error: {message}")
        self.stage_label.setStyleSheet("color: #e74c3c;")
        self.cancel_btn.setVisible(False)
        self._worker = None
        self._update_run_enabled()
        if self._pitch_id is not None:
            self.analysis_completed.emit(self._pitch_id)

    def _on_cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
            self._worker = None
            if self._pitch_id is not None:
                self.db.update_pitch_failed(self._pitch_id, "Cancelled by user")
            self.stage_label.setText("Cancelled")
            self.stage_label.setStyleSheet("color: #f39c12;")
            self.cancel_btn.setVisible(False)
            self._update_run_enabled()
