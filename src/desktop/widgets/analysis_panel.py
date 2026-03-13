"""New Analysis tab: video selection, configuration, and analysis execution."""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.desktop.models import AnalysisSession, Database, Player
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

_STAGE_LABELS = {
    "pose_estimation": "Pose Estimation",
    "event_detection": "Event Detection",
    "feature_extraction": "Feature Extraction",
    "benchmarking": "Benchmarking",
    "coaching": "Coaching Report",
    "report_generation": "Report Generation",
}


class AnalysisPanel(QWidget):
    """Tab for configuring and running a new analysis."""

    analysis_completed = pyqtSignal(int)  # session_id

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self._player: Player | None = None
        self._video_path: Path | None = None
        self._worker: AnalysisWorker | None = None
        self._session_id: int | None = None
        self._setup_ui()
        self._update_run_enabled()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

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

    def set_player(self, player: Player | None):
        self._player = player
        self._update_run_enabled()

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

    def _on_backend_changed(self, text: str):
        is_yolo = text == "YOLOv8"
        self.model_combo.setEnabled(is_yolo)

    def _on_conf_changed(self, value: int):
        self.conf_label.setText(f"{value / 100:.2f}")

    def _update_run_enabled(self):
        enabled = self._player is not None and self._video_path is not None
        self.run_btn.setEnabled(enabled and self._worker is None)

    def _on_run(self):
        if self._player is None or self._video_path is None:
            return

        backend = _BACKEND_MAP[self.backend_combo.currentText()]
        model_size = _MODEL_SIZE_MAP[self.model_combo.currentText()]
        confidence = self.conf_slider.value() / 100.0

        # Build pipeline config from player profile
        config = PipelineConfig(
            backend=backend,
            model_size=model_size,
            confidence_threshold=confidence,
            throws=self._player.throws,
            age=int(self._player.age) if self._player.age else None,
            height_inches=self._player.height_inches,
            weight_lbs=self._player.weight_lbs,
        )

        # Create a session record
        session = AnalysisSession(
            player_id=self._player.id,
            video_path=str(self._video_path),
            video_filename=self._video_path.name,
            backend=backend,
            model_size=model_size,
            confidence_threshold=confidence,
            status="running",
        )
        self._session_id = self.db.add_session(session)
        self.db.update_session_running(self._session_id)

        # Update UI
        self.run_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Starting...")

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
        # Serialize metrics for storage
        metrics_dict = {}
        for attr in vars(result.metrics):
            val = getattr(result.metrics, attr)
            if val is not None and not attr.startswith("_"):
                metrics_dict[attr] = val

        self.db.update_session_completed(
            self._session_id,
            report_html=result.report_html,
            results_json=json.dumps(metrics_dict, default=str),
            output_dir=str(result.output_dir) if result.output_dir else "",
        )

        self.progress_bar.setValue(100)
        self.stage_label.setText("Complete!")
        self.stage_label.setStyleSheet("color: #27ae60;")
        self.cancel_btn.setVisible(False)
        self._worker = None
        self._update_run_enabled()
        self.analysis_completed.emit(self._session_id)

    def _on_error(self, message: str):
        if self._session_id is not None:
            self.db.update_session_failed(self._session_id, message)
        self.stage_label.setText(f"Error: {message}")
        self.stage_label.setStyleSheet("color: #e74c3c;")
        self.cancel_btn.setVisible(False)
        self._worker = None
        self._update_run_enabled()
        self.analysis_completed.emit(self._session_id)

    def _on_cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
            self._worker = None
            if self._session_id is not None:
                self.db.update_session_failed(self._session_id, "Cancelled by user")
            self.stage_label.setText("Cancelled")
            self.stage_label.setStyleSheet("color: #f39c12;")
            self.cancel_btn.setVisible(False)
            self._update_run_enabled()
