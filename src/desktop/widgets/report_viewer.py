"""Report Viewer tab: displays self-contained HTML reports."""

import tempfile
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.desktop.models import Database


class ReportViewer(QWidget):
    """Tab for displaying HTML analysis reports."""

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self._current_html: str = ""
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(12, 8, 12, 4)

        self.export_btn = QPushButton("Export HTML")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        toolbar.addWidget(self.export_btn)

        self.print_btn = QPushButton("Print")
        self.print_btn.clicked.connect(self._on_print)
        self.print_btn.setEnabled(False)
        toolbar.addWidget(self.print_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Web view
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view, stretch=1)

        # Placeholder
        self.placeholder = QLabel("Select a completed analysis to view its report")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #a0a0b0; font-size: 14px; padding: 40px;")
        layout.addWidget(self.placeholder)

        self.web_view.setVisible(False)

    def load_pitch(self, pitch_id: int):
        """Load and display the report for a given pitch."""
        pitch = self.db.get_pitch(pitch_id)
        if pitch is None or not pitch.report_html:
            self.show_placeholder("No report available for this pitch")
            return
        self._current_html = pitch.report_html
        # Write to temp file and load via URL to avoid QWebEngineView's
        # ~2MB setHtml() limit (reports contain base64 images/charts).
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8",
        )
        tmp.write(self._current_html)
        tmp.close()
        self.web_view.load(QUrl.fromLocalFile(tmp.name))
        self.web_view.setVisible(True)
        self.placeholder.setVisible(False)
        self.export_btn.setEnabled(True)
        self.print_btn.setEnabled(True)

    def show_placeholder(self, text: str = "Select a completed analysis to view its report"):
        self.placeholder.setText(text)
        self.placeholder.setVisible(True)
        self.web_view.setVisible(False)
        self.export_btn.setEnabled(False)
        self.print_btn.setEnabled(False)
        self._current_html = ""

    def _on_export(self):
        if not self._current_html:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "report.html",
            "HTML Files (*.html);;All Files (*)",
        )
        if path:
            Path(path).write_text(self._current_html)

    def _on_print(self):
        if self._current_html:
            self.web_view.page().printToPdf("report.pdf")
