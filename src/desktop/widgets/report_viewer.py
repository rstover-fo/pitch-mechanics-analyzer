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

_TAB_BASE_STYLE = (
    "padding: 6px 16px; border: none; border-radius: 4px; font-size: 13px;"
)
_TAB_ACTIVE_STYLE = f"{_TAB_BASE_STYLE} background: #4A90D9; color: #fff;"
_TAB_INACTIVE_STYLE = f"{_TAB_BASE_STYLE} background: #2a2a3e; color: #a0a0b0;"


class ReportViewer(QWidget):
    """Tab for displaying HTML analysis reports."""

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self._current_html: str = ""
        self._technical_html: str = ""
        self._parent_html: str = ""
        self._active_tab: str = ""  # "technical" or "parent"
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(12, 8, 12, 4)

        # Report toggle buttons
        self.tab_technical = QPushButton("Technical Report")
        self.tab_technical.clicked.connect(lambda: self._switch_tab("technical"))
        self.tab_technical.setVisible(False)
        toolbar.addWidget(self.tab_technical)

        self.tab_parent = QPushButton("Parent Report")
        self.tab_parent.clicked.connect(lambda: self._switch_tab("parent"))
        self.tab_parent.setVisible(False)
        toolbar.addWidget(self.tab_parent)

        toolbar.addStretch()

        self.export_btn = QPushButton("Export HTML")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        toolbar.addWidget(self.export_btn)

        self.print_btn = QPushButton("Print")
        self.print_btn.clicked.connect(self._on_print)
        self.print_btn.setEnabled(False)
        toolbar.addWidget(self.print_btn)

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

        self._technical_html = pitch.report_html
        self._parent_html = getattr(pitch, "parent_report_html", "") or ""

        has_both = bool(self._technical_html and self._parent_html)
        self.tab_technical.setVisible(has_both)
        self.tab_parent.setVisible(has_both)

        # Default to parent report if available, otherwise technical
        if self._parent_html:
            self._switch_tab("parent")
        else:
            self._switch_tab("technical")

    def _switch_tab(self, tab: str):
        """Switch between technical and parent report views."""
        self._active_tab = tab
        if tab == "parent":
            self._current_html = self._parent_html
            self.tab_parent.setStyleSheet(_TAB_ACTIVE_STYLE)
            self.tab_technical.setStyleSheet(_TAB_INACTIVE_STYLE)
        else:
            self._current_html = self._technical_html
            self.tab_technical.setStyleSheet(_TAB_ACTIVE_STYLE)
            self.tab_parent.setStyleSheet(_TAB_INACTIVE_STYLE)

        self._display_html(self._current_html)

    def _display_html(self, html: str):
        """Write HTML to a temp file and display in the web view."""
        # Write to temp file and load via URL to avoid QWebEngineView's
        # ~2MB setHtml() limit (reports contain base64 images/charts).
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8",
        )
        tmp.write(html)
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
        self.tab_technical.setVisible(False)
        self.tab_parent.setVisible(False)
        self.export_btn.setEnabled(False)
        self.print_btn.setEnabled(False)
        self._current_html = ""
        self._technical_html = ""
        self._parent_html = ""
        self._active_tab = ""

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
