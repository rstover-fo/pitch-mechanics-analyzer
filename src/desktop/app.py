"""Main application window for the Pitch Mechanics Analyzer."""

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QWidget,
)

from src.desktop.models import Database
from src.desktop.widgets.analysis_panel import AnalysisPanel
from src.desktop.widgets.player_panel import PlayerPanel
from src.desktop.widgets.report_viewer import ReportViewer
from src.desktop.widgets.session_list import SessionList


_STYLES_PATH = Path(__file__).parent / "resources" / "styles.qss"


class MainWindow(QMainWindow):
    """Main application window with left sidebar + right tabbed content area."""

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.setWindowTitle("Pitch Mechanics Analyzer")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        # Central splitter: left sidebar | right tabs
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # Left sidebar: player panel
        self.player_panel = PlayerPanel(self.db)
        self.player_panel.setMinimumWidth(240)
        self.player_panel.setMaximumWidth(350)
        splitter.addWidget(self.player_panel)

        # Right side: tab widget
        self.tabs = QTabWidget()

        self.analysis_panel = AnalysisPanel(self.db)
        self.tabs.addTab(self.analysis_panel, "New Analysis")

        self.session_list = SessionList(self.db)
        self.tabs.addTab(self.session_list, "Analysis History")

        self.report_viewer = ReportViewer(self.db)
        self.tabs.addTab(self.report_viewer, "Report Viewer")

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def _connect_signals(self):
        # Player selection updates analysis panel and session list
        self.player_panel.player_selected.connect(self._on_player_selected)

        # Analysis completion refreshes session list
        self.analysis_panel.analysis_completed.connect(self._on_analysis_completed)

        # Double-clicking a session opens the report
        self.session_list.session_selected.connect(self._on_view_report)

    def _on_player_selected(self, player):
        self.analysis_panel.set_player(player)
        if player is not None:
            self.session_list.set_player(player.id)
        else:
            self.session_list.set_player(None)
        self.report_viewer.show_placeholder()

    def _on_analysis_completed(self, pitch_id: int):
        # Refresh the history tab
        self.session_list.refresh()
        # Auto-show the report
        pitch = self.db.get_pitch(pitch_id)
        if pitch and pitch.status == "completed":
            self.report_viewer.load_pitch(pitch_id)
            self.tabs.setCurrentWidget(self.report_viewer)

    def _on_view_report(self, pitch_id: int):
        self.report_viewer.load_pitch(pitch_id)
        self.tabs.setCurrentWidget(self.report_viewer)


class PitchAnalyzerApp(QApplication):
    """Application entry point — sets up the database and main window."""

    def __init__(self, argv: list[str]):
        super().__init__(argv)
        self.setApplicationName("Pitch Mechanics Analyzer")

        # Load stylesheet
        if _STYLES_PATH.exists():
            self.setStyleSheet(_STYLES_PATH.read_text())

        # Initialize database
        self.db = Database()
        self.db.initialize()

        # Create and show main window
        self.window = MainWindow(self.db)
        self.window.show()
