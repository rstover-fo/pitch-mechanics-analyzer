"""Analysis History tab: table of past sessions for the selected player."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.desktop.models import AnalysisSession, Database


_STATUS_ICONS = {
    "completed": "\u2713 Complete",
    "running": "\u25b6 Running",
    "failed": "\u2717 Failed",
    "pending": "\u2026 Pending",
}

_STATUS_COLORS = {
    "completed": "#27ae60",
    "running": "#4A90D9",
    "failed": "#e74c3c",
    "pending": "#f39c12",
}


class SessionList(QWidget):
    """Table widget showing analysis history for a player."""

    session_selected = pyqtSignal(int)  # session_id (for viewing report)

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self._player_id: int | None = None
        self._sessions: list[AnalysisSession] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Date", "Video", "Backend", "Model", "Status"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(self._on_double_click)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)

        layout.addWidget(self.table)

        self.empty_label = QLabel("Select a player to view their analysis history")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #a0a0b0; font-size: 14px;")
        layout.addWidget(self.empty_label)

    def set_player(self, player_id: int | None):
        self._player_id = player_id
        self.refresh()

    def refresh(self):
        if self._player_id is None:
            self._sessions = []
            self.table.setRowCount(0)
            self.table.setVisible(False)
            self.empty_label.setVisible(True)
            return

        self.empty_label.setVisible(False)
        self.table.setVisible(True)

        self._sessions = self.db.get_sessions_for_player(self._player_id)
        self.table.setRowCount(len(self._sessions))

        for row, session in enumerate(self._sessions):
            # Date
            date_str = session.created_at or ""
            if date_str and len(date_str) >= 10:
                date_str = date_str[:10]
            date_item = QTableWidgetItem(date_str)
            date_item.setData(Qt.ItemDataRole.UserRole, session.id)
            self.table.setItem(row, 0, date_item)

            # Video
            self.table.setItem(row, 1, QTableWidgetItem(session.video_filename))

            # Backend
            self.table.setItem(row, 2, QTableWidgetItem(session.backend))

            # Model
            self.table.setItem(row, 3, QTableWidgetItem(session.model_size))

            # Status
            status_text = _STATUS_ICONS.get(session.status, session.status)
            status_item = QTableWidgetItem(status_text)
            color = _STATUS_COLORS.get(session.status, "#e0e0e0")
            status_item.setForeground(Qt.GlobalColor.white)
            self.table.setItem(row, 4, status_item)

    def _on_double_click(self, index):
        row = index.row()
        if 0 <= row < len(self._sessions):
            session = self._sessions[row]
            if session.status == "completed":
                self.session_selected.emit(session.id)

    def _on_context_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        if row < 0 or row >= len(self._sessions):
            return

        session = self._sessions[row]
        menu = QMenu(self)

        if session.status == "completed":
            view_action = menu.addAction("View Report")
        else:
            view_action = None

        delete_action = menu.addAction("Delete Session")

        action = menu.exec(self.table.viewport().mapToGlobal(pos))
        if action == view_action and view_action is not None:
            self.session_selected.emit(session.id)
        elif action == delete_action:
            reply = QMessageBox.question(
                self, "Delete Session",
                f"Delete this analysis session for '{session.video_filename}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.db.delete_session(session.id)
                self.refresh()
