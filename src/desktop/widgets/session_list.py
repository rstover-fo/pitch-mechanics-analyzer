"""Analysis History tab: tree view of sessions → pitches for the selected player."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.desktop.models import Database, Pitch, PitchMetric, Session


_STATUS_ICONS = {
    "completed": "\u2713",
    "running": "\u25b6",
    "failed": "\u2717",
    "pending": "\u2026",
}

_STATUS_COLORS = {
    "completed": "#27ae60",
    "running": "#4A90D9",
    "failed": "#e74c3c",
    "pending": "#f39c12",
}


class SessionList(QWidget):
    """Tree widget showing sessions → pitches for a player."""

    session_selected = pyqtSignal(int)  # pitch_id (for viewing report)

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self._player_id: int | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["Date / Pitch #", "Type", "Location / Status", "Pitches", "Highlights"])
        self.tree.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tree.setRootIsDecorated(True)
        self.tree.doubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)

        layout.addWidget(self.tree)

        self.empty_label = QLabel("Select a player to view their analysis history")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #a0a0b0; font-size: 14px;")
        layout.addWidget(self.empty_label)

    def set_player(self, player_id: int | None):
        self._player_id = player_id
        self.refresh()

    def refresh(self):
        self.tree.clear()

        if self._player_id is None:
            self.tree.setVisible(False)
            self.empty_label.setVisible(True)
            return

        self.empty_label.setVisible(False)
        self.tree.setVisible(True)

        sessions = self.db.get_sessions_for_player(self._player_id)

        for session in sessions:
            pitches = self.db.get_pitches_for_session(session.id)
            completed = sum(1 for p in pitches if p.status == "completed")

            # Session row
            session_item = QTreeWidgetItem()
            session_item.setText(0, session.session_date)
            session_item.setText(1, session.session_type)
            session_item.setText(2, session.location)
            session_item.setText(3, f"{completed}/{len(pitches)}")
            session_item.setData(0, Qt.ItemDataRole.UserRole, ("session", session.id))

            # Pitch children
            for pitch in pitches:
                pitch_item = QTreeWidgetItem()
                pitch_item.setText(0, f"Pitch #{pitch.pitch_number or '?'}")
                pitch_item.setText(1, pitch.pitch_type)

                status_icon = _STATUS_ICONS.get(pitch.status, pitch.status)
                pitch_item.setText(2, f"{status_icon} {pitch.status.title()}")

                # Key metric highlight
                if pitch.status == "completed":
                    highlight = self._get_highlight(pitch.id)
                    pitch_item.setText(4, highlight)

                pitch_item.setData(0, Qt.ItemDataRole.UserRole, ("pitch", pitch.id))
                session_item.addChild(pitch_item)

            self.tree.addTopLevelItem(session_item)

        # Expand all by default
        self.tree.expandAll()

    def _get_highlight(self, pitch_id: int) -> str:
        """Get a short metric highlight string for a completed pitch."""
        metrics = self.db.get_metrics_for_pitch(pitch_id)
        # Look for peak ER percentile or a notable metric
        for m in metrics:
            if "external_rotation" in m.metric_name and m.obp_percentile is not None:
                return f"Peak ER: P{m.obp_percentile:.0f}"
        # Fallback: show any metric with a percentile
        for m in metrics:
            if m.obp_percentile is not None:
                return f"{m.display_name}: P{m.obp_percentile:.0f}"
        return ""

    def _on_double_click(self, index):
        item = self.tree.currentItem()
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        kind, obj_id = data
        if kind == "pitch":
            # Check pitch is completed
            pitch = self.db.get_pitch(obj_id)
            if pitch and pitch.status == "completed":
                self.session_selected.emit(obj_id)

    def _on_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        kind, obj_id = data
        menu = QMenu(self)

        if kind == "pitch":
            pitch = self.db.get_pitch(obj_id)
            if pitch and pitch.status == "completed":
                view_action = menu.addAction("View Report")
            else:
                view_action = None
            delete_action = menu.addAction("Delete Pitch")

            action = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if action == view_action and view_action is not None:
                self.session_selected.emit(obj_id)
            elif action == delete_action:
                reply = QMessageBox.question(
                    self, "Delete Pitch",
                    "Delete this pitch and its metrics?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.db.delete_pitch(obj_id)
                    self.refresh()

        elif kind == "session":
            delete_action = menu.addAction("Delete Session")
            action = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if action == delete_action:
                reply = QMessageBox.question(
                    self, "Delete Session",
                    "Delete this session and all its pitches?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.db.delete_session(obj_id)
                    self.refresh()
