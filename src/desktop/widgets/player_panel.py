"""Left sidebar: player list, profile editor, and physical measurements management."""

import sqlite3
from datetime import date

from PyQt6.QtCore import QDate, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.desktop.models import Database, PhysicalSnapshot, Player


# ---------------------------------------------------------------------------
# Log Measurement dialog
# ---------------------------------------------------------------------------

class LogMeasurementDialog(QDialog):
    """Dialog to add a new physical snapshot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Measurement")
        self.setMinimumWidth(300)

        layout = QFormLayout(self)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())
        layout.addRow("Date:", self.date_edit)

        self.age_spin = QDoubleSpinBox()
        self.age_spin.setRange(1, 99)
        self.age_spin.setDecimals(1)
        self.age_spin.setSuffix(" yrs")
        layout.addRow("Age:", self.age_spin)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 96)
        self.height_spin.setDecimals(1)
        self.height_spin.setSuffix(" in")
        layout.addRow("Height:", self.height_spin)

        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(1, 400)
        self.weight_spin.setDecimals(1)
        self.weight_spin.setSuffix(" lbs")
        layout.addRow("Weight:", self.weight_spin)

        self.arm_spin = QDoubleSpinBox()
        self.arm_spin.setRange(0, 48)
        self.arm_spin.setDecimals(1)
        self.arm_spin.setSuffix(" in")
        self.arm_spin.setSpecialValueText("(optional)")
        layout.addRow("Arm length:", self.arm_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def to_snapshot(self, player_id: int) -> PhysicalSnapshot:
        return PhysicalSnapshot(
            player_id=player_id,
            measured_date=self.date_edit.date().toString("yyyy-MM-dd"),
            age_years=self.age_spin.value(),
            height_inches=self.height_spin.value(),
            weight_lbs=self.weight_spin.value(),
            arm_length_inches=self.arm_spin.value() if self.arm_spin.value() > 0 else None,
        )


# ---------------------------------------------------------------------------
# Player panel
# ---------------------------------------------------------------------------

class PlayerPanel(QWidget):
    """Left sidebar with player list, profile editor, and physical measurements."""

    player_selected = pyqtSignal(object)  # Player or None

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self._current_player: Player | None = None
        self._snapshots: list[PhysicalSnapshot] = []
        self._setup_ui()
        self._load_players()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # -- Header --
        header = QLabel("PLAYERS")
        header.setObjectName("sectionHeader")
        layout.addWidget(header)

        # -- Add button --
        self.add_btn = QPushButton("+ Add Player")
        self.add_btn.clicked.connect(self._on_add_player)
        layout.addWidget(self.add_btn)

        # -- Player list --
        self.player_list = QListWidget()
        self.player_list.currentItemChanged.connect(self._on_player_changed)
        self.player_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.player_list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.player_list, stretch=1)

        # -- Separator --
        sep = QLabel("PROFILE")
        sep.setObjectName("sectionHeader")
        layout.addWidget(sep)

        # -- Profile editor (no age/height/weight — those are in snapshots now) --
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.name_edit = QLineEdit()
        form.addRow("Name:", self.name_edit)

        self.throws_combo = QComboBox()
        self.throws_combo.addItems(["R", "L"])
        form.addRow("Throws:", self.throws_combo)

        self.team_edit = QLineEdit()
        form.addRow("Team:", self.team_edit)

        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(60)
        form.addRow("Notes:", self.notes_edit)

        layout.addLayout(form)

        # -- Save / Delete buttons --
        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("successButton")
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self.save_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setObjectName("dangerButton")
        self.delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.delete_btn)

        layout.addLayout(btn_row)

        # -- Physical Measurements section --
        snap_header = QLabel("PHYSICAL MEASUREMENTS")
        snap_header.setObjectName("sectionHeader")
        layout.addWidget(snap_header)

        self.snap_table = QTableWidget()
        self.snap_table.setColumnCount(4)
        self.snap_table.setHorizontalHeaderLabels(["Date", "Age", "Height", "Weight"])
        self.snap_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.snap_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.snap_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.snap_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.snap_table.verticalHeader().setVisible(False)
        self.snap_table.setMaximumHeight(120)
        self.snap_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.snap_table.customContextMenuRequested.connect(self._on_snap_context_menu)
        layout.addWidget(self.snap_table)

        self.log_measurement_btn = QPushButton("Log Measurement...")
        self.log_measurement_btn.clicked.connect(self._on_log_measurement)
        layout.addWidget(self.log_measurement_btn)

        self._set_editor_enabled(False)

    # ------------------------------------------------------------------
    # Player list management
    # ------------------------------------------------------------------

    def _load_players(self):
        self.player_list.blockSignals(True)
        self.player_list.clear()
        for player in self.db.get_all_players():
            item = QListWidgetItem(player.name)
            item.setData(Qt.ItemDataRole.UserRole, player.id)
            self.player_list.addItem(item)
        self.player_list.blockSignals(False)

    def _on_add_player(self):
        player = Player(name="New Player")
        player.id = self.db.add_player(player)
        self._load_players()
        for i in range(self.player_list.count()):
            item = self.player_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == player.id:
                self.player_list.setCurrentItem(item)
                self.name_edit.setFocus()
                self.name_edit.selectAll()
                break

    def _on_player_changed(self, current: QListWidgetItem, _previous):
        if current is None:
            self._current_player = None
            self._clear_editor()
            self._set_editor_enabled(False)
            self.player_selected.emit(None)
            return
        player_id = current.data(Qt.ItemDataRole.UserRole)
        player = self.db.get_player(player_id)
        if player is None:
            return
        self._current_player = player
        self._populate_editor(player)
        self._refresh_snapshots()
        self._set_editor_enabled(True)
        self.player_selected.emit(player)

    def _populate_editor(self, player: Player):
        self.name_edit.setText(player.name)
        self.throws_combo.setCurrentText(player.throws)
        self.team_edit.setText(player.team)
        self.notes_edit.setPlainText(player.notes)

    def _clear_editor(self):
        self.name_edit.clear()
        self.throws_combo.setCurrentIndex(0)
        self.team_edit.clear()
        self.notes_edit.clear()
        self.snap_table.setRowCount(0)

    def _set_editor_enabled(self, enabled: bool):
        for w in (self.name_edit, self.throws_combo, self.team_edit,
                  self.notes_edit, self.save_btn, self.delete_btn,
                  self.log_measurement_btn):
            w.setEnabled(enabled)

    def _on_save(self):
        if self._current_player is None:
            return
        p = self._current_player
        p.name = self.name_edit.text().strip() or "Unnamed"
        p.throws = self.throws_combo.currentText()
        p.team = self.team_edit.text().strip()
        p.notes = self.notes_edit.toPlainText().strip()
        self.db.update_player(p)
        self._load_players()
        for i in range(self.player_list.count()):
            item = self.player_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == p.id:
                self.player_list.setCurrentItem(item)
                break

    def _on_delete(self):
        if self._current_player is None:
            return
        reply = QMessageBox.question(
            self, "Delete Player",
            f"Delete '{self._current_player.name}' and all their data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.db.delete_player(self._current_player.id)
            self._current_player = None
            self._clear_editor()
            self._set_editor_enabled(False)
            self._load_players()
            self.player_selected.emit(None)

    def _on_context_menu(self, pos):
        item = self.player_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        action = menu.exec(self.player_list.mapToGlobal(pos))
        if action == rename_action:
            self.player_list.setCurrentItem(item)
            self.name_edit.setFocus()
            self.name_edit.selectAll()
        elif action == delete_action:
            self.player_list.setCurrentItem(item)
            self._on_delete()

    # ------------------------------------------------------------------
    # Physical snapshots
    # ------------------------------------------------------------------

    def _refresh_snapshots(self):
        if self._current_player is None:
            self._snapshots = []
            self.snap_table.setRowCount(0)
            return
        self._snapshots = self.db.get_snapshots_for_player(self._current_player.id)
        self.snap_table.setRowCount(len(self._snapshots))
        for row, snap in enumerate(self._snapshots):
            self.snap_table.setItem(row, 0, QTableWidgetItem(snap.measured_date))
            self.snap_table.setItem(row, 1, QTableWidgetItem(f"{snap.age_years:.1f}"))
            self.snap_table.setItem(row, 2, QTableWidgetItem(f"{snap.height_inches:.1f}\""))
            self.snap_table.setItem(row, 3, QTableWidgetItem(f"{snap.weight_lbs:.0f} lbs"))

    def _on_log_measurement(self):
        if self._current_player is None:
            return
        dlg = LogMeasurementDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        snap = dlg.to_snapshot(self._current_player.id)
        if snap.age_years <= 0 or snap.height_inches <= 0 or snap.weight_lbs <= 0:
            QMessageBox.warning(self, "Invalid", "Age, height, and weight are required.")
            return
        try:
            self.db.add_snapshot(snap)
        except sqlite3.IntegrityError:
            reply = QMessageBox.question(
                self, "Measurement Exists",
                f"A measurement for {snap.measured_date} already exists. Update it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.db.update_snapshot_by_date(snap)
            else:
                return
        self._refresh_snapshots()

    def _on_snap_context_menu(self, pos):
        index = self.snap_table.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        if row < 0 or row >= len(self._snapshots):
            return
        snap = self._snapshots[row]
        menu = QMenu(self)
        delete_action = menu.addAction("Delete Measurement")
        action = menu.exec(self.snap_table.viewport().mapToGlobal(pos))
        if action == delete_action:
            reply = QMessageBox.question(
                self, "Delete Measurement",
                f"Delete measurement from {snap.measured_date}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.db.delete_snapshot(snap.id)
                self._refresh_snapshots()

    @property
    def current_player(self) -> Player | None:
        return self._current_player
