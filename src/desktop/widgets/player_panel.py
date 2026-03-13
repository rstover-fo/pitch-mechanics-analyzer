"""Left sidebar: player list and profile editor."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.desktop.models import Database, Player


class PlayerPanel(QWidget):
    """Left sidebar with player list and profile editor."""

    player_selected = pyqtSignal(object)  # Player or None

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self._current_player: Player | None = None
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

        # -- Profile editor --
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.name_edit = QLineEdit()
        form.addRow("Name:", self.name_edit)

        self.throws_combo = QComboBox()
        self.throws_combo.addItems(["R", "L"])
        form.addRow("Throws:", self.throws_combo)

        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 99)
        self.age_spin.setSpecialValueText("-")
        form.addRow("Age:", self.age_spin)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0, 96)
        self.height_spin.setDecimals(1)
        self.height_spin.setSuffix(" in")
        self.height_spin.setSpecialValueText("-")
        form.addRow("Height:", self.height_spin)

        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0, 400)
        self.weight_spin.setDecimals(1)
        self.weight_spin.setSuffix(" lbs")
        self.weight_spin.setSpecialValueText("-")
        form.addRow("Weight:", self.weight_spin)

        self.team_edit = QLineEdit()
        form.addRow("Team:", self.team_edit)

        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
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

        self._set_editor_enabled(False)

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
        # Select the new player
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
        self._set_editor_enabled(True)
        self.player_selected.emit(player)

    def _populate_editor(self, player: Player):
        self.name_edit.setText(player.name)
        self.throws_combo.setCurrentText(player.throws)
        self.age_spin.setValue(int(player.age) if player.age else 0)
        self.height_spin.setValue(player.height_inches if player.height_inches else 0)
        self.weight_spin.setValue(player.weight_lbs if player.weight_lbs else 0)
        self.team_edit.setText(player.team)
        self.notes_edit.setPlainText(player.notes)

    def _clear_editor(self):
        self.name_edit.clear()
        self.throws_combo.setCurrentIndex(0)
        self.age_spin.setValue(0)
        self.height_spin.setValue(0)
        self.weight_spin.setValue(0)
        self.team_edit.clear()
        self.notes_edit.clear()

    def _set_editor_enabled(self, enabled: bool):
        for w in (self.name_edit, self.throws_combo, self.age_spin, self.height_spin,
                  self.weight_spin, self.team_edit, self.notes_edit, self.save_btn,
                  self.delete_btn):
            w.setEnabled(enabled)

    def _on_save(self):
        if self._current_player is None:
            return
        p = self._current_player
        p.name = self.name_edit.text().strip() or "Unnamed"
        p.throws = self.throws_combo.currentText()
        p.age = self.age_spin.value() if self.age_spin.value() > 0 else None
        p.height_inches = self.height_spin.value() if self.height_spin.value() > 0 else None
        p.weight_lbs = self.weight_spin.value() if self.weight_spin.value() > 0 else None
        p.team = self.team_edit.text().strip()
        p.notes = self.notes_edit.toPlainText().strip()
        self.db.update_player(p)
        self._load_players()
        # Re-select the player
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
            f"Delete '{self._current_player.name}' and all their analysis sessions?",
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

    @property
    def current_player(self) -> Player | None:
        return self._current_player
