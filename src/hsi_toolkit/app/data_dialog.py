"""Dataset selection dialog (step 1): cube, references, annotations JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from hsi_toolkit.app.settings import last_data_dir, set_last_data_dir
from hsi_toolkit.core.annotations import annotations_path_for
from hsi_toolkit.core.dataset import find_reference_hdrs

HDR_FILTER = "ENVI header (*.hdr);;All files (*)"
JSON_FILTER = "JSON (*.json);;All files (*)"


class DataDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Открыть набор данных")

        self.cube_edit = QLineEdit()
        self.white_edit = QLineEdit()
        self.dark_edit = QLineEdit()
        self.annotations_edit = QLineEdit()

        grid = QFormLayout()
        grid.addRow("Куб (HDR):", self._row(self.cube_edit, self._browse_cube))
        grid.addRow(
            "Белый референс:",
            self._row(self.white_edit, lambda: self._browse(self.white_edit)),
        )
        grid.addRow(
            "Тёмный референс:",
            self._row(self.dark_edit, lambda: self._browse(self.dark_edit)),
        )
        grid.addRow(
            "Разметка JSON:",
            self._row(
                self.annotations_edit,
                lambda: self._browse(self.annotations_edit, JSON_FILTER),
            ),
        )

        self.auto_button = QToolButton()
        self.auto_button.setText("Найти референсы автоматически")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        self.setMinimumWidth(560)
        layout.addLayout(grid)
        layout.addWidget(self.auto_button)
        layout.addWidget(buttons)

        self.auto_button.clicked.connect(self._autodetect)
        self.cube_edit.editingFinished.connect(self._autodetect)

    @staticmethod
    def _row(edit: QLineEdit, callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, stretch=1)
        button = QPushButton("Обзор…")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return row

    def _start_dir(self, edit: QLineEdit) -> str:
        """Field's own path, else the last selected dataset folder, else home."""
        text = edit.text().strip()
        if text:
            return text
        saved = last_data_dir()
        if saved is not None and saved.is_dir():
            return str(saved)
        return str(Path.home())

    def _browse(self, edit: QLineEdit, file_filter: str = HDR_FILTER) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", self._start_dir(edit), file_filter
        )
        if path:
            edit.setText(path)

    def _browse_cube(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", self._start_dir(self.cube_edit), HDR_FILTER
        )
        if path:
            self.cube_edit.setText(path)
            # Remember the folder only when an image was actually selected;
            # a cancelled dialog must not touch the stored value.
            set_last_data_dir(Path(path).parent)
        self._autodetect()

    def _autodetect(self) -> None:
        cube = self.cube_path
        if cube is None:
            return
        white, dark = find_reference_hdrs(cube)
        if white is not None and not self.white_edit.text():
            self.white_edit.setText(str(white))
        if dark is not None and not self.dark_edit.text():
            self.dark_edit.setText(str(dark))
        default_ann = annotations_path_for(cube)
        if default_ann.is_file() and not self.annotations_edit.text():
            self.annotations_edit.setText(str(default_ann))

    def _on_accept(self) -> None:
        if self.cube_path is None:
            QMessageBox.warning(self, "Нет куба", "Укажите HDR-файл гиперкуба.")
            return
        has_white = bool(self.white_edit.text().strip())
        has_dark = bool(self.dark_edit.text().strip())
        if has_white != has_dark:
            QMessageBox.warning(
                self,
                "Неполные референсы",
                "Указан только один референс: для коррекции нужны оба "
                "(белый и тёмный). Оставьте оба поля пустыми или заполните оба.",
            )
            return
        self.accept()

    # -- properties -------------------------------------------------------------

    @property
    def cube_path(self) -> Optional[Path]:
        text = self.cube_edit.text().strip()
        return Path(text) if text else None

    @property
    def white_path(self) -> Optional[Path]:
        text = self.white_edit.text().strip()
        return Path(text) if text else None

    @property
    def dark_path(self) -> Optional[Path]:
        text = self.dark_edit.text().strip()
        return Path(text) if text else None

    @property
    def annotations_path(self) -> Optional[Path]:
        text = self.annotations_edit.text().strip()
        return Path(text) if text else None
