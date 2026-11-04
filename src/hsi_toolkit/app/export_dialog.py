"""NPZ export dialog: object selection, corrected/raw, ribbon width, path."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from hsi_toolkit.app.workers import FunctionWorker, WorkerPool
from hsi_toolkit.core.annotations import AnnotationSet
from hsi_toolkit.core.exporter import export_npz


class ExportDialog(QDialog):
    def __init__(
        self, ws, annotations: AnnotationSet, pool: WorkerPool, parent=None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Экспорт участков в NPZ")
        self._ws = ws
        self._annotations = annotations
        self._pool = pool
        self._worker: Optional[FunctionWorker] = None

        self.list_widget = QListWidget()
        for ann in annotations:
            item = QListWidgetItem(f"{ann.name} ({ann.kind})")
            item.setData(Qt.ItemDataRole.UserRole, ann.id)
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked if ann.visible else Qt.CheckState.Unchecked
            )
            self.list_widget.addItem(item)

        self.corrected_radio = QRadioButton("Корректированный куб (float32, 0..1)")
        self.raw_radio = QRadioButton("Сырой куб (uint16, DN)")
        if ws.has_references:
            self.corrected_radio.setChecked(True)
        else:
            self.raw_radio.setChecked(True)
            self.corrected_radio.setEnabled(False)
            self.corrected_radio.setToolTip("Нужны белый и тёмный референсы")
        radio_box = QGroupBox("Данные")
        radio_layout = QVBoxLayout(radio_box)
        radio_layout.addWidget(self.corrected_radio)
        radio_layout.addWidget(self.raw_radio)

        self.ribbon_spin = QSpinBox()
        self.ribbon_spin.setRange(1, 101)
        self.ribbon_spin.setValue(10)
        ribbon_box = QGroupBox("Лента полилиний")
        ribbon_layout = QHBoxLayout(ribbon_box)
        ribbon_layout.addWidget(QLabel("Ширина, пикс:"))
        ribbon_layout.addWidget(self.ribbon_spin)
        ribbon_layout.addStretch(1)

        self.path_edit = QLineEdit(
            str(Path(ws.hdr_path).with_suffix(".export.npz"))
        )
        path_row = QHBoxLayout()
        path_row.addWidget(new := QLabel("Файл:"))
        path_row.addWidget(self.path_edit, stretch=1)
        browse = QPushButton("Обзор…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(browse)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Экспорт")
        buttons.accepted.connect(self._on_export)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons

        layout = QVBoxLayout(self)
        layout.setMinimumWidth(520)
        layout.addWidget(QLabel("Объекты:"))
        layout.addWidget(self.list_widget, stretch=1)
        layout.addWidget(radio_box)
        layout.addWidget(ribbon_box)
        layout.addLayout(path_row)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Файл NPZ", self.path_edit.text(), "NumPy archive (*.npz)"
        )
        if path:
            self.path_edit.setText(path)

    def selected_ids(self) -> List[str]:
        ids = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def _on_export(self) -> None:
        ids = self.selected_ids()
        if not ids:
            QMessageBox.warning(self, "Нет объектов", "Отметьте хотя бы один объект.")
            return
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Нет пути", "Укажите файл экспорта.")
            return
        corrected = self.corrected_radio.isChecked()
        if corrected and not self._ws.has_references:
            QMessageBox.warning(
                self, "Нет референсов", "Корректированный экспорт требует white+dark."
            )
            return
        self._buttons.setEnabled(False)
        ws = self._ws
        annotations = list(self._annotations)
        ribbon_width = self.ribbon_spin.value()

        def compute(progress):
            progress.emit(10, "Экспорт…")
            keys = export_npz(
                ws,
                annotations,
                path,
                corrected=corrected,
                selected=ids,
                ribbon_width=ribbon_width,
            )
            progress.emit(100, "Готово")
            return keys

        self._worker = self._pool.register(FunctionWorker(compute))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def reject(self) -> None:
        # Esc/must not destroy the dialog (and its worker reference) mid-export.
        if self._worker is not None and self._worker.isRunning():
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            event.ignore()
            return
        super().closeEvent(event)

    def _on_done(self, keys) -> None:
        self._buttons.setEnabled(True)
        QMessageBox.information(
            self,
            "Экспорт завершён",
            f"Экспортировано объектов: {len(keys)}\n"
            + "\n".join(f"• {k}" for k in keys),
        )
        self.accept()

    def _on_failed(self, message: str) -> None:
        self._buttons.setEnabled(True)
        QMessageBox.critical(self, "Ошибка экспорта", message)
