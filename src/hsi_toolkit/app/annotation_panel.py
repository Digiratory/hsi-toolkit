"""Annotations panel: object list with names, comments and visibility."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QPlainTextEdit,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hsi_toolkit.core.annotations import AnnotationSet

COMMENT_DEBOUNCE_MS = 300


class AnnotationPanel(QWidget):
    selection_changed = Signal(list)        # ordered annotation ids
    annotation_changed = Signal(str)        # id after name/comment/visibility edit
    annotations_deleted = Signal(list)      # ids

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._annotations = AnnotationSet()
        self._updating = False
        self._comment_source_id: Optional[str] = None

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Имя объекта")
        self.comment_edit = QPlainTextEdit()
        self.comment_edit.setPlaceholderText("Комментарий")
        self.comment_edit.setMaximumHeight(96)
        self.delete_button = QPushButton("Удалить")

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget, stretch=1)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.comment_edit)
        layout.addWidget(self.delete_button)

        self._comment_timer = QTimer(self)
        self._comment_timer.setSingleShot(True)
        self._comment_timer.setInterval(COMMENT_DEBOUNCE_MS)
        self._comment_timer.timeout.connect(self._apply_comment)

        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.name_edit.editingFinished.connect(self._apply_name)
        self.comment_edit.textChanged.connect(self._on_comment_changed)
        self.delete_button.clicked.connect(self._on_delete)

    # -- data ----------------------------------------------------------------

    def set_annotations(self, annotations: AnnotationSet) -> None:
        self._annotations = annotations
        self.refresh()

    def refresh(self) -> None:
        self._updating = True
        selected = set(self.selected_ids())
        self.list_widget.clear()
        for ann in self._annotations:
            pixmap = QPixmap(12, 12)
            pixmap.fill(QColor(ann.color or "#808080"))
            item = QListWidgetItem(QIcon(pixmap), ann.name)
            item.setData(Qt.ItemDataRole.UserRole, ann.id)
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked if ann.visible else Qt.CheckState.Unchecked
            )
            if not ann.visible:
                item.setForeground(QColor("#999999"))
            item.setToolTip(f"{ann.name}\n{ann.comment}")
            self.list_widget.addItem(item)
            if ann.id in selected:
                item.setSelected(True)
        self._updating = False
        self._sync_fields()

    def selected_ids(self) -> List[str]:
        ids = []
        for item in self.list_widget.selectedItems():
            ids.append(item.data(Qt.ItemDataRole.UserRole))
        order = {ann.id: i for i, ann in enumerate(self._annotations)}
        ids.sort(key=lambda ann_id: order.get(ann_id, 0))
        return ids

    def set_selection(self, ids: List[str]) -> None:
        self._updating = True
        wanted = set(ids)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setSelected(item.data(Qt.ItemDataRole.UserRole) in wanted)
        self._updating = False
        self._sync_fields()

    def _current_single(self):
        ids = self.selected_ids()
        if len(ids) == 1:
            return self._annotations.get(ids[0])
        return None

    # -- field syncing -------------------------------------------------------

    def _sync_fields(self) -> None:
        ann = self._current_single()
        self._flush_pending_comment(ann)
        self._comment_source_id = ann.id if ann is not None else None
        if ann is None:
            self.name_edit.clear()
            self.comment_edit.clear()
            self.name_edit.setEnabled(False)
            self.comment_edit.setEnabled(False)
        else:
            self.name_edit.setEnabled(True)
            self.comment_edit.setEnabled(True)
            if self.name_edit.text() != ann.name:
                self.name_edit.setText(ann.name)
            if self.comment_edit.toPlainText() != ann.comment:
                self.comment_edit.setPlainText(ann.comment)

    def _flush_pending_comment(self, new_ann) -> None:
        """Store a debounced comment edit before the fields move off it."""
        pending_id = self._comment_source_id
        if pending_id is None or (new_ann is not None and new_ann.id == pending_id):
            return
        self._comment_timer.stop()
        old = self._annotations.get(pending_id)
        text = self.comment_edit.toPlainText()
        if old is not None and text != old.comment:
            old.comment = text
            self.annotation_changed.emit(old.id)
        self._comment_source_id = None

    def _apply_name(self) -> None:
        ann = self._current_single()
        if ann is None or self._updating:
            return
        name = self.name_edit.text().strip()
        if name and name != ann.name:
            ann.name = name
            self.refresh()
            self.annotation_changed.emit(ann.id)

    def _on_comment_changed(self) -> None:
        if not self._updating and self._current_single() is not None:
            self._comment_timer.start()

    def _apply_comment(self) -> None:
        ann = self._current_single()
        if ann is None or ann.id != self._comment_source_id:
            return
        text = self.comment_edit.toPlainText()
        if text != ann.comment:
            ann.comment = text
            self.annotation_changed.emit(ann.id)

    # -- list events ----------------------------------------------------------

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        self._sync_fields()
        self.selection_changed.emit(self.selected_ids())

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        ann = self._annotations.get(item.data(Qt.ItemDataRole.UserRole))
        if ann is None:
            return
        visible = item.checkState() == Qt.CheckState.Checked
        if visible != ann.visible:
            ann.visible = visible
            self.refresh()
            self.annotation_changed.emit(ann.id)

    def _on_delete(self) -> None:
        ids = self.selected_ids()
        if ids:
            self.annotations_deleted.emit(ids)
