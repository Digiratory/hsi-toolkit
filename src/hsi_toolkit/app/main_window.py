"""Main window: menus/toolbar, docks (data info, objects, spectra), preview."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QDialog,
    QDockWidget,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QToolBar,
    QWidget,
)

from hsi_toolkit.app.annotation_panel import AnnotationPanel
from hsi_toolkit.app.canvas.preview_canvas import (
    MODE_POLYGON,
    MODE_POLYLINE,
    MODE_SELECT,
)
from hsi_toolkit.app.data_dialog import DataDialog
from hsi_toolkit.app.export_dialog import ExportDialog
from hsi_toolkit.app.preview_panel import PreviewPanel
from hsi_toolkit.app.spectrum_panel import SpectrumPanel
from hsi_toolkit.app.workers import FunctionWorker, WorkerPool
from hsi_toolkit.core.annotations import (
    Annotation,
    AnnotationSet,
    annotations_path_for,
    assign_color,
    new_id,
)
from hsi_toolkit.core.dataset import HSIWorkspace

LOWRES_FACTOR = 4


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HSI Toolkit — гиперспектральные данные")
        self.resize(1400, 900)

        self.workspace: Optional[HSIWorkspace] = None
        self.annotations = AnnotationSet()
        self.selected_ids: List[str] = []
        self._loader: Optional[FunctionWorker] = None
        self._lowres_worker: Optional[FunctionWorker] = None
        self.worker_pool = WorkerPool()
        self._workspace_generation = 0

        self.preview_panel = PreviewPanel()
        self.preview_panel.annotation_created.connect(self._on_annotation_created)
        self.preview_panel.annotation_selected.connect(self._on_canvas_selected)
        self.preview_panel.tool_cancelled.connect(self._on_tool_cancelled)

        self.annotation_panel = AnnotationPanel()
        self.annotation_panel.selection_changed.connect(self._on_selection_changed)
        self.annotation_panel.annotation_changed.connect(self._on_annotation_changed)
        self.annotation_panel.annotations_deleted.connect(self._on_annotations_deleted)

        self.spectrum_panel = SpectrumPanel()

        self.setCentralWidget(self.preview_panel)
        self._info_label = QLabel()
        self._info_label.setTextFormat(Qt.TextFormat.RichText)
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        info_dock = self._dock("Данные", self._info_label)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, info_dock)

        objects_dock = self._dock("Объекты", self.annotation_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, objects_dock)

        spectra_dock = self._dock("Спектры", self.spectrum_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, spectra_dock)

        self._build_toolbar()
        self._status_progress = QProgressBar()
        self._status_progress.setMaximumWidth(220)
        self._status_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._status_progress)
        self.statusBar().showMessage("Откройте набор данных")

    # -- UI construction --------------------------------------------------------

    def _dock(self, title: str, widget: QWidget) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        return dock

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Главная")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        act_open = QAction("Открыть…", self)
        act_open.triggered.connect(self.open_dataset)
        act_save = QAction("Сохранить разметку", self)
        act_save.triggered.connect(self.save_annotations)
        act_load = QAction("Загрузить разметку…", self)
        act_load.triggered.connect(self.load_annotations)
        act_export = QAction("Экспорт NPZ…", self)
        act_export.triggered.connect(self.export_npz)
        for action in (act_open, act_save, act_load, act_export):
            toolbar.addAction(action)
        self._act_save = act_save
        self._act_export = act_export

        toolbar.addSeparator()
        self._tool_actions = {}
        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)
        for mode, label in (
            (MODE_SELECT, "Выбор"),
            (MODE_POLYGON, "Полигон"),
            (MODE_POLYLINE, "Полилиния"),
        ):
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked, m=mode: self._set_tool(m) if checked else None
            )
            tool_group.addAction(action)
            toolbar.addAction(action)
            self._tool_actions[mode] = action
        self._tool_actions[MODE_SELECT].setChecked(True)

        menu = self.menuBar().addMenu("Файл")
        menu.addAction(act_open)
        menu.addAction(act_save)
        menu.addAction(act_load)
        menu.addAction(act_export)

    # -- dataset opening -----------------------------------------------------------

    def open_dataset(self) -> None:
        dialog = DataDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.statusBar().showMessage("Загрузка набора данных…")
        self._status_progress.setVisible(True)
        self._status_progress.setRange(0, 100)

        cube = dialog.cube_path
        white = dialog.white_path
        dark = dialog.dark_path
        annotations_path = dialog.annotations_path

        def load(progress):
            progress.emit(5, "Чтение куба и референсов…")
            ws = HSIWorkspace.open(cube, white, dark)
            progress.emit(70, "Референсы сведены к профилям")
            return ws, annotations_path

        self._loader = self.worker_pool.register(FunctionWorker(load))
        self._loader.progressed.connect(self._on_progress)
        self._loader.finished_ok.connect(self._on_dataset_loaded)
        self._loader.failed.connect(self._on_dataset_failed)
        self._loader.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self._status_progress.setValue(percent)
        self.statusBar().showMessage(message)

    def _on_dataset_failed(self, message: str) -> None:
        self._status_progress.setVisible(False)
        self.statusBar().showMessage("Ошибка загрузки")
        QMessageBox.critical(self, "Ошибка загрузки", message)

    def _on_dataset_loaded(self, result) -> None:
        ws, annotations_path = result
        self._status_progress.setVisible(False)
        self.apply_workspace(ws)
        if annotations_path is not None and Path(annotations_path).is_file():
            self._apply_annotation_set(
                AnnotationSet.load(annotations_path), warn_on_error=True
            )
        self.statusBar().showMessage("Набор загружен")

    def apply_workspace(self, ws: HSIWorkspace) -> None:
        """Attach a ready workspace and start the low-res preview build."""
        self._workspace_generation += 1
        # Never close the old workspace's memmaps under threads still reading
        # them: join in-flight spectrum and lowres workers first.
        self.spectrum_panel.shutdown()
        if self._lowres_worker is not None and self._lowres_worker.isRunning():
            self._lowres_worker.wait(60000)
        if self.workspace is not None:
            self.workspace.close()
        self.workspace = ws
        self.annotations = AnnotationSet(image_info=self._image_info())
        self.selected_ids = []
        self._update_info()
        self.annotation_panel.set_annotations(self.annotations)
        self.spectrum_panel.set_workspace(ws)
        self.spectrum_panel.set_annotation_source(self.annotations)
        self.preview_panel.canvas.set_annotations(self.annotations)
        self.preview_panel.set_workspace(ws)
        self._act_save.setEnabled(True)
        self._act_export.setEnabled(True)

        self._status_progress.setVisible(True)
        self._status_progress.setRange(0, 0)
        generation = self._workspace_generation

        def build(progress):
            return ws.build_lowres_cube(factor=LOWRES_FACTOR)

        self._lowres_worker = self.worker_pool.register(FunctionWorker(build))
        self._lowres_worker.progressed.connect(self._on_progress)
        self._lowres_worker.finished_ok.connect(
            lambda _cube, gen=generation: self._on_lowres_ready(gen)
        )
        self._lowres_worker.failed.connect(self._on_lowres_failed)
        self._lowres_worker.start()

    def _on_lowres_ready(self, generation: int) -> None:
        if generation != self._workspace_generation:
            return
        self._status_progress.setVisible(False)
        if self.workspace is not None:
            self.preview_panel.refresh()
            self.statusBar().showMessage("Предпросмотр готов")

    def _on_lowres_failed(self, message: str) -> None:
        self._status_progress.setVisible(False)
        self.statusBar().showMessage("Ошибка построения предпросмотра")
        QMessageBox.warning(self, "Предпросмотр", message)

    def _image_info(self) -> dict:
        if self.workspace is None:
            return {}
        ws = self.workspace
        return {
            "hdr": str(ws.hdr_path),
            "samples": int(ws.samples),
            "lines": int(ws.lines),
            "wavelengths_nm": [float(w) for w in ws.wavelengths],
        }

    def _update_info(self) -> None:
        ws = self.workspace
        if ws is None:
            self._info_label.setText("Набор не загружен")
            return
        wl_min, wl_max = ws.wavelength_range
        self._info_label.setText(
            f"<b>Куб:</b> {ws.hdr_path.name}<br><br>"
            f"Размер: {ws.samples} × {ws.lines} × {ws.bands}<br>"
            f"<small>(samples × lines × bands)</small><br><br>"
            f"λ: {wl_min:.1f}–{wl_max:.1f} нм<br>"
            f"Каналов: {ws.bands}<br><br>"
            f"Коррекция: {'да (white+dark)' if ws.has_references else '<b>нет</b> — сырые данные'}"
        )

    # -- annotations ------------------------------------------------------------

    def add_annotation(self, annotation: Annotation) -> Annotation:
        """Add an annotation programmatically and select it."""
        self.annotations.add(annotation)
        self.annotation_panel.refresh()
        self.preview_panel.canvas.set_annotations(self.annotations)
        self.annotation_panel.set_selection([annotation.id])
        self._on_selection_changed([annotation.id])
        return annotation

    def _on_annotation_created(self, kind: str, points) -> None:
        count = sum(1 for a in self.annotations if a.kind == kind) + 1
        default_name = ("Полигон %d" if kind == MODE_POLYGON else "Линия %d") % count
        name, ok = QInputDialog.getText(
            self, "Новый объект", "Имя объекта:", text=default_name
        )
        self._set_tool(MODE_SELECT)
        if not ok:
            return
        annotation = Annotation(
            id=new_id(),
            kind=kind,
            name=name.strip() or default_name,
            points=[(float(x), float(y)) for x, y in points],
            color=assign_color(len(self.annotations)),
        )
        self.add_annotation(annotation)
        self.statusBar().showMessage(f"Добавлен объект: {annotation.name}")

    def _on_canvas_selected(self, annotation_id: str) -> None:
        ids = [annotation_id] if annotation_id else []
        self.annotation_panel.set_selection(ids)
        self._on_selection_changed(ids)

    def _on_selection_changed(self, ids) -> None:
        self.selected_ids = list(ids)
        self.preview_panel.canvas.set_selected(ids[-1] if ids else "")
        annotations = [
            self.annotations.get(i) for i in ids if self.annotations.get(i) is not None
        ]
        self.spectrum_panel.update_spectra(annotations)

    def _on_annotation_changed(self, _annotation_id: str) -> None:
        self.preview_panel.canvas.set_annotations(self.annotations)
        self.annotation_panel.refresh()
        # Names/colors/visibility affect only rendering — redraw the cached
        # spectra without recomputing them.
        self.spectrum_panel.refresh_annotations()

    def _on_annotations_deleted(self, ids) -> None:
        for annotation_id in ids:
            self.annotations.remove(annotation_id)
        self.selected_ids = []
        self.annotation_panel.refresh()
        self.preview_panel.canvas.set_annotations(self.annotations)
        self.preview_panel.canvas.set_selected("")
        self.spectrum_panel.update_spectra([])

    def _set_tool(self, mode: str) -> None:
        self.preview_panel.set_mode(mode)

    def _on_tool_cancelled(self) -> None:
        self._tool_actions[MODE_SELECT].setChecked(True)
        self._set_tool(MODE_SELECT)

    # -- annotation file save/load --------------------------------------------------

    def save_annotations(self) -> None:
        if self.workspace is None:
            QMessageBox.information(self, "Нет данных", "Сначала откройте набор данных.")
            return
        from PySide6.QtWidgets import QFileDialog

        default = annotations_path_for(self.workspace.hdr_path)
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить разметку", str(default), "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        self.annotations.image_info = self._image_info()
        self.annotations.save(path)
        self.statusBar().showMessage(f"Разметка сохранена: {path}")

    def load_annotations(self) -> None:
        if self.workspace is None:
            QMessageBox.information(self, "Нет данных", "Сначала откройте набор данных.")
            return
        from PySide6.QtWidgets import QFileDialog

        default = annotations_path_for(self.workspace.hdr_path)
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить разметку", str(default), "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        self._apply_annotation_set(AnnotationSet.load(path))

    def _apply_annotation_set(self, annotation_set: AnnotationSet, warn_on_error: bool = False) -> None:
        info = annotation_set.image_info
        if self.workspace is not None and info:
            if (
                info.get("samples") != self.workspace.samples
                or info.get("lines") != self.workspace.lines
            ):
                QMessageBox.warning(
                    self,
                    "Размер не совпадает",
                    "Разметка относится к изображению другого размера "
                    f"({info.get('samples')}×{info.get('lines')}). "
                    "Координаты объектов могут не совпасть.",
                )
        self.annotations = annotation_set
        self.selected_ids = []
        self.annotation_panel.set_annotations(self.annotations)
        self.preview_panel.canvas.set_annotations(self.annotations)
        self.preview_panel.canvas.set_selected("")
        self.spectrum_panel.set_annotation_source(self.annotations)
        self.spectrum_panel.update_spectra([])

    # -- export -----------------------------------------------------------------

    def export_npz(self) -> None:
        if self.workspace is None:
            QMessageBox.information(self, "Нет данных", "Сначала откройте набор данных.")
            return
        if len(self.annotations) == 0:
            QMessageBox.information(self, "Нет объектов", "Сначала создайте разметку.")
            return
        dialog = ExportDialog(self.workspace, self.annotations, self.worker_pool, self)
        dialog.exec()

    # -- shutdown ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self.spectrum_panel.shutdown()
        self.worker_pool.wait_all(10000)
        if self.workspace is not None:
            self.workspace.close()
        super().closeEvent(event)
