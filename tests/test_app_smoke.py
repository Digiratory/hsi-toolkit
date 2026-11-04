"""Offscreen GUI smoke test: open dataset, annotate, compute spectra, save, export."""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from conftest import BANDS


def wait_for(condition, qapp, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met in time")
        qapp.processEvents()
        time.sleep(0.02)


@pytest.fixture
def window(qapp, synthetic_dataset, tmp_path):
    from hsi_toolkit.app.main_window import MainWindow
    from hsi_toolkit.core.dataset import HSIWorkspace

    win = MainWindow()
    ws = HSIWorkspace.open(
        synthetic_dataset["cube"],
        synthetic_dataset["white"],
        synthetic_dataset["dark"],
    )
    win.apply_workspace(ws)
    yield win, ws, tmp_path
    win.close()
    ws.close()


def test_smoke_workflow(window, qapp):
    win, ws, tmp_path = window

    # Low-res preview built in background, then rendered
    wait_for(lambda: win.preview_panel.has_image, qapp)

    # Create annotations via the same entry point the drawing tools use
    from hsi_toolkit.core.annotations import Annotation

    polygon = Annotation(
        id="t1",
        kind="polygon",
        name="Тестовая зона",
        points=[(3.2, 4.2), (20.8, 4.2), (20.8, 12.8), (3.2, 12.8)],
        color="#d62728",
    )
    line = Annotation(
        id="t2",
        kind="polyline",
        name="Профиль",
        points=[(2.0, 5.0), (25.0, 5.0)],
        color="#1f77b4",
    )
    win.add_annotation(polygon)
    win.add_annotation(line)

    # Visible annotations are plotted automatically (per-object cache)
    wait_for(lambda: not win.spectrum_panel.is_busy, qapp)
    assert win.spectrum_panel.has_content
    assert [r[0] for r in win.spectrum_panel._result_polygons] == ["t1"]
    assert [r[0] for r in win.spectrum_panel._result_lines] == ["t2"]

    # 3D view toggle
    win.spectrum_panel.view_combo.setCurrentText("3D")
    qapp.processEvents()
    wait_for(lambda: win.spectrum_panel.has_content, qapp)
    assert any("3d" in ax.name.lower() for ax in win.spectrum_panel.figure.axes)

    # Save annotations to JSON and load back
    ann_path = tmp_path / "annotations.json"
    win.annotations.image_info = win._image_info()
    win.annotations.save(ann_path)
    from hsi_toolkit.core.annotations import AnnotationSet

    loaded = AnnotationSet.load(ann_path)
    assert len(loaded) == 2
    assert {a.id for a in loaded} == {"t1", "t2"}
    win._apply_annotation_set(loaded)
    assert len(win.annotations) == 2

    # Export NPZ and verify structure
    from hsi_toolkit.core.exporter import export_npz

    out_path = tmp_path / "export.npz"
    keys = export_npz(ws, [polygon, line], out_path, corrected=True)
    assert keys == ["01_testovaya_zona", "02_profil"]
    data = np.load(out_path)
    assert data["wavelengths"].shape == (BANDS,)
    assert data["01_testovaya_zona"].shape[0] == BANDS
    assert data["01_testovaya_zona__mask"].dtype == bool
    assert data["02_profil"].shape[1:] == (23, 10)
    assert data["02_profil__positions"].shape == (23,)
    meta = __import__("json").loads(str(data["objects"]))
    assert [m["key"] for m in meta] == keys


def test_canvas_selection_and_delete(window, qapp):
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    from hsi_toolkit.core.annotations import Annotation

    polygon = Annotation(
        id="c1",
        kind="polygon",
        name="Зона",
        points=[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)],
        color="#2ca02c",
    )
    win.add_annotation(polygon)

    # Simulate a click inside the polygon in data coordinates
    win.preview_panel.canvas.annotation_selected.emit("c1")
    qapp.processEvents()
    assert win.selected_ids == ["c1"]
    assert win.preview_panel.canvas._selected_id == "c1"

    # Delete via panel signal
    win._on_annotations_deleted(["c1"])
    assert len(win.annotations) == 0
    assert win.selected_ids == []


def test_tool_mode_switch(window, qapp):
    win, ws, tmp_path = window
    canvas = win.preview_panel.canvas
    win._set_tool("polygon")
    assert canvas.mode == "polygon"
    assert not canvas._toolbar.isEnabled()
    win._set_tool("polyline")
    assert canvas.mode == "polyline"
    assert canvas._polyline_tool.active
    win._on_tool_cancelled()
    assert canvas.mode == "select"
    assert canvas._toolbar.isEnabled()
    assert not canvas._polyline_tool.active


def test_data_dialog_constructs(window, qapp):
    # Regression: missing QWidget import crashed Файл → Открыть…
    from hsi_toolkit.app.data_dialog import DataDialog
    from PySide6.QtWidgets import QToolBar

    win, ws, tmp_path = window
    dialog = DataDialog(win)
    assert dialog.cube_path is None
    tools = [b for b in win.findChildren(QToolBar) if b.windowTitle() == "Инструменты"]
    assert len(tools) == 1
    labels = [a.text() for a in tools[0].actions()]
    assert labels == ["Выбор", "Полигон", "Полилиния"]


def test_polygon_tool_draws(window, qapp, monkeypatch):
    # Regression: matplotlib PolygonSelector could not be completed by a
    # double-click; the custom polygon tool must finish on double-click
    # anywhere with >= 3 vertices, same as the polyline tool.
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    QInputDialog_getText = (
        lambda *a, **k: ("Тестовый полигон", True)
    )
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(QInputDialog_getText))

    canvas = win.preview_panel.canvas
    mc = canvas._canvas
    win._set_tool("polygon")
    assert canvas.mode == "polygon"

    created = []
    canvas.annotation_created.connect(lambda k, p: created.append((k, p)))

    def qt(kind, dx, dy):
        pos = QPointF(dx, dy)
        ev_type = {
            "move": QEvent.Type.MouseMove,
            "press": QEvent.Type.MouseButtonPress,
            "release": QEvent.Type.MouseButtonRelease,
            "dbl": QEvent.Type.MouseButtonDblClick,
        }[kind]
        buttons = Qt.MouseButton.NoButton if kind == "release" else Qt.MouseButton.LeftButton
        ev = QMouseEvent(ev_type, pos, pos, Qt.MouseButton.LeftButton, buttons,
                         Qt.KeyboardModifier.NoModifier)
        {"press": mc.mousePressEvent, "move": mc.mouseMoveEvent,
         "release": mc.mouseReleaseEvent, "dbl": mc.mouseDoubleClickEvent}[kind](ev)
        qapp.processEvents()

    def to_widget(x, y):
        dx, dy = canvas.ax.transData.transform((x, y))
        return dx, mc.get_width_height()[1] - dy

    for x, y in [(5, 5), (20, 5), (20, 15)]:
        dx, dy = to_widget(x, y)
        qt("move", dx, dy)
        qt("press", dx, dy)
        qt("release", dx, dy)

    # Double-click at the LAST vertex (not the first) must close the polygon.
    dx, dy = to_widget(20, 15)
    qt("move", dx, dy)
    qt("dbl", dx, dy)
    qt("release", dx, dy)
    qapp.processEvents()

    assert created and created[0][0] == "polygon"
    assert len(created[0][1]) == 3
    polygon = win.annotations.get([a.id for a in win.annotations][-1])
    assert polygon is not None and polygon.kind == "polygon"
    assert len(polygon.points) == 3
    assert canvas.mode == "select"


def test_preview_image_not_flipped(qapp):
    from hsi_toolkit.app.canvas.preview_canvas import PreviewCanvas

    canvas = PreviewCanvas()
    image = np.zeros((8, 6, 3), dtype=np.uint8)
    canvas.set_image(image, extent=(-0.5, 5.5, 7.5, -0.5))
    assert canvas.ax.yaxis_inverted(), "row 0 must render at the top"
    canvas.set_image(np.zeros((8, 6, 3), dtype=np.uint8))
    assert canvas.ax.yaxis_inverted(), "no-extent path must also be inverted"


def test_spectrum_recompute_skipped_for_same_selection(window, qapp):
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    from hsi_toolkit.core.annotations import Annotation

    line = Annotation(
        id="d1", kind="polyline", name="Линия", points=[(2.0, 5.0), (20.0, 5.0)],
        color="#1f77b4",
    )
    win.add_annotation(line)
    wait_for(lambda: not win.spectrum_panel.is_busy, qapp)
    assert win.spectrum_panel._result_lines is not None
    result_before = win.spectrum_panel._result_lines
    workers_before = len(win.spectrum_panel._running_workers)

    # Same visible set: served from the per-object cache, no new worker.
    win._update_spectra()
    qapp.processEvents()
    assert win.spectrum_panel._result_lines == result_before
    assert len(win.spectrum_panel._running_workers) == workers_before

    # Changed geometry must recompute.
    line.points = [(2.0, 5.0), (25.0, 5.0)]
    win._update_spectra()
    wait_for(lambda: not win.spectrum_panel.is_busy, qapp)
    assert win.spectrum_panel._result_lines is not result_before
    assert win.spectrum_panel._result_lines[0][1][-1] == 1.0

    # Unchecking visibility removes the object from the plot without workers.
    line.visible = False
    win._on_annotation_changed(line.id)
    qapp.processEvents()
    assert not win.spectrum_panel._result_lines
    assert not win.spectrum_panel.has_content
