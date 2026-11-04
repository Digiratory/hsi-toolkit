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

    # Kind switch: both kinds present -> the combo decides what is displayed.
    assert win.spectrum_panel.kind_combo.currentText() == "Полигоны"
    # One curve per polygon; the CI band is a PolyCollection, not a line.
    assert len(win.spectrum_panel.figure.axes[0].lines) == 1

    # Changing the confidence-interval type must recompute, not just redraw.
    lower_before = win.spectrum_panel._result_polygons[0][2]
    win.spectrum_panel.ci_combo.setCurrentIndex(1)  # percentiles 2.5–97.5
    wait_for(lambda: not win.spectrum_panel.is_busy, qapp)
    lower_after = win.spectrum_panel._result_polygons[0][2]
    assert lower_after is not lower_before and not np.allclose(lower_after, lower_before)

    win.spectrum_panel.kind_combo.setCurrentText("Полилинии")
    win.spectrum_panel.view_combo.setCurrentText("3D")
    qapp.processEvents()
    assert any("3d" in ax.name.lower() for ax in win.spectrum_panel.figure.axes)
    assert not win.spectrum_panel.view_combo.isEnabled() is False

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


def test_window_mode_spectrum_strip(window, qapp):
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)
    panel = win.preview_panel

    # RGB mode (default): the spectrum strip is hidden.
    assert not panel._bottom.isVisibleTo(panel)

    # Switch to window mode: strip shows the integral spectrum with shading.
    panel.mode_combo.setCurrentText("Спектральное окно")
    qapp.processEvents()
    assert panel._bottom.isVisibleTo(panel)
    panel._draw_spectrum_strip(ws)
    ax = panel.spectrum_figure.axes[0]
    spans = [p for p in ax.patches]
    assert spans, "window shading must be present"
    span_x0 = spans[0].get_x()
    assert ax.get_xlim()[0] <= span_x0 < ax.get_xlim()[1]

    # Moving the center slider moves the shaded window.
    panel.center_slider.setValue(int(panel.center_slider.minimum()) + 2)
    panel.update_preview()
    qapp.processEvents()
    new_span_x0 = panel.spectrum_figure.axes[0].patches[0].get_x()
    assert new_span_x0 != span_x0

    # Back to RGB: strip hides again.
    panel.mode_combo.setCurrentText("Псевдо-RGB")
    qapp.processEvents()
    assert not panel._bottom.isVisibleTo(panel)


def test_zoom_preserved_and_region_strip(window, qapp):
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)
    panel = win.preview_panel
    canvas = panel.canvas

    # Settings changes (re-set image) must not reset user zoom.
    canvas.ax.set_xlim(5, 15)
    canvas.ax.set_ylim(10, 20)
    panel.update_preview()
    qapp.processEvents()
    assert tuple(canvas.ax.get_xlim()) == (5.0, 15.0)
    assert tuple(canvas.ax.get_ylim()) == (10.0, 20.0)

    # The strip's integral spectrum is computed over the visible region.
    panel.mode_combo.setCurrentText("Спектральное окно")
    panel.update_preview()
    qapp.processEvents()
    ax = panel.spectrum_figure.axes[0]
    assert "Область: x 5–15, y 10–20" in ax.get_title()

    region = panel._view_region(ws)
    assert region == (5, 15, 10, 20)
    low = ws.lowres
    fx = ws.samples / low.shape[2]
    fy = ws.lines / low.shape[1]
    x0l = int(np.floor(5 / fx)); x1l = int(np.ceil(15 / fx))
    y0l = int(np.floor(10 / fy)); y1l = int(np.ceil(20 / fy))
    expected = low[:, y0l:y1l, x0l:x1l].mean(axis=(1, 2))
    np.testing.assert_allclose(panel._integral, expected, rtol=1e-4)

    # Zooming out again updates the region (no worker, just the strip).
    canvas.ax.set_xlim(-0.5, ws.samples - 0.5)
    canvas.ax.set_ylim(ws.lines - 0.5, -0.5)
    panel.update_preview()
    qapp.processEvents()
    assert panel._view_region(ws) == (0, ws.samples, 0, ws.lines)
    ax = panel.spectrum_figure.axes[0]
    assert "Область" not in ax.get_title()


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


def test_polyline_tool_two_points(window, qapp, monkeypatch):
    # Regression: a 2-point polyline finished the tool but was silently
    # dropped — the polygon's >= 3 gate was applied to polylines too.
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("Линия 1", True))
    )

    canvas = win.preview_panel.canvas
    mc = canvas._canvas
    win._set_tool("polyline")
    assert canvas.mode == "polyline"

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

    for x, y in [(5, 5), (20, 5)]:
        dx, dy = to_widget(x, y)
        qt("move", dx, dy)
        qt("press", dx, dy)
        qt("release", dx, dy)

    dx, dy = to_widget(20, 5)
    qt("move", dx, dy)
    qt("dbl", dx, dy)
    qt("release", dx, dy)
    qapp.processEvents()

    assert created and created[0][0] == "polyline"
    assert len(created[0][1]) == 2
    polylines = [a for a in win.annotations if a.kind == "polyline"]
    assert len(polylines) == 1 and len(polylines[0].points) == 2
    assert canvas.mode == "select"


def test_opening_dataset_during_build_defers_close(window, qapp, monkeypatch):
    # Regression: apply_workspace used to block the GUI thread joining the
    # in-flight lowres build; the old workspace close is now deferred instead.
    win, ws0, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    import threading

    from hsi_toolkit.core.dataset import HSIWorkspace

    original = HSIWorkspace.build_lowres_cube
    release = threading.Event()
    started = threading.Event()

    def slow_build(self, factor=4):
        started.set()
        if not release.wait(10):
            raise TimeoutError("slow build not released")
        return original(self, factor=factor)

    monkeypatch.setattr(HSIWorkspace, "build_lowres_cube", slow_build)

    try:
        hdr = ws0.hdr_path
        ws1 = HSIWorkspace.open(hdr)
        win.apply_workspace(ws1)
        wait_for(lambda: started.is_set(), qapp)  # build in flight

        ws2 = HSIWorkspace.open(hdr)
        win.apply_workspace(ws2)  # must not block on the running build
        qapp.processEvents()

        assert win.workspace is ws2
        assert not ws1._closed and ws1 in win._pending_close

        release.set()
        wait_for(lambda: ws1._closed, qapp)  # closed once the reader finished
        assert win.workspace is ws2 and not win.workspace._closed
    finally:
        release.set()
