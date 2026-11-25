"""Offscreen GUI smoke test: open dataset, annotate, compute spectra, save, export."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from conftest import BANDS
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog


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
    assert win.spectrum_panel.view_combo.isEnabled() is not False

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
    x0l = int(np.floor(5 / fx))
    x1l = int(np.ceil(15 / fx))
    y0l = int(np.floor(10 / fy))
    y1l = int(np.ceil(20 / fy))
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
    from PySide6.QtWidgets import QToolBar

    from hsi_toolkit.app.data_dialog import DataDialog

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

    def QInputDialog_getText(*a, **k):
        return ("Тестовый полигон", True)

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
        ev = QMouseEvent(
            ev_type, pos, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier
        )
        {
            "press": mc.mousePressEvent,
            "move": mc.mouseMoveEvent,
            "release": mc.mouseReleaseEvent,
            "dbl": mc.mouseDoubleClickEvent,
        }[kind](ev)
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
        id="d1",
        kind="polyline",
        name="Линия",
        points=[(2.0, 5.0), (20.0, 5.0)],
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

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Линия 1", True)))

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
        ev = QMouseEvent(
            ev_type, pos, pos, Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier
        )
        {
            "press": mc.mousePressEvent,
            "move": mc.mouseMoveEvent,
            "release": mc.mouseReleaseEvent,
            "dbl": mc.mouseDoubleClickEvent,
        }[kind](ev)
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


def _stats_polygons():
    from hsi_toolkit.core.annotations import Annotation

    dark = Annotation(
        id="s1",
        kind="polygon",
        name="Тёмная",
        points=[(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)],
        color="#d62728",
    )
    bright = Annotation(
        id="s2",
        kind="polygon",
        name="Светлая",
        points=[(16.0, 16.0), (24.0, 16.0), (24.0, 24.0), (16.0, 24.0)],
        color="#2ca02c",
    )
    return dark, bright


def test_stats_tab_workflow(window, qapp, monkeypatch):
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    # Empty state: hint instead of the table, exports disabled.
    assert win.stats_panel._result is None
    assert not win.stats_panel.table.isVisibleTo(win.stats_panel)
    assert win.stats_panel.status_label.text() == "Нужно ≥2 видимых полигона"
    assert not win.stats_panel.csv_button.isEnabled()

    dark, bright = _stats_polygons()
    win.add_annotation(dark)
    win.add_annotation(bright)
    wait_for(lambda: not win.stats_panel.is_busy, qapp)

    result = win.stats_panel._result
    assert result is not None
    assert result.pairs == ("Тёмная vs Светлая",)
    assert win.stats_panel.table.isVisibleTo(win.stats_panel)
    assert win.stats_panel.table.rowCount() == BANDS
    assert win.stats_panel.table.columnCount() == 1
    assert win.stats_panel.csv_button.isEnabled()

    # The dark quadrant differs from the bright zone: significant + AUC 0.
    assert (result.significant == True).all()  # noqa: E712
    assert result.auc[0, 0] == pytest.approx(0.0, abs=1e-9)
    highlighted = [
        b
        for b in range(BANDS)
        if win.stats_panel.table.item(b, 0).background().style() == Qt.BrushStyle.SolidPattern
    ]
    assert len(highlighted) == BANDS

    # A rename must refresh labels via redraw, not a recomputation.
    dark.name = "Тёмная зона"
    result_before = win.stats_panel._result
    workers_before = len(win.stats_panel._running_workers)
    win._on_annotation_changed(dark.id)
    qapp.processEvents()
    assert win.stats_panel._result is result_before
    assert len(win.stats_panel._running_workers) == workers_before
    header = win.stats_panel.table.horizontalHeaderItem(0)
    assert header.text() == "Тёмная зона vs Светлая"

    # Metric switch is a pure redraw as well.
    win.stats_panel.metric_combo.setCurrentIndex(2)  # ΔR, %
    qapp.processEvents()
    assert win.stats_panel._result is result_before
    assert len(win.stats_panel._running_workers) == workers_before
    assert win.stats_panel.table.item(0, 0).text() != "—"

    # Settings changes must recompute.
    win.stats_panel.test_combo.setCurrentIndex(1)  # Welch
    wait_for(lambda: not win.stats_panel.is_busy, qapp)
    assert win.stats_panel._result is not result_before
    assert win.stats_panel._result.test == "welch"

    win.stats_panel.correction_combo.setCurrentIndex(2)  # Без поправки
    wait_for(lambda: not win.stats_panel.is_busy, qapp)
    assert win.stats_panel._result.correction == "none"

    win.stats_panel.alpha_combo.setCurrentIndex(1)  # α = 0.01
    wait_for(lambda: not win.stats_panel.is_busy, qapp)
    assert win.stats_panel._result.alpha == 0.01

    # Reference switch must recompute with the new reference and show it.
    win.stats_panel.ref_combo.setCurrentIndex(1)  # Светлая (default was 0)
    wait_for(lambda: not win.stats_panel.is_busy, qapp)
    assert win.stats_panel._result.ref_index == 1
    assert "Эталон: «Светлая»" in win.stats_panel.status_label.text()

    # CSV export from the panel writes a file with a BOM and a header.
    out_path = tmp_path / "exported.stats.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "CSV (*.csv)")),
    )
    win.stats_panel._export_csv()
    wait_for(lambda: not win.stats_panel.is_busy and not win.stats_panel._export_busy, qapp)
    assert out_path.exists()
    raw = out_path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert raw.split(b"\r\n")[0].decode("utf-8-sig").startswith("channel,wavelength_nm")
    assert win.stats_panel.status_label.text().startswith("Экспортировано:")


def test_stats_reader_defers_close(window, qapp, monkeypatch):
    # Regression guard: a running stats extraction is a memmap reader, so
    # apply_workspace must defer the old workspace close until it finishes.
    win, ws0, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    import threading

    from hsi_toolkit.app import stats_panel as stats_module
    from hsi_toolkit.core.dataset import HSIWorkspace

    dark, bright = _stats_polygons()
    win.add_annotation(dark)
    win.add_annotation(bright)
    wait_for(lambda: not win.stats_panel.is_busy, qapp)

    original = stats_module.mask_band_values
    release = threading.Event()
    started = threading.Event()

    def slow_extract(ws, mask):
        started.set()
        if not release.wait(10):
            raise TimeoutError("slow extraction not released")
        return original(ws, mask)

    monkeypatch.setattr(stats_module, "mask_band_values", slow_extract)

    # A geometry change forces a fresh extraction (setting-only changes are
    # served from the extraction cache without touching the memmaps).
    dark.points = [(0.0, 0.0), (9.0, 0.0), (9.0, 9.0), (0.0, 9.0)]
    win._on_annotation_changed(dark.id)
    wait_for(lambda: started.is_set(), qapp)

    ws1 = HSIWorkspace.open(ws0.hdr_path)
    win.apply_workspace(ws1)
    qapp.processEvents()

    assert win.workspace is ws1
    assert win.stats_panel.is_busy
    assert not ws0._closed and ws0 in win._pending_close

    release.set()
    wait_for(lambda: ws0._closed, qapp)
    assert win.workspace is ws1 and not win.workspace._closed
    assert not win.stats_panel.is_busy


def test_stats_skipped_roi_and_fast_recompute(window, qapp):
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    from hsi_toolkit.core.annotations import Annotation

    dark, bright = _stats_polygons()
    outside = Annotation(
        id="s3",
        kind="polygon",
        name="Вне кадра",
        points=[(100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 110.0)],
        color="#9467bd",
    )
    win.add_annotation(dark)
    win.add_annotation(bright)
    win.add_annotation(outside)
    wait_for(lambda: not win.stats_panel.is_busy, qapp)

    result = win.stats_panel._result
    assert result is not None
    assert result.pairs == ("Тёмная vs Светлая",)
    assert "Вне кадра" in win.stats_panel.status_label.text()

    # Selecting the out-of-image ROI as reference: it is skipped, and the
    # combo must resync to the reference actually used for ΔR%.
    outside_index = win.stats_panel.ref_combo.findData("s3")
    assert outside_index >= 0
    win.stats_panel.ref_combo.setCurrentIndex(outside_index)
    wait_for(lambda: not win.stats_panel.is_busy, qapp)
    qapp.processEvents()
    assert win.stats_panel.ref_combo.currentData() == "s1"
    assert win.stats_panel._result.ref_index == 0
    assert "эталон" in win.stats_panel.status_label.text()

    # A rename must refresh labels via the cached redraw even though one ROI
    # was skipped (labels resolve by id, not by position).
    bright.name = "Светлая зона"
    result_before = win.stats_panel._result
    workers_before = len(win.stats_panel._running_workers)
    win._on_annotation_changed(bright.id)
    qapp.processEvents()
    assert win.stats_panel._result is result_before
    assert len(win.stats_panel._running_workers) == workers_before
    assert win.stats_panel.table.horizontalHeaderItem(0).text() == "Тёмная vs Светлая зона"

    # Correction/alpha changes re-adjust the cached p-values synchronously:
    # no worker starts, no memmap extraction happens.
    workers_before = len(win.stats_panel._running_workers)
    result_before = win.stats_panel._result
    win.stats_panel.alpha_combo.setCurrentIndex(1)  # α = 0.01
    qapp.processEvents()
    assert win.stats_panel._result.alpha == 0.01
    assert win.stats_panel._result is not result_before
    assert len(win.stats_panel._running_workers) == workers_before

    win.stats_panel.correction_combo.setCurrentIndex(2)  # без поправки
    qapp.processEvents()
    assert win.stats_panel._result.correction == "none"
    np.testing.assert_array_equal(win.stats_panel._result.p_adjusted, win.stats_panel._result.p_raw)


def test_stats_help_button(window, qapp):
    win, ws, tmp_path = window
    from PySide6.QtWidgets import QTextBrowser

    button = win.stats_panel.help_button
    assert not button.text()  # icon-only
    assert not button.icon().isNull()

    dialog = win.stats_panel._build_help_dialog()
    assert dialog.windowTitle() == "О статистиках"
    content = dialog.findChild(QTextBrowser).toPlainText()
    for keyword in (
        "ΔR,%",
        "Cohen's d",
        "AUC",
        "Холм",
        "Уэлча",
        "эталон",
        "автокорреляция",
    ):
        assert keyword in content, keyword
    dialog.deleteLater()


def test_spectrum_toolbar(window, qapp):
    win, ws, tmp_path = window
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    toolbar = win.spectrum_panel.toolbar
    assert isinstance(toolbar, NavigationToolbar2QT)
    # The subplot editor is hidden: it fights the clf()+tight_layout redraws.
    customize = toolbar._actions.get("edit_parameters")
    assert customize is not None
    assert not customize.isVisible()

    # Pan toggles the canvas tool state and back, like in the preview panel.
    pan = toolbar._actions["pan"]
    pan.trigger()
    qapp.processEvents()
    assert toolbar.mode.name == "PAN"
    pan.trigger()
    qapp.processEvents()
    assert toolbar.mode.name == "NONE"


def test_annotation_label_controls(window, qapp):
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    from hsi_toolkit.core.annotations import Annotation

    polygon = Annotation(
        id="lbl1",
        kind="polygon",
        name="Зона",
        points=[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)],
        color="#2ca02c",
    )
    win.add_annotation(polygon)
    qapp.processEvents()
    canvas = win.preview_panel.canvas

    def label_artists():
        return [t for t in canvas.ax.texts if t.get_text() == "Зона"]

    assert len(label_artists()) == 1
    assert label_artists()[0].get_fontsize() == 8.0  # default

    # Font size control applies to the rendered label.
    win.preview_panel.label_size_spin.setValue(14)
    qapp.processEvents()
    assert len(label_artists()) == 1
    assert label_artists()[0].get_fontsize() == 14.0

    # The checkbox hides the label text entirely.
    win.preview_panel.labels_check.setChecked(False)
    qapp.processEvents()
    assert not label_artists()

    # Back on with a new size, and settings survive selection redraws.
    win.preview_panel.labels_check.setChecked(True)
    win.preview_panel.label_size_spin.setValue(6)
    qapp.processEvents()
    assert label_artists()
    assert label_artists()[0].get_fontsize() == 6.0
    canvas.set_selected("lbl1")
    qapp.processEvents()
    assert label_artists()
    assert label_artists()[0].get_fontsize() == 6.0


def test_data_dialog_remembers_last_dir(qapp, tmp_path, monkeypatch):
    # Isolate QSettings in a temp ini file so the test never touches the
    # user's real persisted settings (native registry store).
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QFileDialog

    from hsi_toolkit.app import data_dialog
    from hsi_toolkit.app import settings as settings_module

    ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        settings_module,
        "QSettings",
        lambda *a, **k: QSettings(str(ini), QSettings.Format.IniFormat),
    )

    picked = tmp_path / "data" / "cube.hdr"
    picked.parent.mkdir()
    picked.write_text("ENVI\n", encoding="ascii")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(picked), "ENVI header (*.hdr)")),
    )

    dialog = data_dialog.DataDialog()
    dialog._browse_cube()
    qapp.processEvents()
    assert dialog.cube_edit.text() == str(picked)
    assert settings_module.last_data_dir() == picked.parent

    # A cancelled file dialog must not change the stored folder.
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
    other = tmp_path / "other"
    other.mkdir()
    settings_module.set_last_data_dir(other)
    dialog._browse_cube()
    assert settings_module.last_data_dir() == other
    dialog.deleteLater()

    # A fresh dialog starts browsing in the remembered folder.
    starts = []

    def capture_start(*args, **kwargs):
        starts.append(args[2])
        return ("", "")

    dialog2 = data_dialog.DataDialog()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(capture_start))
    dialog2._browse(dialog2.white_edit)
    assert starts and Path(starts[0]) == other
    dialog2.deleteLater()


def test_backspace_mid_drag_drops_drag(window, qapp):
    # Regression: Backspace during a vertex drag removed the point but left
    # _drag_idx pointing past the list; the next motion raised IndexError.
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    from types import SimpleNamespace

    canvas = win.preview_panel.canvas
    win._set_tool("polyline")
    tool = canvas._polyline_tool
    assert tool.active

    def ev(**kw):
        base = {
            "inaxes": canvas.ax,
            "button": 1,
            "dblclick": False,
            "xdata": None,
            "ydata": None,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    tool._on_press(ev(xdata=5.0, ydata=5.0))
    tool._on_press(ev(xdata=15.0, ydata=15.0))
    assert len(tool._points) == 2
    tool._on_release(ev())
    # Pressing on the last vertex starts a drag of it.
    tool._on_press(ev(xdata=15.2, ydata=14.8))
    assert tool._drag_idx == 1

    # Backspace removes the dragged vertex and drops the drag.
    tool._on_key(SimpleNamespace(key="backspace"))
    assert len(tool._points) == 1
    assert tool._drag_idx == -1

    # The stale drag index must not raise IndexError on the next motion.
    tool._on_motion(ev(xdata=14.0, ydata=14.0))
    assert len(tool._points) == 1

    win._on_tool_cancelled()


def test_comment_debounce_survives_refresh(qapp):
    # Regression: a same-annotation refresh inside the 300 ms debounce window
    # overwrote the editor with the stored comment, discarding in-progress text.
    from hsi_toolkit.app.annotation_panel import AnnotationPanel
    from hsi_toolkit.core.annotations import Annotation, AnnotationSet

    def make(id_, name, comment=""):
        return Annotation(
            id=id_,
            kind="polygon",
            name=name,
            points=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
            comment=comment,
            color="#d62728",
        )

    annotations = AnnotationSet()
    ann1 = make("a1", "Первая", "старый")
    ann2 = make("a2", "Вторая", "")
    annotations.add(ann1)
    annotations.add(ann2)
    panel = AnnotationPanel()
    panel.set_annotations(annotations)
    panel.set_selection(["a1"])
    qapp.processEvents()
    assert panel.comment_edit.toPlainText() == "старый"

    # The user starts typing: the debounce timer is running.
    panel.comment_edit.setPlainText("новый комментарий")
    assert panel._comment_timer.isActive()

    # A same-annotation refresh lands inside the debounce window (visibility
    # toggle, rename, external annotation_changed): the typed text survives.
    panel.refresh()
    assert panel.comment_edit.toPlainText() == "новый комментарий"

    # Moving the fields off the annotation still flushes the pending edit.
    panel.set_selection(["a2"])
    assert ann1.comment == "новый комментарий"

    # The debounce path applies the text via _apply_comment as well.
    panel.comment_edit.setPlainText("второй")
    panel._apply_comment()
    assert ann2.comment == "второй"


def test_spectrum_skips_degenerate_annotation(window, qapp):
    # Regression: one degenerate annotation (zero-pixel mask) failed the whole
    # compute worker and show_message wiped the figure; the batch must continue
    # and report the skipped object without touching the figure (has_content
    # stays true, so no warning text may be drawn into it).
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    from hsi_toolkit.core.annotations import Annotation

    outside = Annotation(
        id="bad1",
        kind="polygon",
        name="Вне кадра",
        points=[(100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 110.0)],
        color="#9467bd",
    )
    good = Annotation(
        id="ok1",
        kind="polygon",
        name="Зона",
        points=[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)],
        color="#2ca02c",
    )

    skipped_events = []
    win.spectrum_panel.compute_skipped.connect(lambda s: skipped_events.append(list(s)))
    win.add_annotation(outside)
    win.add_annotation(good)
    wait_for(lambda: not win.spectrum_panel.is_busy, qapp)

    assert win.spectrum_panel.has_content
    assert [r[0] for r in win.spectrum_panel._result_polygons] == ["ok1"]
    assert skipped_events
    last_names = [name for name, _reason in skipped_events[-1]]
    assert "Вне кадра" in last_names
    assert "Пропущены объекты" in win.statusBar().currentMessage()
    assert "Вне кадра" in win.statusBar().currentMessage()

    # The failed object is cached as failed: unrelated updates (a rename, a
    # settings-only redraw) must not retry it or re-announce the skip.
    good.name = "Зона 2"
    win._on_annotation_changed(good.id)
    qapp.processEvents()
    wait_for(lambda: not win.spectrum_panel.is_busy, qapp)
    assert len(skipped_events) == 1
    assert [r[0] for r in win.spectrum_panel._result_polygons] == ["ok1"]

    # Fixing the geometry gives the object a new cache key: it is retried.
    outside.points = [(5.0, 20.0), (10.0, 20.0), (10.0, 25.0), (5.0, 25.0)]
    win._on_annotation_changed(outside.id)
    wait_for(lambda: not win.spectrum_panel.is_busy, qapp)
    assert [r[0] for r in win.spectrum_panel._result_polygons] == ["bad1", "ok1"]
    assert len(skipped_events) == 1  # the retry succeeded: no new skip report


def test_toolbar_syncs_after_creation(window, qapp, monkeypatch):
    # Regression: the creation flow reset the canvas to select mode but left
    # Полигон checked in the exclusive tool group.
    win, ws, tmp_path = window
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Полигон 1", True)))
    win._set_tool("polygon")
    assert win._tool_actions["polygon"].isChecked()

    win._on_annotation_created("polygon", [(2.0, 2.0), (10.0, 2.0), (10.0, 8.0)])
    qapp.processEvents()

    assert win.preview_panel.canvas.mode == "select"
    assert win._tool_actions["select"].isChecked()
    assert not win._tool_actions["polygon"].isChecked()


def test_load_annotations_corrupt_json_keeps_current(window, qapp, monkeypatch):
    # Regression: AnnotationSet.load on a corrupt file raised out of
    # load_annotations; it must warn and keep the current annotations instead.
    win, ws, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from hsi_toolkit.core.annotations import Annotation

    polygon = Annotation(
        id="k1",
        kind="polygon",
        name="Зона",
        points=[(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)],
        color="#2ca02c",
    )
    win.add_annotation(polygon)

    bad = tmp_path / "broken.json"
    bad.write_text("{ not json at all", encoding="utf-8")
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(bad), "JSON (*.json)")),
    )
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))

    win.load_annotations()
    qapp.processEvents()

    assert warned, "a corrupt file must warn instead of raising"
    assert [a.id for a in win.annotations] == ["k1"]

    # Structurally invalid (but parseable) JSON must degrade the same way:
    # an object without "points" raises KeyError inside AnnotationSet.load.
    bad_structure = tmp_path / "structure.json"
    bad_structure.write_text(
        '{"schema": "hsi-toolkit.annotations/1",'
        ' "objects": [{"id": "x", "kind": "polygon", "name": "X"}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(bad_structure), "JSON (*.json)")),
    )
    warned.clear()
    win.load_annotations()
    qapp.processEvents()

    assert warned, "a structurally invalid file must warn instead of raising"
    assert [a.id for a in win.annotations] == ["k1"]


def test_close_event_defers_workspace_close(window, qapp, monkeypatch):
    # Regression: closeEvent used to close the workspace memmaps even when a
    # reader outlived the shutdown wait (Windows access-violation risk). The
    # workspace must go through the deferred-close machinery instead.
    win, ws0, tmp_path = window
    wait_for(lambda: win.preview_panel.has_image, qapp)

    import threading

    from hsi_toolkit.app import stats_panel as stats_module
    from hsi_toolkit.app.workers import FunctionWorker

    dark, bright = _stats_polygons()
    win.add_annotation(dark)
    win.add_annotation(bright)
    wait_for(lambda: not win.stats_panel.is_busy, qapp)

    original = stats_module.mask_band_values
    release = threading.Event()
    started = threading.Event()

    def slow_extract(ws, mask):
        started.set()
        if not release.wait(10):
            raise TimeoutError("slow extraction not released")
        return original(ws, mask)

    monkeypatch.setattr(stats_module, "mask_band_values", slow_extract)
    # Simulate the closeEvent wait timing out instead of blocking for 30 s:
    # the timed-out worker must stay registered and keep the workspace open.
    monkeypatch.setattr(FunctionWorker, "wait", lambda self, ms=None: None)

    dark.points = [(0.0, 0.0), (9.0, 0.0), (9.0, 9.0), (0.0, 9.0)]
    win._on_annotation_changed(dark.id)
    wait_for(lambda: started.is_set(), qapp)

    win.close()
    qapp.processEvents()

    assert win.stats_panel.is_busy  # timed-out worker stays registered
    assert not ws0._closed
    assert ws0 in win._pending_close

    release.set()
    wait_for(lambda: ws0._closed, qapp)  # closed via the pending flush
    assert ws0 not in win._pending_close
    assert not win.stats_panel.is_busy
