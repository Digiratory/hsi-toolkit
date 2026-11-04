"""Preview image canvas: imshow, annotation overlays, drawing tools."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtGui import QCursor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from hsi_toolkit.app.canvas.tools import PolylineTool
from hsi_toolkit.core.annotations import AnnotationSet

MODE_SELECT = "select"
MODE_POLYGON = "polygon"
MODE_POLYLINE = "polyline"

PICK_THRESHOLD_PX = 8.0


def _point_in_polygon(x: float, y: float, pts: np.ndarray) -> bool:
    """Even-odd ray casting; pts is an (n, 2) array of (x, y)."""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _distance_point_segment(px, py, x0, y0, x1, y1) -> float:
    dx, dy = x1 - x0, y1 - y0
    seg2 = dx * dx + dy * dy
    if seg2 == 0.0:
        return float(np.hypot(px - x0, py - y0))
    t = ((px - x0) * dx + (py - y0) * dy) / seg2
    t = min(max(t, 0.0), 1.0)
    return float(np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)))


class PreviewCanvas(QWidget):
    annotation_created = Signal(str, list)  # (kind, points)
    annotation_selected = Signal(str)       # annotation id, "" clears selection
    tool_cancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self.ax = self._figure.add_subplot(1, 1, 1)
        self._image_artist = None
        self._message_artist = None
        self._current_extent: Optional[tuple] = None
        self._annotations = AnnotationSet()
        self._overlay_artists: dict = {}
        self._selected_id = ""
        self._mode = MODE_SELECT
        self._polyline_tool = PolylineTool(self, min_points=2, closed=False)
        self._polygon_tool = PolylineTool(self, min_points=3, closed=True)
        for tool, kind in (
            (self._polyline_tool, MODE_POLYLINE),
            (self._polygon_tool, MODE_POLYGON),
        ):
            tool.finished.connect(
                lambda points, tool_kind=kind: self._on_tool_finished(tool_kind, points)
            )
            tool.cancelled.connect(self.tool_cancelled.emit)
        self._canvas.mpl_connect("button_press_event", self._on_button_press)
        self.show_message("Откройте набор данных (Файл → Открыть…)")

    # -- image ------------------------------------------------------------

    def set_image(self, image: np.ndarray, extent=None) -> None:
        self._clear_message()
        # Preserve the user's pan/zoom across redraws as long as the data
        # extent is unchanged (new dataset -> reset to the full view).
        had_image = self._image_artist is not None
        prev_limits = (
            (self.ax.get_xlim(), self.ax.get_ylim()) if had_image else None
        )
        if self._image_artist is not None:
            self._image_artist.remove()
        self._image_artist = self.ax.imshow(
            image, interpolation="nearest", origin="upper", extent=extent
        )
        if extent is None:
            # (left, right, bottom, top): row 0 at the top of the screen.
            extent = (-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5)
        x0, x1, y0, y1 = extent
        if had_image and self._current_extent == tuple(extent):
            self.ax.set_xlim(*prev_limits[0])
            self.ax.set_ylim(*prev_limits[1])
        else:
            self._current_extent = tuple(extent)
            self.ax.set_xlim(x0, x1)
            self.ax.set_ylim(y0, y1)
        self.ax.set_xlabel("x, пиксели")
        self.ax.set_ylabel("y, пиксели")
        self._canvas.draw_idle()

    def show_message(self, text: str) -> None:
        self._clear_message()
        self._message_artist = self.ax.text(
            0.5, 0.5, text, transform=self.ax.transAxes,
            ha="center", va="center", fontsize=12, color="#666666",
        )
        self._canvas.draw_idle()

    def _clear_message(self) -> None:
        if getattr(self, "_message_artist", None) is not None:
            self._message_artist.remove()
            self._message_artist = None

    # -- annotations --------------------------------------------------------

    def set_annotations(self, annotations: AnnotationSet) -> None:
        self._annotations = annotations
        self._redraw_overlays()

    def set_selected(self, annotation_id: str) -> None:
        self._selected_id = annotation_id
        self._redraw_overlays()

    def _redraw_overlays(self) -> None:
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon as MplPolygon

        for artists in self._overlay_artists.values():
            for artist in artists:
                artist.remove()
        self._overlay_artists.clear()
        for ann in self._annotations:
            if not ann.visible:
                continue
            selected = ann.id == self._selected_id
            color = ann.color or "#1f77b4"
            linewidth = 2.5 if selected else 1.4
            fill_alpha = 0.30 if selected else 0.15
            pts = np.asarray(ann.points, dtype=float)
            artists: List = []
            if ann.kind == MODE_POLYGON:
                patch = MplPolygon(
                    pts, closed=True, edgecolor=color,
                    facecolor=to_rgba(color, fill_alpha), linewidth=linewidth,
                    zorder=3,
                )
                self.ax.add_patch(patch)
                artists.append(patch)
            else:
                line = Line2D(
                    pts[:, 0], pts[:, 1], color=color, linewidth=linewidth,
                    marker="o", markersize=5.0 if selected else 3.5,
                    markerfacecolor="white", markeredgecolor=color, zorder=3,
                )
                self.ax.add_line(line)
                artists.append(line)
            label = self.ax.text(
                pts[0, 0], pts[0, 1], ann.name, color=color,
                fontsize=8, fontweight="bold" if selected else "normal",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.0),
                zorder=4,
            )
            artists.append(label)
            self._overlay_artists[ann.id] = artists
        self._canvas.draw_idle()

    # -- tools ----------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        self._polyline_tool.stop()
        self._polygon_tool.stop()
        self._mode = mode
        self._toolbar.setEnabled(mode == MODE_SELECT)
        if mode == MODE_POLYGON:
            self._polygon_tool.start()
        elif mode == MODE_POLYLINE:
            self._polyline_tool.start()
        else:
            self.set_cursor(False)

    def set_cursor(self, drawing: bool) -> None:
        shape = Qt.CursorShape.CrossCursor if drawing else Qt.CursorShape.ArrowCursor
        self._canvas.setCursor(QCursor(shape))

    def _on_tool_finished(self, kind: str, points) -> None:
        if kind == MODE_POLYLINE:
            self._polyline_tool.stop()
        else:
            self._polygon_tool.stop()
        if len(points) >= 3:
            self.annotation_created.emit(
                kind, [(float(x), float(y)) for x, y in points]
            )

    # -- selection ----------------------------------------------------------

    def _on_button_press(self, event) -> None:
        if self._mode != MODE_SELECT or event.button != 1:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        self.annotation_selected.emit(self._hit_test(event.xdata, event.ydata))

    def _hit_test(self, x: float, y: float) -> str:
        cursor = self.ax.transData.transform((x, y))
        best_id = ""
        best_distance = PICK_THRESHOLD_PX
        for ann in reversed(list(self._annotations)):
            if not ann.visible:
                continue
            pts = np.asarray(ann.points, dtype=float)
            disp = self.ax.transData.transform(pts)
            distance = np.min(np.hypot(disp[:, 0] - cursor[0], disp[:, 1] - cursor[1]))
            for i in range(len(disp)):
                x0, y0 = disp[i]
                x1, y1 = disp[(i + 1) % len(disp)]
                if ann.kind == MODE_POLYLINE and i == len(disp) - 1:
                    break
                distance = min(
                    distance,
                    _distance_point_segment(cursor[0], cursor[1], x0, y0, x1, y1),
                )
            if (
                ann.kind == MODE_POLYGON
                and distance > best_distance
                and _point_in_polygon(x, y, pts)
            ):
                distance = 0.0
            if distance < best_distance:
                best_distance = distance
                best_id = ann.id
        return best_id
