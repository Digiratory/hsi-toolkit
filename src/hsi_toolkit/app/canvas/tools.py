"""Interactive polyline drawing tool.

Click adds a vertex, press+drag on an existing vertex edits it, Backspace
removes the last vertex, Enter or double-click finishes (>= 2 points),
Escape cancels. Vertices are stored in data (image pixel) coordinates.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

Point = Tuple[float, float]

DRAG_RADIUS_PX = 8.0


class PolylineTool(QObject):
    finished = Signal(list)
    cancelled = Signal()

    def __init__(self, owner) -> None:
        super().__init__(owner)
        self._owner = owner  # PreviewCanvas: provides .ax, ._canvas, set_cursor()
        self._active = False
        self._points: List[Point] = []
        self._drag_idx = -1
        self._line = None
        self._preview = None
        self._cids: list = []

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self.stop()
        self._active = True
        self._points = []
        self._drag_idx = -1
        canvas = self._owner._canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("key_press_event", self._on_key),
        ]
        self._ensure_artists()
        self._owner.set_cursor(True)
        canvas.draw_idle()

    def stop(self) -> None:
        if self._cids:
            for cid in self._cids:
                self._owner._canvas.mpl_disconnect(cid)
            self._cids = []
        self._active = False
        self._points = []
        self._drag_idx = -1
        for artist in (self._line, self._preview):
            if artist is not None:
                artist.remove()
        self._line = None
        self._preview = None
        self._owner.set_cursor(False)
        self._owner._canvas.draw_idle()

    # -- event handlers ------------------------------------------------

    def _on_press(self, event) -> None:
        if not self._active or event.inaxes is not self._owner.ax or event.button != 1:
            return
        if event.dblclick:
            self._finish()
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._points:
            idx = self._vertex_near(event.xdata, event.ydata)
            if idx >= 0:
                self._drag_idx = idx
                return
        self._points.append((float(event.xdata), float(event.ydata)))
        self._update_line_artist()
        self._owner._canvas.draw_idle()

    def _on_release(self, event) -> None:
        self._drag_idx = -1

    def _on_motion(self, event) -> None:
        if not self._active or event.inaxes is not self._owner.ax:
            return
        if self._drag_idx >= 0 and event.xdata is not None and event.ydata is not None:
            self._points[self._drag_idx] = (float(event.xdata), float(event.ydata))
            self._update_line_artist()
            self._owner._canvas.draw_idle()
        elif self._points and self._preview is not None:
            if event.xdata is None or event.ydata is None:
                return
            last = self._points[-1]
            self._preview.set_data([last[0], event.xdata], [last[1], event.ydata])
            self._preview.set_visible(True)
            self._owner._canvas.draw_idle()

    def _on_key(self, event) -> None:
        if not self._active:
            return
        if event.key == "enter":
            self._finish()
        elif event.key == "escape":
            self.cancelled.emit()
            self.stop()
        elif event.key in ("backspace", "delete") and self._points:
            self._points.pop()
            self._update_line_artist()
            self._owner._canvas.draw_idle()

    # -- helpers --------------------------------------------------------

    def _vertex_near(self, x: float, y: float) -> int:
        if not self._points:
            return -1
        pts = np.asarray(self._points, dtype=float)
        disp = self._owner.ax.transData.transform(pts)
        cursor = self._owner.ax.transData.transform((x, y))
        distances = np.hypot(disp[:, 0] - cursor[0], disp[:, 1] - cursor[1])
        idx = int(np.argmin(distances))
        return idx if distances[idx] <= DRAG_RADIUS_PX else -1

    def _ensure_artists(self) -> None:
        from matplotlib.lines import Line2D

        self._line = Line2D(
            [], [], color="#d62728", linewidth=1.5, marker="o",
            markersize=4.0, markerfacecolor="white", markeredgecolor="#d62728",
            zorder=5,
        )
        self._preview = Line2D(
            [], [], color="#d62728", linewidth=1.0, linestyle="--", zorder=5,
        )
        self._preview.set_visible(False)
        self._owner.ax.add_line(self._line)
        self._owner.ax.add_line(self._preview)

    def _update_line_artist(self) -> None:
        if self._line is None:
            return
        if self._points:
            xs, ys = zip(*self._points)
        else:
            xs, ys = [], []
        self._line.set_data(xs, ys)

    def _finish(self) -> None:
        points = list(self._points)
        # A double-click first appends a duplicate vertex: drop it.
        if len(points) >= 2 and np.allclose(points[-1], points[-2]):
            points.pop()
        if len(points) >= 2:
            self.finished.emit(points)
        else:
            # Not enough points: notify the owner so it leaves polyline mode.
            self.cancelled.emit()
            self.stop()
