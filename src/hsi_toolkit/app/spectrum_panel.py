"""Spectra panel: polygon median+CI figure and polyline 3D/waterfall plots."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from hsi_toolkit.app.workers import FunctionWorker
from hsi_toolkit.core.annotations import Annotation, rasterize
from hsi_toolkit.core.spectra import (
    DEFAULT_RIBBON_WIDTH,
    polygon_spectrum_stats,
    polyline_profile,
)

VIEW_3D = "3D"
VIEW_WATERFALL = "Водопад"
WATERFALL_MAP = "Карта позиция×λ"
WATERFALL_TRACES = "Трассы"

CI_IQR = "Медиана + IQR (25–75%)"
CI_P975 = "Медиана + перцентили (2.5–97.5%)"
CI_STD = "Среднее ± 3·STD"

POLYLINE_RIBBON = DEFAULT_RIBBON_WIDTH
MAX_TRACES = 30


def _ci_setting(index: int) -> Tuple[str, tuple]:
    return {
        0: ("median", (25.0, 75.0)),
        1: ("median", (2.5, 97.5)),
        2: ("mean", ("std", 3.0)),
    }.get(index, ("median", (25.0, 75.0)))


class SpectrumPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ws = None
        self._generation = 0
        self._worker: Optional[FunctionWorker] = None
        self._running_workers: list = []
        self._result_polygons: Optional[list] = None
        self._result_lines: Optional[list] = None
        self._cancel_current: Optional[dict] = None
        self._cache: dict = {}

        self.view_combo = QComboBox()
        self.view_combo.addItems([VIEW_3D, VIEW_WATERFALL])
        self.waterfall_combo = QComboBox()
        self.waterfall_combo.addItems([WATERFALL_MAP, WATERFALL_TRACES])
        self.ci_combo = QComboBox()
        self.ci_combo.addItems([CI_IQR, CI_P975, CI_STD])

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Полилинии:"))
        bar.addWidget(self.view_combo)
        bar.addWidget(self.waterfall_combo)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Полигонов интервал:"))
        bar.addWidget(self.ci_combo)
        bar.addStretch(1)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, stretch=1)

        self.view_combo.currentIndexChanged.connect(self._redraw)
        self.waterfall_combo.currentTextChanged.connect(self._redraw)
        self.ci_combo.currentIndexChanged.connect(self._redraw)
        self.show_message("Нет видимых объектов — отметьте их галочкой в списке")

    # -- public ---------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return bool(self._running_workers)

    @property
    def has_content(self) -> bool:
        return self.figure.axes and not self.figure.axes[0].texts

    def set_workspace(self, ws) -> None:
        self._generation += 1
        self._ws = ws
        self._cache.clear()
        self._invalidate()

    def shutdown(self) -> None:
        """Request cancellation and wait for running workers (teardown)."""
        if self._cancel_current is not None:
            self._cancel_current["cancelled"] = True
        for worker in list(self._running_workers):
            if worker.isRunning():
                worker.wait(30000)
        self._running_workers.clear()

    @staticmethod
    def _object_key(ws, ann: Annotation, stat: str, ci) -> tuple:
        """Cache key: one computed spectrum per object geometry + settings."""
        key_extra = (POLYLINE_RIBBON,) if ann.kind == "polyline" else ()
        return (id(ws), ann.id, ann.kind, tuple(map(tuple, ann.points)), stat, ci) + key_extra

    def update_spectra(self, annotations: List[Annotation]) -> None:
        """Plot the given (visible) annotations, recomputing only uncached ones."""
        self._generation += 1
        generation = self._generation
        if self._ws is None or not annotations:
            self._invalidate()
            return
        polygons = [a for a in annotations if a.kind == "polygon"]
        lines = [a for a in annotations if a.kind == "polyline"]
        if not polygons and not lines:
            self._invalidate()
            return
        stat, ci = _ci_setting(self.ci_combo.currentIndex())
        ws_id = id(self._ws)
        requested_keys = {self._object_key(self._ws, a, stat, ci) for a in annotations}
        for key in [k for k in self._cache if k[0] == ws_id and k not in requested_keys]:
            del self._cache[key]  # drop deleted/hidden objects and stale settings

        cached_polygons: List[tuple] = []
        cached_lines: List[tuple] = []
        missing: List[Annotation] = []
        for ann in annotations:
            key = self._object_key(self._ws, ann, stat, ci)
            hit = self._cache.get(key)
            if hit is None:
                missing.append(ann)
            elif ann.kind == "polygon":
                cached_polygons.append((ann.id, *hit))
            else:
                cached_lines.append((ann.id, *hit))

        if not missing:
            # Everything cached (e.g., a checkbox toggled back on): no compute.
            self._cancel_current = None
            self._result_polygons, self._result_lines = cached_polygons, cached_lines
            self._redraw()
            return

        # Ask a superseded computation to stop between bands.
        if self._cancel_current is not None:
            self._cancel_current["cancelled"] = True
        cancel = {"cancelled": False}
        self._cancel_current = cancel
        self.show_message("Расчёт спектров…")
        ws = self._ws

        def compute(progress):
            computed = {}
            for i, ann in enumerate(missing):
                if cancel["cancelled"]:
                    return None
                progress.emit(
                    int(50 * i / max(len(missing), 1)), f"Объект {ann.name}"
                )
                key = self._object_key(ws, ann, stat, ci)
                if ann.kind == "polygon":
                    mask = rasterize(ann, (ws.lines, ws.samples))
                    center, lower, upper = polygon_spectrum_stats(ws, mask, stat, ci)
                    computed[key] = (center, lower, upper)
                else:
                    rel, spectra = polyline_profile(
                        ws, ann.points, width=POLYLINE_RIBBON, step=1.0
                    )
                    computed[key] = (rel, spectra)
            progress.emit(100, "Готово")
            return computed

        self._worker = FunctionWorker(compute)
        self._running_workers.append(self._worker)
        self._worker.finished.connect(
            lambda worker=self._worker: self._forget_worker(worker)
        )
        self._worker.finished_ok.connect(
            lambda result, gen=generation, req=list(annotations): self._on_computed(
                gen, result, req
            )
        )
        self._worker.failed.connect(
            lambda message, gen=generation: self._on_failed(gen, message)
        )
        self._worker.start()

    def _forget_worker(self, worker: FunctionWorker) -> None:
        if worker in self._running_workers:
            self._running_workers.remove(worker)

    def _invalidate(self) -> None:
        self._result_polygons = None
        self._result_lines = None
        self._redraw()

    def _on_computed(self, generation: int, result, requested: List[Annotation]) -> None:
        if generation != self._generation or result is None:
            return
        if self._ws is None:
            return
        self._cache.update(result)
        stat, ci = _ci_setting(self.ci_combo.currentIndex())
        self._result_polygons = [
            (a.id, *self._cache[self._object_key(self._ws, a, stat, ci)])
            for a in requested
            if a.kind == "polygon" and self._object_key(self._ws, a, stat, ci) in self._cache
        ]
        self._result_lines = [
            (a.id, *self._cache[self._object_key(self._ws, a, stat, ci)])
            for a in requested
            if a.kind == "polyline" and self._object_key(self._ws, a, stat, ci) in self._cache
        ]
        self._redraw()

    def _on_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self.show_message(f"Ошибка расчёта спектров: {message}")

    # -- drawing ---------------------------------------------------------------

    def show_message(self, text: str) -> None:
        self.figure.clf()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color="#666666")
        self.canvas.draw_idle()

    def _redraw(self) -> None:
        if self._result_lines:
            if self.view_combo.currentText() == VIEW_3D:
                self._draw_lines_3d(self._result_lines)
            else:
                self._draw_lines_waterfall(self._result_lines)
        elif self._result_polygons:
            self._draw_polygons(self._result_polygons)
        else:
            self.show_message("Нет видимых объектов — отметьте их галочкой в списке")

    def _draw_polygons(self, results: list) -> None:
        self.figure.clf()
        ax = self.figure.add_subplot(1, 1, 1)
        drawn = 0
        for ann_id, center, lower, upper in results:
            ann = self._find_annotation(ann_id)
            if ann is not None and not ann.visible:
                continue
            color = ann.color if ann is not None else "#1f77b4"
            name = ann.name if ann is not None else ann_id
            wavelengths = self._ws.wavelengths
            line, = ax.plot(
                wavelengths, center, color=color, linewidth=1.6, label=name
            )
            ax.fill_between(
                wavelengths, lower, upper, color=color, alpha=0.3
            )
            drawn += 1
        ax.set_xlabel("Длина волны, нм")
        ax.set_ylabel("Спектральная мощность")
        ax.set_title("Медианные спектры полигонов с интервалами")
        if drawn:
            ax.legend(loc="best", fontsize=9)
        self.figure.tight_layout(pad=1.2)
        self.canvas.draw_idle()

    def _draw_lines_3d(self, results: list) -> None:
        # Axes3D ignores layout engines and colorbar-steal repositions the
        # axes on draw: use explicit positions and a dedicated cax instead.
        self.figure.clf()
        n = len(results)
        slot = 1.0 / n
        wavelengths = self._ws.wavelengths
        for i, (ann_id, rel, spectra) in enumerate(results):
            ann = self._find_annotation(ann_id)
            if ann is not None and not ann.visible:
                continue
            ax = self.figure.add_subplot(1, n, i + 1, projection="3d")
            ax.set_position([slot * i + 0.015, 0.07, slot - 0.13, 0.86])
            cax = self.figure.add_axes([slot * (i + 1) - 0.105, 0.25, 0.014, 0.5])
            ann = self._find_annotation(ann_id)
            name = ann.name if ann is not None else ann_id
            rel_grid, wl_grid = np.meshgrid(rel, wavelengths, indexing="ij")
            surf = ax.plot_surface(
                rel_grid, wl_grid, spectra,
                cmap="viridis", linewidth=0, antialiased=True,
            )
            self.figure.colorbar(
                surf, cax=cax, label="Спектральная мощность"
            )
            ax.set_xlabel("Отн. позиция")
            ax.set_ylabel("λ, нм")
            ax.set_zlabel("Мощность")
            ax.set_title(name, fontsize=10)
        self.canvas.draw_idle()

    def _draw_lines_waterfall(self, results: list) -> None:
        self.figure.clf()
        n = len(results)
        wavelengths = self._ws.wavelengths
        use_map = self.waterfall_combo.currentText() == WATERFALL_MAP
        for i, (ann_id, rel, spectra) in enumerate(results):
            ann = self._find_annotation(ann_id)
            if ann is not None and not ann.visible:
                continue
            ax = self.figure.add_subplot(1, n, i + 1)
            name = ann.name if ann is not None else ann_id
            if use_map:
                image = ax.imshow(
                    spectra.T,
                    aspect="auto",
                    origin="lower",
                    extent=[
                        float(rel[0]), float(rel[-1]),
                        float(wavelengths[0]), float(wavelengths[-1]),
                    ],
                    cmap="viridis", interpolation="nearest",
                )
                self.figure.colorbar(
                    image, ax=ax, shrink=0.8, pad=0.05, label="Мощность"
                )
                ax.set_xlabel("Отн. позиция")
                ax.set_ylabel("λ, нм")
            else:
                vmax = float(np.nanmax(spectra)) if spectra.size else 1.0
                offset = vmax * 1.05 if vmax > 0 else 1.0
                indices = np.unique(
                    np.linspace(0, spectra.shape[0] - 1, min(spectra.shape[0], MAX_TRACES))
                    .astype(int)
                )
                for j, s in enumerate(indices):
                    ax.plot(
                        wavelengths, spectra[s] + j * offset,
                        linewidth=0.8, color="#1f77b4",
                    )
                ax.set_xlabel("λ, нм")
                ax.set_ylabel("Трассы (позиции)")
                ax.set_yticks([])
            ax.set_title(name, fontsize=10)
        self.figure.tight_layout(pad=1.2)
        self.canvas.draw_idle()

    def _find_annotation(self, ann_id: str) -> Optional[Annotation]:
        return self._annotations_ref.get(ann_id) if self._annotations_ref else None

    @property
    def _annotations_ref(self):
        return getattr(self, "_annotations", None)

    def set_annotation_source(self, annotations) -> None:
        self._annotations = annotations
