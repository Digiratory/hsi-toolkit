"""Preview panel: display mode, spectral-window sliders, stretch, colormap.

In spectral-window mode a strip below the preview shows the integral spectrum
of the whole image (mean intensity per band) with the active window shaded on
top; the center/width sliders sit directly under it so the slider positions
map onto the wavelength axis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hsi_toolkit.app.canvas.preview_canvas import PreviewCanvas
from hsi_toolkit.core import preview as preview_core

MODE_RGB = "Псевдо-RGB"
MODE_WINDOW = "Спектральное окно"

STRETCH_PERCENTILE = "Перцентили 2–98"
STRETCH_MINMAX = "Min–max"

CMAPS = ("gray", "viridis", "inferno", "magma", "turbo", "cividis")

DEBOUNCE_MS = 100
SPECTRUM_STRIP_HEIGHT = 150


class PreviewPanel(QWidget):
    annotation_created = Signal(str, list)
    annotation_selected = Signal(str)
    tool_cancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ws = None
        self._center_nm = 0.0
        self._width_nm = 0.0
        self._integral: Optional[np.ndarray] = None
        self._last_region: Optional[tuple] = None

        self.canvas = PreviewCanvas()
        self.canvas.annotation_created.connect(self.annotation_created.emit)
        self.canvas.annotation_selected.connect(self.annotation_selected.emit)
        self.canvas.tool_cancelled.connect(self.tool_cancelled.emit)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([MODE_RGB, MODE_WINDOW])

        self.center_slider = QSlider(Qt.Orientation.Horizontal)
        self.width_slider = QSlider(Qt.Orientation.Horizontal)
        self.center_label = QLabel()
        self.width_label = QLabel()
        self.stretch_combo = QComboBox()
        self.stretch_combo.addItems([STRETCH_PERCENTILE, STRETCH_MINMAX])
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(CMAPS)
        # Annotation label controls (visibility + font size).
        self.labels_check = QCheckBox("Показывать")
        self.labels_check.setChecked(True)
        self.label_size_spin = QSpinBox()
        self.label_size_spin.setRange(4, 24)
        self.label_size_spin.setValue(8)
        self.label_size_spin.setToolTip("Размер шрифта подписей зон на предпросмотре")

        # Integral-spectrum strip with the window overlay (window mode only).
        self.spectrum_figure = Figure(figsize=(1, 1))
        self.spectrum_canvas = FigureCanvasQTAgg(self.spectrum_figure)
        self.spectrum_canvas.setMinimumHeight(SPECTRUM_STRIP_HEIGHT)

        controls = QFormLayout()
        controls.addRow("Режим:", self.mode_combo)
        controls.addRow("Растяжение:", self.stretch_combo)
        controls.addRow("Цветовая карта:", self.cmap_combo)
        controls.addRow("Подписи зон:", self._labels_row())

        side = QWidget()
        side.setLayout(controls)
        side.setMaximumWidth(320)

        sliders = QFormLayout()
        sliders.addRow("Центр λ:", self._slider_row(self.center_slider, self.center_label))
        sliders.addRow("Ширина окна:", self._slider_row(self.width_slider, self.width_label))

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(4, 0, 4, 0)
        bottom_layout.addWidget(self.spectrum_canvas)
        bottom_layout.addLayout(sliders)
        self._bottom = bottom

        top = QHBoxLayout()
        top.addWidget(self.canvas, stretch=1)
        top.addWidget(side)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(top)
        layout.addWidget(bottom)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.update_preview)

        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.center_slider.valueChanged.connect(self._on_slider_changed)
        self.width_slider.valueChanged.connect(self._on_slider_changed)
        self.stretch_combo.currentTextChanged.connect(self._on_options_changed)
        self.cmap_combo.currentTextChanged.connect(self._on_options_changed)
        self.labels_check.toggled.connect(self._on_labels_changed)
        self.label_size_spin.valueChanged.connect(self._on_labels_changed)
        # Pan/zoom of the preview changes the strip's spatial region.
        self.canvas.ax.callbacks.connect("xlim_changed", self._on_view_limits_changed)
        self.canvas.ax.callbacks.connect("ylim_changed", self._on_view_limits_changed)
        self._set_controls_enabled(False)
        self._bottom.setVisible(False)

    @staticmethod
    def _slider_row(slider: QSlider, label: QLabel) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(label)
        return row

    def _labels_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.labels_check)
        layout.addWidget(QLabel("размер:"))
        layout.addWidget(self.label_size_spin)
        layout.addStretch(1)
        return row

    def _on_labels_changed(self) -> None:
        self.canvas.set_label_settings(self.labels_check.isChecked(), self.label_size_spin.value())

    # -- workspace --------------------------------------------------------

    def set_workspace(self, ws) -> None:
        self._ws = ws
        self._integral = None
        self._last_region = None
        if ws is None:
            self._set_controls_enabled(False)
            self.canvas.show_message("Откройте набор данных (Файл → Открыть…)")
            return
        wl_min, wl_max = ws.wavelength_range
        span = max(wl_max - wl_min, 1.0)
        self.center_slider.setRange(int(np.floor(wl_min)), int(np.ceil(wl_max)))
        self.center_slider.setValue(int(round((wl_min + wl_max) / 2)))
        self.width_slider.setRange(1, int(np.ceil(span)))
        self.width_slider.setValue(max(int(round(span / 4)), 1))
        self._set_controls_enabled(True)
        self.update_preview()

    def refresh(self) -> None:
        """Call when ws.lowres becomes available (rebuild preview)."""
        self._integral = None  # recomputed from the fresh low-res cube
        if self._ws is not None:
            self.update_preview()

    def set_mode(self, mode: str) -> None:
        self.canvas.set_mode(mode)

    @property
    def has_image(self) -> bool:
        return self.canvas._image_artist is not None

    # -- internals ---------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.mode_combo,
            self.center_slider,
            self.width_slider,
            self.stretch_combo,
            self.cmap_combo,
            self.labels_check,
            self.label_size_spin,
        ):
            widget.setEnabled(enabled)

    def _on_mode_changed(self, _text: str) -> None:
        self._bottom.setVisible(self.mode_combo.currentText() == MODE_WINDOW)
        self.update_preview()

    def _on_slider_changed(self, _value: int) -> None:
        self._update_slider_labels()
        self._debounce.start()

    def _on_options_changed(self, _text: str) -> None:
        self._debounce.start()

    def _update_slider_labels(self) -> None:
        if self._ws is None:
            return
        self._center_nm = float(self.center_slider.value())
        self._width_nm = float(self.width_slider.value())
        self.center_label.setText(f"{self._center_nm:.0f} нм")
        self.width_label.setText(f"{self._width_nm:.0f} нм")

    @property
    def _stretch(self):
        return "minmax" if self.stretch_combo.currentText() == STRETCH_MINMAX else (2.0, 98.0)

    def _integral_spectrum(self, ws) -> np.ndarray:
        """Mean intensity per band over the image region currently in view."""
        region = self._view_region(ws)
        if self._integral is None or self._last_region != region:
            source = ws.lowres
            if source is None:
                return np.asarray(ws.wavelengths) * np.nan  # placeholder shape
            fx = ws.samples / source.shape[2]
            fy = ws.lines / source.shape[1]
            x0, x1, y0, y1 = region
            x0l = int(np.clip(np.floor(x0 / fx), 0, source.shape[2] - 1))
            x1l = int(np.clip(np.ceil(x1 / fx), x0l + 1, source.shape[2]))
            y0l = int(np.clip(np.floor(y0 / fy), 0, source.shape[1] - 1))
            y1l = int(np.clip(np.ceil(y1 / fy), y0l + 1, source.shape[1]))
            self._integral = np.asarray(
                source[:, y0l:y1l, x0l:x1l].mean(axis=(1, 2)), dtype=np.float64
            )
            self._last_region = region
        return self._integral

    def _view_region(self, ws) -> tuple:
        """Currently visible image area in full-res pixels, (x0, x1, y0, y1)."""
        x0, x1 = sorted(self.canvas.ax.get_xlim())
        y0, y1 = sorted(self.canvas.ax.get_ylim())
        x0 = int(np.clip(np.floor(x0), 0, ws.samples - 1))
        x1 = int(np.clip(np.ceil(x1), x0 + 1, ws.samples))
        y0 = int(np.clip(np.floor(y0), 0, ws.lines - 1))
        y1 = int(np.clip(np.ceil(y1), y0 + 1, ws.lines))
        return (x0, x1, y0, y1)

    def _on_view_limits_changed(self, _ax) -> None:
        """Pan/zoom of the preview changes the spatial region of the strip."""
        if self._ws is None or self.mode_combo.currentText() != MODE_WINDOW:
            return
        if self._view_region(self._ws) != self._last_region:
            self._debounce.start()

    def _draw_spectrum_strip(self, ws) -> None:
        """Integral spectrum of the visible region with the window shaded."""
        region = self._view_region(ws)
        integral = self._integral_spectrum(ws)
        self.spectrum_figure.clf()
        ax = self.spectrum_figure.add_subplot(1, 1, 1)
        wavelengths = np.asarray(ws.wavelengths, dtype=np.float64)
        valid = np.isfinite(integral)
        if valid.any():
            ax.fill_between(wavelengths[valid], integral[valid], 0, color="#888888", alpha=0.35)
            ax.plot(wavelengths[valid], integral[valid], color="#555555", linewidth=1.0)

        lo = self._center_nm - self._width_nm / 2.0
        hi = self._center_nm + self._width_nm / 2.0
        ax.axvspan(lo, hi, color="#1f77b4", alpha=0.22)
        ax.axvline(self._center_nm, color="#1f77b4", linewidth=1.2)

        # Pseudo-RGB channel positions for reference.
        for target, color in ((650.0, "#d62728"), (550.0, "#2ca02c"), (450.0, "#1f77b4")):
            wl = float(ws.wavelengths[ws.band_index_near(target)])
            ax.axvline(wl, color=color, linewidth=1.0, linestyle="--", alpha=0.8)

        wl_min, wl_max = ws.wavelength_range
        ax.set_xlim(wl_min, wl_max)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("λ, нм", fontsize=8)
        ax.set_ylabel("Средняя интенсивность", fontsize=8)
        x0, x1, y0, y1 = region
        if (x1 - x0, y1 - y0) < (ws.samples, ws.lines):
            ax.set_title(f"Область: x {x0}–{x1}, y {y0}–{y1}", fontsize=8)
        ax.tick_params(labelsize=8)
        self.spectrum_figure.tight_layout(pad=0.8)
        self.spectrum_canvas.draw_idle()

    def update_preview(self) -> None:
        ws = self._ws
        if ws is None:
            return
        self._update_slider_labels()
        window_mode = self.mode_combo.currentText() == MODE_WINDOW
        self._bottom.setVisible(window_mode)
        if ws.lowres is None:
            self.canvas.show_message("Строится уменьшенный куб для предпросмотра…")
            return
        # Data coordinates stay in full-resolution pixels even though the
        # displayed cube is downsampled, so annotations always align.
        extent = (-0.5, ws.samples - 0.5, ws.lines - 0.5, -0.5)
        if not window_mode:
            image = preview_core.pseudo_rgb(ws, stretch=self._stretch)
        else:
            image = preview_core.window_preview(
                ws,
                center=self._center_nm,
                width=self._width_nm,
                stretch=self._stretch,
                cmap=self.cmap_combo.currentText(),
            )
            self._draw_spectrum_strip(ws)
        self.canvas.set_image(image, extent=extent)
