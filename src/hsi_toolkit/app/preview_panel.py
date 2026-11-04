"""Preview panel: display mode, spectral-window sliders, stretch, colormap."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
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


class PreviewPanel(QWidget):
    annotation_created = Signal(str, list)
    annotation_selected = Signal(str)
    tool_cancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ws = None
        self._center_nm = 0.0
        self._width_nm = 0.0

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

        controls = QFormLayout()
        controls.addRow("Режим:", self.mode_combo)
        controls.addRow("Центр λ:", self._slider_row(self.center_slider, self.center_label))
        controls.addRow("Ширина окна:", self._slider_row(self.width_slider, self.width_label))
        controls.addRow("Растяжение:", self.stretch_combo)
        controls.addRow("Цветовая карта:", self.cmap_combo)

        side = QWidget()
        side.setLayout(controls)
        side.setMaximumWidth(360)

        top = QHBoxLayout()
        top.addWidget(self.canvas, stretch=1)
        top.addWidget(side)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(top)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.update_preview)

        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.center_slider.valueChanged.connect(self._on_slider_changed)
        self.width_slider.valueChanged.connect(self._on_slider_changed)
        self.stretch_combo.currentTextChanged.connect(self._on_options_changed)
        self.cmap_combo.currentTextChanged.connect(self._on_options_changed)
        self._set_controls_enabled(False)

    @staticmethod
    def _slider_row(slider: QSlider, label: QLabel) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(label)
        return row

    # -- workspace --------------------------------------------------------

    def set_workspace(self, ws) -> None:
        self._ws = ws
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
        ):
            widget.setEnabled(enabled)

    def _on_mode_changed(self, _text: str) -> None:
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
        return (
            "minmax"
            if self.stretch_combo.currentText() == STRETCH_MINMAX
            else (2.0, 98.0)
        )

    def update_preview(self) -> None:
        ws = self._ws
        if ws is None:
            return
        self._update_slider_labels()
        if ws.lowres is None:
            self.canvas.show_message("Строится уменьшенный куб для предпросмотра…")
            return
        # Data coordinates stay in full-resolution pixels even though the
        # displayed cube is downsampled, so annotations always align.
        extent = (-0.5, ws.samples - 0.5, ws.lines - 0.5, -0.5)
        if self.mode_combo.currentText() == MODE_RGB:
            image = preview_core.pseudo_rgb(ws, stretch=self._stretch)
        else:
            image = preview_core.window_preview(
                ws,
                center=self._center_nm,
                width=self._width_nm,
                stretch=self._stretch,
                cmap=self.cmap_combo.currentText(),
            )
        self.canvas.set_image(image, extent=extent)
