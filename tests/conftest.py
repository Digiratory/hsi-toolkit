"""Shared test fixtures: synthetic ENVI datasets and a QApplication for GUI tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

BANDS, LINES, SAMPLES = 5, 32, 32
# Deliberately unsorted so workspace sorting is exercised.
WAVELENGTHS = [700.0, 500.0, 600.0, 550.0, 650.0]


def write_envi(
    path: Path,
    cube: np.ndarray,
    wavelengths=None,
    interleave: str = "bsq",
    envi_data_type: int = 12,
) -> Path:
    """Write a minimal ENVI hdr + raw pair for a (bands, lines, samples) cube."""
    path = Path(path)
    raw_path = path.with_suffix(".raw")
    dtype = {12: "<u2", 4: "<f4"}[envi_data_type]
    if interleave == "bsq":
        data = np.ascontiguousarray(cube, dtype=dtype)
    elif interleave == "bil":
        data = np.ascontiguousarray(np.moveaxis(cube, 1, 0), dtype=dtype)
    elif interleave == "bip":
        data = np.ascontiguousarray(np.moveaxis(cube, 2, 0), dtype=dtype)
    else:
        raise ValueError(interleave)
    raw_path.write_bytes(data.tobytes())
    bands, lines, samples = cube.shape
    if wavelengths is not None:
        wl = " ".join(f"{v:g}" for v in wavelengths)
        wavelength_lines = [f"wavelength = {{{wl}}}"]
    else:
        wavelength_lines = []
    hdr = "\n".join(
        [
            "ENVI",
            f"samples = {samples}",
            f"lines = {lines}",
            f"bands = {bands}",
            "header offset = 0",
            f"data type = {envi_data_type}",
            "byte order = 0",
            f"interleave = {interleave}",
            f"data file = {raw_path.name}",
            *wavelength_lines,
        ]
    )
    path.write_text(hdr, encoding="ascii")
    return path


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> dict:
    """Cube + white + dark ENVI datasets in tmp_path; returns the HDR paths."""
    xs = np.arange(SAMPLES, dtype=np.float64)
    ys = np.arange(LINES, dtype=np.float64)
    col_ramp = 200.0 + 100.0 * xs / SAMPLES
    row_ramp = 50.0 * ys / LINES
    band_gain = np.array([1.0, 1.2, 0.9, 1.1, 1.3])[:BANDS]

    base = col_ramp[None, :] + row_ramp[:, None]  # (H, W)
    cube = np.stack([(1000.0 + base * gain).astype(np.uint16) for gain in band_gain])
    # A darker quadrant so percentile stretches have some variance.
    cube[:, :8, :8] = (cube[:, :8, :8] * 0.4).astype(np.uint16)

    white_profile = (3800.0 + 100.0 * col_ramp / SAMPLES)
    dark_profile = (50.0 + 10.0 * col_ramp / SAMPLES)
    white_cube = np.stack(
        [np.broadcast_to(white_profile, (LINES, SAMPLES)).copy() for _ in range(BANDS)]
    ).astype(np.uint16)
    dark_cube = np.stack(
        [np.broadcast_to(dark_profile, (LINES, SAMPLES)).copy() for _ in range(BANDS)]
    ).astype(np.uint16)

    cube_hdr = write_envi(tmp_path / "cube.hdr", cube, WAVELENGTHS)
    white_hdr = write_envi(tmp_path / "WHITEREF_cube.hdr", white_cube, WAVELENGTHS)
    dark_hdr = write_envi(tmp_path / "DARKREF_cube.hdr", dark_cube, WAVELENGTHS)
    return {
        "cube": cube_hdr,
        "white": white_hdr,
        "dark": dark_hdr,
        "cube_raw": cube,
        "white_raw": white_cube,
        "dark_raw": dark_cube,
    }


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
