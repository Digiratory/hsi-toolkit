"""Hyperspectral dataset workspace: lazy loading and reference-based correction.

The main cube stays on disk as a memory map; each band is corrected on demand
with a small LRU cache. White/dark reference cubes are never held in memory
whole (they are ~622 MB each for the full-frame data): at open time they are
reduced to per-band, per-column mean profiles ``dark_prof[b, x]`` /
``white_prof[b, x]`` (mean over lines). This is the lazy equivalent of
``bw_correction_broadcast_axis(sample, dark, white, axis=1)`` from the research
notebooks and compensates the along-scan illumination variation per channel and
per column.

Correction formula (per band ``b``, column ``x``)::

    corrected[b, y, x] = clip(
        (raw[b, y, x] - dark_prof[b, x]) / (white_prof[b, x] - dark_prof[b, x] + eps),
        0.0, 1.0,
    )
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from hsi_toolkit.io.envi_hdr import HSIDataset, load_hsi_from_hdr

_EPS32 = np.float32(np.finfo(np.float32).eps)


class HSIWorkspace:
    """Lazy hyperspectral cube with optional white/dark reference correction."""

    def __init__(
        self,
        cube: HSIDataset,
        white: Optional[HSIDataset] = None,
        dark: Optional[HSIDataset] = None,
    ) -> None:
        if (white is None) != (dark is None):
            raise ValueError("Both white and dark references are required for correction")
        cube_wl = np.asarray(cube.wavelengths, dtype=np.float64)
        if cube_wl.size != cube.bands:
            raise ValueError("Cube HDR lacks per-band wavelengths; spectral axis is unknown")
        self.order = np.argsort(cube_wl, kind="stable")
        self.wavelengths = cube_wl[self.order]
        self.hdr_path = Path(cube.hdr_path)
        self._cube = cube
        self._white = white
        self._dark = dark
        self.samples = int(cube.samples)
        self.lines = int(cube.lines)
        self.bands = int(cube.bands)
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_max = 8
        self._closed = False
        self.lowres: Optional[np.ndarray] = None
        self.dark_prof: Optional[np.ndarray] = None
        self.white_prof: Optional[np.ndarray] = None
        if white is not None and dark is not None:
            self.dark_prof = _reference_profile(
                dark, self.wavelengths, self.bands, self.samples, "dark"
            )
            self.white_prof = _reference_profile(
                white, self.wavelengths, self.bands, self.samples, "white"
            )

    @classmethod
    def open(
        cls,
        cube_hdr: Path | str,
        white_hdr: Optional[Path | str] = None,
        dark_hdr: Optional[Path | str] = None,
    ) -> "HSIWorkspace":
        cube = load_hsi_from_hdr(Path(cube_hdr))
        white = load_hsi_from_hdr(Path(white_hdr)) if white_hdr else None
        dark = load_hsi_from_hdr(Path(dark_hdr)) if dark_hdr else None
        return cls(cube, white, dark)

    @property
    def has_references(self) -> bool:
        return self.dark_prof is not None

    @property
    def wavelength_range(self) -> Tuple[float, float]:
        return float(self.wavelengths[0]), float(self.wavelengths[-1])

    @property
    def raw_dtype(self) -> np.dtype:
        """Raw cube dtype in native byte order (from the ENVI header)."""
        return self._cube.dtype.newbyteorder("=")

    def raw_band(self, b: int) -> np.ndarray:
        """Raw band ``b`` (sorted by ascending wavelength) as a memmap view."""
        if self._closed:
            raise ValueError("Workspace is closed")
        return self._cube.get_band(int(self.order[int(b)]))

    def working_band(self, b: int) -> np.ndarray:
        """float32 band used for spectra/stats: corrected if references exist."""
        if self.has_references:
            return self.corrected_band(b)
        return np.asarray(self.raw_band(b), dtype=np.float32)

    def corrected_band(self, b: int) -> np.ndarray:
        """Corrected band ``b`` as float32 (H, W), values clipped to [0, 1]."""
        if not self.has_references:
            raise ValueError("No white/dark references loaded; correction unavailable")
        b = int(b)
        cached = self._cache.get(b)
        if cached is not None:
            self._cache.move_to_end(b)
            return cached
        raw = np.asarray(self.raw_band(b), dtype=np.float32)
        dark = self.dark_prof[b][None, :].astype(np.float32)
        white = self.white_prof[b][None, :].astype(np.float32)
        corrected = np.clip(
            (raw - dark) / (white - dark + _EPS32),
            np.float32(0.0),
            np.float32(1.0),
        )
        corrected = np.ascontiguousarray(corrected)
        self._cache[b] = corrected
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return corrected

    def band_index_near(self, wl: float) -> int:
        """Index of the band nearest to ``wl`` nanometers."""
        return int(np.argmin(np.abs(self.wavelengths - float(wl))))

    def bands_in_window(self, center: float, width: float) -> np.ndarray:
        """Indices of bands inside the spectral window [center ± width / 2]."""
        center = float(center)
        width = max(float(width), 0.0)
        lo, hi = center - width / 2.0, center + width / 2.0
        idx = np.flatnonzero((self.wavelengths >= lo) & (self.wavelengths <= hi))
        if idx.size == 0:
            idx = np.array([self.band_index_near(center)])
        return idx

    def build_lowres_cube(self, factor: int = 4) -> np.ndarray:
        """Downsampled float32 cube for interactive previews (B, H/f, W/f)."""
        factor = max(1, int(factor))
        h = (self.lines // factor) * factor
        w = (self.samples // factor) * factor
        out = np.empty((self.bands, h // factor, w // factor), dtype=np.float32)
        for b in range(self.bands):
            band = self.working_band(b)
            out[b] = (
                band[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
            )
        self.lowres = out
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cache.clear()
        self.lowres = None
        for ds in (self._cube, self._white, self._dark):
            if ds is not None:
                ds.close()


def _reference_profile(
    ds: HSIDataset,
    cube_wl: np.ndarray,
    bands: int,
    samples: int,
    name: str,
) -> np.ndarray:
    """Per-band, per-column mean over lines of a reference cube: (bands, samples)."""
    if ds.bands != bands:
        raise ValueError(f"{name} reference has {ds.bands} bands, the cube has {bands}")
    if ds.samples != samples:
        raise ValueError(f"{name} reference has {ds.samples} samples, the cube has {samples}")
    ds_wl = np.asarray(ds.wavelengths, dtype=np.float64)
    if ds_wl.size == ds.bands:
        order = np.argsort(ds_wl, kind="stable")
        if not np.allclose(ds_wl[order], cube_wl, atol=1.0):
            raise ValueError(f"{name} reference wavelengths differ from the cube wavelengths")
    else:
        order = np.arange(ds.bands)
    profile = np.empty((bands, samples), dtype=np.float64)
    for i, band_idx in enumerate(order):
        band = np.asarray(ds.get_band(int(band_idx)), dtype=np.float64)
        profile[i] = band.mean(axis=0)
    return profile


def find_reference_hdrs(cube_hdr: Path | str) -> Tuple[Optional[Path], Optional[Path]]:
    """Auto-discover WHITEREF_/DARKREF_ HDRs near the cube.

    Checks the cube folder, ``<cube folder>/capture`` and ``<parent>/capture``
    (both known mouse-exp data layouts).
    """
    hdr = Path(cube_hdr)
    candidates_dirs = [hdr.parent, hdr.parent / "capture", hdr.parent.parent / "capture"]
    white: Optional[Path] = None
    dark: Optional[Path] = None
    for folder in candidates_dirs:
        if not folder.is_dir():
            continue
        if white is None:
            white = _pick_reference(folder, "WHITEREF_", hdr.stem)
        if dark is None:
            dark = _pick_reference(folder, "DARKREF_", hdr.stem)
        if white is not None and dark is not None:
            break
    return white, dark


def _pick_reference(folder: Path, prefix: str, cube_stem: str) -> Optional[Path]:
    matches = sorted(folder.glob(f"{prefix}*.hdr"))
    if not matches:
        return None
    same_stem = [m for m in matches if cube_stem in m.stem]
    return (same_stem or matches)[0]
