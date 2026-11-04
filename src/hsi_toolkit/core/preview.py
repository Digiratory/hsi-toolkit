"""Preview image construction: pseudo-RGB and spectral-window previews.

Both functions return display-ready ``uint8 (H, W, 3)`` arrays. When the
workspace has a low-res cube built (``ws.lowres``), it is used as the source;
otherwise bands are computed lazily at full resolution.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

Stretch = Union[Tuple[float, float], str]


def _stretch_u8(channel: np.ndarray, stretch: Stretch) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float32)
    if isinstance(stretch, str) and stretch == "minmax":
        lo = float(channel.min())
        hi = float(channel.max())
    else:
        lo_pct, hi_pct = stretch  # type: ignore[misc]
        lo = float(np.percentile(channel, lo_pct))
        hi = float(np.percentile(channel, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def _apply_cmap(gray: np.ndarray, cmap: str) -> np.ndarray:
    from matplotlib import colormaps

    rgb = colormaps[cmap](gray / 255.0)[..., :3]
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _source_bands(ws, indices: Sequence[int]) -> np.ndarray:
    if ws.lowres is not None:
        return np.stack([ws.lowres[int(i)] for i in indices])
    return np.stack([ws.working_band(int(i)) for i in indices])


def pseudo_rgb(
    ws,
    wl_r: float = 650.0,
    wl_g: float = 550.0,
    wl_b: float = 450.0,
    stretch: Stretch = (2.0, 98.0),
) -> np.ndarray:
    """Nearest-band pseudo-RGB preview, uint8 (H, W, 3)."""
    indices = (
        ws.band_index_near(wl_r),
        ws.band_index_near(wl_g),
        ws.band_index_near(wl_b),
    )
    bands = _source_bands(ws, indices)
    return np.dstack([_stretch_u8(bands[i], stretch) for i in range(3)])


def window_preview(
    ws,
    center: float,
    width: float,
    stretch: Stretch = (2.0, 98.0),
    cmap: str = "gray",
) -> np.ndarray:
    """Mean-intensity preview over the spectral window, uint8 (H, W, 3)."""
    indices = ws.bands_in_window(center, width)
    if ws.lowres is not None:
        img = ws.lowres[indices].mean(axis=0)
    else:
        img = np.mean([ws.working_band(int(i)) for i in indices], axis=0)
    return _apply_cmap(_stretch_u8(np.asarray(img), stretch), cmap)
