"""Spectral statistics over annotation regions.

``polyline_samples`` / ``polyline_profile`` are ports of ``polyline_band_profile``
from ``mouse_hsi.ipynb``: the cube is sampled along the polyline vertices with a
ribbon of ``width`` pixels perpendicular to the line, one cross-section per
~``step`` pixels; the per-step spectrum is the median across the ribbon, and
positions are normalized to 0..1 by the total line length. Coordinates are
clipped to the image bounds exactly as in the notebook.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np

CI = Union[Tuple[float, float], Tuple[str, float]]

DEFAULT_RIBBON_WIDTH = 10

Sample = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (ys (width, n), xs (width, n), positions (n,))


def ribbon_offsets(width: int) -> np.ndarray:
    """Perpendicular cross-section offsets for a ribbon of ``width`` pixels.

    Integer pixel offsets centered on the line (even widths put the extra
    sample above the center), so no offset is a half-integer and np.round's
    ties-to-even can never drop or duplicate a ribbon row.
    """
    width = max(int(width), 1)
    return np.arange(width, dtype=np.float64) - (width - 1) // 2


def segment_perpendicular(x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float]:
    """Unit perpendicular of the segment, notebook convention (caller skips zero segments)."""
    seg_length = float(np.hypot(x1 - x0, y1 - y0))
    return -(y1 - y0) / seg_length, (x1 - x0) / seg_length


def ribbon_pixel_indices(
    cx: np.ndarray,
    cy: np.ndarray,
    px: float,
    py: float,
    width: int,
    shape_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Ribbon pixel index arrays (ys, xs), each (width, n) — shared by the
    profile sampler and the polyline rasterizer so they can never diverge."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    xs = np.clip(np.round(cx[None, :] + ribbon_offsets(width)[:, None] * px).astype(int), 0, w - 1)
    ys = np.clip(np.round(cy[None, :] + ribbon_offsets(width)[:, None] * py).astype(int), 0, h - 1)
    return ys, xs


def polyline_samples(
    points: Sequence[Sequence[float]],
    shape_hw: Tuple[int, int],
    width: int = DEFAULT_RIBBON_WIDTH,
    step: float = 1.0,
) -> Tuple[List[Sample], float]:
    """Split a polyline into per-step ribbon pixel index arrays.

    Returns ``(samples, total_length)``: each sample is ``(ys, xs, positions)``
    with ``ys``/``xs`` shaped ``(width, n_steps)`` (clip-rounded pixel indices)
    and absolute along-line positions in pixels.
    """
    if len(points) < 2:
        raise ValueError("Polyline needs at least 2 points")
    if step <= 0:
        raise ValueError("step must be positive")
    width = max(int(width), 1)
    h, w = int(shape_hw[0]), int(shape_hw[1])
    pts = np.asarray(points, dtype=np.float64)
    segments = list(zip(pts[:-1], pts[1:]))
    total_length = sum(float(np.hypot(x1 - x0, y1 - y0)) for (x0, y0), (x1, y1) in segments)

    samples: List[Sample] = []
    distance = 0.0
    for i, ((x0, y0), (x1, y1)) in enumerate(segments):
        seg_length = float(np.hypot(x1 - x0, y1 - y0))
        if seg_length < 1e-9:
            continue
        steps = max(int(np.ceil(seg_length / step)), 2)
        t = np.linspace(0.0, 1.0, steps)
        if i > 0:
            t = t[1:]
        cx = x0 + t * (x1 - x0)
        cy = y0 + t * (y1 - y0)
        px, py = segment_perpendicular(x0, y0, x1, y1)
        ys, xs = ribbon_pixel_indices(cx, cy, px, py, width, (h, w))
        samples.append((ys, xs, distance + t * seg_length))
        distance += seg_length
    if not samples:
        raise ValueError("Polyline has zero length")
    return samples, total_length


def gather_polyline_ribbon(
    ws,
    points: Sequence[Sequence[float]],
    width: int = DEFAULT_RIBBON_WIDTH,
    step: float = 1.0,
    corrected: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full ribbon along a polyline: ``(data (B, S, width), positions (S,))``.

    No median is applied — the numerical analysis computes whatever statistics
    it needs. Shared by ``polyline_profile`` (display) and the NPZ exporter so
    both always sample the same pixels.
    """
    samples, total_length = polyline_samples(points, (ws.lines, ws.samples), width, step)
    if corrected and not ws.has_references:
        raise ValueError("Corrected ribbon requires white and dark references")
    n_steps = sum(int(ys.shape[1]) for ys, _, _ in samples)
    data = np.empty(
        (ws.bands, n_steps, width),
        dtype=np.float32 if corrected else ws.raw_dtype,
    )
    for b in range(ws.bands):
        band = ws.corrected_band(b) if corrected else np.asarray(ws.raw_band(b))
        col = 0
        for ys, xs, _ in samples:
            chunk = int(ys.shape[1])
            data[b, col : col + chunk, :] = band[ys, xs].T
            col += chunk
    positions = np.concatenate([pos for _, _, pos in samples]) / total_length
    return data, positions


def polyline_profile(
    ws,
    points: Sequence[Sequence[float]],
    width: int = DEFAULT_RIBBON_WIDTH,
    step: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Median spectrum along a polyline ribbon.

    Returns ``(rel_positions (S,), spectra (S, B))`` with positions in 0..1 and
    spectra as float32 in the workspace's working values (corrected or raw).
    """
    ribbon, positions = gather_polyline_ribbon(ws, points, width, step, corrected=ws.has_references)
    spectra = np.transpose(np.median(ribbon, axis=2)).astype(np.float32)
    return positions.astype(np.float64), spectra


def mask_band_values(ws, mask: np.ndarray) -> np.ndarray:
    """Working values of the masked pixels for every band: (B, n_pixels).

    ``working_band`` is corrected (float32, 0..1) when references exist,
    raw (float32) otherwise — the same values the spectra panel plots.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (ws.lines, ws.samples):
        raise ValueError(f"Mask shape {mask.shape} does not match image {(ws.lines, ws.samples)}")
    n_pixels = int(mask.sum())
    if n_pixels == 0:
        raise ValueError("Empty mask")
    values = np.empty((ws.bands, n_pixels), dtype=np.float32)
    for b in range(ws.bands):
        values[b] = ws.working_band(b)[mask]
    return values


def polygon_spectrum_stats(
    ws,
    mask: np.ndarray,
    stat: str = "median",
    ci: CI = (25.0, 75.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-band center + interval statistics over the masked pixels.

    ``stat``: ``"median"`` or ``"mean"``. ``ci``: ``(p_lo, p_hi)`` percentiles
    (for ``"median"`` typically ``(25, 75)`` IQR) or ``("std", k)`` for the
    mean ± ``k``·std interval. Returns ``(center (B,), lower (B,), upper (B,))``.
    """
    values = mask_band_values(ws, mask)
    if stat == "median":
        center = np.median(values, axis=1)
        lower, upper = _percentile_interval(values, ci)
    elif stat == "mean":
        center = values.mean(axis=1, dtype=np.float64)
        lower, upper = _std_interval(values, center, ci)
    else:
        raise ValueError(f"Unknown stat: {stat!r}")
    return center.astype(np.float64), lower.astype(np.float64), upper.astype(np.float64)


def _percentile_interval(values: np.ndarray, ci: CI) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(ci, str) or len(ci) != 2:
        raise ValueError("Percentile CI must be (p_lo, p_hi)")
    lo_pct, hi_pct = float(ci[0]), float(ci[1])
    result = np.percentile(values, [lo_pct, hi_pct], axis=1)
    return result[0], result[1]


def _std_interval(values: np.ndarray, center: np.ndarray, ci: CI) -> Tuple[np.ndarray, np.ndarray]:
    if not (isinstance(ci, tuple) and len(ci) == 2 and ci[0] == "std"):
        raise ValueError('For stat="mean" use ci=("std", k)')
    k = float(ci[1])
    std = values.std(axis=1, dtype=np.float64)
    return center - k * std, center + k * std
