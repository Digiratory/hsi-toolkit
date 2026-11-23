"""NPZ export of annotation image regions (cube cutouts + masks).

Polygons are exported as a bbox cutout of the cube ``(B, H, W)`` plus a boolean
mask ``(H, W)`` of the polygon pixels inside the cutout. Polylines are exported
as the full ribbon ``(B, S, width)`` sampled along the line (no median applied —
the numerical analysis computes whatever statistics it needs) plus relative
positions ``(S,)`` in 0..1. Metadata (id/kind/name/comment/color/points and
bbox or ribbon geometry) is stored as a JSON string under ``objects``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from hsi_toolkit.core.annotations import Annotation, polygon_bounds, rasterize
from hsi_toolkit.core.spectra import (
    DEFAULT_RIBBON_WIDTH,
    gather_polyline_ribbon,
)

_TRANSLIT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def slugify(name: str, max_length: int = 40) -> str:
    """ASCII slug from a (usually Russian) object name."""
    lowered = str(name).strip().lower()
    translit = "".join(_TRANSLIT.get(ch, ch) for ch in lowered)
    slug = re.sub(r"[^a-z0-9]+", "_", translit)
    slug = re.sub(r"_+", "_", slug).strip("_")
    slug = slug[:max_length].strip("_")
    return slug or "object"


def export_npz(
    ws,
    annotations: Iterable[Annotation],
    path: Path | str,
    corrected: bool = True,
    selected: Optional[Sequence[str]] = None,
    ribbon_width: int = DEFAULT_RIBBON_WIDTH,
) -> List[str]:
    """Export annotation regions to ``.npz``; returns the exported object keys."""
    anns = list(annotations)
    if selected is not None:
        wanted = set(selected)
        anns = [a for a in anns if a.id in wanted]
    anns = [a for a in anns if a.visible]
    if not anns:
        raise ValueError("No annotations selected for export")
    if corrected and not ws.has_references:
        raise ValueError("Corrected export requires white and dark references")

    arrays: Dict[str, np.ndarray] = {"wavelengths": np.asarray(ws.wavelengths, dtype=np.float64)}
    used_keys: set = set()
    meta_objects: List[Dict] = []
    for index, ann in enumerate(anns, start=1):
        base = f"{index:02d}_{slugify(ann.name)}"
        base = _unique_key(base, used_keys)
        if ann.kind == "polygon":
            data, mask, bbox = _polygon_arrays(ws, ann, bool(corrected))
            arrays[base] = data
            arrays[base + "__mask"] = mask
            geometry: Dict = {"bbox": [int(v) for v in bbox]}
        else:
            data, positions, width_used = _polyline_arrays(
                ws, ann, bool(corrected), int(ribbon_width)
            )
            arrays[base] = data
            arrays[base + "__positions"] = positions
            geometry = {"ribbon_width": int(width_used)}
        meta_objects.append(
            {
                "key": base,
                "id": ann.id,
                "kind": ann.kind,
                "name": ann.name,
                "comment": ann.comment,
                "color": ann.color,
                "points": [[float(x), float(y)] for x, y in ann.points],
                **geometry,
            }
        )
    arrays["objects"] = np.array(json.dumps(meta_objects, ensure_ascii=False))
    np.savez_compressed(path, **arrays)
    return [obj["key"] for obj in meta_objects]


def _unique_key(base: str, used: set) -> str:
    key = base
    suffix = 2
    while key in used:
        key = f"{base}_{suffix}"
        suffix += 1
    used.add(key)
    return key


def _polygon_arrays(
    ws, ann: Annotation, corrected: bool
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = polygon_bounds(ann.points, (ws.lines, ws.samples))
    if x1 < x0 or y1 < y0:
        raise ValueError(f"Polygon '{ann.name}' is outside the image")
    mask_full = rasterize(ann, (ws.lines, ws.samples))
    data = np.empty(
        (ws.bands, y1 - y0 + 1, x1 - x0 + 1),
        dtype=np.float32 if corrected else ws.raw_dtype,
    )
    for b in range(ws.bands):
        band = ws.corrected_band(b) if corrected else np.asarray(ws.raw_band(b))
        data[b] = band[y0 : y1 + 1, x0 : x1 + 1]
    return data, mask_full[y0 : y1 + 1, x0 : x1 + 1], (x0, y0, x1, y1)


def _polyline_arrays(
    ws, ann: Annotation, corrected: bool, ribbon_width: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    data, positions = gather_polyline_ribbon(ws, ann.points, ribbon_width, 1.0, corrected=corrected)
    return data, positions, ribbon_width
