"""Vector annotations (polygons and polylines) with JSON persistence.

Annotation points are ``(x, y)`` pixel coordinates on the image (``x`` along
samples/columns, ``y`` along lines/rows). Annotations are created in the app
and stored next to the cube as ``<cube name>.annotations.json``.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np

from hsi_toolkit.core.spectra import (
    ribbon_pixel_indices,
    segment_perpendicular,
)

SCHEMA = "hsi-toolkit.annotations/1"

TAB10 = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

Point = Tuple[float, float]
Kind = Literal["polygon", "polyline"]


def assign_color(index: int) -> str:
    return TAB10[index % len(TAB10)]


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Annotation:
    id: str
    kind: Kind
    name: str
    points: List[Point]
    comment: str = ""
    color: str = ""
    visible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "points": [[float(x), float(y)] for x, y in self.points],
            "comment": self.comment,
            "color": self.color,
            "visible": bool(self.visible),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        kind = data.get("kind")
        if kind not in ("polygon", "polyline"):
            raise ValueError(f"Unknown annotation kind: {kind!r}")
        points = [(float(x), float(y)) for x, y in data["points"]]
        if kind == "polygon" and len(points) < 3:
            raise ValueError("Polygon needs at least 3 points")
        if kind == "polyline" and len(points) < 2:
            raise ValueError("Polyline needs at least 2 points")
        return cls(
            id=str(data["id"]),
            kind=kind,
            name=str(data["name"]),
            points=points,
            comment=str(data.get("comment", "")),
            color=str(data.get("color", "")),
            visible=bool(data.get("visible", True)),
        )


class AnnotationSet:
    """Ordered collection of annotations plus the image metadata they refer to."""

    def __init__(self, image_info: Optional[Dict[str, Any]] = None) -> None:
        self.objects: List[Annotation] = []
        self.image_info: Dict[str, Any] = dict(image_info or {})

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterator[Annotation]:
        return iter(self.objects)

    def add(self, annotation: Annotation) -> Annotation:
        if self.get(annotation.id) is not None:
            raise ValueError(f"Duplicate annotation id: {annotation.id}")
        self.objects.append(annotation)
        return annotation

    def remove(self, annotation_id: str) -> Optional[Annotation]:
        for i, ann in enumerate(self.objects):
            if ann.id == annotation_id:
                return self.objects.pop(i)
        return None

    def get(self, annotation_id: str) -> Optional[Annotation]:
        for ann in self.objects:
            if ann.id == annotation_id:
                return ann
        return None

    def save(self, path: Path | str) -> Path:
        payload = {
            "schema": SCHEMA,
            "image": dict(self.image_info),
            "objects": [ann.to_dict() for ann in self.objects],
        }
        path = Path(path)
        # Atomic write: a crash mid-write must not destroy the previous copy.
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "AnnotationSet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = payload.get("schema")
        if schema != SCHEMA:
            raise ValueError(f"Unsupported annotations schema: {schema!r}")
        image_info = payload.get("image") or {}
        annotation_set = cls(image_info=image_info)
        for obj in payload.get("objects", []):
            annotation_set.add(Annotation.from_dict(obj))
        return annotation_set


def annotations_path_for(hdr_path: Path | str) -> Path:
    """Default JSON sidecar path for a cube: ``<cube>.annotations.json``."""
    return Path(hdr_path).with_suffix(".annotations.json")


def polygon_bounds(points: List[Point], shape_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Clamped inclusive pixel bbox (x0, y0, x1, y1) covering all points.

    Single source of truth for the rasterizer and the NPZ exporter so an
    exported cutout can never clip mask pixels at the bbox edges.
    """
    pts = np.asarray(points, dtype=np.float64)
    h, w = int(shape_hw[0]), int(shape_hw[1])
    x0 = max(0, int(np.floor(pts[:, 0].min())))
    x1 = min(w - 1, int(np.ceil(pts[:, 0].max())))
    y0 = max(0, int(np.floor(pts[:, 1].min())))
    y1 = min(h - 1, int(np.ceil(pts[:, 1].max())))
    return x0, y0, x1, y1


def rasterize(
    annotation: Annotation, shape_hw: Tuple[int, int], ribbon_width: int = 10
) -> np.ndarray:
    """Boolean mask (H, W) of the annotation footprint on the image grid."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    mask = np.zeros((h, w), dtype=bool)
    if annotation.kind == "polygon":
        _fill_polygon(mask, annotation.points)
    else:
        _fill_polyline_ribbon(mask, annotation.points, int(ribbon_width))
    return mask


def _fill_polygon(mask: np.ndarray, points: List[Point]) -> None:
    """Scanline even-odd polygon fill over integer pixel centers.

    matplotlib's ``Path.contains_points`` is unreliable for axis-aligned
    rectangles, so the mask is computed directly. A pixel (row y, col x) is
    inside when its center is between an odd/even pair of edge crossings,
    with half-open spans ``[xa, xb)`` for deterministic boundaries.
    """
    if len(points) < 3:
        return
    h, w = mask.shape
    pts = np.asarray(points, dtype=np.float64)
    _, y_min, _, y_max = polygon_bounds(points, mask.shape)
    edges = list(zip(pts, np.roll(pts, -1, axis=0)))
    for y in range(y_min, y_max + 1):
        crossings = []
        for (x0, y0), (x1, y1) in edges:
            if y0 == y1:
                continue
            if (y0 <= y < y1) or (y1 <= y < y0):
                t = (y - y0) / (y1 - y0)
                crossings.append(x0 + t * (x1 - x0))
        if not crossings:
            continue
        crossings.sort()
        row = mask[y]
        for i in range(0, len(crossings) - 1, 2):
            xa, xb = crossings[i], crossings[i + 1]
            start = max(int(np.ceil(xa)), 0)
            stop = min(int(np.ceil(xb)), w)
            if start < stop:
                row[start:stop] = True


def _fill_polyline_ribbon(mask: np.ndarray, points: List[Point], width: int) -> None:
    h, w = mask.shape
    if len(points) < 2 or width < 1:
        return
    width = max(1, int(width))
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        seg_length = float(np.hypot(x1 - x0, y1 - y0))
        if seg_length < 1e-9:
            continue
        # Sample at half-pixel spacing so no pixel center is skipped
        # (polyline_samples keeps the notebook's ~1 px spacing for spectra).
        steps = max(int(np.ceil(seg_length * 2)), 2)
        t = np.linspace(0.0, 1.0, steps)
        cx = x0 + t * (x1 - x0)
        cy = y0 + t * (y1 - y0)
        px, py = segment_perpendicular(x0, y0, x1, y1)
        ys, xs = ribbon_pixel_indices(cx, cy, px, py, width, mask.shape)
        mask[ys, xs] = True
