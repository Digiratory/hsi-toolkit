"""ENVI HDR file loader returning numpy arrays.

Loads ENVI HDR/RAW hyperspectral cubes as lazily memory-mapped
:class:`HSIDataset` objects (per-band access via ``get_band``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ENVI_DTYPE_MAP = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


@dataclass(frozen=True)
class HSIDataset:
    hdr_path: Path
    raw_path: Path
    samples: int
    lines: int
    bands: int
    interleave: str
    dtype: np.dtype
    offset: int
    byte_order: int
    wavelengths: List[float]
    description: str
    mm: np.memmap

    def close(self) -> None:
        mmap_obj = getattr(self.mm, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()

    def get_band(self, band_index: int) -> np.ndarray:
        b = int(band_index)
        if b < 0 or b >= self.bands:
            raise IndexError("band index out of range")
        if self.interleave == "bsq":
            band = self.mm[b, :, :]
        elif self.interleave == "bil":
            band = self.mm[:, b, :]
        elif self.interleave == "bip":
            band = self.mm[:, :, b]
        else:
            raise ValueError(f"Unsupported interleave: {self.interleave}")
        return np.array(band, copy=False)


def _strip_braces(value: str) -> str:
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1].strip()
    return value


def parse_envi_hdr(hdr_path: Path | str) -> Dict[str, Any]:
    path = Path(hdr_path)
    if not path.is_file():
        raise FileNotFoundError(f"HDR not found: {path}")

    props: Dict[str, str] = {}
    collecting_key: Optional[str] = None
    collecting_parts: List[str] = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";") or line == "ENVI":
            continue

        if collecting_key is not None:
            collecting_parts.append(line)
            if "}" in line:
                props[collecting_key] = " ".join(collecting_parts)
                collecting_key = None
                collecting_parts = []
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("{") and "}" not in value:
            collecting_key = key
            collecting_parts = [value]
        else:
            props[key] = value

    if collecting_key is not None:
        raise ValueError(f"Unterminated ENVI list in {path}")

    def as_int(key: str, default: Optional[int] = None) -> Optional[int]:
        raw = props.get(key)
        if raw is None:
            return default
        return int(_strip_braces(raw))

    wavelengths: List[float] = []
    if "wavelength" in props:
        for part in re.split(r"[,\s]+", _strip_braces(props["wavelength"])):
            if part:
                wavelengths.append(float(part))

    return {
        "samples": as_int("samples"),
        "lines": as_int("lines"),
        "bands": as_int("bands"),
        "header offset": as_int("header offset", 0),
        "data type": as_int("data type"),
        "byte order": as_int("byte order", 0),
        "interleave": _strip_braces(props.get("interleave", "bsq")).lower(),
        "data file": _strip_braces(props["data file"]) if "data file" in props else None,
        "wavelength": wavelengths,
        "description": _strip_braces(props.get("description", "")),
    }


def guess_raw_path(hdr_path: Path, hdr_info: Dict[str, Any]) -> Optional[Path]:
    data_file = hdr_info.get("data file")
    if data_file:
        candidate = hdr_path.parent / str(data_file)
        # Never follow the HDR outside its own folder (crafted `data file`
        # values must not turn into arbitrary-file reads).
        parent = hdr_path.parent.resolve()
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = None
        if (
            resolved is not None
            and resolved.is_relative_to(parent)
            and resolved.is_file()
        ):
            return candidate

    for suffix in (".raw", ".dat", ".img", ".bin", ""):
        candidate = hdr_path.with_suffix(suffix)
        if candidate.is_file() and candidate != hdr_path:
            return candidate
    return None


def load_hsi_from_hdr(hdr_path: Path | str) -> HSIDataset:
    hdr = Path(hdr_path)
    info = parse_envi_hdr(hdr)
    samples = info["samples"]
    lines = info["lines"]
    bands = info["bands"]
    data_type = info["data type"]
    interleave = str(info["interleave"]).lower()
    offset = int(info["header offset"] or 0)
    byte_order = int(info["byte order"] or 0)

    if samples is None or lines is None or bands is None or data_type is None:
        raise ValueError(
            "HDR missing required fields: samples/lines/bands/data type")
    if data_type not in ENVI_DTYPE_MAP:
        raise ValueError(f"Unsupported ENVI data type: {data_type}")

    raw_path = guess_raw_path(hdr, info)
    if raw_path is None:
        raise FileNotFoundError("RAW file not found next to HDR")

    dtype = np.dtype(ENVI_DTYPE_MAP[data_type]).newbyteorder(
        ">" if byte_order == 1 else "<")
    if interleave == "bsq":
        shape = (bands, lines, samples)
    elif interleave == "bil":
        shape = (lines, bands, samples)
    elif interleave == "bip":
        shape = (lines, samples, bands)
    else:
        raise ValueError(f"Unsupported interleave: {interleave}")

    expected_bytes = int(np.prod(shape)) * dtype.itemsize + offset
    actual_bytes = raw_path.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"RAW is smaller than HDR shape requires: got {actual_bytes} bytes, need {expected_bytes}"
        )

    mm = np.memmap(raw_path, dtype=dtype, mode="r", offset=offset, shape=shape)
    return HSIDataset(
        hdr_path=hdr,
        raw_path=raw_path,
        samples=int(samples),
        lines=int(lines),
        bands=int(bands),
        interleave=interleave,
        dtype=dtype,
        offset=offset,
        byte_order=byte_order,
        wavelengths=list(info["wavelength"]),
        description=str(info["description"]),
        mm=mm,
    )
