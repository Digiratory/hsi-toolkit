"""ROI pairwise statistics per band: effect sizes, tests, corrections, export.

Pure numpy/scipy computations, no Qt. Pixels inside a polygon are treated as
independent samples; spatial autocorrelation inflates significance (see the
README note the app surfaces as a tooltip).

Multiple-comparison corrections are hand-rolled (statsmodels is not a
dependency): only finite p-values enter the family, NaN passes through as NaN.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

TEST_MANN_WHITNEY = "mann_whitney"
TEST_WELCH = "welch"

CORRECTION_NONE = "none"
CORRECTION_HOLM = "holm"
CORRECTION_BH = "bh"

TESTS = (TEST_MANN_WHITNEY, TEST_WELCH)
CORRECTIONS = (CORRECTION_NONE, CORRECTION_HOLM, CORRECTION_BH)

# Long-format column order shared by table_rows and both exporters.
TABLE_COLUMNS = (
    "channel",
    "wavelength_nm",
    "pair",
    "mean_a",
    "mean_b",
    "delta",
    "delta_pct",
    "median_a",
    "median_b",
    "delta_median",
    "cohens_d",
    "auc",
    "test_statistic",
    "p_value",
    "p_adjusted",
    "significant",
)

_NAN = float("nan")

# Cells beginning with these characters are treated as formulas by Excel and
# openpyxl; exported user text gets a text-forcing treatment instead (CWE-1236).
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-standard-deviation Cohen's d (ddof=1); degenerate -> NaN."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return _NAN
    var_pooled = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    if not np.isfinite(var_pooled) or var_pooled <= 0.0:
        return _NAN
    return float((a.mean() - b.mean()) / math.sqrt(var_pooled))


def mann_whitney(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Two-sided Mann–Whitney U via scipy: ``(u, p, auc)``.

    ``u`` is scipy's U statistic for ``a``, i.e. the number of ``a > b`` pairs
    (ties count 0.5); ``auc = u / (n1 * n2) = P(a > b) + 0.5·P(tie)`` is the
    directional ROC AUC. Samples with fewer than 2 pixels, or identical values
    everywhere (U undefined), -> NaN.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return _NAN, _NAN, _NAN
    both = np.concatenate([a, b])
    if both.size and np.all(both == both[0]):
        return _NAN, _NAN, _NAN
    result = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    u = float(result.statistic)
    return u, float(result.pvalue), u / (n1 * n2)


def welch_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sided Welch t-test (unequal variances) via scipy: ``(t, p)``."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return _NAN, _NAN
    result = ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def _finite_adjust(p: np.ndarray, adjust_sorted) -> np.ndarray:
    """Apply ``adjust_sorted`` to the finite p-values only; NaN stays NaN."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, _NAN)
    finite = np.isfinite(p)
    p_f = p[finite]
    m = int(p_f.size)
    if m == 0:
        return out
    order = np.argsort(p_f, kind="stable")
    adjusted = adjust_sorted(p_f[order], m)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    unsorted = np.empty(m, dtype=np.float64)
    unsorted[order] = adjusted
    out[finite] = unsorted
    return out


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN passes through.

    Sorted ascending, ``p_adj_(i) = max_{j<=i} min(1, (m - j + 1) * p_(j))``
    (running maximum keeps the adjustment monotone in the sorted p's).
    """
    return _finite_adjust(
        p,
        lambda ranked, m: np.maximum.accumulate(
            (m - np.arange(m)) * ranked
        ),
    )


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR adjusted p-values; NaN passes through.

    Sorted ascending, ``p_adj_(i) = min_{j>=i} min(1, m * p_(j) / j)``
    (enforced backwards so the adjustment stays monotone).
    """

    def adjust(ranked, m):
        values = m * ranked / np.arange(1, m + 1)
        return np.minimum.accumulate(values[::-1])[::-1]

    return _finite_adjust(p, adjust)


@dataclass(frozen=True)
class RoiStatsResult:
    """Immutable result of ``pairwise_roi_stats`` for panel and export.

    Per-metric arrays are flat ``(P, B)``: pair-major, band-minor.
    """

    wavelengths: np.ndarray  # (B,)
    names: Tuple[str, ...]  # ROI names in input order
    used_ids: Tuple[str, ...]  # ROI ids in the same order (fresh label lookup)
    ref_index: int
    test: str
    alpha: float
    correction: str
    pairs: Tuple[str, ...]  # "A vs B" labels
    pair_indices: Tuple[Tuple[int, int], ...]
    mean_a: np.ndarray
    mean_b: np.ndarray
    median_a: np.ndarray
    median_b: np.ndarray
    delta: np.ndarray  # mean_a - mean_b
    delta_pct: np.ndarray  # delta / mean_ref * 100
    delta_median: np.ndarray
    cohens_d: np.ndarray
    auc: np.ndarray
    statistic: np.ndarray
    p_raw: np.ndarray
    p_adjusted: np.ndarray
    significant: np.ndarray  # bool (P, B): p_adjusted < alpha


def pairwise_roi_stats(
    samples: Sequence[np.ndarray],
    names: Sequence[str],
    ref_index: int,
    test: str = TEST_MANN_WHITNEY,
    alpha: float = 0.05,
    correction: str = CORRECTION_HOLM,
    wavelengths: Optional[np.ndarray] = None,
    ids: Optional[Sequence[str]] = None,
    cancel: Optional[dict] = None,
) -> Optional[RoiStatsResult]:
    """Pairwise ROI comparisons for every band.

    ``samples``: per-ROI arrays ``(B, n_i)`` of working values (see
    ``spectra.mask_band_values``). Pairs are ``i < j`` in input order. The
    p-value correction runs over the whole family (bands x pairs). ``test``:
    ``TEST_MANN_WHITNEY`` or ``TEST_WELCH``; AUC is rank-based regardless.
    ``ids``: stable ROI ids matching ``names`` (the GUI resolves fresh
    labels through them). ``cancel``: optional ``{"cancelled": bool}`` dict
    checked between pairs; a cancelled run returns ``None``.
    """
    if len(samples) != len(names):
        raise ValueError(
            f"samples ({len(samples)}) and names ({len(names)}) must have the same length"
        )
    if len(samples) < 2:
        raise ValueError("Pairwise stats need at least 2 ROIs")
    if not 0 <= int(ref_index) < len(samples):
        raise ValueError(f"ref_index {ref_index} out of range")
    if test not in TESTS:
        raise ValueError(f"Unknown test: {test!r}")
    if correction not in CORRECTIONS:
        raise ValueError(f"Unknown correction: {correction!r}")
    if ids is not None and len(ids) != len(names):
        raise ValueError(
            f"ids ({len(ids)}) and names ({len(names)}) must have the same length"
        )

    bands = int(np.asarray(samples[0]).shape[0])
    for s in samples:
        if int(np.asarray(s).shape[0]) != bands:
            raise ValueError("All ROI sample arrays must have the same band count")

    names = tuple(str(n) for n in names)
    used_ids = names if ids is None else tuple(str(x) for x in ids)
    pair_indices = tuple(
        (i, j) for i in range(len(samples)) for j in range(i + 1, len(samples))
    )
    pairs = tuple(f"{names[i]} vs {names[j]}" for i, j in pair_indices)
    n_pairs = len(pair_indices)

    mean_ref = np.array(
        [
            float(np.mean(np.asarray(samples[ref_index][b], dtype=np.float64)))
            for b in range(bands)
        ]
    )

    shape = (n_pairs, bands)
    mean_a = np.full(shape, _NAN)
    mean_b = np.full(shape, _NAN)
    median_a = np.full(shape, _NAN)
    median_b = np.full(shape, _NAN)
    delta = np.full(shape, _NAN)
    delta_pct = np.full(shape, _NAN)
    delta_median = np.full(shape, _NAN)
    cohens = np.full(shape, _NAN)
    auc_arr = np.full(shape, _NAN)
    statistic = np.full(shape, _NAN)
    p_raw = np.full(shape, _NAN)

    for p_idx, (i, j) in enumerate(pair_indices):
        if cancel is not None and cancel.get("cancelled"):
            return None
        for b in range(bands):
            a = np.asarray(samples[i][b], dtype=np.float64)
            c = np.asarray(samples[j][b], dtype=np.float64)
            mean_a[p_idx, b] = float(a.mean())
            mean_b[p_idx, b] = float(c.mean())
            median_a[p_idx, b] = float(np.median(a))
            median_b[p_idx, b] = float(np.median(c))
            delta[p_idx, b] = mean_a[p_idx, b] - mean_b[p_idx, b]
            denom = mean_ref[b]
            delta_pct[p_idx, b] = delta[p_idx, b] / denom * 100.0 if denom else _NAN
            delta_median[p_idx, b] = median_a[p_idx, b] - median_b[p_idx, b]
            cohens[p_idx, b] = cohen_d(a, c)
            u, p_mw, auc = mann_whitney(a, c)
            auc_arr[p_idx, b] = auc
            if test == TEST_WELCH:
                statistic[p_idx, b], p_raw[p_idx, b] = welch_test(a, c)
            else:
                statistic[p_idx, b] = u
                p_raw[p_idx, b] = p_mw

    if correction == CORRECTION_NONE:
        p_adjusted = p_raw.copy()
    elif correction == CORRECTION_HOLM:
        p_adjusted = holm_adjust(p_raw)
    else:
        p_adjusted = bh_adjust(p_raw)
    significant = p_adjusted < alpha  # NaN < alpha -> False

    if wavelengths is None:
        wavelengths = np.full(bands, _NAN)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if wavelengths.size != bands:
        raise ValueError(
            f"wavelengths size {wavelengths.size} does not match band count {bands}"
        )

    return RoiStatsResult(
        wavelengths=wavelengths,
        names=names,
        used_ids=used_ids,
        ref_index=int(ref_index),
        test=test,
        alpha=float(alpha),
        correction=correction,
        pairs=pairs,
        pair_indices=pair_indices,
        mean_a=mean_a,
        mean_b=mean_b,
        median_a=median_a,
        median_b=median_b,
        delta=delta,
        delta_pct=delta_pct,
        delta_median=delta_median,
        cohens_d=cohens,
        auc=auc_arr,
        statistic=statistic,
        p_raw=p_raw,
        p_adjusted=p_adjusted,
        significant=significant,
    )


def table_rows(result: RoiStatsResult) -> List[dict]:
    """Long-format rows (one per channel x pair) in TABLE_COLUMNS order."""
    rows: List[dict] = []
    n_pairs = len(result.pairs)
    for b in range(result.wavelengths.size):
        for p_idx in range(n_pairs):
            rows.append(
                {
                    "channel": b + 1,
                    "wavelength_nm": float(result.wavelengths[b]),
                    "pair": result.pairs[p_idx],
                    "mean_a": float(result.mean_a[p_idx, b]),
                    "mean_b": float(result.mean_b[p_idx, b]),
                    "delta": float(result.delta[p_idx, b]),
                    "delta_pct": float(result.delta_pct[p_idx, b]),
                    "median_a": float(result.median_a[p_idx, b]),
                    "median_b": float(result.median_b[p_idx, b]),
                    "delta_median": float(result.delta_median[p_idx, b]),
                    "cohens_d": float(result.cohens_d[p_idx, b]),
                    "auc": float(result.auc[p_idx, b]),
                    "test_statistic": float(result.statistic[p_idx, b]),
                    "p_value": float(result.p_raw[p_idx, b]),
                    "p_adjusted": float(result.p_adjusted[p_idx, b]),
                    "significant": bool(result.significant[p_idx, b]),
                }
            )
    return rows


def _cell_value(value):
    """Empty cell for missing values; numbers/bools/strings as is."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _csv_cell(value):
    """``_cell_value`` plus a quote-prefix so Excel never sees a formula.

    Names come from annotation JSON files; Excel evaluates cells starting
    with ``=``, ``+``, ``-`` or ``@`` as formulas on open (CWE-1236).
    """
    value = _cell_value(value)
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def export_csv(rows: Sequence[dict], path: Path | str) -> Path:
    """Write long-format rows as CSV (utf-8-sig so Excel reads Cyrillic)."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(TABLE_COLUMNS)
        for row in rows:
            writer.writerow([_csv_cell(row.get(col)) for col in TABLE_COLUMNS])
    return path


def export_xlsx(rows: Sequence[dict], path: Path | str, sheet: str = "Статистика") -> Path:
    """Write long-format rows as XLSX (lazy openpyxl import)."""
    try:
        import openpyxl
        from openpyxl.styles import Font
    except ImportError as exc:
        raise RuntimeError(
            "XLSX export requires the 'openpyxl' package (uv add openpyxl)"
        ) from exc

    workbook = openpyxl.Workbook()
    ws = workbook.active
    ws.title = sheet
    bold = Font(bold=True)
    widths = [len(str(name)) for name in TABLE_COLUMNS]
    for col, name in enumerate(TABLE_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col, value=name)
        cell.font = bold
    for r, row in enumerate(rows, start=2):
        for col, name in enumerate(TABLE_COLUMNS, start=1):
            value = _cell_value(row.get(name))
            if value != "":
                widths[col - 1] = max(widths[col - 1], len(str(value)))
            cell = ws.cell(row=r, column=col, value=value if value != "" else None)
            if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
                cell.data_type = "s"  # never let openpyxl store user text as a formula
    for col, width in enumerate(widths, start=1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = min(
            max(width + 2, 9), 60
        )
    ws.freeze_panes = "A2"
    path = Path(path)
    workbook.save(path)
    return path
