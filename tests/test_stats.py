import csv

import numpy as np
import openpyxl
import pytest
from conftest import BANDS, LINES, SAMPLES
from scipy.stats import mannwhitneyu, ttest_ind

from hsi_toolkit.core.annotations import Annotation, rasterize
from hsi_toolkit.core.spectra import mask_band_values
from hsi_toolkit.core.stats import (
    TABLE_COLUMNS,
    TEST_WELCH,
    bh_adjust,
    cohen_d,
    export_csv,
    export_xlsx,
    holm_adjust,
    mann_whitney,
    pairwise_roi_stats,
    table_rows,
    welch_test,
)


def _ws(synthetic_dataset):
    from hsi_toolkit.core.dataset import HSIWorkspace

    return HSIWorkspace.open(
        synthetic_dataset["cube"],
        synthetic_dataset["white"],
        synthetic_dataset["dark"],
    )


def _square_mask(points):
    ann = Annotation(id="m", kind="polygon", name="m", points=points)
    return rasterize(ann, (LINES, SAMPLES))


# Masks over the deliberately dark quadrant and bright regions of the
# synthetic cube (conftest scales cube[:, :8, :8] by 0.4).
DARK_POINTS = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]
BRIGHT_POINTS = [(16.0, 16.0), (24.0, 16.0), (24.0, 24.0), (16.0, 24.0)]
MID_POINTS = [(8.0, 8.0), (16.0, 8.0), (16.0, 16.0), (8.0, 16.0)]


class TestCohenD:
    def test_matches_manual_pooled_formula(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([3.0, 4.0, 5.0, 7.0])
        sp = np.sqrt((3 * a.var(ddof=1) + 3 * b.var(ddof=1)) / (4 + 4 - 2))
        expected = (a.mean() - b.mean()) / sp
        assert cohen_d(a, b) == pytest.approx(expected, abs=1e-12)

    def test_zero_variance_is_nan(self):
        assert np.isnan(cohen_d([2.0, 2.0, 2.0], [3.0, 3.0, 3.0]))
        assert np.isnan(cohen_d([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]))

    def test_too_few_samples_is_nan(self):
        assert np.isnan(cohen_d([1.0], [1.0, 2.0]))
        assert np.isnan(cohen_d([1.0, 2.0], []))

    def test_sign_follows_group_order(self):
        assert cohen_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) < 0
        assert cohen_d([4.0, 5.0, 6.0], [1.0, 2.0, 3.0]) > 0


class TestMannWhitney:
    def test_matches_scipy(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0.0, 1.0, 30)
        b = rng.normal(0.7, 1.0, 25)
        u, p, auc = mann_whitney(a, b)
        ref = mannwhitneyu(a, b, alternative="two-sided", method="auto")
        assert u == pytest.approx(float(ref.statistic))
        assert p == pytest.approx(float(ref.pvalue))
        assert auc == pytest.approx(u / (30 * 25))

    def test_full_separation_auc_one(self):
        a = np.arange(1.0, 11.0)
        b = np.arange(11.0, 21.0)
        # AUC is directional: P(a > b) = 0, P(b > a) = 1.
        _, _, auc = mann_whitney(a, b)
        assert auc == pytest.approx(0.0)
        _, _, auc_rev = mann_whitney(b, a)
        assert auc_rev == pytest.approx(1.0)

    def test_identical_distributions_auc_half(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u, _, auc = mann_whitney(a, b)
        assert auc == pytest.approx(0.5)
        assert u == pytest.approx(12.5)

    def test_degenerate_is_nan(self):
        assert all(np.isnan(v) for v in mann_whitney([5.0, 5.0], [5.0, 5.0]))
        assert all(np.isnan(v) for v in mann_whitney([1.0], [2.0, 3.0]))


class TestWelch:
    def test_matches_scipy(self):
        rng = np.random.default_rng(7)
        a = rng.normal(1.0, 2.0, 40)
        b = rng.normal(0.0, 1.0, 33)
        t, p = welch_test(a, b)
        ref = ttest_ind(a, b, equal_var=False)
        assert t == pytest.approx(float(ref.statistic))
        assert p == pytest.approx(float(ref.pvalue))

    def test_too_few_samples_is_nan(self):
        assert np.isnan(welch_test([1.0], [1.0, 2.0])[1])


class TestHolmBH:
    def test_holm_manual_example(self):
        adjusted = holm_adjust(np.array([0.01, 0.03, 0.04]))
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_bh_manual_example(self):
        adjusted = bh_adjust(np.array([0.01, 0.03, 0.04]))
        np.testing.assert_allclose(adjusted, [0.03, 0.04, 0.04])

    def test_holm_is_monotone_in_sorted_p(self):
        p = np.array([0.001, 0.004, 0.02, 0.04, 0.3])
        adjusted = holm_adjust(p)
        assert (np.diff(adjusted) >= 0).all()

    def test_bh_is_monotone_in_sorted_p(self):
        p = np.array([0.001, 0.004, 0.02, 0.04, 0.3])
        adjusted = bh_adjust(p)
        assert (np.diff(adjusted) >= 0).all()

    def test_clipped_to_one(self):
        np.testing.assert_allclose(holm_adjust(np.array([0.5, 0.6, 0.7])), [1.0, 1.0, 1.0])

    def test_nan_passes_through_and_family_is_finite_only(self):
        p = np.array([np.nan, 0.01, 0.03])
        np.testing.assert_allclose(holm_adjust(p), [np.nan, 0.02, 0.03])
        np.testing.assert_allclose(bh_adjust(p), [np.nan, 0.02, 0.03])

        # Multipliers use the count of finite p's, not the array length.
        p = np.array([0.5, np.nan, 0.4])
        np.testing.assert_allclose(holm_adjust(p), [0.8, np.nan, 0.8], equal_nan=True)

    def test_correction_over_flattened_family_is_order_independent(self):
        p = np.array([[0.01, 0.5, np.nan], [0.2, 0.04, 0.6]])
        expected = bh_adjust(p.ravel()).reshape(2, 3)
        np.testing.assert_allclose(
            np.sort(expected[~np.isnan(expected)]),
            [0.05, 0.1, 0.33333333, 0.6, 0.6],
        )


class TestMaskBandValues:
    def test_matches_manual_extraction(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        mask = _square_mask([(4.0, 5.0), (15.0, 5.0), (15.0, 12.0), (4.0, 12.0)])
        values = mask_band_values(ws, mask)
        n_pixels = int(mask.sum())
        assert values.shape == (BANDS, n_pixels)
        assert values.dtype == np.float32
        for b in range(BANDS):
            np.testing.assert_array_equal(values[b], ws.working_band(b)[mask])
        ws.close()

    def test_validation_same_errors_as_polygon_stats(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        with pytest.raises(ValueError, match="does not match image"):
            mask_band_values(ws, np.zeros((4, 4), dtype=bool))
        with pytest.raises(ValueError, match="Empty mask"):
            mask_band_values(ws, np.zeros((LINES, SAMPLES), dtype=bool))
        ws.close()


class TestPairwiseRoiStats:
    def test_dark_vs_bright_significant_auc_one(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        masks = [_square_mask(DARK_POINTS), _square_mask(BRIGHT_POINTS)]
        samples = [mask_band_values(ws, m) for m in masks]
        result = pairwise_roi_stats(
            samples,
            ["Тёмная", "Светлая"],
            ref_index=1,
            wavelengths=ws.wavelengths,
        )
        assert result.pairs == ("Тёмная vs Светлая",)
        assert result.p_raw.shape == (1, BANDS)
        assert (result.significant == True).all()  # noqa: E712
        # Directional AUC: pair is (dark, bright) so P(dark > bright) = 0;
        # the reversed pair must give 1.
        np.testing.assert_allclose(result.auc, 0.0, atol=1e-9)
        assert (result.p_adjusted < 1e-6).all()
        assert (result.delta_pct < 0).all()  # dark zone below bright reference

        reversed_result = pairwise_roi_stats(samples[::-1], ["Светлая", "Тёмная"], ref_index=0)
        np.testing.assert_allclose(reversed_result.auc, 1.0, atol=1e-9)
        ws.close()

    def test_delta_pct_formula(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        masks = [_square_mask(DARK_POINTS), _square_mask(BRIGHT_POINTS)]
        samples = [mask_band_values(ws, m) for m in masks]
        result = pairwise_roi_stats(samples, ["A", "B"], ref_index=1)
        va, vb = samples
        mean_a = va.mean(axis=1, dtype=np.float64)
        mean_b = vb.mean(axis=1, dtype=np.float64)
        expected_pct = (mean_a - mean_b) / mean_b * 100.0
        np.testing.assert_allclose(result.delta_pct[0], expected_pct, atol=1e-4)
        np.testing.assert_allclose(result.delta[0], mean_a - mean_b, atol=1e-4)
        np.testing.assert_allclose(
            result.delta_median[0],
            np.median(va, axis=1) - np.median(vb, axis=1),
            atol=1e-4,
        )
        ws.close()

    def test_three_rois_three_pairs_in_order(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        masks = [
            _square_mask(DARK_POINTS),
            _square_mask(BRIGHT_POINTS),
            _square_mask(MID_POINTS),
        ]
        samples = [mask_band_values(ws, m) for m in masks]
        result = pairwise_roi_stats(samples, ["Тёмная", "Светлая", "Средняя"], ref_index=0)
        assert result.pairs == (
            "Тёмная vs Светлая",
            "Тёмная vs Средняя",
            "Светлая vs Средняя",
        )
        assert result.pair_indices == ((0, 1), (0, 2), (1, 2))
        for metric in ("mean_a", "mean_b", "p_raw", "auc", "cohens_d"):
            assert getattr(result, metric).shape == (3, BANDS)
        ws.close()

    def test_welch_changes_statistic_not_auc(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        masks = [_square_mask(DARK_POINTS), _square_mask(BRIGHT_POINTS)]
        samples = [mask_band_values(ws, m) for m in masks]
        result = pairwise_roi_stats(samples, ["A", "B"], ref_index=0, test=TEST_WELCH)
        assert result.test == TEST_WELCH
        ref = ttest_ind(
            samples[0].astype(np.float64),
            samples[1].astype(np.float64),
            axis=1,
            equal_var=False,
        )
        np.testing.assert_allclose(result.statistic[0], ref.statistic, atol=1e-6)
        np.testing.assert_allclose(result.p_raw[0], ref.pvalue, atol=1e-12)
        # AUC stays rank-based: pair (dark, bright) -> 0.
        np.testing.assert_allclose(result.auc, 0.0, atol=1e-9)
        ws.close()

    def test_correction_none_and_bh(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        masks = [
            _square_mask(DARK_POINTS),
            _square_mask(BRIGHT_POINTS),
            _square_mask(MID_POINTS),
        ]
        samples = [mask_band_values(ws, m) for m in masks]
        raw = pairwise_roi_stats(samples, ["A", "B", "C"], ref_index=0, correction="none")
        np.testing.assert_array_equal(raw.p_adjusted, raw.p_raw)

        bh = pairwise_roi_stats(samples, ["A", "B", "C"], ref_index=0, correction="bh")
        np.testing.assert_array_equal(
            bh.p_adjusted, bh_adjust(bh.p_raw).reshape(bh.p_adjusted.shape)
        )
        assert bh.correction == "bh"
        ws.close()

    def test_validation(self):
        samples = [np.ones((3, 5)), np.ones((3, 7))]
        with pytest.raises(ValueError):
            pairwise_roi_stats(samples[:1], ["A"], ref_index=0)
        with pytest.raises(ValueError):
            pairwise_roi_stats(samples, ["A"], ref_index=0)
        with pytest.raises(ValueError):
            pairwise_roi_stats(samples, ["A", "B"], ref_index=5)
        with pytest.raises(ValueError, match="Unknown test"):
            pairwise_roi_stats(samples, ["A", "B"], ref_index=0, test="anova")
        with pytest.raises(ValueError, match="Unknown correction"):
            pairwise_roi_stats(samples, ["A", "B"], ref_index=0, correction="bonferroni")
        with pytest.raises(ValueError):
            pairwise_roi_stats([np.ones((3, 5))], ["A", "B"], ref_index=0)


class TestTableRowsAndExport:
    def _samples_with_nan_pair(self):
        # Band 1 is identical in both ROIs -> NaN test/AUC/d in that channel.
        a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [7.0, 7.0, 7.0, 7.0, 7.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0, 5.0, 6.0, 7.0], [7.0, 7.0, 7.0, 7.0, 7.0]], dtype=np.float32)
        return [a, b]

    def test_table_rows_long_format(self):
        samples = self._samples_with_nan_pair()
        result = pairwise_roi_stats(samples, ["A", "B"], ref_index=0, wavelengths=[500.0, 600.0])
        rows = table_rows(result)
        assert len(rows) == 2 * 1
        assert list(rows[0].keys()) == list(TABLE_COLUMNS)
        assert rows[0]["channel"] == 1
        assert rows[0]["wavelength_nm"] == pytest.approx(500.0)
        assert rows[0]["pair"] == "A vs B"
        assert rows[1]["channel"] == 2
        assert rows[1]["significant"] is False  # NaN p-value never significant
        assert np.isnan(rows[1]["p_value"])
        # Channel 1 samples overlap -> not significant even before correction.
        assert rows[0]["significant"] is False
        assert rows[0]["p_value"] < 0.2

    def test_export_csv_utf8_sig_and_roundtrip(self, tmp_path):
        samples = self._samples_with_nan_pair()
        result = pairwise_roi_stats(samples, ["A", "B"], ref_index=0, wavelengths=[500.0, 600.0])
        rows = table_rows(result)
        path = export_csv(rows, tmp_path / "stats.csv")
        raw = path.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")
        with open(path, newline="", encoding="utf-8-sig") as f:
            parsed = list(csv.reader(f))
        assert parsed[0] == list(TABLE_COLUMNS)
        assert len(parsed) == 1 + len(rows)
        # NaN cells are written as empty strings.
        nan_row = parsed[2]
        nan_cols = [TABLE_COLUMNS.index(c) for c in ("p_value", "p_adjusted", "auc", "cohens_d")]
        assert all(nan_row[i] == "" for i in nan_cols)
        assert float(nan_row[TABLE_COLUMNS.index("mean_a")]) == pytest.approx(7.0)

    def test_export_xlsx(self, tmp_path):
        samples = self._samples_with_nan_pair()
        result = pairwise_roi_stats(samples, ["A", "B"], ref_index=0, wavelengths=[500.0, 600.0])
        rows = table_rows(result)
        path = export_xlsx(rows, tmp_path / "stats.xlsx")
        wb = openpyxl.load_workbook(path)
        ws = wb.active
        assert ws.title == "Статистика"
        assert ws.freeze_panes == "A2"
        assert ws["A1"].value == "channel"
        assert ws["D1"].value == "mean_a"
        assert ws["D1"].font.bold is True
        assert ws["D2"].value == pytest.approx(3.0)  # mean_a, channel 1
        nan_col = list(TABLE_COLUMNS).index("p_value") + 1
        assert ws.cell(row=3, column=nan_col).value is None  # NaN -> empty cell
        assert ws.max_row == 1 + len(rows)


class TestExportFormulaInjection:
    def _samples(self):
        a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
        return [a, b]

    def _pair_col(self):
        return list(TABLE_COLUMNS).index("pair")

    def test_csv_escapes_formula_prefix(self, tmp_path):
        result = pairwise_roi_stats(
            self._samples(), ["=SUM(A1)", "B"], ref_index=0, wavelengths=[500.0]
        )
        path = export_csv(table_rows(result), tmp_path / "stats.csv")
        with open(path, newline="", encoding="utf-8-sig") as f:
            parsed = list(csv.reader(f))
        col = self._pair_col()
        assert parsed[1][col] == "'=SUM(A1) vs B"

    def test_csv_leaves_normal_names_untouched(self, tmp_path):
        result = pairwise_roi_stats(self._samples(), ["A", "B"], ref_index=0, wavelengths=[500.0])
        path = export_csv(table_rows(result), tmp_path / "stats.csv")
        with open(path, newline="", encoding="utf-8-sig") as f:
            parsed = list(csv.reader(f))
        assert parsed[1][self._pair_col()] == "A vs B"

    def test_xlsx_never_stores_user_text_as_formula(self, tmp_path):
        result = pairwise_roi_stats(
            self._samples(), ["=SUM(A1)", "B"], ref_index=0, wavelengths=[500.0]
        )
        path = export_xlsx(table_rows(result), tmp_path / "stats.xlsx")
        col = self._pair_col() + 1
        cell = openpyxl.load_workbook(path).active.cell(row=2, column=col)
        assert cell.data_type != "f"
        # data_only=True returns None for real formulas without cached values;
        # text must survive as the original string.
        value_only = (
            openpyxl.load_workbook(path, data_only=True).active.cell(row=2, column=col).value
        )
        assert value_only == "=SUM(A1) vs B"


class TestPairwiseCancelAndIds:
    def _samples(self):
        a = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0, 5.0, 6.0, 7.0]], dtype=np.float32)
        return [a, b]

    def test_cancelled_returns_none(self):
        assert (
            pairwise_roi_stats(self._samples(), ["A", "B"], 0, cancel={"cancelled": True}) is None
        )

    def test_active_cancel_token_runs(self):
        result = pairwise_roi_stats(self._samples(), ["A", "B"], 0, cancel={"cancelled": False})
        assert result is not None

    def test_used_ids_default_to_names(self):
        result = pairwise_roi_stats(self._samples(), ["A", "B"], 0)
        assert result.used_ids == ("A", "B")

    def test_used_ids_explicit(self):
        result = pairwise_roi_stats(self._samples(), ["A", "B"], 0, ids=["x", "y"])
        assert result.used_ids == ("x", "y")

    def test_used_ids_length_mismatch(self):
        with pytest.raises(ValueError):
            pairwise_roi_stats(self._samples(), ["A", "B"], 0, ids=["x"])
