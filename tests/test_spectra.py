import numpy as np
import pytest

from hsi_toolkit.core.spectra import (
    polygon_spectrum_stats,
    polyline_profile,
    polyline_samples,
)

from conftest import BANDS, LINES, SAMPLES


def _ws(synthetic_dataset):
    from hsi_toolkit.core.dataset import HSIWorkspace

    return HSIWorkspace.open(
        synthetic_dataset["cube"],
        synthetic_dataset["white"],
        synthetic_dataset["dark"],
    )


class TestPolylineSamples:
    def test_horizontal_line(self):
        samples, total = polyline_samples([(2, 5), (12, 5)], (LINES, SAMPLES), width=3, step=1.0)
        assert total == pytest.approx(10.0)
        assert len(samples) == 1
        ys, xs, positions = samples[0]
        # ceil(seg_length / step) samples spanning the segment (notebook semantics)
        assert ys.shape == (3, 10) and xs.shape == (3, 10)
        # Ribbon rows: offsets -1, 0, +1 perpendicular to the horizontal line
        np.testing.assert_array_equal(ys, np.array([[4] * 10, [5] * 10, [6] * 10]))
        assert xs[0, 0] == 2 and xs[0, -1] == 12
        np.testing.assert_allclose(positions, np.linspace(0.0, 10.0, 10))

    def test_positions_normalized(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        rel, spectra = polyline_profile(ws, [(2, 5), (12, 5)], width=3, step=1.0)
        assert rel.shape == (10,)
        assert rel[0] == pytest.approx(0.0)
        assert rel[-1] == pytest.approx(1.0)
        ws.close()

    def test_clips_to_bounds(self):
        samples, _ = polyline_samples([(0, 0), (100, 100)], (LINES, SAMPLES), width=3, step=1.0)
        for ys, xs, _ in samples:
            assert ys.min() >= 0 and ys.max() < LINES
            assert xs.min() >= 0 and xs.max() < SAMPLES

    def test_multi_segment_continuity(self):
        points = [(0, 0), (10, 0), (10, 10)]
        samples, total = polyline_samples(points, (32, 32), width=3, step=1.0)
        assert len(samples) == 2
        total_steps = sum(ys.shape[1] for ys, _, _ in samples)
        positions = np.concatenate([p for _, _, p in samples])
        assert total_steps == positions.size
        # No duplicated sample at the joint
        assert total == pytest.approx(20.0)
        assert positions.max() == pytest.approx(20.0)

    def test_validation(self):
        with pytest.raises(ValueError):
            polyline_samples([(0, 0)], (32, 32))
        with pytest.raises(ValueError):
            polyline_samples([(0, 0), (1, 1)], (32, 32), step=0.0)
        with pytest.raises(ValueError):
            polyline_samples([(3, 3), (3, 3)], (32, 32))


class TestPolylineProfile:
    def test_matches_manual_median(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        rel, spectra = polyline_profile(ws, [(2, 5), (12, 5)], width=3, step=1.0)
        assert spectra.shape == (10, BANDS)
        assert spectra.dtype == np.float32
        xs_expected = np.round(2.0 + np.linspace(0.0, 1.0, 10) * 10.0).astype(int)
        for s in range(10):
            x = int(xs_expected[s])
            for b in range(BANDS):
                band = ws.corrected_band(b)
                expected = np.median(band[4:7, x])
                assert spectra[s, b] == pytest.approx(expected, abs=1e-6)
        ws.close()

    def test_spectra_within_unit_interval(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        _, spectra = polyline_profile(ws, [(0, 0), (20, 31)], width=5, step=1.0)
        assert spectra.min() >= 0.0
        assert spectra.max() <= 1.0
        ws.close()


class TestPolygonSpectrumStats:
    def test_median_iqr_matches_manual(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        mask = np.zeros((LINES, SAMPLES), dtype=bool)
        mask[5:15, 8:20] = True
        center, lower, upper = polygon_spectrum_stats(ws, mask, stat="median", ci=(25, 75))
        assert center.shape == (BANDS,)
        for b in range(BANDS):
            vals = ws.corrected_band(b)[mask]
            assert center[b] == pytest.approx(np.median(vals), abs=1e-6)
            q25, q75 = np.percentile(vals, [25, 75])
            assert lower[b] == pytest.approx(q25, abs=1e-6)
            assert upper[b] == pytest.approx(q75, abs=1e-6)
            assert lower[b] <= center[b] <= upper[b]
        ws.close()

    def test_median_wide_percentiles(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        mask = np.zeros((LINES, SAMPLES), dtype=bool)
        mask[5:15, 8:20] = True
        _, lower, upper = polygon_spectrum_stats(ws, mask, ci=(2.5, 97.5))
        _, lower_iqr, upper_iqr = polygon_spectrum_stats(ws, mask, ci=(25, 75))
        assert (lower <= lower_iqr).all()
        assert (upper >= upper_iqr).all()
        ws.close()

    def test_mean_std(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        mask = np.zeros((LINES, SAMPLES), dtype=bool)
        mask[5:15, 8:20] = True
        center, lower, upper = polygon_spectrum_stats(ws, mask, stat="mean", ci=("std", 3.0))
        for b in range(BANDS):
            vals = ws.corrected_band(b)[mask]
            assert center[b] == pytest.approx(vals.mean(), abs=1e-6)
            assert lower[b] == pytest.approx(center[b] - 3 * vals.std(), abs=1e-6)
        ws.close()

    def test_validation(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        with pytest.raises(ValueError):
            polygon_spectrum_stats(ws, np.zeros((4, 4), dtype=bool))
        with pytest.raises(ValueError):
            polygon_spectrum_stats(ws, np.zeros((LINES, SAMPLES), dtype=bool))
        with pytest.raises(ValueError):
            polygon_spectrum_stats(ws, np.ones((LINES, SAMPLES), dtype=bool), stat="mode")
        with pytest.raises(ValueError):
            polygon_spectrum_stats(
                ws, np.ones((LINES, SAMPLES), dtype=bool), stat="mean", ci=(25, 75)
            )
        ws.close()
