import numpy as np
import pytest
from conftest import BANDS, LINES, SAMPLES, WAVELENGTHS, write_envi

from hsi_toolkit.core.dataset import HSIWorkspace, find_reference_hdrs


def manual_corrected(band_raw, dark_prof, white_prof):
    eps = np.float32(np.finfo(np.float32).eps)
    d = dark_prof[None, :].astype(np.float32)
    w = white_prof[None, :].astype(np.float32)
    return np.clip((band_raw.astype(np.float32) - d) / (w - d + eps), 0, 1)


def manual_profile(cube):
    return cube.astype(np.float64).mean(axis=1)  # (B, W) mean over lines


class TestOpenAndSort:
    def test_sorted_wavelengths(self, synthetic_dataset):
        ws = HSIWorkspace.open(
            synthetic_dataset["cube"],
            synthetic_dataset["white"],
            synthetic_dataset["dark"],
        )
        assert ws.bands == BANDS
        assert ws.lines == LINES
        assert ws.samples == SAMPLES
        assert np.all(np.diff(ws.wavelengths) > 0)
        assert np.allclose(ws.wavelengths, sorted(WAVELENGTHS))
        ws.close()

    def test_raw_band_follows_sorting(self, synthetic_dataset):
        ws = HSIWorkspace.open(synthetic_dataset["cube"])
        cube = synthetic_dataset["cube_raw"]
        order = np.argsort(WAVELENGTHS)
        for b in range(BANDS):
            np.testing.assert_array_equal(ws.raw_band(b), cube[order[b]])
        ws.close()

    def test_requires_both_references(self, synthetic_dataset):
        with pytest.raises(ValueError):
            HSIWorkspace.open(synthetic_dataset["cube"], white_hdr=synthetic_dataset["white"])
        with pytest.raises(ValueError):
            HSIWorkspace.open(synthetic_dataset["cube"], dark_hdr=synthetic_dataset["dark"])


class TestCorrection:
    def test_corrected_band_matches_manual_formula(self, synthetic_dataset):
        ws = HSIWorkspace.open(
            synthetic_dataset["cube"],
            synthetic_dataset["white"],
            synthetic_dataset["dark"],
        )
        dark_prof = manual_profile(synthetic_dataset["dark_raw"])
        white_prof = manual_profile(synthetic_dataset["white_raw"])
        cube = synthetic_dataset["cube_raw"]
        order = np.argsort(WAVELENGTHS)
        for b in range(BANDS):
            expected = manual_corrected(cube[order[b]], dark_prof[order[b]], white_prof[order[b]])
            got = ws.corrected_band(b)
            assert got.dtype == np.float32
            assert got.shape == (LINES, SAMPLES)
            np.testing.assert_allclose(got, expected, atol=1e-5)
        ws.close()

    def test_correction_normalizes_references(self, synthetic_dataset):
        # Sample == white -> ~1 everywhere; sample == dark -> 0 everywhere.
        ws_white = HSIWorkspace.open(
            synthetic_dataset["white"], synthetic_dataset["white"], synthetic_dataset["dark"]
        )
        np.testing.assert_allclose(ws_white.corrected_band(0), 1.0, atol=1e-6)
        ws_white.close()
        ws_dark = HSIWorkspace.open(
            synthetic_dataset["dark"], synthetic_dataset["white"], synthetic_dataset["dark"]
        )
        np.testing.assert_allclose(ws_dark.corrected_band(0), 0.0, atol=1e-6)
        ws_dark.close()

    def test_without_references_raises(self, synthetic_dataset):
        ws = HSIWorkspace.open(synthetic_dataset["cube"])
        assert not ws.has_references
        with pytest.raises(ValueError):
            ws.corrected_band(0)
        # working_band falls back to raw float32
        band = ws.working_band(0)
        assert band.dtype == np.float32
        ws.close()

    def test_reference_dimension_mismatch(self, synthetic_dataset, tmp_path):
        narrow_white = write_envi(
            tmp_path / "WHITEREF_small.hdr",
            synthetic_dataset["white_raw"][:, :, :16],
            WAVELENGTHS,
        )
        with pytest.raises(ValueError):
            HSIWorkspace.open(
                synthetic_dataset["cube"],
                white_hdr=narrow_white,
                dark_hdr=synthetic_dataset["dark"],
            )


class TestSpectralAxis:
    def test_band_index_near(self, synthetic_dataset):
        ws = HSIWorkspace.open(synthetic_dataset["cube"])
        assert ws.band_index_near(500.0) == list(sorted(WAVELENGTHS)).index(500.0)
        assert ws.band_index_near(520.0) == list(sorted(WAVELENGTHS)).index(500.0)
        assert ws.band_index_near(10000.0) == BANDS - 1
        ws.close()

    def test_bands_in_window(self, synthetic_dataset):
        ws = HSIWorkspace.open(synthetic_dataset["cube"])
        wl = sorted(WAVELENGTHS)
        idx = ws.bands_in_window(600.0, 110.0)
        assert list(idx) == [wl.index(550.0), wl.index(600.0), wl.index(650.0)]
        idx = ws.bands_in_window(600.0, 0.0)
        assert list(idx) == [wl.index(600.0)]
        idx = ws.bands_in_window(605.0, 1.0)  # empty window -> nearest band
        assert list(idx) == [wl.index(600.0)]
        ws.close()


class TestCacheAndLowres:
    def test_lru_cache(self, synthetic_dataset):
        ws = HSIWorkspace.open(
            synthetic_dataset["cube"],
            synthetic_dataset["white"],
            synthetic_dataset["dark"],
        )
        first = ws.corrected_band(0)
        assert ws.corrected_band(0) is first
        for b in range(BANDS + 4):  # exceeds the cache size
            ws.corrected_band(b % BANDS)
        assert len(ws._cache) <= ws._cache_max
        again = ws.corrected_band(0)
        np.testing.assert_array_equal(again, first)
        ws.close()

    def test_build_lowres_cube(self, synthetic_dataset):
        ws = HSIWorkspace.open(
            synthetic_dataset["cube"],
            synthetic_dataset["white"],
            synthetic_dataset["dark"],
        )
        low = ws.build_lowres_cube(factor=4)
        assert low.shape == (BANDS, LINES // 4, SAMPLES // 4)
        assert low.dtype == np.float32
        assert ws.lowres is low
        full = ws.corrected_band(0)
        block = full.reshape(8, 4, 8, 4).mean(axis=(1, 3))
        np.testing.assert_allclose(low[0], block, atol=1e-5)
        ws.close()

    def test_close(self, synthetic_dataset):
        ws = HSIWorkspace.open(synthetic_dataset["cube"])
        ws.close()
        with pytest.raises(ValueError):
            ws.raw_band(0)


class TestFindReferences:
    def test_flat_layout(self, synthetic_dataset):
        white, dark = find_reference_hdrs(synthetic_dataset["cube"])
        assert white is not None and white.name == "WHITEREF_cube.hdr"
        assert dark is not None and dark.name == "DARKREF_cube.hdr"

    def test_data_file_traversal_is_blocked(self, synthetic_dataset, tmp_path):
        # A crafted HDR must not make the loader read files outside its folder.
        crafted_dir = tmp_path / "crafted"
        crafted_dir.mkdir()
        outside = tmp_path / "secret.raw"
        outside.write_bytes(synthetic_dataset["cube_raw"].tobytes())
        hdr = crafted_dir / "cube.hdr"
        hdr.write_text(
            "\n".join(
                [
                    "ENVI",
                    f"samples = {SAMPLES}",
                    f"lines = {LINES}",
                    f"bands = {BANDS}",
                    "header offset = 0",
                    "data type = 12",
                    "byte order = 0",
                    "interleave = bsq",
                    "data file = ../secret.raw",
                    "wavelength = {500.0 550.0 600.0 650.0 700.0}",
                ]
            ),
            encoding="ascii",
        )
        from hsi_toolkit.io.envi_hdr import load_hsi_from_hdr

        with pytest.raises(FileNotFoundError):
            load_hsi_from_hdr(hdr)

    def test_capture_layout(self, synthetic_dataset, tmp_path):
        base = tmp_path / "capture-layout"
        capture = base / "capture"
        capture.mkdir(parents=True)
        white, dark = find_reference_hdrs(capture / "cube.hdr")
        assert white is None and dark is None
        hdr = write_envi(capture / "cube.hdr", synthetic_dataset["cube_raw"], WAVELENGTHS)
        write_envi(capture / "WHITEREF_cube.hdr", synthetic_dataset["white_raw"], WAVELENGTHS)
        write_envi(capture / "DARKREF_cube.hdr", synthetic_dataset["dark_raw"], WAVELENGTHS)
        white, dark = find_reference_hdrs(hdr)
        assert white is not None and dark is not None

    def test_bil_interleave_roundtrip(self, synthetic_dataset, tmp_path):
        hdr = write_envi(
            tmp_path / "bil.hdr", synthetic_dataset["cube_raw"], WAVELENGTHS, interleave="bil"
        )
        ws = HSIWorkspace.open(hdr)
        cube = synthetic_dataset["cube_raw"]
        order = np.argsort(WAVELENGTHS)
        for b in range(BANDS):
            np.testing.assert_array_equal(ws.raw_band(b), cube[order[b]])
        ws.close()
