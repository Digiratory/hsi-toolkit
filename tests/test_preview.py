import numpy as np

from hsi_toolkit.core.preview import pseudo_rgb, window_preview


class TestPseudoRgb:
    def test_shape_dtype_range(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        rgb = pseudo_rgb(ws, wl_r=700, wl_g=600, wl_b=500)
        assert rgb.shape == (ws.lines, ws.samples, 3)
        assert rgb.dtype == np.uint8
        ws.close()

    def test_deterministic(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        rgb1 = pseudo_rgb(ws)
        rgb2 = pseudo_rgb(ws)
        np.testing.assert_array_equal(rgb1, rgb2)
        ws.close()

    def test_distinct_channel_bands(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        # The cube differs per band, so R and B channels must differ.
        rgb = pseudo_rgb(ws, wl_r=700, wl_g=700, wl_b=500)
        assert not np.array_equal(rgb[..., 0], rgb[..., 2])
        ws.close()

    def test_uses_lowres_when_built(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        full = pseudo_rgb(ws)
        ws.build_lowres_cube(factor=4)
        low = pseudo_rgb(ws)
        assert low.shape == (ws.lines // 4, ws.samples // 4, 3)
        assert not np.array_equal(full, low)
        ws.close()


class TestWindowPreview:
    def test_shape_dtype(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        img = window_preview(ws, center=600.0, width=100.0)
        assert img.shape == (ws.lines, ws.samples, 3)
        assert img.dtype == np.uint8
        ws.close()

    def test_full_window_is_grayscale(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        wl_lo, wl_hi = ws.wavelength_range
        img = window_preview(ws, center=(wl_lo + wl_hi) / 2, width=(wl_hi - wl_lo) + 10)
        # gray cmap maps equal values to R=G=B
        np.testing.assert_array_equal(img[..., 0], img[..., 1])
        np.testing.assert_array_equal(img[..., 1], img[..., 2])
        ws.close()

    def test_narrow_vs_wide_window(self, synthetic_dataset):
        ws = _ws(synthetic_dataset)
        narrow = window_preview(ws, center=600.0, width=1.0)
        wide = window_preview(ws, center=600.0, width=200.0)
        assert not np.array_equal(narrow, wide)
        ws.close()


def _ws(synthetic_dataset):
    from hsi_toolkit.core.dataset import HSIWorkspace

    return HSIWorkspace.open(
        synthetic_dataset["cube"],
        synthetic_dataset["white"],
        synthetic_dataset["dark"],
    )
