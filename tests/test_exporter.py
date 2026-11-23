import json

import numpy as np
import pytest
from conftest import BANDS, WAVELENGTHS, write_envi

from hsi_toolkit.core.annotations import Annotation
from hsi_toolkit.core.exporter import export_npz, slugify


def _ws(synthetic_dataset):
    from hsi_toolkit.core.dataset import HSIWorkspace

    return HSIWorkspace.open(
        synthetic_dataset["cube"],
        synthetic_dataset["white"],
        synthetic_dataset["dark"],
    )


def _polygon(name="Зона"):
    return Annotation(
        id="p1",
        kind="polygon",
        name=name,
        points=[(3.2, 4.2), (9.8, 4.2), (9.8, 8.8), (3.2, 8.8)],
        comment="первый",
        color="#d62728",
    )


def _polyline(name="Линия"):
    return Annotation(
        id="l1",
        kind="polyline",
        name=name,
        points=[(2.0, 5.0), (12.0, 5.0)],
        color="#1f77b4",
    )


class TestSlugify:
    def test_cyrillic_transliterated(self):
        assert slugify("Коагуляция") == "koagulyatsiya"
        assert slugify("Линия 1") == "liniya_1"

    def test_special_chars(self):
        assert slugify("Зона! @#травмы-2") == "zona_travmy_2"
        assert slugify("!!!") == "object"

    def test_truncated(self):
        assert len(slugify("а" * 100)) <= 40


class TestExportPolygon:
    def test_shapes_and_values(self, synthetic_dataset, tmp_path):
        ws = _ws(synthetic_dataset)
        ann = _polygon()
        path = tmp_path / "export.npz"
        keys = export_npz(ws, [ann], path, corrected=True)
        assert keys == ["01_zona"]

        data = np.load(path, allow_pickle=False)
        assert data["wavelengths"].shape == (BANDS,)
        np.testing.assert_allclose(data["wavelengths"], sorted(ws.wavelengths))
        cutout = data["01_zona"]
        mask = data["01_zona__mask"]
        assert cutout.shape == (BANDS, 6, 8)
        assert cutout.dtype == np.float32
        assert mask.dtype == bool
        assert mask.shape == (6, 8)
        assert mask.sum() > 0
        for b in range(BANDS):
            np.testing.assert_allclose(cutout[b], ws.corrected_band(b)[4:10, 3:11], atol=1e-6)
        meta = json.loads(str(data["objects"]))
        assert meta[0]["key"] == "01_zona"
        assert meta[0]["id"] == "p1"
        assert meta[0]["kind"] == "polygon"
        assert meta[0]["bbox"] == [3, 4, 10, 9]
        assert meta[0]["points"][0] == [3.2, 4.2]
        ws.close()

    def test_raw_export_uint16(self, synthetic_dataset, tmp_path):
        ws = _ws(synthetic_dataset)
        path = tmp_path / "export_raw.npz"
        export_npz(ws, [_polygon()], path, corrected=False)
        data = np.load(path)
        cutout = data["01_zona"]
        assert cutout.dtype == np.uint16
        order = np.argsort(WAVELENGTHS)
        np.testing.assert_array_equal(
            cutout[0], synthetic_dataset["cube_raw"][order[0]][4:10, 3:11]
        )
        ws.close()

    def test_raw_export_preserves_float32(self, tmp_path):
        # Regression: the raw path used to hardcode a uint16 buffer, silently
        # truncating/wrapping non-uint16 ENVI cubes (e.g. float32 reflectance).
        from hsi_toolkit.core.dataset import HSIWorkspace

        cube = np.full((BANDS, 16, 16), 0.7, dtype=np.float32)
        cube[0, 5, 5] = 70000.5  # wrapped modulo 2**16 in the old uint16 buffer
        hdr = write_envi(tmp_path / "f32.hdr", cube, WAVELENGTHS, envi_data_type=4)
        ws = HSIWorkspace.open(hdr)
        try:
            path = tmp_path / "f32.npz"
            export_npz(ws, [_polygon()], path, corrected=False)
            cutout = np.load(path)["01_zona"]
            assert cutout.dtype == np.float32
            assert float(cutout.max()) == pytest.approx(70000.5)
            assert float(cutout.min()) == pytest.approx(0.7)

            path2 = tmp_path / "f32_line.npz"
            export_npz(ws, [_polyline()], path2, corrected=False)
            ribbon = np.load(path2)["01_liniya"]
            assert ribbon.dtype == np.float32
            assert float(ribbon.max()) == pytest.approx(70000.5)
            assert float(ribbon.min()) == pytest.approx(0.7)
        finally:
            ws.close()


class TestExportPolyline:
    def test_ribbon_and_positions(self, synthetic_dataset, tmp_path):
        ws = _ws(synthetic_dataset)
        ann = _polyline()
        path = tmp_path / "export_line.npz"
        keys = export_npz(ws, [ann], path, corrected=True, ribbon_width=3)
        assert keys == ["01_liniya"]
        data = np.load(path)
        ribbon = data["01_liniya"]
        positions = data["01_liniya__positions"]
        assert ribbon.shape == (BANDS, 10, 3)
        assert ribbon.dtype == np.float32
        assert positions.shape == (10,)
        assert positions[0] == pytest.approx(0.0)
        assert positions[-1] == pytest.approx(1.0)
        xs_expected = np.round(2.0 + np.linspace(0.0, 1.0, 10) * 10.0).astype(int)
        for b in range(BANDS):
            band = ws.corrected_band(b)
            for s, x in enumerate(xs_expected):
                np.testing.assert_allclose(ribbon[b, s, :], band[4:7, int(x)], atol=1e-6)
        ws.close()


class TestExportSelection:
    def test_selected_ids_only(self, synthetic_dataset, tmp_path):
        ws = _ws(synthetic_dataset)
        path = tmp_path / "sel.npz"
        keys = export_npz(ws, [_polygon(), _polyline()], path, selected=["l1"])
        assert keys == ["01_liniya"]
        data = np.load(path)
        assert "01_liniya" in data and "01_zona" not in data
        ws.close()

    def test_invisible_excluded(self, synthetic_dataset, tmp_path):
        ws = _ws(synthetic_dataset)
        ann = _polygon()
        ann.visible = False
        with pytest.raises(ValueError):
            export_npz(ws, [ann], tmp_path / "inv.npz")
        ws.close()

    def test_invisible_but_selected_exports(self, synthetic_dataset, tmp_path):
        # An explicit selection (dialog checkboxes) wins over visibility:
        # hidden objects start unchecked but are user-checkable.
        ws = _ws(synthetic_dataset)
        hidden = _polygon("Скрытая")
        hidden.visible = False
        path = tmp_path / "hidden_sel.npz"
        keys = export_npz(ws, [hidden, _polyline()], path, selected=["p1"])
        assert keys == ["01_skrytaya"]
        data = np.load(path)
        assert "01_skrytaya" in data and "01_liniya" not in data
        ws.close()

    def test_no_annotations_raises(self, synthetic_dataset, tmp_path):
        ws = _ws(synthetic_dataset)
        with pytest.raises(ValueError):
            export_npz(ws, [], tmp_path / "empty.npz")
        ws.close()

    def test_corrected_without_refs_raises(self, synthetic_dataset, tmp_path):
        from hsi_toolkit.core.dataset import HSIWorkspace

        ws = HSIWorkspace.open(synthetic_dataset["cube"])
        with pytest.raises(ValueError):
            export_npz(ws, [_polygon()], tmp_path / "x.npz", corrected=True)
        export_npz(ws, [_polygon()], tmp_path / "y.npz", corrected=False)
        ws.close()

    def test_duplicate_name_keys_unique(self, synthetic_dataset, tmp_path):
        ws = _ws(synthetic_dataset)
        ann1 = _polygon("Зона")
        ann2 = _polygon("Зона")
        ann2.id = "p2"
        path = tmp_path / "dup.npz"
        keys = export_npz(ws, [ann1, ann2], path)
        assert len(set(keys)) == len(keys)
        data = np.load(path)
        for key in keys:
            assert key in data
            assert key + "__mask" in data
        ws.close()
