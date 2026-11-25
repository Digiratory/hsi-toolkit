import json

import numpy as np
import pytest

from hsi_toolkit.core.annotations import (
    SCHEMA,
    Annotation,
    AnnotationSet,
    assign_color,
    rasterize,
)


def make_polygon():
    return Annotation(
        id="a1",
        kind="polygon",
        name="Тест",
        points=[(3.2, 4.2), (9.8, 4.2), (9.8, 8.8), (3.2, 8.8)],
        comment="комментарий",
        color="#d62728",
    )


def make_polyline():
    return Annotation(
        id="a2",
        kind="polyline",
        name="Линия 1",
        points=[(2.0, 5.0), (12.0, 5.0)],
        color="#1f77b4",
    )


class TestAnnotation:
    def test_defaults(self):
        ann = Annotation(id="x", kind="polygon", name="n", points=[(0, 0), (1, 0), (1, 1)])
        assert ann.comment == ""
        assert ann.color == ""
        assert ann.visible is True

    def test_from_dict_validates(self):
        with pytest.raises(ValueError):
            Annotation.from_dict({"id": "x", "kind": "line", "name": "n", "points": []})
        with pytest.raises(ValueError):
            Annotation.from_dict(
                {"id": "x", "kind": "polygon", "name": "n", "points": [[0, 0], [1, 1]]}
            )
        with pytest.raises(ValueError):
            Annotation.from_dict({"id": "x", "kind": "polyline", "name": "n", "points": [[0, 0]]})

    def test_roundtrip_dict(self):
        ann = make_polygon()
        restored = Annotation.from_dict(ann.to_dict())
        assert restored == ann


class TestAnnotationSet:
    def test_add_remove_get(self):
        ann_set = AnnotationSet()
        ann = ann_set.add(make_polygon())
        assert len(ann_set) == 1
        assert ann_set.get("a1") is ann
        assert ann_set.remove("a1") is ann
        assert len(ann_set) == 0
        assert ann_set.remove("missing") is None
        with pytest.raises(ValueError):
            ann_set.add(make_polygon())
            ann_set.add(make_polygon())

    def test_json_roundtrip(self, tmp_path):
        ann_set = AnnotationSet(
            image_info={
                "hdr": "cube.hdr",
                "samples": 32,
                "lines": 32,
                "wavelengths_nm": [500.0, 600.0],
            }
        )
        ann_set.add(make_polygon())
        ann_set.add(make_polyline())
        path = ann_set.save(tmp_path / "out.json")

        assert not (tmp_path / "out.json.tmp").exists()  # atomic write, no leftovers
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == SCHEMA
        assert payload["image"]["samples"] == 32
        assert len(payload["objects"]) == 2

        loaded = AnnotationSet.load(path)
        assert loaded.image_info == ann_set.image_info
        assert list(loaded) == list(ann_set)

    def test_save_overwrites_atomically(self, tmp_path):
        ann_set = AnnotationSet()
        ann_set.add(make_polygon())
        path = tmp_path / "ann.json"
        ann_set.save(path)
        first = path.read_text(encoding="utf-8")
        ann_set.objects.clear()
        ann_set.save(path)
        second = path.read_text(encoding="utf-8")
        assert first != second
        assert json.loads(second)["objects"] == []

    def test_load_bad_schema(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema": "other/9", "objects": []}), encoding="utf-8")
        with pytest.raises(ValueError):
            AnnotationSet.load(path)

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AnnotationSet.load(tmp_path / "nope.json")


class TestRasterizePolygon:
    def test_axis_aligned_polygon(self):
        ann = make_polygon()
        mask = rasterize(ann, (16, 16))
        assert mask.dtype == bool
        expected = np.zeros((16, 16), dtype=bool)
        expected[5:9, 4:10] = True
        np.testing.assert_array_equal(mask, expected)

    def test_polygon_clipped_to_shape(self):
        ann = Annotation(
            id="p",
            kind="polygon",
            name="x",
            points=[(-5.0, -5.0), (50.0, -5.0), (50.0, 50.0), (-5.0, 50.0)],
        )
        mask = rasterize(ann, (8, 8))
        assert mask.all()

    def test_too_few_points_empty(self):
        ann = Annotation(id="p", kind="polygon", name="x", points=[(1, 1), (2, 2)])
        mask = rasterize(ann, (8, 8))
        assert not mask.any()


class TestRasterizePolyline:
    def test_horizontal_ribbon_width3(self):
        ann = make_polyline()  # y=5, x 2..12
        mask = rasterize(ann, (16, 16), ribbon_width=3)
        expected = np.zeros((16, 16), dtype=bool)
        expected[4:7, 2:13] = True
        np.testing.assert_array_equal(mask, expected)

    def test_horizontal_ribbon_width5(self):
        ann = make_polyline()
        mask = rasterize(ann, (16, 16), ribbon_width=5)
        assert mask[3:8, 5:10].all()
        assert not mask[:3].any()
        assert not mask[8:].any()

    def test_diagonal_endpoints_covered(self):
        ann = Annotation(id="d", kind="polyline", name="diag", points=[(0.0, 0.0), (10.0, 10.0)])
        mask = rasterize(ann, (16, 16), ribbon_width=3)
        assert mask[0, 0]
        assert mask[10, 10]
        assert mask.sum() >= 20

    def test_width1_line(self):
        ann = Annotation(id="d", kind="polyline", name="line", points=[(2.0, 6.0), (10.0, 6.0)])
        mask = rasterize(ann, (16, 16), ribbon_width=1)
        assert mask[6, 2:11].all()
        assert not mask[5].any()
        assert not mask[7].any()


def test_assign_color_cycles():
    assert assign_color(0) != assign_color(1)
    assert assign_color(10) == assign_color(0)
