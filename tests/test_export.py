"""Tests for the mesh writers.

The GLB test parses the file back and walks its accessors, because a
byte-for-byte wrong glTF still produces a plausible-looking file and only fails
later, inside whatever tool the user imports it into.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from image_to_model.errors import ExportError
from image_to_model.export import export_many, export_mesh, format_for_path
from image_to_model.export.axes import convert_up_axis, prepare_for_target
from image_to_model.types import Mesh

_COMPONENT_TYPES = {5121: ("u1", 1), 5123: ("<u2", 2), 5125: ("<u4", 4), 5126: ("<f4", 4)}
_TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _parse_glb(path):
    raw = path.read_bytes()
    magic, version, total = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67, "missing glTF magic"
    assert version == 2
    assert total == len(raw), "header length disagrees with the file size"

    offset, chunks = 12, {}
    while offset < len(raw):
        length, kind = struct.unpack("<II", raw[offset : offset + 8])
        offset += 8
        assert length % 4 == 0, "chunk length must be 4-byte aligned"
        chunks[kind] = raw[offset : offset + length]
        offset += length
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


def _read_accessor(gltf, binary, index):
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype, size = _COMPONENT_TYPES[accessor["componentType"]]
    columns = _TYPE_COUNTS[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    assert start % size == 0, "accessor is not aligned to its component size"
    count = accessor["count"] * columns * size
    return np.frombuffer(binary[start : start + count], dtype=dtype).reshape(-1, columns)


@pytest.fixture
def textured_cube(unit_cube):
    mesh = unit_cube.copy()
    mesh.uvs = np.random.default_rng(0).random((8, 2)).astype(np.float32)
    return mesh.with_normals()


@pytest.fixture
def texture():
    return (np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8)


class TestFormatDispatch:
    def test_infers_format_from_extension(self):
        assert format_for_path("a/b/model.GLB") == "glb"

    def test_rejects_unknown_extension(self):
        with pytest.raises(ExportError, match="Unsupported export format"):
            format_for_path("model.blend")

    def test_rejects_missing_extension(self):
        with pytest.raises(ExportError, match="No file extension"):
            format_for_path("model")


class TestGLB:
    def test_round_trips_geometry(self, textured_cube, texture, tmp_path):
        path = tmp_path / "cube.glb"
        export_mesh(textured_cube, path, texture=texture)
        gltf, binary = _parse_glb(path)

        primitive = gltf["meshes"][0]["primitives"][0]
        positions = _read_accessor(gltf, binary, primitive["attributes"]["POSITION"])
        indices = _read_accessor(gltf, binary, primitive["indices"]).reshape(-1)

        assert positions.shape == (8, 3)
        assert np.allclose(positions, textured_cube.vertices, atol=1e-6)
        assert indices.tolist() == textured_cube.faces.reshape(-1).tolist()

    def test_position_accessor_declares_bounds(self, textured_cube, tmp_path):
        path = tmp_path / "cube.glb"
        export_mesh(textured_cube, path)
        gltf, binary = _parse_glb(path)
        accessor = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
        positions = _read_accessor(gltf, binary, gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"])
        assert np.allclose(accessor["min"], positions.min(axis=0))
        assert np.allclose(accessor["max"], positions.max(axis=0))

    def test_embeds_texture_as_png(self, textured_cube, texture, tmp_path):
        path = tmp_path / "cube.glb"
        export_mesh(textured_cube, path, texture=texture)
        gltf, binary = _parse_glb(path)

        view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
        png = binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert "target" not in view, "an image bufferView must not declare a target"
        assert "baseColorTexture" in gltf["materials"][0]["pbrMetallicRoughness"]

    def test_omits_texture_when_mesh_has_no_uvs(self, unit_cube, texture, tmp_path):
        path = tmp_path / "cube.glb"
        export_mesh(unit_cube.with_normals(), path, texture=texture)
        gltf, _ = _parse_glb(path)
        assert "images" not in gltf

    def test_vertex_colors_can_be_excluded(self, textured_cube, tmp_path):
        path = tmp_path / "cube.glb"
        export_mesh(textured_cube, path, include_vertex_colors=False)
        gltf, _ = _parse_glb(path)
        assert "COLOR_0" not in gltf["meshes"][0]["primitives"][0]["attributes"]

    def test_uses_32_bit_indices_for_large_meshes(self, tmp_path):
        count = 70_000  # past the 16-bit index ceiling
        vertices = np.random.default_rng(0).random((count, 3)).astype(np.float32)
        faces = np.arange(count - (count % 3), dtype=np.int32).reshape(-1, 3)
        path = tmp_path / "big.glb"
        export_mesh(Mesh(vertices=vertices, faces=faces), path)
        gltf, _ = _parse_glb(path)
        indices = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["indices"]]
        assert indices["componentType"] == 5125


class TestOBJ:
    def test_writes_sidecar_files_when_textured(self, textured_cube, texture, tmp_path):
        written = export_mesh(textured_cube, tmp_path / "cube.obj", texture=texture)
        names = {path.name for path in written}
        assert names == {"cube.obj", "cube.mtl", "cube_texture.png"}
        assert "map_Kd cube_texture.png" in (tmp_path / "cube.mtl").read_text()

    def test_face_indices_are_one_based(self, textured_cube, tmp_path):
        export_mesh(textured_cube, tmp_path / "cube.obj")
        faces = [
            line for line in (tmp_path / "cube.obj").read_text().splitlines() if line.startswith("f ")
        ]
        assert len(faces) == 12
        indices = [int(token.split("/")[0]) for line in faces for token in line.split()[1:]]
        assert min(indices) == 1
        assert max(indices) == 8

    def test_flips_v_coordinate_for_obj_convention(self, tmp_path):
        mesh = Mesh(
            vertices=np.eye(3, dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            uvs=np.array([[0.25, 0.75], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        )
        export_mesh(mesh, tmp_path / "t.obj")
        vt_lines = [
            line for line in (tmp_path / "t.obj").read_text().splitlines() if line.startswith("vt ")
        ]
        assert vt_lines[0].split()[1:] == ["0.250000", "0.250000"]

    def test_vertex_colors_are_opt_in(self, unit_cube, tmp_path):
        export_mesh(unit_cube, tmp_path / "plain.obj", include_vertex_colors=False)
        first = [
            line for line in (tmp_path / "plain.obj").read_text().splitlines() if line.startswith("v ")
        ][0]
        assert len(first.split()) == 4  # "v x y z" only


class TestPLYandSTL:
    def test_ply_header_and_counts(self, unit_cube, tmp_path):
        path = tmp_path / "cube.ply"
        export_mesh(unit_cube.with_normals(), path)
        raw = path.read_bytes()
        header = raw[: raw.index(b"end_header")].decode("ascii")
        assert "format binary_little_endian 1.0" in header
        assert "element vertex 8" in header
        assert "element face 12" in header
        assert "property uchar red" in header

    def test_stl_record_size(self, unit_cube, tmp_path):
        path = tmp_path / "cube.stl"
        export_mesh(unit_cube, path)
        raw = path.read_bytes()
        count = struct.unpack("<I", raw[80:84])[0]
        assert count == 12
        assert len(raw) == 84 + 12 * 50

    def test_empty_mesh_is_rejected(self, tmp_path):
        empty = Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
        with pytest.raises(ExportError):
            export_mesh(empty, tmp_path / "empty.stl")


class TestAxesAndBatch:
    def test_y_up_is_a_no_op(self, unit_cube):
        assert convert_up_axis(unit_cube, "y") is unit_cube

    def test_z_up_conversion_preserves_volume_and_handedness(self, unit_cube):
        converted = convert_up_axis(unit_cube, "z")
        assert converted.volume() == pytest.approx(unit_cube.volume())

    def test_z_up_moves_y_into_z(self):
        mesh = Mesh(
            vertices=np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int32),
        )
        converted = convert_up_axis(mesh, "z")
        assert np.allclose(converted.vertices[0], [0.0, 0.0, 2.0])
        assert np.allclose(converted.vertices[1], [3.0, 0.0, 0.0])  # X is unchanged

    def test_rejects_unknown_axis(self, unit_cube):
        with pytest.raises(ValueError, match="up_axis"):
            convert_up_axis(unit_cube, "x")

    def test_prepare_for_target_scales_then_rotates(self, unit_cube):
        prepared = prepare_for_target(unit_cube, up_axis="z", size_units=10.0)
        assert float(prepared.extent().max()) == pytest.approx(10.0)

    def test_export_many_writes_every_format(self, textured_cube, texture, tmp_path):
        written = export_many(
            textured_cube, tmp_path, "cube", ["obj", "glb", "ply", "stl"], texture=texture
        )
        suffixes = {path.suffix for path in written}
        assert suffixes == {".obj", ".mtl", ".png", ".glb", ".ply", ".stl"}
        assert all(path.stat().st_size > 0 for path in written)
