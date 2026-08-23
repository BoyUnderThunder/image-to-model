"""Tests for the core data types."""

from __future__ import annotations

import numpy as np
import pytest

from image_to_model.types import CameraIntrinsics, DepthMap, Mesh, Subject


class TestMesh:
    def test_basic_properties(self, unit_cube):
        assert unit_cube.n_vertices == 8
        assert unit_cube.n_faces == 12
        assert not unit_cube.is_empty

    def test_closed_cube_is_watertight_and_genus_zero(self, unit_cube):
        assert unit_cube.is_watertight()
        assert unit_cube.euler_characteristic() == 2

    def test_unit_cube_measurements(self, unit_cube):
        assert unit_cube.surface_area() == pytest.approx(6.0)
        assert unit_cube.volume() == pytest.approx(1.0)
        assert np.allclose(unit_cube.extent(), [1.0, 1.0, 1.0])
        assert np.allclose(unit_cube.centroid(), [0.0, 0.0, 0.0])

    def test_flipped_winding_inverts_volume(self, unit_cube):
        assert unit_cube.flipped_winding().volume() == pytest.approx(-1.0)

    def test_normalize_scales_longest_axis(self, unit_cube):
        stretched = unit_cube.copy()
        stretched.vertices = stretched.vertices * np.array([4.0, 1.0, 1.0], dtype=np.float32)
        normalized = stretched.normalized(target_size=2.0)
        assert float(normalized.extent().max()) == pytest.approx(2.0)
        assert np.allclose(normalized.centroid(), [0, 0, 0], atol=1e-6)

    def test_normals_are_unit_length_and_outward(self, unit_cube):
        with_normals = unit_cube.with_normals()
        lengths = np.linalg.norm(with_normals.vertex_normals, axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-5)
        # On a cube centred at the origin every normal points away from it.
        outward = (with_normals.vertex_normals * with_normals.vertices).sum(axis=1)
        assert np.all(outward > 0)

    def test_rejects_out_of_range_face_index(self):
        with pytest.raises(ValueError, match="out of range"):
            Mesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 9]]))

    def test_rejects_mismatched_attribute_length(self):
        with pytest.raises(ValueError, match="one entry per vertex"):
            Mesh(
                vertices=np.zeros((3, 3)),
                faces=np.array([[0, 1, 2]]),
                vertex_colors=np.zeros((2, 3), dtype=np.uint8),
            )

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            Mesh(vertices=np.zeros((3, 2)), faces=np.zeros((0, 3)))

    def test_empty_mesh_is_reported_empty(self):
        mesh = Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
        assert mesh.is_empty
        assert not mesh.is_watertight()

    def test_copy_is_independent(self, unit_cube):
        duplicate = unit_cube.copy()
        duplicate.vertices[0] = [9.0, 9.0, 9.0]
        assert not np.allclose(duplicate.vertices[0], unit_cube.vertices[0])

    def test_stats_reports_expected_keys(self, unit_cube):
        stats = unit_cube.stats()
        assert stats["watertight"] is True
        assert stats["faces"] == 12
        assert stats["has_vertex_colors"] is True


class TestDepthMap:
    def test_normalized_maps_into_unit_range(self):
        depth = DepthMap(depth=np.linspace(5.0, 25.0, 100).reshape(10, 10))
        values = depth.normalized().depth
        assert values.min() == pytest.approx(0.0, abs=1e-5)
        assert values.max() == pytest.approx(1.0, abs=1e-5)

    def test_normalized_is_robust_to_outliers(self):
        values = np.full((20, 20), 10.0, dtype=np.float32)
        values[0, 0] = 1e6  # a single speckle should not flatten everything else
        normalized = DepthMap(depth=values).normalized().depth
        assert normalized[5, 5] == pytest.approx(normalized[9, 9])

    def test_constant_depth_does_not_divide_by_zero(self):
        normalized = DepthMap(depth=np.full((8, 8), 3.0)).normalized().depth
        assert np.all(np.isfinite(normalized))

    def test_inverted_flips_ordering(self):
        depth = DepthMap(depth=np.array([[0.0, 1.0], [2.0, 3.0]]))
        assert depth.inverted().depth[0, 0] > depth.inverted().depth[1, 1]

    def test_mask_shape_is_validated(self):
        with pytest.raises(ValueError, match="does not match"):
            DepthMap(depth=np.zeros((4, 4)), mask=np.zeros((3, 3)))


class TestSubject:
    def test_coverage_and_bounding_box(self):
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[2:6, 3:8] = 1.0
        subject = Subject(image=np.zeros((10, 10, 3), dtype=np.uint8), mask=mask)
        assert subject.coverage == pytest.approx(20 / 100)
        assert subject.bounding_box() == (3, 2, 8, 6)

    def test_empty_mask_falls_back_to_full_frame(self):
        subject = Subject(
            image=np.zeros((6, 8, 3), dtype=np.uint8), mask=np.zeros((6, 8), dtype=np.float32)
        )
        assert subject.bounding_box() == (0, 0, 8, 6)

    def test_mask_shape_is_validated(self):
        with pytest.raises(ValueError, match="does not match"):
            Subject(image=np.zeros((4, 4, 3), dtype=np.uint8), mask=np.zeros((5, 5)))


class TestCameraIntrinsics:
    def test_from_fov_centres_principal_point(self):
        intrinsics = CameraIntrinsics.from_fov(200, 100, 90.0)
        assert intrinsics.cx == pytest.approx(100.0)
        assert intrinsics.cy == pytest.approx(50.0)
        # At 90 degrees horizontal FOV, fx equals half the image width.
        assert intrinsics.fx == pytest.approx(100.0)

    def test_narrower_fov_gives_longer_focal_length(self):
        wide = CameraIntrinsics.from_fov(100, 100, 100.0)
        narrow = CameraIntrinsics.from_fov(100, 100, 30.0)
        assert narrow.fx > wide.fx

    def test_rejects_degenerate_size(self):
        with pytest.raises(ValueError):
            CameraIntrinsics.from_fov(0, 100)
