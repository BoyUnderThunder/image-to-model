"""Tests for surface construction, mesh cleanup and decimation."""

from __future__ import annotations

import numpy as np
import pytest

from image_to_model.errors import ReconstructionError
from image_to_model.geometry.decimate import decimate
from image_to_model.geometry.lift import camera_distance, project_to_world, world_to_pixel
from image_to_model.geometry.mesh_ops import (
    boundary_vertices,
    clean_mesh,
    compact_vertices,
    compute_vertex_normals,
    connected_components,
    keep_largest_components,
    remove_degenerate_faces,
    remove_duplicate_faces,
    taubin_smooth,
    weld_vertices,
)
from image_to_model.geometry.surface import RIM_PROFILES, build_surface, choose_stride
from image_to_model.types import CameraIntrinsics, Mesh


def _disc_mask(size: int = 64, radius: float = 0.7) -> np.ndarray:
    axis = (np.arange(size, dtype=np.float32) - (size - 1) / 2) / (size / 2)
    grid = axis[None, :] ** 2 + axis[:, None] ** 2
    return (grid <= radius**2).astype(np.float32)


def _dome_depth(mask: np.ndarray) -> np.ndarray:
    """A smooth dome, normalised so larger means further away."""
    size = mask.shape[0]
    axis = (np.arange(size, dtype=np.float32) - (size - 1) / 2) / (size / 2)
    height = np.clip(1.0 - (axis[None, :] ** 2 + axis[:, None] ** 2), 0.0, 1.0) ** 0.5
    return (1.0 - height).astype(np.float32)


class TestSurface:
    @pytest.fixture
    def disc_surface(self):
        mask = _disc_mask()
        return build_surface(_dome_depth(mask), mask, stride=1)

    def test_closed_surface_is_watertight(self, disc_surface):
        assert disc_surface.mesh.is_watertight()

    def test_closed_surface_is_genus_zero(self, disc_surface):
        assert disc_surface.mesh.euler_characteristic() == 2

    def test_winding_faces_outward(self, disc_surface):
        # A positive signed volume means the winding is consistently outward.
        assert disc_surface.mesh.volume() > 0

    def test_front_and_back_vertices_are_both_present(self, disc_surface):
        assert set(np.unique(disc_surface.side).tolist()) == {0, 1}

    def test_open_surface_has_a_boundary(self):
        mask = _disc_mask()
        result = build_surface(_dome_depth(mask), mask, close_back=False, stride=1)
        assert not result.mesh.is_watertight()
        assert boundary_vertices(result.mesh).any()

    def test_thickness_controls_depth(self):
        mask = _disc_mask()
        depth = _dome_depth(mask)
        thin = build_surface(depth, mask, thickness=0.05, stride=1).mesh
        thick = build_surface(depth, mask, thickness=1.0, stride=1).mesh
        assert thick.extent()[2] > thin.extent()[2]

    def test_relief_scale_controls_depth(self):
        mask = _disc_mask()
        depth = _dome_depth(mask)
        shallow = build_surface(depth, mask, relief_scale=0.1, stride=1).mesh
        deep = build_surface(depth, mask, relief_scale=0.8, stride=1).mesh
        assert deep.extent()[2] > shallow.extent()[2]

    def test_stride_reduces_triangle_count(self):
        mask = _disc_mask()
        depth = _dome_depth(mask)
        fine = build_surface(depth, mask, stride=1).mesh
        coarse = build_surface(depth, mask, stride=3).mesh
        assert coarse.n_faces < fine.n_faces
        assert coarse.is_watertight()

    def test_empty_mask_raises(self):
        mask = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(ReconstructionError, match="mask is empty"):
            build_surface(np.zeros((32, 32), dtype=np.float32), mask, stride=1)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ReconstructionError, match="does not match"):
            build_surface(np.zeros((8, 8)), np.ones((4, 4)), stride=1)

    def test_pixel_coords_align_with_vertices(self, disc_surface):
        assert len(disc_surface.pixel_coords) == disc_surface.mesh.n_vertices


class TestChooseStride:
    def test_larger_budget_allows_finer_grid(self):
        mask = _disc_mask(128)
        assert choose_stride(mask > 0.5, 100_000) <= choose_stride(mask > 0.5, 1_000)

    def test_zero_budget_uses_full_resolution(self):
        assert choose_stride(_disc_mask() > 0.5, 0) == 1

    def test_empty_mask_is_safe(self):
        assert choose_stride(np.zeros((16, 16), dtype=bool), 5000) == 1


class TestLift:
    def test_projection_round_trips_to_pixels(self):
        intrinsics = CameraIntrinsics.from_fov(200, 150, 55.0)
        distance = 5.0
        cols = np.array([10.0, 100.0, 190.0], dtype=np.float32)
        rows = np.array([20.0, 75.0, 140.0], dtype=np.float32)
        relief = np.array([0.0, 0.3, -0.2], dtype=np.float32)

        points = project_to_world(cols, rows, relief, intrinsics, distance)
        recovered = world_to_pixel(points, intrinsics, distance)

        assert np.allclose(recovered[:, 0], cols, atol=1e-3)
        assert np.allclose(recovered[:, 1], rows, atol=1e-3)

    def test_image_rows_map_to_descending_y(self):
        intrinsics = CameraIntrinsics.from_fov(100, 100, 55.0)
        points = project_to_world(
            np.array([50.0, 50.0]), np.array([10.0, 90.0]), np.zeros(2), intrinsics, 4.0
        )
        assert points[0, 1] > points[1, 1]

    def test_camera_distance_follows_pinhole_relation(self):
        intrinsics = CameraIntrinsics.from_fov(100, 100, 55.0)
        distance = camera_distance(intrinsics, 50.0, 1.0)
        assert distance == pytest.approx(intrinsics.fx / 50.0)

    def test_rejects_zero_subject(self):
        with pytest.raises(ValueError):
            camera_distance(CameraIntrinsics.from_fov(10, 10), 0.0, 1.0)


class TestMeshOps:
    def test_normals_point_outward_on_a_cube(self, unit_cube):
        normals = compute_vertex_normals(unit_cube.vertices, unit_cube.faces)
        assert np.all((normals * unit_cube.vertices).sum(axis=1) > 0)

    def test_weld_merges_coincident_vertices(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        mesh = Mesh(vertices=vertices, faces=np.array([[0, 1, 2], [3, 1, 2]], dtype=np.int32))
        welded = weld_vertices(mesh, 1e-6)
        assert welded.n_vertices == 3

    def test_degenerate_faces_are_dropped(self):
        mesh = Mesh(
            vertices=np.eye(3, dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 1, 1]], dtype=np.int32),
        )
        assert remove_degenerate_faces(mesh).n_faces == 1

    def test_duplicate_faces_are_dropped_entirely(self):
        # Both copies must go: keeping one would leave its neighbours attached
        # to a face that is really a zero-thickness flap.
        mesh = Mesh(
            vertices=np.eye(3, dtype=np.float32),
            faces=np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int32),
        )
        assert remove_duplicate_faces(mesh).n_faces == 0

    def test_compact_reports_kept_indices(self):
        mesh = Mesh(
            vertices=np.zeros((5, 3), dtype=np.float32),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )
        compacted, kept = compact_vertices(mesh)
        assert compacted.n_vertices == 3
        assert kept.tolist() == [0, 1, 2]

    def test_connected_components_counts_islands(self, unit_cube):
        shifted = unit_cube.copy()
        merged = Mesh(
            vertices=np.concatenate([unit_cube.vertices, shifted.vertices + 10.0]),
            faces=np.concatenate([unit_cube.faces, shifted.faces + unit_cube.n_vertices]),
        )
        _, count = connected_components(merged)
        assert count == 2

    def test_small_islands_are_removed(self, unit_cube):
        # Components are ranked by vertex count, so the island needs fewer
        # vertices than the cube rather than merely being physically smaller.
        island_vertices = np.array(
            [[20, 0, 0], [21, 0, 0], [20, 1, 0], [20, 0, 1]], dtype=np.float32
        )
        island_faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32)
        merged = Mesh(
            vertices=np.concatenate([unit_cube.vertices, island_vertices]),
            faces=np.concatenate([unit_cube.faces, island_faces + unit_cube.n_vertices]),
        )
        cleaned = keep_largest_components(merged, min_ratio=0.6)
        assert cleaned.n_faces == unit_cube.n_faces
        assert cleaned.n_vertices == unit_cube.n_vertices

    def test_islands_above_the_ratio_are_kept(self, unit_cube):
        island_vertices = np.array(
            [[20, 0, 0], [21, 0, 0], [20, 1, 0], [20, 0, 1]], dtype=np.float32
        )
        island_faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32)
        merged = Mesh(
            vertices=np.concatenate([unit_cube.vertices, island_vertices]),
            faces=np.concatenate([unit_cube.faces, island_faces + unit_cube.n_vertices]),
        )
        assert keep_largest_components(merged, min_ratio=0.4).n_faces == merged.n_faces

    def test_smoothing_preserves_topology(self):
        mask = _disc_mask()
        mesh = build_surface(_dome_depth(mask), mask, stride=1).mesh
        smoothed = taubin_smooth(mesh, iterations=5)
        assert smoothed.is_watertight()
        assert smoothed.n_vertices == mesh.n_vertices

    def test_taubin_barely_shrinks_volume(self):
        # This is the whole point of alternating lambda/mu rather than
        # running a plain Laplacian, which collapses the model steadily.
        mask = _disc_mask()
        mesh = build_surface(_dome_depth(mask), mask, stride=1).mesh
        smoothed = taubin_smooth(mesh, iterations=10)
        assert smoothed.volume() == pytest.approx(mesh.volume(), rel=0.10)

    def test_clean_mesh_is_idempotent(self):
        mask = _disc_mask()
        mesh = build_surface(_dome_depth(mask), mask, stride=1).mesh
        once = clean_mesh(mesh, weld_tolerance=0.0)
        twice = clean_mesh(once, weld_tolerance=0.0)
        assert once.n_faces == twice.n_faces


class TestDecimate:
    @pytest.fixture
    def dome(self):
        mask = _disc_mask(64)
        return build_surface(_dome_depth(mask), mask, stride=1).mesh

    def test_reaches_the_face_budget(self, dome):
        result = decimate(dome, 800)
        assert result.mesh.n_faces <= 800

    def test_preserves_watertightness(self, dome):
        assert decimate(dome, 800).mesh.is_watertight()

    def test_preserves_genus(self, dome):
        assert decimate(dome, 800).mesh.euler_characteristic() == 2

    def test_preserves_volume_closely(self, dome):
        result = decimate(dome, 1000)
        assert result.mesh.volume() == pytest.approx(dome.volume(), rel=0.05)

    def test_vertex_map_indexes_the_original_mesh(self, dome):
        result = decimate(dome, 800)
        assert len(result.vertex_map) == result.mesh.n_vertices
        assert int(result.vertex_map.max()) < dome.n_vertices

    def test_mesh_already_under_budget_is_untouched(self, unit_cube):
        result = decimate(unit_cube, 1000)
        assert result.mesh is unit_cube
        assert result.vertex_map.tolist() == list(range(unit_cube.n_vertices))

    def test_zero_budget_disables_decimation(self, dome):
        assert decimate(dome, 0).mesh is dome


class TestRimProfile:
    """The rim profile sets the silhouette's cross-section.

    ``fillet`` is the default because a real object's surface turns tangent to
    the view direction at its silhouette, which a smoothstep does not do.
    """

    @staticmethod
    def _build(profile: str):
        mask = _disc_mask()
        return build_surface(_dome_depth(mask), mask, stride=1, rim_profile=profile)

    @pytest.mark.parametrize("profile", sorted(RIM_PROFILES))
    def test_every_profile_closes_the_surface(self, profile):
        mesh = self._build(profile).mesh
        assert mesh.is_watertight()
        assert mesh.euler_characteristic() == 2

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown rim profile"):
            self._build("bevel")

    def test_fillet_starts_vertical_and_ends_flat(self):
        t = np.linspace(0.0, 1.0, 101)
        values = RIM_PROFILES["fillet"](t)
        assert values[0] == pytest.approx(0.0)
        assert values[-1] == pytest.approx(1.0)
        # A vertical tangent at the silhouette: the first step is far larger
        # than the last, which is what rounds the edge instead of cornering it.
        assert (values[1] - values[0]) > 10 * (values[-1] - values[-2])

    def test_smoothstep_starts_flat(self):
        t = np.linspace(0.0, 1.0, 101)
        values = RIM_PROFILES["smoothstep"](t)
        assert (values[1] - values[0]) < 0.01

    def test_fillet_reaches_full_relief_sooner_than_smoothstep(self):
        # The fillet lifts the surface away from the silhouette immediately,
        # so the rounding is confined to a narrow band at the edge instead of
        # flattening the whole outer region.
        t = np.linspace(0.0, 1.0, 201)
        fillet = RIM_PROFILES["fillet"](t)
        smoothstep = RIM_PROFILES["smoothstep"](t)
        assert np.all(fillet >= smoothstep - 1e-6)
        assert t[np.argmax(fillet >= 0.9)] < t[np.argmax(smoothstep >= 0.9)]

    def test_fillet_bulges_further_forward(self):
        # Perspective narrows the cross-section as the surface comes towards
        # the camera, so the fillet's fuller profile shows up as extra depth
        # rather than extra volume.
        fillet = self._build("fillet").mesh
        smoothstep = self._build("smoothstep").mesh
        assert fillet.extent()[2] >= smoothstep.extent()[2] - 1e-6

    def test_all_profiles_are_bounded(self):
        t = np.linspace(-0.5, 1.5, 51)
        for name, fn in RIM_PROFILES.items():
            values = np.asarray(fn(t))
            assert values.min() >= 0.0 and values.max() <= 1.0, name
