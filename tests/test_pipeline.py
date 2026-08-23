"""End-to-end tests, target profiles and configuration."""

from __future__ import annotations

import numpy as np
import pytest

from image_to_model.backends import all_backend_names, get_backend, register_backend
from image_to_model.backends.base import BackendOutput, ReconstructionBackend
from image_to_model.config import PRESETS, ReconstructionConfig, preset
from image_to_model.errors import BackendNotFound
from image_to_model.imaging import load_image, resize_longest, save_image
from image_to_model.pipeline import reconstruct
from image_to_model.samples import make_sample
from image_to_model.targets import ROBLOX, available_targets, get_target
from image_to_model.types import Mesh


class TestConfig:
    def test_defaults_target_roblox(self):
        assert ReconstructionConfig().target == "roblox"

    def test_face_budget_comes_from_the_target(self):
        assert ReconstructionConfig(target="roblox").resolve_target_faces() == 10_000
        assert ReconstructionConfig(target="roblox-avatar").resolve_target_faces() == 4_000

    def test_explicit_budget_wins(self):
        assert ReconstructionConfig(target_faces=1234).resolve_target_faces() == 1234

    def test_negative_budget_disables_decimation(self):
        assert ReconstructionConfig(target_faces=-1).resolve_target_faces() == 0

    def test_texture_size_is_capped_by_the_target(self):
        # Roblox allows at most 1024, so a larger request must be clamped.
        assert ReconstructionConfig(target="roblox", texture_size=4096).resolve_texture_size() == 1024

    def test_roblox_forces_texture_baking(self):
        assert ReconstructionConfig(target="roblox").should_bake_texture()

    def test_generic_target_prefers_vertex_colours(self):
        assert not ReconstructionConfig(target="generic").should_bake_texture()

    def test_bake_can_be_forced_on_or_off(self):
        assert ReconstructionConfig(target="generic", bake_texture="always").should_bake_texture()
        assert not ReconstructionConfig(target="roblox", bake_texture="never").should_bake_texture()

    def test_no_texture_disables_baking(self):
        assert not ReconstructionConfig(target="roblox", texture=False).should_bake_texture()

    def test_up_axis_follows_the_target(self):
        assert ReconstructionConfig(target="roblox").resolve_up_axis() == "y"
        assert ReconstructionConfig(target="print").resolve_up_axis() == "z"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"working_resolution": 8},
            {"mask_threshold": 2.0},
            {"fov_degrees": 400.0},
            {"relief_scale": 0.0},
            {"thickness": -1.0},
            {"device": "tpu"},
            {"target": "playstation"},
            {"bake_texture": "sometimes"},
            {"up_axis": "w"},
        ],
    )
    def test_invalid_values_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ReconstructionConfig(**kwargs)

    def test_from_dict_ignores_unknown_keys(self):
        config = ReconstructionConfig.from_dict({"thickness": 0.3, "nonsense": 1, "target": None})
        assert config.thickness == 0.3
        assert config.target == "roblox"  # None was skipped, so the default held

    def test_merged_ignores_none(self):
        base = ReconstructionConfig(thickness=0.3)
        assert base.merged(thickness=None).thickness == 0.3

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_every_preset_builds(self, name):
        assert isinstance(preset(name), ReconstructionConfig)

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            preset("ultra")


class TestTargets:
    def test_roblox_limits_match_the_documented_specification(self):
        assert ROBLOX.triangle_budget == 10_000
        assert ROBLOX.triangle_hard_limit == 21_000
        assert ROBLOX.max_texture_size == 1024
        assert ROBLOX.up_axis == "y"
        assert ROBLOX.unit_name == "stud"
        assert ROBLOX.meters_per_unit == pytest.approx(0.28)

    def test_roblox_does_not_support_vertex_colours(self):
        assert ROBLOX.supports_vertex_colors is False

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            get_target("gamecube")

    def test_lookup_is_case_insensitive(self):
        assert get_target("RoBloX") is ROBLOX

    def test_all_targets_resolve(self):
        assert all(get_target(name) for name in available_targets())

    def test_over_limit_mesh_is_an_error(self, unit_cube):
        mesh = unit_cube.copy()
        mesh.faces = np.tile(mesh.faces, (2000, 1))
        report = ROBLOX.validate(mesh)
        assert not report.ok
        assert any(issue.code == "triangle_limit" for issue in report.errors)

    def test_missing_uvs_are_an_error_for_roblox(self, unit_cube):
        report = ROBLOX.validate(unit_cube)
        assert any(issue.code == "missing_uvs" for issue in report.errors)

    def test_oversized_texture_is_an_error(self, unit_cube):
        mesh = unit_cube.copy()
        mesh.uvs = np.zeros((8, 2), dtype=np.float32)
        report = ROBLOX.validate(mesh, texture_size=(2048, 2048))
        assert any(issue.code == "texture_too_large" for issue in report.errors)

    def test_empty_mesh_is_an_error(self):
        empty = Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
        assert not ROBLOX.validate(empty).ok

    def test_open_mesh_is_only_a_warning(self, unit_cube):
        mesh = unit_cube.copy()
        mesh.uvs = np.zeros((8, 2), dtype=np.float32)
        mesh.faces = mesh.faces[:6]  # tear it open
        report = ROBLOX.validate(mesh)
        assert any(issue.code == "not_watertight" for issue in report.warnings)


class TestImaging:
    def test_resize_longest_preserves_aspect(self):
        resized = resize_longest(np.zeros((100, 200, 3), dtype=np.uint8), 100)
        assert resized.shape[:2] == (50, 100)

    def test_small_images_are_not_upscaled(self):
        original = np.zeros((40, 30, 3), dtype=np.uint8)
        assert resize_longest(original, 500).shape == original.shape

    def test_round_trip_through_disk(self, tmp_path, ball_image):
        path = tmp_path / "ball.png"
        save_image(ball_image, path)
        loaded = load_image(path)
        assert loaded.shape[2] == 4
        assert np.array_equal(loaded[..., :3], ball_image[..., :3])

    def test_missing_file_raises(self, tmp_path):
        from image_to_model.errors import InputError

        with pytest.raises(InputError, match="not found"):
            load_image(tmp_path / "nope.png")


class TestBackendRegistry:
    def test_depth_backend_is_registered(self):
        assert "depth" in all_backend_names()

    def test_unknown_backend_raises(self):
        with pytest.raises(BackendNotFound, match="Unknown backend"):
            get_backend("neural-magic")

    def test_custom_backend_can_be_registered(self, unit_cube):
        class StubBackend(ReconstructionBackend):
            name = "stub-for-test"

            def reconstruct(self, subject, config):
                return BackendOutput(mesh=unit_cube)

        register_backend(StubBackend)
        assert isinstance(get_backend("stub-for-test"), StubBackend)


class TestReconstruction:
    @pytest.fixture(scope="class")
    def roblox_result(self):
        return reconstruct(
            make_sample("ball", 128),
            working_resolution=128,
            target_faces=600,
            smooth_iterations=2,
            texture_size=128,
        )

    def test_produces_a_closed_solid(self, roblox_result):
        # Closure is checked after merging coincident vertices: the UV seam
        # splits vertices so the front and back tiles can have their own
        # texture coordinates, which breaks index-level watertightness while
        # leaving the solid geometrically closed.
        assert roblox_result.mesh.is_closed_surface()

    def test_respects_the_face_budget(self, roblox_result):
        assert roblox_result.mesh.n_faces <= 600

    def test_passes_target_validation(self, roblox_result):
        assert roblox_result.validation.ok, str(roblox_result.validation)

    def test_bakes_a_texture_and_uvs_for_roblox(self, roblox_result):
        assert roblox_result.texture is not None
        assert roblox_result.mesh.uvs is not None
        assert roblox_result.mesh.uvs.min() >= 0.0
        assert roblox_result.mesh.uvs.max() <= 1.0

    def test_scales_to_stud_units(self, roblox_result):
        assert float(roblox_result.mesh.extent().max()) == pytest.approx(4.0, rel=1e-3)

    def test_has_normals(self, roblox_result):
        assert roblox_result.mesh.vertex_normals is not None

    def test_summary_mentions_the_target(self, roblox_result):
        assert "Roblox" in roblox_result.summary()

    def test_accepts_an_rgb_array(self):
        rgb = make_sample("ball", 96)[..., :3]
        result = reconstruct(rgb, working_resolution=96, target_faces=400, smooth_iterations=1)
        assert result.mesh.n_faces > 0

    def test_size_units_override_is_honoured(self):
        result = reconstruct(
            make_sample("ball", 96),
            working_resolution=96,
            target_faces=400,
            smooth_iterations=1,
            size_units=10.0,
        )
        assert float(result.mesh.extent().max()) == pytest.approx(10.0, rel=1e-3)

    def test_print_target_is_z_up_and_millimetres(self):
        result = reconstruct(
            make_sample("ball", 96),
            target="print",
            working_resolution=96,
            target_faces=400,
            smooth_iterations=1,
        )
        assert result.config.resolve_up_axis() == "z"
        assert float(result.mesh.extent().max()) == pytest.approx(100.0, rel=1e-3)

    def test_open_back_is_not_watertight(self):
        result = reconstruct(
            make_sample("ball", 96),
            working_resolution=96,
            target_faces=400,
            smooth_iterations=1,
            close_back=False,
        )
        assert not result.mesh.is_watertight()

    def test_saving_writes_roblox_guides(self, tmp_path, roblox_result):
        written = roblox_result.save(tmp_path)
        names = {path.name for path in written}
        assert "ROBLOX_IMPORT.md" in names
        assert any(name.endswith(".luau") for name in names)
        assert any(name.endswith(".obj") for name in names)
        assert any(name.endswith(".glb") for name in names)

    def test_saving_to_an_explicit_file_uses_that_format(self, tmp_path, roblox_result):
        written = roblox_result.save(tmp_path / "thing.glb")
        assert written[0].name == "thing.glb"

    def test_debug_output_is_written(self, tmp_path, roblox_result):
        written = roblox_result.save_debug(tmp_path / "debug")
        assert {path.name for path in written} == {"mask.png", "depth.png", "texture.png"}
