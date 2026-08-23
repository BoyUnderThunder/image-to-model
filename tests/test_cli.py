"""Tests for the command-line interface and the viewer/preview outputs."""

from __future__ import annotations

import numpy as np
import pytest

from image_to_model.cli import main
from image_to_model.imaging import save_image
from image_to_model.preview import render_mesh, render_turntable
from image_to_model.samples import SAMPLES, make_sample
from image_to_model.viewer import viewer_html, write_viewer

_FAST_ARGS = [
    "--working-resolution", "96",
    "--target-faces", "400",
    "--smooth-iterations", "1",
    "--texture-size", "128",
]


@pytest.fixture
def ball_png(tmp_path):
    path = tmp_path / "ball.png"
    save_image(make_sample("ball", 96), path)
    return path


class TestSamples:
    @pytest.mark.parametrize("name", sorted(SAMPLES))
    def test_every_sample_renders_to_rgba(self, name):
        image = make_sample(name, 64)
        assert image.shape == (64, 64, 4)
        assert image.dtype == np.uint8

    def test_unknown_sample_raises(self):
        with pytest.raises(ValueError, match="Unknown sample"):
            make_sample("dragon")


class TestCLI:
    def test_info_succeeds(self, capsys):
        assert main(["info"]) == 0
        output = capsys.readouterr().out
        assert "roblox" in output
        assert "21,000" in output  # the Roblox hard limit is surfaced
        assert "Export formats" in output

    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])
        assert excinfo.value.code == 0

    def test_bare_image_argument_implies_run(self, ball_png, tmp_path, capsys):
        assert main([str(ball_png), "-o", str(tmp_path / "out"), *_FAST_ARGS]) == 0
        assert "Triangles" in capsys.readouterr().out

    def test_run_writes_expected_files(self, ball_png, tmp_path):
        out_dir = tmp_path / "out"
        assert main(["run", str(ball_png), "-o", str(out_dir), *_FAST_ARGS]) == 0
        names = {path.name for path in out_dir.iterdir()}
        assert {"ball.obj", "ball.mtl", "ball_texture.png", "ball.glb"} <= names

    def test_explicit_format_selection(self, ball_png, tmp_path):
        out_dir = tmp_path / "out"
        assert main(["run", str(ball_png), "-o", str(out_dir), "-f", "ply", *_FAST_ARGS]) == 0
        assert {path.suffix for path in out_dir.iterdir()} == {".ply", ".md", ".luau"}

    def test_verbose_flag_after_subcommand(self, ball_png, tmp_path):
        # -v used to be rejected here because only the top-level parser took it.
        assert main(["run", str(ball_png), "-o", str(tmp_path / "out"), "-v", *_FAST_ARGS]) == 0

    def test_verbose_flag_before_subcommand(self, ball_png, tmp_path):
        assert main(["-v", "run", str(ball_png), "-o", str(tmp_path / "out"), *_FAST_ARGS]) == 0

    def test_demo_produces_a_model(self, tmp_path):
        out_dir = tmp_path / "demo"
        assert main(["demo", "--sample", "star", "--size", "96", "-o", str(out_dir), *_FAST_ARGS]) == 0
        assert any(path.suffix == ".glb" for path in out_dir.iterdir())

    def test_preview_and_viewer_flags(self, ball_png, tmp_path):
        out_dir = tmp_path / "out"
        code = main(
            ["run", str(ball_png), "-o", str(out_dir), "--preview", "--viewer", *_FAST_ARGS]
        )
        assert code == 0
        assert (out_dir / "preview.png").exists()
        assert (out_dir / "viewer.html").exists()

    def test_debug_dir_flag(self, ball_png, tmp_path):
        debug_dir = tmp_path / "dbg"
        main(["run", str(ball_png), "-o", str(tmp_path / "out"), "--debug-dir", str(debug_dir), *_FAST_ARGS])
        assert (debug_dir / "depth.png").exists()

    def test_missing_file_reports_an_error(self, tmp_path, capsys):
        assert main(["run", str(tmp_path / "absent.png"), "-o", str(tmp_path / "o")]) == 2
        assert "error:" in capsys.readouterr().err

    def test_preset_is_applied(self, ball_png, tmp_path):
        assert main(
            ["run", str(ball_png), "-o", str(tmp_path / "out"), "--preset", "fast", *_FAST_ARGS]
        ) == 0


class TestPreview:
    @pytest.fixture(scope="class")
    def rendered(self, request):
        from image_to_model.pipeline import reconstruct

        result = reconstruct(
            make_sample("ball", 96), working_resolution=96, target_faces=400, smooth_iterations=1
        )
        return result, render_mesh(result.mesh, size=96, texture=result.texture)

    def test_render_has_the_requested_size(self, rendered):
        _, image = rendered
        assert image.shape == (96, 96, 3)

    def test_render_draws_something(self, rendered):
        _, image = rendered
        # More than the flat background colour must be present.
        assert len(np.unique(image.reshape(-1, 3), axis=0)) > 10

    def test_turntable_is_a_strip(self, rendered):
        result, _ = rendered
        sheet = render_turntable(result.mesh, size=64, views=4, texture=result.texture)
        assert sheet.shape == (64, 256, 3)

    def test_empty_mesh_renders_background_only(self):
        from image_to_model.types import Mesh

        empty = Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
        image = render_mesh(empty, size=32)
        assert len(np.unique(image.reshape(-1, 3), axis=0)) == 1


class TestViewer:
    def test_html_is_self_contained(self, unit_cube):
        html = viewer_html(unit_cube.with_normals(), title="Cube")
        assert "<title>Cube</title>" in html
        assert "http://" not in html and "https://" not in html
        assert "webgl" in html.lower()

    def test_payload_carries_the_geometry(self, unit_cube):
        html = viewer_html(unit_cube.with_normals())
        assert '"positions"' in html and '"indices"' in html

    def test_write_viewer_creates_the_file(self, unit_cube, tmp_path):
        path = write_viewer(unit_cube.with_normals(), tmp_path / "nested" / "v.html")
        assert path.exists()
        assert path.read_text().startswith("<!doctype html>")
