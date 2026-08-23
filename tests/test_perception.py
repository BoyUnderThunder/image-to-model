"""Tests for segmentation, depth estimation and texture baking."""

from __future__ import annotations

import numpy as np
import pytest

from image_to_model.depth import available_models, create_estimator, estimate_depth
from image_to_model.depth.heuristic import HeuristicDepthEstimator, distance_transform
from image_to_model.depth.learned import LearnedDepthEstimator, resolve_model
from image_to_model.errors import ImageToModelError, SegmentationError
from image_to_model.morphology import (
    binary_close,
    binary_dilate,
    binary_erode,
    fill_holes,
    largest_component,
    otsu_threshold,
)
from image_to_model.samples import make_sample
from image_to_model.segmentation import has_real_alpha, segment
from image_to_model.texture import (
    AtlasLayout,
    build_atlas,
    inpaint_background,
    sample_vertex_colors,
)


class TestMorphology:
    def test_erode_then_dilate_restores_a_solid_block(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        assert binary_dilate(binary_erode(mask, 1), 1).sum() == mask.sum()

    def test_erode_shrinks_and_dilate_grows(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        assert binary_erode(mask, 1).sum() < mask.sum() < binary_dilate(mask, 1).sum()

    def test_close_seals_a_small_gap(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        mask[9:11, 10] = False
        assert binary_close(mask, 2).sum() > mask.sum()

    def test_fill_holes_closes_interior_only(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[4:16, 4:16] = True
        mask[9:11, 9:11] = False
        filled = fill_holes(mask)
        assert filled[9, 9]
        assert not filled[0, 0]  # the exterior stays background

    def test_largest_component_drops_speckle(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:10, 2:10] = True
        mask[18, 18] = True
        assert largest_component(mask).sum() == 64

    def test_otsu_separates_two_clusters(self):
        values = np.concatenate([np.zeros(500), np.ones(500)])
        assert 0.0 < otsu_threshold(values) < 1.0

    def test_otsu_handles_constant_input(self):
        assert np.isfinite(otsu_threshold(np.full(100, 0.5)))


class TestSegmentation:
    def test_detects_a_real_alpha_channel(self, star_image):
        assert has_real_alpha(star_image)

    def test_opaque_image_has_no_real_alpha(self, ball_image):
        assert not has_real_alpha(ball_image)

    def test_alpha_is_used_when_present(self, star_image):
        _, method = segment(star_image)
        assert method == "alpha"

    def test_ball_coverage_matches_its_geometry(self, ball_image):
        # The sample is a disc of radius 0.62 in normalised coordinates, so it
        # covers pi * 0.62^2 / 4 of the frame.
        mask, method = segment(ball_image)
        assert method == "border"
        assert mask.mean() == pytest.approx(np.pi * 0.62**2 / 4, abs=0.03)

    def test_gradient_background_does_not_swallow_the_frame(self):
        # The rocket sits on a vertical gradient, which a single global colour
        # model gets badly wrong.
        mask, _ = segment(make_sample("rocket", 128))
        assert 0.02 < mask.mean() < 0.45

    def test_none_method_returns_full_coverage(self, ball_image):
        mask, method = segment(ball_image, method="none")
        assert method == "none"
        assert mask.min() == 1.0

    def test_mask_is_soft_and_bounded(self, ball_image):
        mask, _ = segment(ball_image)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0 and mask.max() <= 1.0

    def test_unknown_method_raises(self, ball_image):
        with pytest.raises(SegmentationError, match="Unknown segmentation method"):
            segment(ball_image, method="magic")

    def test_rejects_non_image_input(self):
        with pytest.raises(SegmentationError, match="Expected an RGB"):
            segment(np.zeros((5, 5), dtype=np.uint8))


class TestDepth:
    def test_heuristic_is_always_available(self):
        assert "heuristic" in available_models()

    def test_distance_transform_peaks_at_the_centre(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        distance = distance_transform(mask)
        assert distance[16, 16] > distance[9, 9]
        assert distance[0, 0] == 0

    def test_heuristic_produces_relief_inside_the_mask(self, ball_image):
        mask, _ = segment(ball_image)
        depth = HeuristicDepthEstimator().estimate(ball_image[..., :3], mask).normalized()
        inside = depth.depth[mask > 0.5]
        assert inside.max() - inside.min() > 0.2

    def test_depth_convention_is_near_is_small(self, ball_image):
        # The centre of a ball is nearest the camera, so its depth must be the
        # smallest value in the subject.
        mask, _ = segment(ball_image)
        depth = HeuristicDepthEstimator().estimate(ball_image[..., :3], mask).normalized().depth
        centre = depth[ball_image.shape[0] // 2, ball_image.shape[1] // 2]
        assert centre < np.median(depth[mask > 0.5])

    def test_depth_shape_matches_input(self, ball_image):
        result = estimate_depth(ball_image[..., :3], model="heuristic")
        assert result.depth.shape == ball_image.shape[:2]

    def test_auto_falls_back_without_torch(self):
        estimator = create_estimator("auto")
        if not LearnedDepthEstimator.is_available():
            assert isinstance(estimator, HeuristicDepthEstimator)

    def test_bad_model_falls_back_rather_than_failing(self, ball_image):
        result = estimate_depth(
            ball_image[..., :3], model="definitely/not-a-real-model", allow_fallback=True
        )
        assert result.source == "heuristic"

    def test_fallback_can_be_disabled(self, ball_image):
        if not LearnedDepthEstimator.is_available():
            pytest.skip("torch/transformers not installed, so no learned model to fail")
        with pytest.raises(ImageToModelError):
            estimate_depth(
                ball_image[..., :3], model="definitely/not-a-real-model", allow_fallback=False
            )

    def test_known_aliases_resolve_to_model_ids(self):
        model_id, is_disparity = resolve_model("depth-anything")
        assert "Depth-Anything" in model_id
        assert is_disparity is True

    def test_unknown_names_pass_through_as_hub_ids(self):
        model_id, _ = resolve_model("some-org/some-model")
        assert model_id == "some-org/some-model"


class TestTexture:
    def test_inpaint_fills_outside_the_mask(self):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[6:10, 6:10] = [200, 100, 50]
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[6:10, 6:10] = 1.0
        filled = inpaint_background(rgb, mask, iterations=6)
        assert filled[5, 7].sum() > 0  # colour spread outwards
        assert np.allclose(filled[6:10, 6:10], rgb[6:10, 6:10])

    def test_atlas_is_square_and_two_tiles_tall(self, ball_image):
        mask, _ = segment(ball_image)
        atlas, _ = build_atlas(ball_image[..., :3], mask, texture_size=128)
        assert atlas.shape == (128, 128, 3)

    def test_back_tile_is_darker_than_the_front(self, ball_image):
        mask, _ = segment(ball_image)
        atlas, _ = build_atlas(ball_image[..., :3], mask, texture_size=128, backface_darkening=0.5)
        front, back = atlas[:64], atlas[64:]
        assert back.mean() < front.mean()

    def test_uvs_stay_inside_their_own_tile(self):
        layout = AtlasLayout(x0=0.0, y0=0.0, width=100.0, height=100.0)
        coords = np.array([[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]], dtype=np.float32)

        front = layout.uv(coords, np.zeros(3, dtype=bool))
        back = layout.uv(coords, np.ones(3, dtype=bool))

        assert front[:, 1].max() < 0.5
        assert back[:, 1].min() > 0.5
        assert front.min() >= 0.0 and back.max() <= 1.0

    def test_uvs_are_clamped_for_out_of_range_pixels(self):
        layout = AtlasLayout(x0=0.0, y0=0.0, width=10.0, height=10.0)
        uv = layout.uv(np.array([[-50.0, 500.0]], dtype=np.float32), np.zeros(1, dtype=bool))
        assert 0.0 <= uv[0, 0] <= 1.0 and 0.0 <= uv[0, 1] <= 1.0

    def test_vertex_colors_sample_the_image(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[5, 5] = [255, 0, 0]
        colors = sample_vertex_colors(rgb, np.array([[5.0, 5.0]], dtype=np.float32))
        assert colors[0, 0] > 200

    def test_back_faces_are_darkened(self):
        rgb = np.full((10, 10, 3), 200, dtype=np.uint8)
        coords = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        colors = sample_vertex_colors(
            rgb, coords, is_back=np.array([False, True]), backface_darkening=0.5
        )
        assert colors[1].max() < colors[0].max()
