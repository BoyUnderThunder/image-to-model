"""The built-in depth-to-mesh reconstruction backend.

Stages, in order:

1. predict a relative depth map for the subject;
2. smooth it, since depth models are noisy at the pixel level and every bit of
   that noise would become surface roughness;
3. mesh the depth as a front surface, mirror it into a back surface and stitch
   the two into a closed solid;
4. clean, smooth and decimate the result down to the target's polygon budget;
5. bake a texture and assign UVs by projecting vertices back into the photo.

Texturing happens before unit scaling, because the projection that generates
UVs is only valid while the mesh is still in the camera space it was built in.
"""

from __future__ import annotations

import numpy as np

from ..config import ReconstructionConfig
from ..depth import estimate_depth
from ..geometry.decimate import decimate
from ..geometry.lift import world_to_pixel
from ..geometry.mesh_ops import (
    compact_vertices,
    keep_largest_components,
    remove_degenerate_faces,
    remove_duplicate_faces,
    taubin_smooth,
)
from ..geometry.surface import BACK, build_surface, choose_stride
from ..imaging import gaussian_blur, joint_bilateral_filter
from ..logging import get_logger, stage
from ..morphology import binary_erode
from ..texture import build_atlas, compute_uvs, sample_vertex_colors
from ..types import Subject
from .base import BackendOutput, ReconstructionBackend

__all__ = ["DepthBackend"]

log = get_logger("backends.depth")


class DepthBackend(ReconstructionBackend):
    """Monocular depth estimation followed by height-field meshing."""

    name = "depth"
    description = "Predict depth, mesh it as a closed solid, bake a texture."

    def reconstruct(self, subject: Subject, config: ReconstructionConfig) -> BackendOutput:
        image = subject.image
        mask = subject.mask

        with stage("Estimating depth", log):
            depth_map = estimate_depth(
                image, mask, model=config.depth_model, device=config.device
            )
            if config.depth_invert:
                depth_map = depth_map.inverted()
            depth_map = depth_map.normalized()

        depth = depth_map.depth
        if config.depth_smoothing > 0 and config.depth_filter != "none":
            if config.depth_filter == "bilateral":
                # Guided by the photograph's luminance, so smoothing stops at
                # edges the image shows rather than rounding them away.
                guide = (image.astype(np.float32) / 255.0) @ np.array(
                    [0.2126, 0.7152, 0.0722], dtype=np.float32
                )
                depth = joint_bilateral_filter(
                    depth, guide, config.depth_smoothing, config.depth_filter_range
                )
            else:
                depth = gaussian_blur(depth, config.depth_smoothing)

        # Trimming the mask edge keeps the soft alpha fringe, which blends
        # subject and background colours, out of the geometry.
        binary = mask > config.mask_threshold
        if config.mask_erode > 0:
            eroded = binary_erode(binary, config.mask_erode)
            if eroded.any():
                binary = eroded
            else:
                log.warning("Mask erosion removed the whole subject; keeping the original mask")
        work_mask = binary.astype(np.float32)

        target_faces = config.resolve_target_faces()
        stride = choose_stride(binary, target_faces)

        with stage("Building surface", log):
            surface = build_surface(
                depth,
                work_mask,
                relief_scale=config.relief_scale,
                relief_mode=config.relief_mode,
                thickness=config.thickness,
                close_back=config.close_back,
                fov_degrees=config.fov_degrees,
                mask_threshold=0.5,
                stride=stride,
                rim_profile=config.rim_profile,
                min_thickness=config.min_thickness,
                orthographic=config.projection == "orthographic",
            )

        mesh = surface.mesh
        side = surface.side

        mesh = remove_duplicate_faces(remove_degenerate_faces(mesh))
        if config.min_component_ratio > 0:
            mesh = keep_largest_components(mesh, config.min_component_ratio)
        mesh, kept = compact_vertices(mesh)
        side = side[kept]

        if config.smooth_iterations > 0:
            with stage("Smoothing", log):
                mesh = taubin_smooth(
                    mesh,
                    iterations=config.smooth_iterations,
                    lam=config.smooth_lambda,
                    mu=config.smooth_mu,
                )

        if target_faces > 0 and mesh.n_faces > target_faces:
            with stage(f"Decimating to {target_faces:,} faces", log):
                result = decimate(mesh, target_faces)
                mesh = result.mesh
                side = side[result.vertex_map]

        is_back = side == BACK
        texture = None

        # Still in camera space here, so back-projection is exact.
        pixel_coords = world_to_pixel(
            mesh.vertices, surface.intrinsics, surface.distance, surface.orthographic
        )

        if config.should_bake_texture():
            with stage("Baking texture", log):
                texture, layout = build_atlas(
                    image,
                    mask,
                    texture_size=config.resolve_texture_size(),
                    backface_darkening=config.backface_darkening,
                )
                mesh.uvs = compute_uvs(
                    mesh.vertices,
                    is_back,
                    surface.intrinsics,
                    surface.distance,
                    layout,
                    orthographic=surface.orthographic,
                )

        if config.texture:
            mesh.vertex_colors = sample_vertex_colors(
                image, pixel_coords, is_back, config.backface_darkening
            )

        return BackendOutput(
            mesh=mesh,
            texture=texture,
            depth=depth_map,
            metadata={
                "backend": self.name,
                "depth_source": depth_map.source,
                "grid_stride": stride,
                "target_faces": target_faces,
                "front_vertices": int((~is_back).sum()),
                "back_vertices": int(is_back.sum()),
                "camera_distance": surface.distance,
            },
        )
