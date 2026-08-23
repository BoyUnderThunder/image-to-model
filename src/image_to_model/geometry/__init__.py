"""Geometry construction and refinement."""

from __future__ import annotations

from .decimate import DecimationResult, decimate
from .lift import camera_distance, project_to_world, world_to_pixel
from .mesh_ops import (
    boundary_vertices,
    clean_mesh,
    compact_vertices,
    compute_vertex_normals,
    connected_components,
    keep_largest_components,
    remove_degenerate_faces,
    remove_duplicate_faces,
    remove_unreferenced_vertices,
    taubin_smooth,
    weld_vertices,
)
from .surface import BACK, FRONT, SurfaceResult, build_surface, choose_stride

__all__ = [
    "BACK",
    "FRONT",
    "DecimationResult",
    "SurfaceResult",
    "boundary_vertices",
    "build_surface",
    "camera_distance",
    "choose_stride",
    "clean_mesh",
    "compact_vertices",
    "compute_vertex_normals",
    "connected_components",
    "decimate",
    "keep_largest_components",
    "project_to_world",
    "remove_degenerate_faces",
    "remove_duplicate_faces",
    "remove_unreferenced_vertices",
    "taubin_smooth",
    "weld_vertices",
    "world_to_pixel",
]
