"""A small software renderer for previewing results.

Producing a 3D model you cannot look at is not much use, and pulling in a GPU
renderer for a sanity check would be a heavy dependency. This is a plain
z-buffered triangle rasteriser in NumPy: enough to confirm the shape is right,
to render a turntable contact sheet, and to give the CLI a ``--preview`` flag.
"""

from __future__ import annotations

import numpy as np

from .imaging import bilinear_sample
from .types import Mesh

__all__ = ["render_mesh", "render_turntable"]


def _rotation(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)

    cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
    yaw = np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]], dtype=np.float32)

    cos_e, sin_e = np.cos(elevation), np.sin(elevation)
    pitch = np.array([[1.0, 0.0, 0.0], [0.0, cos_e, -sin_e], [0.0, sin_e, cos_e]], dtype=np.float32)

    return (pitch @ yaw).astype(np.float32)


def render_mesh(
    mesh: Mesh,
    size: int = 512,
    azimuth: float = 25.0,
    elevation: float = 15.0,
    texture: np.ndarray | None = None,
    background: tuple[int, int, int] = (24, 26, 32),
    light_direction: tuple[float, float, float] = (-0.4, 0.6, 0.8),
) -> np.ndarray:
    """Render ``mesh`` to an RGB image using an orthographic camera.

    Colour comes from the texture when the mesh has UVs and a texture is given,
    otherwise from vertex colours, otherwise a neutral grey.
    """
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[:] = np.asarray(background, dtype=np.float32) / 255.0
    if mesh.is_empty:
        return (canvas * 255).astype(np.uint8)

    rotation = _rotation(azimuth, elevation)
    vertices = mesh.vertices @ rotation.T

    # Fit the model to the frame with a small margin.
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) * 0.5
    span = float(np.max(high - low))
    if span < 1e-9:
        return (canvas * 255).astype(np.uint8)
    scale = (size * 0.86) / span

    screen_x = (vertices[:, 0] - center[0]) * scale + size * 0.5
    screen_y = -(vertices[:, 1] - center[1]) * scale + size * 0.5
    depth = vertices[:, 2]

    normals = mesh.vertex_normals
    if normals is None:
        from .geometry.mesh_ops import compute_vertex_normals

        normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    normals = normals @ rotation.T

    light = np.asarray(light_direction, dtype=np.float32)
    light /= max(float(np.linalg.norm(light)), 1e-8)

    if texture is not None and mesh.uvs is not None:
        texture_f = texture.astype(np.float32) / 255.0
        height, width = texture_f.shape[:2]
        base_colors = bilinear_sample(
            texture_f, mesh.uvs[:, 0] * (width - 1), mesh.uvs[:, 1] * (height - 1)
        )
    elif mesh.vertex_colors is not None:
        base_colors = mesh.vertex_colors.astype(np.float32) / 255.0
    else:
        base_colors = np.full((mesh.n_vertices, 3), 0.72, dtype=np.float32)

    # Lambert with a lifted ambient term, so faces turned away stay readable.
    lambert = np.clip(normals @ light, 0.0, 1.0)
    shading = (0.25 + 0.75 * lambert)[:, None]
    shaded = np.clip(base_colors * shading, 0.0, 1.0)

    zbuffer = np.full((size, size), np.inf, dtype=np.float32)

    faces = mesh.faces
    tri_x, tri_y = screen_x[faces], screen_y[faces]
    tri_z, tri_c = depth[faces], shaded[faces]

    # Backface culling in screen space: keep the winding that faces the camera.
    area = (tri_x[:, 1] - tri_x[:, 0]) * (tri_y[:, 2] - tri_y[:, 0]) - (
        tri_x[:, 2] - tri_x[:, 0]
    ) * (tri_y[:, 1] - tri_y[:, 0])
    visible = np.flatnonzero(np.abs(area) > 1e-9)

    for index in visible:
        x0, x1, x2 = tri_x[index]
        y0, y1, y2 = tri_y[index]

        min_x = max(int(np.floor(min(x0, x1, x2))), 0)
        max_x = min(int(np.ceil(max(x0, x1, x2))), size - 1)
        min_y = max(int(np.floor(min(y0, y1, y2))), 0)
        max_y = min(int(np.ceil(max(y0, y1, y2))), size - 1)
        if min_x > max_x or min_y > max_y:
            continue

        xs = np.arange(min_x, max_x + 1, dtype=np.float32)
        ys = np.arange(min_y, max_y + 1, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        inverse_area = 1.0 / area[index]
        w0 = ((x1 - grid_x) * (y2 - grid_y) - (x2 - grid_x) * (y1 - grid_y)) * inverse_area
        w1 = ((x2 - grid_x) * (y0 - grid_y) - (x0 - grid_x) * (y2 - grid_y)) * inverse_area
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * tri_z[index, 0] + w1 * tri_z[index, 1] + w2 * tri_z[index, 2]
        window = zbuffer[min_y : max_y + 1, min_x : max_x + 1]
        # Larger Z is nearer the camera in this convention.
        write = inside & ((window == np.inf) | (z > -window))
        if not write.any():
            continue

        color = (
            w0[..., None] * tri_c[index, 0]
            + w1[..., None] * tri_c[index, 1]
            + w2[..., None] * tri_c[index, 2]
        )
        window[write] = -z[write]
        canvas[min_y : max_y + 1, min_x : max_x + 1][write] = color[write]

    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def render_turntable(
    mesh: Mesh,
    size: int = 320,
    views: int = 4,
    elevation: float = 15.0,
    texture: np.ndarray | None = None,
) -> np.ndarray:
    """Render several angles side by side as a single contact-sheet image."""
    views = max(1, int(views))
    frames = [
        render_mesh(
            mesh,
            size=size,
            azimuth=360.0 * i / views,
            elevation=elevation,
            texture=texture,
        )
        for i in range(views)
    ]
    return np.concatenate(frames, axis=1)
