"""Axis and unit conversion at export time.

Internally the pipeline always produces geometry in the glTF convention:
right-handed, **Y up**, with +Z pointing towards the camera (so the front of
the object faces the viewer). Roblox and most game engines share that
convention, so nothing has to move for them. CAD and 3D-printing formats are
conventionally Z up, which is the one conversion this module performs.
"""

from __future__ import annotations

import numpy as np

from ..types import Mesh

__all__ = ["convert_up_axis", "scale_to_units", "prepare_for_target"]


# Y-up -> Z-up: (x, y, z) becomes (x, -z, y). This is a pure rotation about X,
# so winding order and therefore surface orientation are preserved.
_Y_TO_Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def convert_up_axis(mesh: Mesh, up_axis: str) -> Mesh:
    """Rotate a Y-up mesh into the requested convention."""
    axis = (up_axis or "y").lower()
    if axis == "y":
        return mesh
    if axis != "z":
        raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")

    out = mesh.copy()
    out.vertices = (out.vertices @ _Y_TO_Z.T).astype(np.float32)
    if out.vertex_normals is not None:
        out.vertex_normals = (out.vertex_normals @ _Y_TO_Z.T).astype(np.float32)
    return out


def scale_to_units(mesh: Mesh, size_units: float) -> Mesh:
    """Centre the mesh and scale its longest axis to ``size_units``.

    ``size_units <= 0`` leaves the mesh in raw camera-space units.
    """
    if size_units <= 0:
        return mesh
    return mesh.normalized(target_size=float(size_units))


def prepare_for_target(mesh: Mesh, up_axis: str = "y", size_units: float = 0.0) -> Mesh:
    """Apply unit scaling then axis conversion, in that order.

    Scaling first keeps the "longest axis" measurement independent of which
    axis ends up pointing up.
    """
    return convert_up_axis(scale_to_units(mesh, size_units), up_axis)
