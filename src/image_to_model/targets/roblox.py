"""Roblox target profile.

Numbers here follow Roblox's published mesh specifications:

* The 3D Importer accepts ``.obj``, ``.fbx`` and ``.gltf``/``.glb``.
* A single imported mesh is capped at 21,000 triangles. Batch imports and
  avatar/humanoid meshes are held to the stricter 10,000, so that is the
  budget the pipeline aims for by default.
* Uploaded textures are limited to 1024x1024.
* Roblox is Y-up and measures in studs, where one stud is about 0.28 m.
* MeshPart geometry carries UVs and a texture; per-vertex colour is dropped on
  import, which is why texture baking is not optional for this target.
"""

from __future__ import annotations

from .base import TargetProfile

__all__ = ["ROBLOX", "ROBLOX_AVATAR", "import_instructions", "luau_snippet"]


ROBLOX = TargetProfile(
    name="Roblox",
    description="Roblox Studio MeshPart, imported through the 3D Importer.",
    triangle_budget=10_000,
    triangle_hard_limit=21_000,
    max_texture_size=1024,
    supports_vertex_colors=False,
    up_axis="y",
    unit_name="stud",
    meters_per_unit=0.28,
    default_size_units=4.0,
    preferred_formats=("obj", "glb"),
)


ROBLOX_AVATAR = TargetProfile(
    name="Roblox avatar",
    description="Roblox avatar or accessory mesh, held to the stricter batch-import cap.",
    triangle_budget=4_000,
    triangle_hard_limit=10_000,
    max_texture_size=1024,
    supports_vertex_colors=False,
    up_axis="y",
    unit_name="stud",
    meters_per_unit=0.28,
    default_size_units=1.5,
    preferred_formats=("obj", "glb"),
)


def import_instructions(mesh_filename: str, texture_filename: str | None = None) -> str:
    """Step-by-step guide written alongside the exported files."""
    texture_step = (
        f"3. The importer picks up `{texture_filename}` automatically when it sits next to "
        f"`{mesh_filename}`. If it does not, upload the PNG separately and set the MeshPart's "
        "`TextureID` to the resulting asset ID."
        if texture_filename
        else "3. This export has no texture map; set the MeshPart's `Color` property instead."
    )
    return f"""# Importing into Roblox Studio

1. Open your place in Roblox Studio and choose **Avatar -> 3D Importer**.
2. Select `{mesh_filename}`.
{texture_step}
4. Check the preview panel: the triangle count must stay under 21,000 for a single
   import, and under 10,000 if you are batch importing or building an avatar item.
5. Click **Import**. The mesh lands in the Workspace as a `MeshPart`.

## Scale

The model is exported in studs with Y up, so it arrives at roughly the right size
and orientation. Adjust the `Size` property to taste; changing `Size` on a MeshPart
rescales the mesh rather than cropping it.

## Publishing

To use the mesh in a live game, right-click the MeshPart and choose
**Save to Roblox** so it gets an asset ID. Meshes and textures are moderated, so
allow time for approval before the asset renders for other players.
"""


def luau_snippet(mesh_asset_id: str = "YOUR_MESH_ID", texture_asset_id: str = "") -> str:
    """A Luau template for spawning the imported mesh from a script."""
    texture_line = (
        f'part.TextureID = "rbxassetid://{texture_asset_id}"'
        if texture_asset_id
        else 'part.TextureID = "rbxassetid://YOUR_TEXTURE_ID" -- fill in after uploading'
    )
    return f"""--!strict
-- Spawns the generated mesh. Replace the asset IDs with the ones Roblox
-- assigns after you upload the mesh and its texture.

local part = Instance.new("MeshPart")
part.MeshId = "rbxassetid://{mesh_asset_id}"
{texture_line}
part.Size = Vector3.new(4, 4, 4)
part.Anchored = true
part.CanCollide = true
part.Position = Vector3.new(0, 5, 0)
part.Parent = workspace
"""
