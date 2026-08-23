"""Target platform profiles and lookup."""

from __future__ import annotations

from .base import TargetProfile, ValidationIssue, ValidationReport
from .roblox import ROBLOX, ROBLOX_AVATAR, import_instructions, luau_snippet

__all__ = [
    "TargetProfile",
    "ValidationIssue",
    "ValidationReport",
    "ROBLOX",
    "ROBLOX_AVATAR",
    "GENERIC",
    "GAME_ENGINE",
    "PRINT_3D",
    "TARGETS",
    "get_target",
    "available_targets",
    "import_instructions",
    "luau_snippet",
]


GENERIC = TargetProfile(
    name="generic",
    description="No platform constraints; keeps full detail and vertex colour.",
    triangle_budget=200_000,
    triangle_hard_limit=0,
    max_texture_size=4096,
    supports_vertex_colors=True,
    up_axis="y",
    unit_name="meter",
    meters_per_unit=1.0,
    default_size_units=1.0,
    preferred_formats=("glb", "obj", "ply"),
)

GAME_ENGINE = TargetProfile(
    name="game engine",
    description="Unity or Unreal style budget: real-time friendly, textured.",
    triangle_budget=50_000,
    triangle_hard_limit=0,
    max_texture_size=2048,
    supports_vertex_colors=False,
    up_axis="y",
    unit_name="meter",
    meters_per_unit=1.0,
    default_size_units=1.0,
    preferred_formats=("glb", "obj"),
)

PRINT_3D = TargetProfile(
    name="3D print",
    description="Watertight solid in millimetres, no texture.",
    triangle_budget=300_000,
    triangle_hard_limit=0,
    max_texture_size=4096,
    supports_vertex_colors=True,
    up_axis="z",
    unit_name="millimeter",
    meters_per_unit=0.001,
    default_size_units=100.0,
    preferred_formats=("stl", "ply", "obj"),
)


TARGETS: dict[str, TargetProfile] = {
    "roblox": ROBLOX,
    "roblox-avatar": ROBLOX_AVATAR,
    "generic": GENERIC,
    "game": GAME_ENGINE,
    "print": PRINT_3D,
}


def available_targets() -> list[str]:
    return sorted(TARGETS)


def get_target(name: str) -> TargetProfile:
    """Look up a profile by name, case-insensitively."""
    key = (name or "").strip().lower()
    if key not in TARGETS:
        raise ValueError(
            f"Unknown target {name!r}. Choose from: {', '.join(available_targets())}"
        )
    return TARGETS[key]
