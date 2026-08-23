"""Configuration for a reconstruction run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from .targets import TargetProfile, get_target

__all__ = ["ReconstructionConfig", "PRESETS", "preset"]


@dataclass
class ReconstructionConfig:
    """Every knob the pipeline exposes, with defaults tuned for a single object photo.

    Several fields default to ``0`` or ``"auto"``, meaning "take this from the
    target platform profile". Call :meth:`resolved` to get a config with those
    filled in from :attr:`target`.
    """

    # -- destination ------------------------------------------------------
    target: str = "roblox"
    """Platform the model is built for: ``roblox``, ``roblox-avatar``, ``game``,
    ``print`` or ``generic``. Drives polygon budget, texture size, units and axes."""

    backend: str = "depth"
    """Reconstruction backend name. ``depth`` is the built-in pipeline."""

    # -- input handling ---------------------------------------------------
    working_resolution: int = 512
    """Longest edge the image is resized to before depth prediction. Cost grows
    roughly quadratically with this, so it is the main speed/detail dial."""

    segmentation: str = "auto"
    """``auto``, ``alpha``, ``rembg``, ``border`` or ``none``."""

    mask_threshold: float = 0.5
    """Alpha above which a pixel counts as subject when building geometry."""

    mask_erode: int = 1
    """Pixels trimmed from the mask edge to avoid meshing background fringe."""

    # -- depth ------------------------------------------------------------
    depth_model: str = "auto"
    """``auto`` picks the best available estimator; also ``depth-anything``,
    ``dpt``, ``heuristic``, or any HuggingFace depth model id."""

    depth_invert: bool = False
    """Set when the estimator returns disparity where larger means nearer."""

    # -- geometry ---------------------------------------------------------
    fov_degrees: float = 55.0
    """Assumed horizontal field of view used to lift pixels into camera space."""

    relief_scale: float = 0.35
    """Depth range of the front surface as a fraction of the subject's width.
    Higher values make the object bulge further towards the camera."""

    thickness: float = 0.55
    """Back-surface depth as a fraction of the front relief. 0 leaves the model
    open (a 2.5D relief); values near 1 produce a roughly symmetric solid."""

    close_back: bool = True
    """Generate and stitch a back surface to make the mesh watertight."""

    depth_smoothing: float = 1.2
    """Filter radius (in pixels) applied to depth before meshing."""

    depth_filter: str = "bilateral"
    """``bilateral`` smooths depth while keeping the edges the photo shows,
    ``gaussian`` blurs everything equally, ``none`` skips filtering."""

    depth_filter_range: float = 0.08
    """How different the guide image must look before the bilateral filter
    stops smoothing across it. Lower keeps more edges, at the cost of noise."""

    rim_profile: str = "fillet"
    """Silhouette cross-section: ``fillet`` rounds the edge like a real object,
    ``smoothstep`` leaves it flatter with a harder corner, ``linear`` chamfers."""

    # -- mesh refinement --------------------------------------------------
    smooth_iterations: int = 12
    """Taubin smoothing passes. Removes stair-stepping without shrinking."""

    smooth_lambda: float = 0.5
    smooth_mu: float = -0.53

    target_faces: int = 0
    """Decimation budget. 0 takes the target profile's triangle budget;
    a negative value disables decimation entirely."""

    min_component_ratio: float = 0.05
    """Connected components smaller than this fraction of the largest one are
    discarded as reconstruction noise."""

    # -- texturing --------------------------------------------------------
    texture: bool = True
    """Sample colour from the source photograph."""

    bake_texture: str = "auto"
    """``auto`` bakes a UV texture image when the target ignores vertex colour,
    ``always`` forces baking, ``never`` keeps vertex colours only."""

    texture_size: int = 0
    """Baked texture resolution in pixels. 0 takes the target's maximum."""

    backface_darkening: float = 0.55
    """Multiplier applied to back-surface colours, since the reverse of the
    object is never observed and a flat mirror of the front reads as wrong."""

    # -- output -----------------------------------------------------------
    size_units: float = 0.0
    """Longest axis of the exported model, in the target's units (studs for
    Roblox). 0 takes the target default; a negative value keeps raw units."""

    up_axis: str = "auto"
    """``auto`` follows the target profile; otherwise ``y`` or ``z``."""

    # -- misc -------------------------------------------------------------
    seed: int = 0
    device: str = "auto"
    """``auto``, ``cpu`` or ``cuda`` for learned components."""

    def __post_init__(self) -> None:
        self.validate()

    # -- validation and resolution ---------------------------------------

    def validate(self) -> None:
        get_target(self.target)  # raises on an unknown target name
        if self.working_resolution < 32:
            raise ValueError("working_resolution must be at least 32")
        if self.working_resolution > 4096:
            raise ValueError("working_resolution above 4096 is not supported")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be within [0, 1]")
        if self.mask_erode < 0:
            raise ValueError("mask_erode must be non-negative")
        if not 1.0 <= self.fov_degrees <= 179.0:
            raise ValueError("fov_degrees must be within [1, 179]")
        if self.relief_scale <= 0.0:
            raise ValueError("relief_scale must be positive")
        if self.thickness < 0.0:
            raise ValueError("thickness must be non-negative")
        if self.depth_smoothing < 0.0:
            raise ValueError("depth_smoothing must be non-negative")
        if self.depth_filter not in {"bilateral", "gaussian", "none"}:
            raise ValueError("depth_filter must be 'bilateral', 'gaussian' or 'none'")
        if self.depth_filter_range <= 0.0:
            raise ValueError("depth_filter_range must be positive")
        if self.rim_profile not in {"fillet", "smoothstep", "linear"}:
            raise ValueError("rim_profile must be 'fillet', 'smoothstep' or 'linear'")
        if self.smooth_iterations < 0:
            raise ValueError("smooth_iterations must be non-negative")
        if not 0.0 <= self.min_component_ratio < 1.0:
            raise ValueError("min_component_ratio must be within [0, 1)")
        if not 0.0 <= self.backface_darkening <= 1.0:
            raise ValueError("backface_darkening must be within [0, 1]")
        if self.texture_size < 0:
            raise ValueError("texture_size must be non-negative")
        if self.bake_texture not in {"auto", "always", "never"}:
            raise ValueError("bake_texture must be 'auto', 'always' or 'never'")
        if self.up_axis not in {"auto", "y", "z"}:
            raise ValueError("up_axis must be 'auto', 'y' or 'z'")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu' or 'cuda'")

    @property
    def profile(self) -> TargetProfile:
        """The resolved target platform profile."""
        return get_target(self.target)

    def resolve_target_faces(self) -> int:
        """Face budget to decimate to; 0 means no decimation."""
        if self.target_faces < 0:
            return 0
        if self.target_faces > 0:
            return self.target_faces
        return self.profile.triangle_budget

    def resolve_texture_size(self) -> int:
        if self.texture_size > 0:
            return min(self.texture_size, self.profile.max_texture_size)
        return self.profile.max_texture_size

    def resolve_size_units(self) -> float:
        """Longest axis of the export; 0 means leave raw camera units alone."""
        if self.size_units < 0:
            return 0.0
        if self.size_units > 0:
            return self.size_units
        return self.profile.default_size_units

    def resolve_up_axis(self) -> str:
        return self.profile.up_axis if self.up_axis == "auto" else self.up_axis

    def should_bake_texture(self) -> bool:
        if not self.texture:
            return False
        if self.bake_texture == "always":
            return True
        if self.bake_texture == "never":
            return False
        return not self.profile.supports_vertex_colors

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ReconstructionConfig:
        """Build a config from a mapping, ignoring keys that are not fields.

        Lets the CLI hand over its whole argument namespace without filtering.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known and v is not None})

    def merged(self, **overrides: Any) -> ReconstructionConfig:
        """Return a copy with ``overrides`` applied, dropping ``None`` values."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "working_resolution": 320,
        "smooth_iterations": 6,
        "depth_smoothing": 1.5,
    },
    "balanced": {},
    "detailed": {
        "working_resolution": 768,
        "smooth_iterations": 16,
        "depth_smoothing": 0.9,
    },
    "relief": {
        # A flat-backed bas-relief, useful for signage and wall props.
        "close_back": True,
        "thickness": 0.05,
        "relief_scale": 0.25,
    },
    "roblox-prop": {
        "target": "roblox",
        "working_resolution": 512,
        "smooth_iterations": 14,
    },
    "roblox-detail": {
        # A single prop imported on its own may use the full 21,000-triangle
        # allowance, rather than the conservative batch-safe budget.
        "target": "roblox",
        "target_faces": 21_000,
        "working_resolution": 768,
        "smooth_iterations": 16,
        "depth_smoothing": 0.8,
        "texture_size": 1024,
    },
    "roblox-accessory": {
        "target": "roblox-avatar",
        "working_resolution": 512,
        "smooth_iterations": 16,
    },
}


def preset(name: str, **overrides: Any) -> ReconstructionConfig:
    """Build a config from a named preset with optional overrides."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Choose from: {', '.join(sorted(PRESETS))}")
    return ReconstructionConfig(**PRESETS[name]).merged(**overrides)
