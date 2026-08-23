"""Backend registry.

Register a custom backend to swap in a different reconstruction strategy — a
feed-forward image-to-3D model, say — while keeping the rest of the pipeline
(segmentation, unit scaling, texture baking, export, target validation)::

    from image_to_model.backends import ReconstructionBackend, register_backend

    class MyBackend(ReconstructionBackend):
        name = "my-backend"

        def reconstruct(self, subject, config):
            ...

    register_backend(MyBackend)
"""

from __future__ import annotations

from ..errors import BackendNotFound
from .base import BackendOutput, ReconstructionBackend
from .depth_backend import DepthBackend

__all__ = [
    "BackendOutput",
    "ReconstructionBackend",
    "DepthBackend",
    "register_backend",
    "get_backend",
    "available_backends",
    "all_backend_names",
]

_REGISTRY: dict[str, type[ReconstructionBackend]] = {}


def register_backend(backend: type[ReconstructionBackend], name: str | None = None) -> None:
    """Add a backend class to the registry."""
    key = (name or backend.name).strip().lower()
    if not key:
        raise ValueError("A backend needs a non-empty name")
    _REGISTRY[key] = backend


def all_backend_names() -> list[str]:
    """Every registered backend, whether or not its dependencies are installed."""
    return sorted(_REGISTRY)


def available_backends() -> list[str]:
    """Backends that can actually run in this environment."""
    return sorted(name for name, cls in _REGISTRY.items() if cls.is_available())


def get_backend(name: str) -> ReconstructionBackend:
    """Instantiate a backend by name."""
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise BackendNotFound(name, all_backend_names())
    backend_cls = _REGISTRY[key]
    if not backend_cls.is_available():
        raise BackendNotFound(name, available_backends())
    return backend_cls()


register_backend(DepthBackend)
