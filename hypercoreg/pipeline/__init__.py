"""Internal pipeline package for the refactored HyperCoreg workflow."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "api_client",
    "auth",
    "baseline",
    "coreg_math",
    "migration",
    "orchestrator",
    "raster_io",
    "reporting",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"hypercoreg.pipeline.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
