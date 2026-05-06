"""Detector branch and tie-point orchestration helpers."""

from __future__ import annotations

from hypercoreg.pipeline import runtime as _runtime

__all__ = [
    "_build_detector_branch_plan",
    "_collect_multiband_tiepoints",
    "_harmonize_detector_branch_grids",
    "_process_detector_branch_candidate",
    "_recombine_detector_branches_windowed",
    "_write_hs_narrowband_windowed",
    "_write_branch_raster_windowed",
]

for _name in __all__:
    globals()[_name] = getattr(_runtime, _name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
