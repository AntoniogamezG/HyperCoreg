"""Coregistration math, scoring, and QA helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

__all__ = [
    "MIN_TIE_POINTS_FOR_POLYNOMIAL",
    "PROCESSING_NODATA",
    "SCENE_CLUSTER_HULL_RATIO_THRESHOLD",
    "SCENE_CLUSTER_SPREAD_THRESHOLD",
    "_S2_SRF_TABLE_CACHE",
    "_apply_spatial_stratification",
    "_assess_gcp_geometry_for_order2",
    "_build_consensus_group_ids",
    "_build_gcps_from_tiepoints",
    "_coerce_global_coreg_attempt_ladder",
    "_coerce_window_size",
    "_compute_quality_score",
    "_compute_srf_weights",
    "_compute_tiepoint_residuals",
    "_count_occupied_cells",
    "_decide_polynomial_order",
    "_derive_scene_rmse_metrics",
    "_estimate_translation_phasecorr",
    "_extract_finite_xy",
    "_extract_tiepoint_model_arrays",
    "_filter_frame_by_min_distance",
    "_fit_transform_model",
    "_load_sentinel2_srf_table",
    "_merge_tiepoints",
    "_minimum_gcps_for_polynomial_order",
    "_model_edge_stability_ok",
    "_parse_arosics_output",
    "_poly_design_matrix",
    "_predict_transform_model",
    "_quality_tier_from_score",
    "_resolve_band_indices_for_matching",
    "_resolve_fixed_band_pair",
    "_resolve_sensor_matcher_profile",
    "_run_postwarp_phasecorr_qa",
    "_safe_optional_float",
    "_select_transform_model_cv",
    "_summarize_scene_tiepoint_quality",
    "_tiepoint_model_residuals",
    "_trapezoid",
    "_validate_coreg_hybrid",
]

# Defined only in this module (no runtime.py equivalent); everything else
# in __all__ resolves to hypercoreg.pipeline.runtime on attribute access.
_LOCAL_NAMES = frozenset({
    "_filter_frame_by_min_distance",
})


def _runtime():
    # Imported lazily: runtime pulls in AROSICS/geoarray and is edited independently,
    # so names are looked up on every access instead of being copied at import time.
    return import_module("hypercoreg.pipeline.runtime")


def _filter_frame_by_min_distance(frame: Any, min_distance: float, seed_frame: Any = None):
    """Greedily keep rows that stay at least ``min_distance`` apart in image space."""
    min_dist = float(max(0.0, min_distance))
    if frame is None or len(frame) == 0:
        return frame
    if min_dist <= 0.0:
        return frame.copy() if hasattr(frame, "copy") else frame

    active_x: list[float] = []
    active_y: list[float] = []
    min_dist_sq = min_dist * min_dist

    def _seed_active_points(seed_df: Any) -> None:
        if seed_df is None or len(seed_df) == 0 or not hasattr(seed_df, "iterrows"):
            return
        for _idx, row in seed_df.iterrows():
            try:
                cand_x = float(row["X_IM"])
                cand_y = float(row["Y_IM"])
            except (TypeError, ValueError, KeyError):
                continue
            if np.isfinite(cand_x) and np.isfinite(cand_y):
                active_x.append(cand_x)
                active_y.append(cand_y)

    _seed_active_points(seed_frame)

    kept_indices: list[Any] = []
    for idx, row in frame.iterrows():
        try:
            cand_x = float(row["X_IM"])
            cand_y = float(row["Y_IM"])
        except (TypeError, ValueError, KeyError):
            continue
        if not (np.isfinite(cand_x) and np.isfinite(cand_y)):
            continue

        if active_x:
            dx = np.asarray(active_x, dtype=float) - cand_x
            dy = np.asarray(active_y, dtype=float) - cand_y
            dists_sq = dx * dx + dy * dy
            if float(np.min(dists_sq)) <= min_dist_sq:
                continue

        kept_indices.append(idx)
        active_x.append(cand_x)
        active_y.append(cand_y)

    if hasattr(frame, "loc"):
        return frame.loc[kept_indices]
    return frame


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_runtime(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
