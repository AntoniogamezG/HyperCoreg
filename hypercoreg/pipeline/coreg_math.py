"""Coregistration math, scoring, and QA helpers."""

from __future__ import annotations

import os
import re
import csv
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from hypercoreg.config import (
    DEFAULT_CONFIG,
    LOCAL_GRID_RES_M,
    MULTIBAND_S2_WAVELENGTHS,
    PRISMA_FIXED_BAND_PAIRS,
    S2_APPROX_FWHM_NM,
)

SCENE_CLUSTER_SPREAD_THRESHOLD = 0.35
SCENE_CLUSTER_HULL_RATIO_THRESHOLD = 0.20
PROCESSING_NODATA = -9999.0
MIN_TIE_POINTS_FOR_POLYNOMIAL = 10


def _parse_arosics_output(stdout_text):
    result = {
        "reliability": None,
        "ssim_before": None,
        "ssim_after": None,
        "ssim_delta": None,
        "x_shift_m": None,
        "y_shift_m": None,
        "shift_magnitude": None,
        "parsed_success": False,
    }
    rel_match = re.search(r"Estimated reliability[^0-9\n\r]*?([0-9]+\.?[0-9]*)%?", stdout_text)
    if rel_match:
        result["reliability"] = float(rel_match.group(1))
        result["parsed_success"] = True
    ssim_match = re.search(r"SSIM.*?([0-9]+\.[0-9]+)\s*=>\s*([0-9]+\.[0-9]+)", stdout_text)
    if ssim_match:
        result["ssim_before"] = float(ssim_match.group(1))
        result["ssim_after"] = float(ssim_match.group(2))
        result["ssim_delta"] = result["ssim_after"] - result["ssim_before"]
    shift_match = re.search(r"Calculated map shifts \(X,Y\):\s*(-?[0-9]+\.?[0-9]*)/(-?[0-9]+\.?[0-9]*)", stdout_text)
    if shift_match:
        x_shift, y_shift = float(shift_match.group(1)), float(shift_match.group(2))
        result["x_shift_m"], result["y_shift_m"] = x_shift, y_shift
        result["shift_magnitude"] = np.sqrt(x_shift**2 + y_shift**2)
        result["parsed_success"] = True
    return result


def _validate_coreg_hybrid(CRG, stdout_text, sensor_type="PRISMA", max_displacement=350.0):
    result = {
        "is_valid": False,
        "confidence": 0.0,
        "shift_m": 0.0,
        "message": "",
        "reliability_parsed": None,
        "ssim_before": None,
        "ssim_after": None,
        "ssim_delta": None,
    }
    try:
        coreg_info = getattr(CRG, "coreg_info", {})
        if not coreg_info.get("success", False):
            result["message"] = "AROSICS reported success=False"
            return result
        parsed = _parse_arosics_output(stdout_text)
        reliability = parsed.get("reliability")
        shift_magnitude = parsed.get("shift_magnitude")
        if shift_magnitude is None:
            x_shift = coreg_info.get("x_shift_m")
            y_shift = coreg_info.get("y_shift_m")
            if x_shift is not None and y_shift is not None:
                shift_magnitude = np.sqrt(float(x_shift) ** 2 + float(y_shift) ** 2)
        shift_magnitude = shift_magnitude if shift_magnitude is not None else 0.0
        result["shift_m"] = shift_magnitude
        result["ssim_before"] = parsed.get("ssim_before")
        result["ssim_after"] = parsed.get("ssim_after")
        result["ssim_delta"] = parsed.get("ssim_delta")
        if shift_magnitude > max_displacement:
            result["message"] = f"Shift {shift_magnitude:.1f}m exceeds max {max_displacement:.0f}m"
            return result
        if reliability is not None:
            result["reliability_parsed"] = reliability
            result["confidence"] = reliability / 100.0
            if reliability >= 40.0:
                result["is_valid"] = True
                result["message"] = f"Reliability {reliability:.1f}% >= 40% (shift: {shift_magnitude:.1f}m)"
            else:
                result["message"] = f"Reliability {reliability:.1f}% < 40% (shift: {shift_magnitude:.1f}m)"
        else:
            result["message"] = f"No reliability available (shift: {shift_magnitude:.1f}m)"
        return result
    except Exception as exc:
        result["message"] = f"Validation error: {exc}"
        return result


def _compute_tiepoint_residuals(tie_points_df: Any, pixel_size_m: float = 30.0) -> Dict[str, Any]:
    result = {
        "n_tiepoints_used": 0,
        "residual_mean_m": None,
        "residual_median_m": None,
        "residual_rmse_m": None,
        "residual_p90_m": None,
        "residual_source": "unavailable",
    }
    if tie_points_df is None:
        return result
    try:
        df = tie_points_df.copy()
    except Exception:
        return result
    if hasattr(df, "empty") and df.empty:
        return result

    for col in ("L1_OUTLIER", "L2_OUTLIER", "L3_OUTLIER"):
        if col in df.columns:
            try:
                outlier_mask = df[col].astype("boolean").fillna(False).to_numpy(dtype=bool)
                df = df[~outlier_mask]
            except Exception:
                pass
    if hasattr(df, "empty") and df.empty:
        return result

    result["n_tiepoints_used"] = len(df)

    if "ABS_SHIFT_M" in df.columns:
        try:
            residuals = df["ABS_SHIFT_M"].dropna().astype(float).values
            if len(residuals) > 0:
                result["residual_source"] = "ABS_SHIFT_M"
                result["residual_mean_m"] = float(np.mean(residuals))
                result["residual_median_m"] = float(np.median(residuals))
                result["residual_rmse_m"] = float(np.sqrt(np.mean(residuals**2)))
                result["residual_p90_m"] = float(np.percentile(residuals, 90))
                return result
        except Exception:
            pass

    if "X_SHIFT_M" in df.columns and "Y_SHIFT_M" in df.columns:
        try:
            x_shift = df["X_SHIFT_M"].fillna(0).astype(float).values
            y_shift = df["Y_SHIFT_M"].fillna(0).astype(float).values
            residuals = np.sqrt(x_shift**2 + y_shift**2)
            valid_mask = np.isfinite(residuals)
            residuals = residuals[valid_mask]
            if len(residuals) > 0:
                result["residual_source"] = "XY_SHIFT_M"
                result["residual_mean_m"] = float(np.mean(residuals))
                result["residual_median_m"] = float(np.median(residuals))
                result["residual_rmse_m"] = float(np.sqrt(np.mean(residuals**2)))
                result["residual_p90_m"] = float(np.percentile(residuals, 90))
        except Exception:
            pass
    return result


def _coerce_window_size(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, str):
        cleaned = value.replace("x", ",")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) == 2:
            try:
                return (max(32, int(parts[0])), max(32, int(parts[1])))
            except Exception:
                return default
        return default
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return (max(32, int(value[0])), max(32, int(value[1])))
        except Exception:
            return default
    return default


def _coerce_global_coreg_attempt_ladder(raw: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        raw = fallback
    for item in raw:
        if not isinstance(item, dict):
            continue
        ws = item.get("window_size", item.get("ws"))
        ws_tuple = _coerce_window_size(ws, default=(256, 256))
        try:
            max_shift = float(item.get("max_shift", 150))
        except Exception:
            max_shift = 150.0
        out.append({"ws": ws_tuple, "max_shift": max(5.0, max_shift)})
    if not out:
        out = [{"ws": (256, 256), "max_shift": 150.0}]
    return out


def _resolve_sensor_matcher_profile(config: Optional[Dict[str, Any]], sensor_type: str) -> Dict[str, Any]:
    cfg = config or {}
    sensor_key = str(sensor_type or "").strip().upper() or "DEFAULT"
    default_profiles = DEFAULT_CONFIG.get("global_coreg_profiles_by_sensor", {})
    profile_dict = cfg.get("global_coreg_profiles_by_sensor", default_profiles)
    if not isinstance(profile_dict, dict):
        profile_dict = default_profiles
    ladder_override = cfg.get("global_coreg_attempt_ladder", None)
    if isinstance(ladder_override, list) and ladder_override:
        global_ladder = _coerce_global_coreg_attempt_ladder(ladder_override, [])
        profile_source = "global_coreg_attempt_ladder override"
    else:
        sensor_profiles = profile_dict.get(sensor_key) or profile_dict.get("DEFAULT") or []
        if not sensor_profiles and isinstance(default_profiles, dict):
            sensor_profiles = default_profiles.get(sensor_key) or default_profiles.get("DEFAULT") or []
        global_ladder = _coerce_global_coreg_attempt_ladder(sensor_profiles, [])
        profile_source = f"global_coreg_profiles_by_sensor:{sensor_key if profile_dict.get(sensor_key) else 'DEFAULT'}"
    local_shift_defaults = DEFAULT_CONFIG.get("local_max_shift_by_sensor", {"DEFAULT": 50})
    local_shift_map = cfg.get("local_max_shift_by_sensor", local_shift_defaults)
    if not isinstance(local_shift_map, dict):
        local_shift_map = local_shift_defaults
    local_max_shift = max(5.0, float(local_shift_map.get(sensor_key, local_shift_map.get("DEFAULT", 50))))
    local_grid_res = max(30, int(cfg.get("local_coreg_grid_res", DEFAULT_CONFIG.get("local_coreg_grid_res", LOCAL_GRID_RES_M))))
    local_window = _coerce_window_size(cfg.get("local_coreg_window_size", DEFAULT_CONFIG.get("local_coreg_window_size", (256, 256))), default=(256, 256))
    tiep_filter = max(0, int(cfg.get("local_coreg_tieP_filter_level", DEFAULT_CONFIG.get("local_coreg_tieP_filter_level", 1))))
    max_iter_raw = cfg.get("local_coreg_max_iter", DEFAULT_CONFIG.get("local_coreg_max_iter"))
    local_max_iter = None if max_iter_raw in (None, "", False) else max(1, int(max_iter_raw))
    return {
        "sensor": sensor_key,
        "global_attempt_ladder": global_ladder,
        "global_profile_source": profile_source,
        "local_max_shift": local_max_shift,
        "local_grid_res": local_grid_res,
        "local_window_size": local_window,
        "local_tieP_filter_level": tiep_filter,
        "local_max_iter": local_max_iter,
        "local_align_grids": bool(sensor_key != "PRISMA"),
    }


def _resolve_fixed_band_pair(
    hs_wavelengths: np.ndarray,
    band_label: str,
    sensor_type: str,
    prefer_fixed_band_pairs: bool = True,
    fixed_band_pairs_by_sensor: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    sensor_key = str(sensor_type or "").strip().upper()
    for suffix in ("_VNIR", "_SWIR"):
        if sensor_key.endswith(suffix):
            sensor_key = sensor_key[: -len(suffix)]
            break
    sensor_key = sensor_key or "DEFAULT"
    fixed_map = fixed_band_pairs_by_sensor or {}
    sensor_map: Dict[str, int] = {}
    if isinstance(fixed_map, dict):
        sensor_map = fixed_map.get(sensor_key) or fixed_map.get("DEFAULT") or {}
    if not sensor_map and sensor_key == "PRISMA":
        sensor_map = dict(PRISMA_FIXED_BAND_PAIRS)
    out: Dict[str, Any] = {"resolved": False, "mode": "window", "indices": [], "reason": None, "band_label": str(band_label), "sensor": sensor_key}
    if not prefer_fixed_band_pairs:
        out["reason"] = "fixed mapping disabled by config"
        return out
    idx = sensor_map.get(str(band_label))
    if idx is None:
        out["reason"] = f"no fixed mapping for band {band_label}"
        return out
    idx_1based = int(idx)
    if idx_1based < 1 or idx_1based > int(len(hs_wavelengths)):
        out["reason"] = f"fixed index {idx_1based} out of range for {len(hs_wavelengths)} bands"
        return out
    out["resolved"] = True
    out["mode"] = "fixed"
    out["indices"] = [idx_1based]
    return out


def _resolve_band_indices_for_matching(
    hs_wavelengths: np.ndarray,
    target_wavelength_nm: float,
    band_label: str,
    sensor_type: str,
    prefer_fixed_band_pairs: bool = True,
    fixed_band_pairs_by_sensor: Optional[Dict[str, Dict[str, int]]] = None,
    wavelength_window_nm: float = 20.0,
) -> Dict[str, Any]:
    hs_wl = np.asarray(hs_wavelengths).flatten().astype(float, copy=False)
    out = _resolve_fixed_band_pair(hs_wavelengths=hs_wl, band_label=band_label, sensor_type=sensor_type, prefer_fixed_band_pairs=prefer_fixed_band_pairs, fixed_band_pairs_by_sensor=fixed_band_pairs_by_sensor)
    if out.get("resolved", False):
        return out
    win = float(max(0.1, wavelength_window_nm))
    half = win / 2.0
    band_mask = (hs_wl >= float(target_wavelength_nm) - half) & (hs_wl <= float(target_wavelength_nm) + half)
    indices = (np.where(band_mask)[0] + 1).astype(int).tolist()
    if indices:
        out["resolved"] = True
        out["mode"] = "window"
        out["indices"] = indices
        return out
    finite = np.isfinite(hs_wl)
    if not np.any(finite):
        out["reason"] = "no finite wavelengths in source band table"
        return out
    nearest = int(np.nanargmin(np.abs(hs_wl - float(target_wavelength_nm))))
    out["resolved"] = True
    out["mode"] = "nearest"
    out["indices"] = [nearest + 1]
    out["reason"] = out.get("reason") or "window match unavailable; using nearest wavelength"
    return out


_S2_SRF_TABLE_CACHE: Optional[Dict[str, list[tuple[float, float]]]] = None


def _load_sentinel2_srf_table() -> Dict[str, list[tuple[float, float]]]:
    global _S2_SRF_TABLE_CACHE
    if _S2_SRF_TABLE_CACHE is not None:
        return _S2_SRF_TABLE_CACHE
    path = Path(__file__).resolve().parents[1] / "data" / "sentinel2_msi_srf.csv"
    table: Dict[str, list[tuple[float, float]]] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
            for row in reader:
                platform = str(row.get("platform", "S2A")).strip().upper() or "S2A"
                band = str(row.get("band", "")).strip().upper()
                if not band:
                    continue
                wl = float(row.get("wavelength_nm", "nan"))
                response = float(row.get("response", "nan"))
                if np.isfinite(wl) and np.isfinite(response):
                    table.setdefault(f"{platform}:{band}", []).append((wl, max(0.0, response)))
    except Exception:
        table = {}
    for key in list(table.keys()):
        table[key] = sorted(table[key], key=lambda item: item[0])
    _S2_SRF_TABLE_CACHE = table
    return table


def _compute_srf_weights(
    hs_wavelengths: Sequence[float],
    hs_fwhm: Optional[Sequence[float]],
    band_label: str,
    platform: str = "S2A",
) -> Dict[str, Any]:
    hs_wl = np.asarray(hs_wavelengths, dtype=float).reshape(-1)
    out: Dict[str, Any] = {
        "resolved": False,
        "mode": "srf_weighted",
        "indices": [],
        "weights": [],
        "band_label": str(band_label),
        "platform": str(platform or "S2A").upper(),
        "reason": None,
    }
    finite = np.isfinite(hs_wl)
    if not np.any(finite):
        out["reason"] = "no finite hyperspectral wavelengths"
        return out
    if hs_fwhm is not None:
        fwhm = np.asarray(hs_fwhm, dtype=float).reshape(-1)
        if fwhm.size != hs_wl.size:
            fwhm = None
    else:
        fwhm = None
    if fwhm is None:
        finite_wl = np.sort(hs_wl[finite])
        spacing = float(np.nanmedian(np.diff(finite_wl))) if finite_wl.size > 1 else 10.0
        if not np.isfinite(spacing) or spacing <= 0:
            spacing = 10.0
        fwhm = np.full(hs_wl.shape, spacing, dtype=float)
    valid_fwhm = np.isfinite(fwhm) & (fwhm > 0)
    fallback_width = float(np.nanmedian(fwhm[valid_fwhm])) if np.any(valid_fwhm) else 10.0
    fwhm = np.where(valid_fwhm, fwhm, fallback_width)

    band_key = str(band_label).upper()
    table = _load_sentinel2_srf_table()
    samples = table.get(f"{str(platform or 'S2A').upper()}:{band_key}") or table.get(f"S2A:{band_key}")
    if not samples:
        center = MULTIBAND_S2_WAVELENGTHS.get(band_key, {}).get("wavelength")
        approx_fwhm = S2_APPROX_FWHM_NM.get(band_key, 20.0)
        if center is None:
            out["reason"] = f"no SRF samples for {band_label}"
            return out
        grid = np.arange(float(center) - 2.0 * approx_fwhm, float(center) + 2.0 * approx_fwhm + 1.0, 1.0)
        sigma = float(approx_fwhm) / 2.355
        response = np.exp(-0.5 * ((grid - float(center)) / sigma) ** 2)
        samples = list(zip(grid.astype(float).tolist(), response.astype(float).tolist()))
        out["reason"] = "using approximate Gaussian SRF fallback"
    srf_wl = np.asarray([item[0] for item in samples], dtype=float)
    srf_resp = np.asarray([item[1] for item in samples], dtype=float)
    raw_weights = np.zeros(hs_wl.size, dtype=float)
    for idx, (center_wl, width) in enumerate(zip(hs_wl, fwhm)):
        if not np.isfinite(center_wl) or not np.isfinite(width) or width <= 0:
            continue
        lo = float(center_wl) - float(width) / 2.0
        hi = float(center_wl) + float(width) / 2.0
        if hi < float(np.nanmin(srf_wl)) or lo > float(np.nanmax(srf_wl)):
            continue
        mask = (srf_wl >= lo) & (srf_wl <= hi)
        if np.count_nonzero(mask) >= 2:
            raw_weights[idx] = float(np.trapz(srf_resp[mask], srf_wl[mask]))
        else:
            raw_weights[idx] = max(0.0, float(np.interp(float(center_wl), srf_wl, srf_resp, left=0.0, right=0.0))) * float(width)
    positive = raw_weights > 0
    if not np.any(positive):
        out["reason"] = f"SRF {band_label} has no overlap with hyperspectral bands"
        return out
    out["resolved"] = True
    out["indices"] = (np.where(positive)[0] + 1).astype(int).tolist()
    out["weights"] = (raw_weights[positive] / float(np.sum(raw_weights[positive]))).astype(float).tolist()
    return out


def _safe_optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _quality_tier_from_score(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= 0.8:
        return "High"
    if score >= 0.6:
        return "Medium"
    return "Low"


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


def _extract_finite_xy(df: Any, x_col: str = "X_IM", y_col: str = "Y_IM"):
    if df is None or not hasattr(df, "columns") or x_col not in df.columns or y_col not in df.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.asarray(df[x_col], dtype=float).reshape(-1)
    y = np.asarray(df[y_col], dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return x[valid], y[valid]


def _count_occupied_cells(
    df: Any,
    grid_rows: int,
    grid_cols: int,
    x_col: str = "X_IM",
    y_col: str = "Y_IM",
    extent: Optional[Sequence[float]] = None,
) -> int:
    if df is None or len(df) == 0 or x_col not in df.columns or y_col not in df.columns:
        return 0
    x = np.asarray(df[x_col], dtype=float)
    y = np.asarray(df[y_col], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return 0
    x = x[valid]
    y = y[valid]
    gx = max(1, int(grid_cols))
    gy = max(1, int(grid_rows))
    if extent is not None and len(extent) >= 4:
        x_min = float(min(float(extent[0]), float(extent[2])))
        x_max = float(max(float(extent[0]), float(extent[2])))
        y_min = float(min(float(extent[1]), float(extent[3])))
        y_max = float(max(float(extent[1]), float(extent[3])))
    else:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    cx = np.floor((x - x_min) / x_span * gx).astype(int)
    cy = np.floor((y - y_min) / y_span * gy).astype(int)
    cx = np.clip(cx, 0, gx - 1)
    cy = np.clip(cy, 0, gy - 1)
    return int(len(np.unique(cy * gx + cx)))


def _summarize_scene_tiepoint_quality(
    tie_points_df: Any,
    grid_rows: int,
    grid_cols: int,
    stratification_extent: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    from shapely.geometry import MultiPoint

    summary = {"quality_score": None, "quality_tier": None, "spatial_spread_score": None, "hull_bbox_ratio": None, "is_clustered": None}
    if tie_points_df is None or len(tie_points_df) == 0:
        return summary
    quality_score = None
    if hasattr(tie_points_df, "columns") and "QUALITY_SCORE" in tie_points_df.columns:
        scores = np.asarray(tie_points_df["QUALITY_SCORE"], dtype=float).reshape(-1)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size > 0:
            quality_score = float(np.mean(finite_scores))
    if quality_score is None and hasattr(tie_points_df, "columns") and "RELIABILITY" in tie_points_df.columns:
        reliabilities = np.asarray(tie_points_df["RELIABILITY"], dtype=float).reshape(-1)
        finite_rel = reliabilities[np.isfinite(reliabilities)]
        if finite_rel.size > 0:
            quality_score = float(np.mean(finite_rel) / 100.0)
    if quality_score is not None:
        quality_score = float(np.clip(quality_score, 0.0, 1.0))
        summary["quality_score"] = quality_score
        summary["quality_tier"] = _quality_tier_from_score(quality_score)
    occupied_cells = _count_occupied_cells(
        tie_points_df,
        grid_rows=max(1, int(grid_rows)),
        grid_cols=max(1, int(grid_cols)),
        extent=stratification_extent,
    )
    total_cells = max(1, int(max(1, int(grid_rows)) * max(1, int(grid_cols))))
    summary["spatial_spread_score"] = float(np.clip(float(occupied_cells) / float(total_cells), 0.0, 1.0))
    x_vals, y_vals = _extract_finite_xy(tie_points_df, x_col="X_IM", y_col="Y_IM")
    if x_vals.size > 0:
        hull_ratio = 0.0
        if x_vals.size >= 3:
            x_span = float(np.max(x_vals) - np.min(x_vals))
            y_span = float(np.max(y_vals) - np.min(y_vals))
            bbox_area = float(max(0.0, x_span) * max(0.0, y_span))
            if bbox_area > 0:
                hull_area = float(MultiPoint(list(zip(x_vals.tolist(), y_vals.tolist()))).convex_hull.area)
                hull_ratio = float(np.clip(hull_area / bbox_area, 0.0, 1.0))
        summary["hull_bbox_ratio"] = hull_ratio
    spread_is_clustered = summary["spatial_spread_score"] < float(SCENE_CLUSTER_SPREAD_THRESHOLD)
    hull_is_clustered = (summary["hull_bbox_ratio"] or 0.0) < float(SCENE_CLUSTER_HULL_RATIO_THRESHOLD)
    summary["is_clustered"] = bool(spread_is_clustered or hull_is_clustered)
    return summary


def _derive_scene_rmse_metrics(final_validation: Optional[Dict[str, Any]], tp_residuals: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    rmse_global = _safe_optional_float(final_validation.get("shift_m")) if isinstance(final_validation, dict) else None
    rmse_local = _safe_optional_float(tp_residuals.get("residual_rmse_m")) if isinstance(tp_residuals, dict) else None
    rmse_improvement_pct = None
    if rmse_global is not None and rmse_local is not None and rmse_global > 0.0:
        rmse_improvement_pct = float((rmse_global - rmse_local) / rmse_global * 100.0)
    return {"rmse_global": rmse_global, "rmse_local": rmse_local, "rmse_improvement_pct": rmse_improvement_pct}


def _minimum_gcps_for_polynomial_order(order: int) -> int:
    order_i = max(1, int(order))
    return int((order_i + 1) * (order_i + 2) / 2)


def _assess_gcp_geometry_for_order2(merged_df) -> Dict[str, Any]:
    out = {"ok": False, "reason": "missing geometry columns", "condition_number": None}
    if merged_df is None or len(merged_df) == 0 or "X_IM" not in merged_df.columns or "Y_IM" not in merged_df.columns:
        return out
    x = np.asarray(merged_df["X_IM"], dtype=float)
    y = np.asarray(merged_df["Y_IM"], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(valid)) < 6:
        out["reason"] = "insufficient finite image-space coordinates"
        return out
    x = x[valid]
    y = y[valid]
    x_span = float(np.percentile(x, 95) - np.percentile(x, 5))
    y_span = float(np.percentile(y, 95) - np.percentile(y, 5))
    if x_span < 20.0 or y_span < 20.0:
        out["reason"] = f"point spread too clustered (x_span={x_span:.2f}, y_span={y_span:.2f})"
        return out
    design = np.column_stack([np.ones_like(x), x, y, x * y, x * x, y * y])
    cond = float(np.linalg.cond(design))
    out["condition_number"] = cond
    if np.isfinite(cond) and cond < 1.0e8:
        out["ok"] = True
        out["reason"] = "geometry conditioning acceptable"
    else:
        out["reason"] = f"poor geometry conditioning (cond={cond:.2e})"
    return out


def _decide_polynomial_order(
    merged_df,
    preferred_order: int = 2,
    auto_downgrade: bool = True,
    min_gcps_order2: int = 12,
    min_cells_order2: int = 6,
    grid_rows: int = 4,
    grid_cols: int = 4,
    stratification_extent: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    order_target = 1 if int(preferred_order) <= 1 else 2
    n_gcps = int(len(merged_df)) if merged_df is not None else 0
    occupied_cells = _count_occupied_cells(
        merged_df,
        grid_rows=max(1, int(grid_rows)),
        grid_cols=max(1, int(grid_cols)),
        extent=stratification_extent,
    )
    geom = _assess_gcp_geometry_for_order2(merged_df)
    decision = {
        "preferred_order": order_target,
        "order_used": order_target,
        "downgraded": False,
        "reason": "preferred order honored",
        "n_gcps": n_gcps,
        "occupied_cells": occupied_cells,
        "geometry_ok": bool(geom.get("ok", False)),
        "geometry_reason": geom.get("reason"),
        "geometry_condition_number": geom.get("condition_number"),
    }
    if order_target <= 1:
        decision["reason"] = "preferred_polynomial_order=1"
        return decision
    if not auto_downgrade:
        decision["reason"] = "auto-downgrade disabled"
        return decision
    fail_reasons = []
    if n_gcps < int(max(6, min_gcps_order2)):
        fail_reasons.append(f"gcps {n_gcps} < min_gcps_order2 {int(max(6, min_gcps_order2))}")
    if occupied_cells < int(max(1, min_cells_order2)):
        fail_reasons.append(f"occupied_cells {occupied_cells} < min_cells_order2 {int(max(1, min_cells_order2))}")
    if not bool(geom.get("ok", False)):
        fail_reasons.append(str(geom.get("reason", "geometry check failed")))
    if fail_reasons:
        decision["order_used"] = 1
        decision["downgraded"] = True
        decision["reason"] = "; ".join(fail_reasons)
    return decision


def _estimate_translation_phasecorr(
    reference_raster_path: str,
    candidate_raster_path: str,
    max_dim: int = 1024,
    reference_band: int = 1,
    candidate_band: int = 1,
    reference_nodata: float = PROCESSING_NODATA,
    candidate_nodata: float = PROCESSING_NODATA,
) -> Dict[str, Any]:
    import rasterio
    from rasterio.enums import Resampling

    out: Dict[str, Any] = {
        "ok": False,
        "shift_x_px": None,
        "shift_y_px": None,
        "shift_magnitude_px": None,
        "valid_pixels": 0,
        "error": None,
    }
    try:
        with rasterio.open(candidate_raster_path) as cand_ds:
            cand_band_i = int(max(1, min(int(candidate_band), int(cand_ds.count))))
            cw = int(cand_ds.width)
            ch = int(cand_ds.height)
            longest = max(cw, ch)
            tgt = int(max(64, int(max_dim)))
            if longest > tgt:
                scale = float(tgt) / float(longest)
                out_w = max(64, int(round(cw * scale)))
                out_h = max(64, int(round(ch * scale)))
            else:
                out_w = cw
                out_h = ch
            cand = cand_ds.read(cand_band_i, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(np.float32, copy=False)
            nodata_c = cand_ds.nodata
            nodata_c = float(nodata_c) if nodata_c is not None and np.isfinite(float(nodata_c)) else float(candidate_nodata)

        with rasterio.open(reference_raster_path) as ref_ds:
            ref_band_i = int(max(1, min(int(reference_band), int(ref_ds.count))))
            ref = ref_ds.read(ref_band_i, out_shape=(cand.shape[0], cand.shape[1]), resampling=Resampling.bilinear).astype(np.float32, copy=False)
            nodata_r = ref_ds.nodata
            nodata_r = float(nodata_r) if nodata_r is not None and np.isfinite(float(nodata_r)) else float(reference_nodata)

        valid = np.isfinite(ref) & np.isfinite(cand)
        if np.isfinite(nodata_r):
            valid &= ref != nodata_r
        if np.isfinite(nodata_c):
            valid &= cand != nodata_c

        valid_count = int(np.count_nonzero(valid))
        out["valid_pixels"] = valid_count
        if valid_count < 256:
            out["error"] = "Insufficient overlapping valid pixels for phase-correlation residual check."
            return out

        ref_work = ref.copy()
        cand_work = cand.copy()
        ref_work[~valid] = float(np.median(ref_work[valid]))
        cand_work[~valid] = float(np.median(cand_work[valid]))
        ref_work -= float(np.mean(ref_work))
        cand_work -= float(np.mean(cand_work))

        h, w = ref_work.shape
        window = np.outer(np.hanning(h).astype(np.float32), np.hanning(w).astype(np.float32))
        ref_work *= window
        cand_work *= window

        cross = np.fft.fft2(ref_work) * np.conj(np.fft.fft2(cand_work))
        denom = np.abs(cross)
        denom[denom == 0] = 1.0
        corr = np.fft.ifft2(cross / denom)
        corr_abs = np.abs(corr)
        peak_idx = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
        py = int(peak_idx[0])
        px = int(peak_idx[1])

        def _parabolic_delta(center: float, before: float, after: float) -> float:
            denom = float(before) - 2.0 * float(center) + float(after)
            if abs(denom) < 1e-12:
                return 0.0
            return float(np.clip(0.5 * (float(before) - float(after)) / denom, -1.0, 1.0))

        dx_sub = _parabolic_delta(corr_abs[py, px], corr_abs[py, (px - 1) % w], corr_abs[py, (px + 1) % w])
        dy_sub = _parabolic_delta(corr_abs[py, px], corr_abs[(py - 1) % h, px], corr_abs[(py + 1) % h, px])
        peak_x = float(px) + float(dx_sub)
        peak_y = float(py) + float(dy_sub)
        shift_y = peak_y if peak_y <= (h / 2.0) else peak_y - float(h)
        shift_x = peak_x if peak_x <= (w / 2.0) else peak_x - float(w)
        out["ok"] = True
        out["shift_x_px"] = float(shift_x)
        out["shift_y_px"] = float(shift_y)
        out["shift_magnitude_px"] = float(np.hypot(float(shift_x), float(shift_y)))
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _run_postwarp_phasecorr_qa(
    enabled: bool,
    reference_raster_path: Optional[str],
    candidate_raster_path: Optional[str],
    reference_band: int,
    candidate_band: int,
    warn_threshold_px: float,
    reject_threshold_px: float,
    reject_bad: bool,
    max_dim: int,
    estimator: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"enabled": bool(enabled), "ok": False, "warning": False, "reject": False, "shift_magnitude_px": None, "error": None}
    if not bool(enabled):
        return result
    if not reference_raster_path or not candidate_raster_path:
        result["error"] = "missing reference or candidate raster path"
        return result
    if not os.path.exists(reference_raster_path) or not os.path.exists(candidate_raster_path):
        result["error"] = "reference or candidate raster does not exist"
        return result
    fn = estimator or _estimate_translation_phasecorr
    qa = fn(
        reference_raster_path=reference_raster_path,
        candidate_raster_path=candidate_raster_path,
        max_dim=int(max(128, int(max_dim))),
        reference_band=int(max(1, int(reference_band))),
        candidate_band=int(max(1, int(candidate_band))),
        reference_nodata=0.0,
        candidate_nodata=PROCESSING_NODATA,
    )
    if isinstance(qa, dict):
        result.update(qa)
    shift = float(result.get("shift_magnitude_px", 0.0) or 0.0)
    result["warning"] = bool(result.get("ok", False) and shift > float(max(0.0, warn_threshold_px)))
    reject_limit = max(float(max(0.0, warn_threshold_px)), float(max(0.0, reject_threshold_px)))
    result["reject"] = bool(result.get("ok", False) and bool(reject_bad) and shift > reject_limit)
    return result


def _build_consensus_group_ids(df, rounding_px: float = 1.0):
    import pandas as pd

    round_px = float(max(0.1, rounding_px))
    if "POINT_ID" in df.columns:
        ids = df["POINT_ID"].astype(str)
        missing = ids.isna() | (ids.str.strip() == "") | (ids.str.lower() == "nan")
        if not missing.any():
            return ids
    else:
        ids = pd.Series([""] * len(df), index=df.index, dtype=str)
        missing = pd.Series([True] * len(df), index=df.index)
    if "X_IM" in df.columns and "Y_IM" in df.columns:
        x = pd.to_numeric(df["X_IM"], errors="coerce")
        y = pd.to_numeric(df["Y_IM"], errors="coerce")
        xq = np.round(x / round_px).astype("Int64")
        yq = np.round(y / round_px).astype("Int64")
        fallback = xq.astype(str) + "_" + yq.astype(str)
        ids = ids.where(~missing, fallback)
        missing = ids.isna() | (ids.str.strip() == "") | (ids.str.lower() == "nan")
    if missing.any():
        seq = pd.Series(np.arange(len(df)), index=df.index).astype(str)
        ids = ids.where(~missing, "row_" + seq)
    return ids.astype(str)


def _compute_quality_score(df):
    import pandas as pd

    work = df.copy()

    def _normalize_series(series, higher_is_better=True):
        values = pd.to_numeric(series, errors="coerce").astype(float)
        valid = np.isfinite(values)
        out = np.zeros(len(values), dtype=float)
        if not np.any(valid):
            return out
        valid_vals = values[valid]
        lo = float(np.nanmin(valid_vals))
        hi = float(np.nanmax(valid_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-9:
            out[valid] = 0.5
            return out if higher_is_better else (1.0 - out)
        norm = (values - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0)
        out[valid] = norm[valid]
        return out if higher_is_better else (1.0 - out)

    metric_specs = [
        ("RELIABILITY", 0.45, True),
        ("SSIM_IMPRO", 0.25, True),
        ("SSIM_AFTER", 0.20, True),
        ("LAST_ERR", 0.10, False),
        ("ABS_SHIFT", 0.01, False),
        ("ABS_SHIFT_M", 0.01, False),
    ]
    score = np.zeros(len(work), dtype=float)
    weight_sum = 0.0
    for col, weight, higher_is_better in metric_specs:
        if col in work.columns:
            comp = _normalize_series(work[col], higher_is_better=higher_is_better)
            score += weight * comp
            weight_sum += weight
    score = (score / weight_sum) if weight_sum > 0 else np.zeros(len(work), dtype=float)
    if "BAND_SUPPORT" in work.columns:
        support = pd.to_numeric(work["BAND_SUPPORT"], errors="coerce").fillna(1.0).astype(float)
        score += np.clip((support - 1.0) * 0.03, 0.0, 0.30)
    penalty = np.zeros(len(work), dtype=float)
    for col in ("L1_OUTLIER", "L2_OUTLIER", "L3_OUTLIER"):
        if col in work.columns:
            flagged = work[col].astype("boolean").fillna(False).to_numpy(dtype=bool)
            penalty[flagged] -= 10.0
    work["QUALITY_SCORE"] = score + penalty
    return work


def _tiepoint_model_residuals(df: Any, order: int = 1) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": False, "residuals": None, "valid_mask": None, "order": int(1 if int(order) <= 1 else 2), "error": None}
    required = ("X_IM", "Y_IM", "X_MAP", "Y_MAP", "X_SHIFT_M", "Y_SHIFT_M")
    if df is None or not hasattr(df, "columns"):
        out["error"] = "missing dataframe"
        return out
    missing = [col for col in required if col not in df.columns]
    if missing:
        out["error"] = f"missing columns: {missing}"
        return out
    try:
        x = np.asarray(df["X_IM"], dtype=float).reshape(-1)
        y = np.asarray(df["Y_IM"], dtype=float).reshape(-1)
        dst_x = np.asarray(df["X_MAP"], dtype=float).reshape(-1) + np.asarray(df["X_SHIFT_M"], dtype=float).reshape(-1)
        dst_y = np.asarray(df["Y_MAP"], dtype=float).reshape(-1) + np.asarray(df["Y_SHIFT_M"], dtype=float).reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(dst_x) & np.isfinite(dst_y)
        if int(np.count_nonzero(valid)) < 4:
            out["error"] = "insufficient finite tiepoints for model residuals"
            return out
        xv = x[valid]
        yv = y[valid]
        if int(out["order"]) <= 1:
            design = np.column_stack([np.ones_like(xv), xv, yv])
        else:
            if int(np.count_nonzero(valid)) < 7:
                out["order"] = 1
                design = np.column_stack([np.ones_like(xv), xv, yv])
            else:
                design = np.column_stack([np.ones_like(xv), xv, yv, xv * yv, xv * xv, yv * yv])
        if int(np.count_nonzero(valid)) <= int(design.shape[1]):
            out["error"] = "not enough tiepoints for overdetermined model fit"
            return out
        coef_x, *_ = np.linalg.lstsq(design, dst_x[valid], rcond=None)
        coef_y, *_ = np.linalg.lstsq(design, dst_y[valid], rcond=None)
        pred_x = design @ coef_x
        pred_y = design @ coef_y
        residual_valid = np.hypot(pred_x - dst_x[valid], pred_y - dst_y[valid])
        residuals = np.full(len(x), np.nan, dtype=float)
        residuals[valid] = residual_valid
        out["ok"] = True
        out["residuals"] = residuals
        out["valid_mask"] = valid
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _extract_tiepoint_model_arrays(df: Any) -> Dict[str, Any]:
    required = ("X_IM", "Y_IM", "X_MAP", "Y_MAP", "X_SHIFT_M", "Y_SHIFT_M")
    out: Dict[str, Any] = {"ok": False, "error": None}
    if df is None or not hasattr(df, "columns"):
        out["error"] = "missing dataframe"
        return out
    missing = [col for col in required if col not in df.columns]
    if missing:
        out["error"] = f"missing columns: {missing}"
        return out
    try:
        x = np.asarray(df["X_IM"], dtype=float).reshape(-1)
        y = np.asarray(df["Y_IM"], dtype=float).reshape(-1)
        dst_x = np.asarray(df["X_MAP"], dtype=float).reshape(-1) + np.asarray(df["X_SHIFT_M"], dtype=float).reshape(-1)
        dst_y = np.asarray(df["Y_MAP"], dtype=float).reshape(-1) + np.asarray(df["Y_SHIFT_M"], dtype=float).reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(dst_x) & np.isfinite(dst_y)
        if int(np.count_nonzero(valid)) < 4:
            out["error"] = "insufficient finite tiepoints"
            return out
        out.update({"ok": True, "x": x[valid], "y": y[valid], "dst_x": dst_x[valid], "dst_y": dst_y[valid]})
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _poly_design_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    if int(order) <= 1:
        return np.column_stack([np.ones_like(x), x, y])
    return np.column_stack([np.ones_like(x), x, y, x * y, x * x, y * y])


def _fit_transform_model(kind: str, x: np.ndarray, y: np.ndarray, dst_x: np.ndarray, dst_y: np.ndarray) -> Dict[str, Any]:
    if kind == "affine":
        design = _poly_design_matrix(x, y, 1)
        if x.size <= design.shape[1]:
            return {"ok": False, "error": "insufficient affine train points"}
        coef_x, *_ = np.linalg.lstsq(design, dst_x, rcond=None)
        coef_y, *_ = np.linalg.lstsq(design, dst_y, rcond=None)
        return {"ok": True, "kind": kind, "coef_x": coef_x, "coef_y": coef_y, "order": 1}
    if kind == "order2":
        design = _poly_design_matrix(x, y, 2)
        if x.size <= design.shape[1]:
            return {"ok": False, "error": "insufficient order2 train points"}
        coef_x, *_ = np.linalg.lstsq(design, dst_x, rcond=None)
        coef_y, *_ = np.linalg.lstsq(design, dst_y, rcond=None)
        return {"ok": True, "kind": kind, "coef_x": coef_x, "coef_y": coef_y, "order": 2}
    if kind == "tps":
        try:
            from scipy.interpolate import Rbf

            return {
                "ok": True,
                "kind": kind,
                "rbf_x": Rbf(x, y, dst_x, function="thin_plate", smooth=0.0),
                "rbf_y": Rbf(x, y, dst_y, function="thin_plate", smooth=0.0),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    return {"ok": False, "error": f"unknown model kind {kind}"}


def _predict_transform_model(model: Dict[str, Any], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kind = str(model.get("kind"))
    if kind in {"affine", "order2"}:
        design = _poly_design_matrix(np.asarray(x, dtype=float), np.asarray(y, dtype=float), int(model.get("order", 1)))
        return design @ np.asarray(model["coef_x"], dtype=float), design @ np.asarray(model["coef_y"], dtype=float)
    if kind == "tps":
        return np.asarray(model["rbf_x"](x, y), dtype=float), np.asarray(model["rbf_y"](x, y), dtype=float)
    raise ValueError(f"Unknown transform model kind: {kind}")


def _model_edge_stability_ok(
    model: Dict[str, Any],
    *,
    x: np.ndarray,
    y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
    factor: float,
) -> tuple[bool, str]:
    corners_x = np.asarray([np.nanmin(x), np.nanmin(x), np.nanmax(x), np.nanmax(x)], dtype=float)
    corners_y = np.asarray([np.nanmin(y), np.nanmax(y), np.nanmin(y), np.nanmax(y)], dtype=float)
    try:
        pred_x, pred_y = _predict_transform_model(model, corners_x, corners_y)
    except Exception as exc:
        return False, f"edge prediction failed: {exc}"
    if not (np.all(np.isfinite(pred_x)) and np.all(np.isfinite(pred_y))):
        return False, "non-finite edge prediction"
    diag = float(np.hypot(np.nanmax(dst_x) - np.nanmin(dst_x), np.nanmax(dst_y) - np.nanmin(dst_y)))
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0
    margin = max(100.0, float(factor) * diag)
    if (
        np.nanmin(pred_x) < np.nanmin(dst_x) - margin
        or np.nanmax(pred_x) > np.nanmax(dst_x) + margin
        or np.nanmin(pred_y) < np.nanmin(dst_y) - margin
        or np.nanmax(pred_y) > np.nanmax(dst_y) + margin
    ):
        return False, "edge prediction outside expanded target extent"
    return True, "stable"


def _select_transform_model_cv(
    tiepoints_df: Any,
    *,
    grid_rows: int = 4,
    grid_cols: int = 4,
    stratification_extent: Optional[Sequence[float]] = None,
    folds: int = 5,
    repeats: int = 3,
    holdout_fraction: float = 0.25,
    seed: int = 1337,
    min_tps_gcps: int = 20,
    min_tps_cells: int = 8,
    tps_min_p90_improvement_m: float = 1.0,
    edge_instability_factor: float = 2.5,
) -> Dict[str, Any]:
    arrays = _extract_tiepoint_model_arrays(tiepoints_df)
    if not arrays.get("ok", False):
        return {"ok": False, "model": "rule_based", "order_used": 1, "use_tps": False, "reason": arrays.get("error")}
    x = np.asarray(arrays["x"], dtype=float)
    y = np.asarray(arrays["y"], dtype=float)
    dst_x = np.asarray(arrays["dst_x"], dtype=float)
    dst_y = np.asarray(arrays["dst_y"], dtype=float)
    n = int(x.size)
    occupied = _count_occupied_cells(tiepoints_df, grid_rows=max(1, int(grid_rows)), grid_cols=max(1, int(grid_cols)), extent=stratification_extent)
    candidates = ["affine"]
    if n >= 8:
        candidates.append("order2")
    if n >= int(min_tps_gcps) and occupied >= int(min_tps_cells):
        candidates.append("tps")
    rng = np.random.default_rng(int(seed))
    holdout_n = int(max(3, round(float(np.clip(holdout_fraction, 0.05, 0.5)) * n)))
    holdout_n = min(max(1, holdout_n), max(1, n - 4))
    split_count = max(1, int(folds)) * max(1, int(repeats))
    metrics: Dict[str, Dict[str, Any]] = {}
    for kind in candidates:
        residuals: list[float] = []
        errors: list[str] = []
        for _split in range(split_count):
            perm = rng.permutation(n)
            test_idx = perm[:holdout_n]
            train_idx = perm[holdout_n:]
            min_train = 4 if kind == "affine" else (8 if kind == "order2" else max(8, int(min_tps_gcps // 2)))
            if int(train_idx.size) < min_train:
                errors.append("insufficient train points")
                continue
            fit = _fit_transform_model(kind, x[train_idx], y[train_idx], dst_x[train_idx], dst_y[train_idx])
            if not fit.get("ok", False):
                errors.append(str(fit.get("error", "fit failed")))
                continue
            try:
                pred_x, pred_y = _predict_transform_model(fit, x[test_idx], y[test_idx])
                res = np.hypot(pred_x - dst_x[test_idx], pred_y - dst_y[test_idx])
                residuals.extend([float(v) for v in res if np.isfinite(v)])
            except Exception as exc:
                errors.append(str(exc))
        full_fit = _fit_transform_model(kind, x, y, dst_x, dst_y)
        stable, stable_reason = (
            _model_edge_stability_ok(full_fit, x=x, y=y, dst_x=dst_x, dst_y=dst_y, factor=float(edge_instability_factor))
            if full_fit.get("ok", False)
            else (False, str(full_fit.get("error", "full fit failed")))
        )
        if residuals:
            arr = np.asarray(residuals, dtype=float)
            metrics[kind] = {
                "ok": bool(stable),
                "median_residual_m": float(np.median(arr)),
                "p90_residual_m": float(np.percentile(arr, 90)),
                "n_residuals": int(arr.size),
                "edge_stable": bool(stable),
                "edge_reason": stable_reason,
                "errors": errors[:5],
            }
        else:
            metrics[kind] = {
                "ok": False,
                "median_residual_m": None,
                "p90_residual_m": None,
                "n_residuals": 0,
                "edge_stable": bool(stable),
                "edge_reason": stable_reason,
                "errors": errors[:5] or ["no held-out residuals"],
            }
    valid_metrics = {k: v for k, v in metrics.items() if v.get("ok") and v.get("p90_residual_m") is not None}
    if not valid_metrics:
        return {"ok": False, "model": "rule_based", "order_used": 1, "use_tps": False, "reason": "no stable CV model", "cv_metrics": metrics, "n_gcps": n, "occupied_cells": int(occupied)}
    non_tps = {k: v for k, v in valid_metrics.items() if k != "tps"}
    best_non_tps_name = min(non_tps, key=lambda k: float(non_tps[k]["p90_residual_m"])) if non_tps else None
    best_name = min(valid_metrics, key=lambda k: float(valid_metrics[k]["p90_residual_m"]))
    if "tps" in valid_metrics and best_non_tps_name is not None:
        tps_p90 = float(valid_metrics["tps"]["p90_residual_m"])
        non_p90 = float(valid_metrics[best_non_tps_name]["p90_residual_m"])
        if (non_p90 - tps_p90) >= float(tps_min_p90_improvement_m):
            best_name = "tps"
        elif best_name == "tps":
            best_name = best_non_tps_name
    model_label = {"affine": "affine", "order2": "polynomial_order2", "tps": "tps"}[best_name]
    return {
        "ok": True,
        "model": model_label,
        "order_used": 1 if best_name == "affine" else 2,
        "use_tps": bool(best_name == "tps"),
        "reason": f"selected {model_label} by held-out P90 residual",
        "cv_metrics": metrics,
        "n_gcps": n,
        "occupied_cells": int(occupied),
    }


def _apply_spatial_stratification(
    df,
    grid_rows: int,
    grid_cols: int,
    max_points_per_cell: int,
    min_points_required: int,
    min_distance: float = 60.0,
    stratification_extent: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    import pandas as pd

    selected_default = df.copy() if hasattr(df, "copy") else df
    result: Dict[str, Any] = {
        "selected_df": selected_default,
        "used_stratification": False,
        "quota_used": int(max(1, max_points_per_cell)),
        "fallback_notes": [],
        "occupied_cells_selected": 0,
        "occupied_cells_total": 0,
    }
    if df is None or len(df) == 0:
        return result
    if "X_IM" not in df.columns or "Y_IM" not in df.columns:
        result["fallback_notes"].append("stratification disabled: missing X_IM/Y_IM")
        return result
    rank_col = "QUALITY_SCORE" if "QUALITY_SCORE" in df.columns else None
    if rank_col is None and "RELIABILITY" in df.columns:
        rank_col = "RELIABILITY"
    work = df.copy().sort_values(rank_col, ascending=False, kind="mergesort") if rank_col else df.copy()
    x = np.asarray(work["X_IM"], dtype=float)
    y = np.asarray(work["Y_IM"], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        result["fallback_notes"].append("stratification disabled: no finite coordinates")
        return result

    gx = max(1, int(grid_cols))
    gy = max(1, int(grid_rows))
    x_valid = x[valid]
    y_valid = y[valid]
    if stratification_extent is not None and len(stratification_extent) >= 4:
        x_min = float(min(float(stratification_extent[0]), float(stratification_extent[2])))
        x_max = float(max(float(stratification_extent[0]), float(stratification_extent[2])))
        y_min = float(min(float(stratification_extent[1]), float(stratification_extent[3])))
        y_max = float(max(float(stratification_extent[1]), float(stratification_extent[3])))
    else:
        x_min, x_max = float(np.min(x_valid)), float(np.max(x_valid))
        y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    cell_x = np.floor((x - x_min) / x_span * gx).astype(int)
    cell_y = np.floor((y - y_min) / y_span * gy).astype(int)
    cell_x = np.clip(cell_x, 0, gx - 1)
    cell_y = np.clip(cell_y, 0, gy - 1)
    work["_STRAT_CELL"] = (cell_y * gx + cell_x).astype(int)
    result["occupied_cells_total"] = int(work["_STRAT_CELL"].nunique())

    def _select_with_quota(quota: int):
        return work.groupby("_STRAT_CELL", sort=False).head(int(max(1, quota)))

    quota = int(max(1, max_points_per_cell))
    selected = _select_with_quota(quota)
    result["used_stratification"] = True

    min_dist = float(max(0.0, min_distance))
    if min_dist > 0.0 and len(selected) > 1:
        selected = _filter_frame_by_min_distance(selected, min_dist)

    while len(selected) < int(max(1, min_points_required)) and quota < len(work):
        prev_quota = quota
        quota = min(len(work), max(quota + 1, quota * 2))
        if quota == prev_quota:
            break
        selected = _select_with_quota(quota)
        if min_dist > 0.0 and len(selected) > 1:
            selected = _filter_frame_by_min_distance(selected, min_dist)
        result["fallback_notes"].append(
            f"stratification quota relaxed from {prev_quota} to {quota} due to low point count"
        )
        if len(selected) >= int(max(1, min_points_required)):
            break

    required = int(max(1, min_points_required))
    if len(selected) < required:
        needed = required - len(selected)
        remainder = work.loc[~work.index.isin(selected.index)].copy()
        if needed > 0 and len(remainder) > 0:
            rem_rank_col = "QUALITY_SCORE" if "QUALITY_SCORE" in remainder.columns else None
            if rem_rank_col is None and "RELIABILITY" in remainder.columns:
                rem_rank_col = "RELIABILITY"
            if rem_rank_col is not None:
                remainder = remainder.sort_values(rem_rank_col, ascending=False, kind="mergesort")
            added_df = _filter_frame_by_min_distance(remainder, min_dist, seed_frame=selected)
            if len(added_df) > 0:
                if len(added_df) > needed:
                    added_df = added_df.head(needed)
                selected = added_df if len(selected) == 0 else pd.concat([selected, added_df], axis=0)
        if len(selected) < required:
            deficit = required - len(selected)
            remainder = work.loc[~work.index.isin(selected.index)].copy()
            if deficit > 0 and len(remainder) > 0:
                rem_rank_col = "QUALITY_SCORE" if "QUALITY_SCORE" in remainder.columns else None
                if rem_rank_col is None and "RELIABILITY" in remainder.columns:
                    rem_rank_col = "RELIABILITY"
                if rem_rank_col is not None:
                    remainder = remainder.sort_values(rem_rank_col, ascending=False, kind="mergesort")
                added_df = remainder.head(deficit)
                if len(added_df) > 0:
                    selected = added_df if len(selected) == 0 else pd.concat([selected, added_df], axis=0)
                    result["fallback_notes"].append(
                        f"distance-based fallback relaxed after exhausting spread candidates; added {len(added_df)} points"
                    )
        if len(selected) < required:
            result["fallback_notes"].append(
                f"distance-based fallback exhausted with {required - len(selected)} points deficit"
            )

    if "QUALITY_SCORE" in selected.columns:
        selected = selected.sort_values("QUALITY_SCORE", ascending=False, kind="mergesort")
    elif "RELIABILITY" in selected.columns:
        selected = selected.sort_values("RELIABILITY", ascending=False, kind="mergesort")
    if "_STRAT_CELL" in selected.columns:
        selected = selected.drop(columns=["_STRAT_CELL"])
    result["quota_used"] = quota
    result["selected_df"] = selected
    result["occupied_cells_selected"] = _count_occupied_cells(selected, gy, gx, extent=stratification_extent)
    return result


def _merge_tiepoints(
    tiepoint_dfs,
    min_reliability: float = 75.0,
    trim_by_residual: bool = True,
    residual_mad_factor: float = 3.0,
    min_band_support: int = 2,
    allow_single_band_fallback: bool = True,
    consensus_group_rounding_px: float = 1.0,
    grid_rows: int = 4,
    grid_cols: int = 4,
    max_points_per_cell: int = 3,
    min_points_required: int = MIN_TIE_POINTS_FOR_POLYNOMIAL,
    spatial_fallback_min_distance: float = 60.0,
    stratification_extent: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    import pandas as pd

    result = {
        "success": False,
        "merged_df": None,
        "visualization_df": None,
        "n_total_before_merge": 0,
        "n_after_outlier_filter": 0,
        "n_after_reliability_filter": 0,
        "n_after_residual_trim": 0,
        "reliability_threshold_used": None,
        "residual_threshold_m": None,
        "residual_basis": None,
        "error_message": None,
        "stage_counts": {},
        "fallback_notes": [],
        "consensus_fallback_used": False,
        "occupied_cells_selected": 0,
        "occupied_cells_total": 0,
    }
    if not tiepoint_dfs:
        result["error_message"] = "No tie point DataFrames provided"
        return result
    try:
        combined_df = pd.concat(tiepoint_dfs, ignore_index=True)
        result["n_total_before_merge"] = int(len(combined_df))
        result["stage_counts"]["01_concat_sanitize"] = int(len(combined_df))
        if len(combined_df) == 0:
            result["error_message"] = "No tiepoints after concatenation"
            return result
        for col in ("X_IM", "Y_IM", "X_SHIFT_M", "Y_SHIFT_M", "LAST_ERR", "ABS_SHIFT", "ABS_SHIFT_M"):
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")
        if "RELIABILITY" in combined_df.columns:
            combined_df["RELIABILITY"] = pd.to_numeric(combined_df["RELIABILITY"], errors="coerce")
            combined_df = combined_df[combined_df["RELIABILITY"].notna() & (combined_df["RELIABILITY"] > -9990)]
            if len(combined_df) > 0 and float(combined_df["RELIABILITY"].max()) <= 1.0:
                combined_df["RELIABILITY"] = combined_df["RELIABILITY"] * 100.0
        result["stage_counts"]["01_concat_sanitize"] = int(len(combined_df))
        if len(combined_df) == 0:
            result["error_message"] = "All tie points removed during sanitation"
            return result
        for col in ("L1_OUTLIER", "L2_OUTLIER", "L3_OUTLIER"):
            if col in combined_df.columns:
                outlier_mask = combined_df[col].astype("boolean").fillna(False).to_numpy(dtype=bool)
                combined_df = combined_df[~outlier_mask]
        result["n_after_outlier_filter"] = int(len(combined_df))
        result["stage_counts"]["02_remove_outliers"] = int(len(combined_df))
        if len(combined_df) == 0:
            result["error_message"] = "All tie points removed by outlier filter"
            return result
        combined_df = combined_df.copy()
        combined_df["CONSENSUS_GROUP_ID"] = _build_consensus_group_ids(combined_df, rounding_px=float(max(0.1, consensus_group_rounding_px)))
        band_support = combined_df.groupby("CONSENSUS_GROUP_ID")["BAND_LABEL"].nunique() if "BAND_LABEL" in combined_df.columns else combined_df.groupby("CONSENSUS_GROUP_ID").size()
        combined_df["BAND_SUPPORT"] = combined_df["CONSENSUS_GROUP_ID"].map(band_support).astype(int)
        min_support = max(1, int(min_band_support))
        consensus_df = combined_df[combined_df["BAND_SUPPORT"] >= min_support]
        if len(consensus_df) == 0:
            if allow_single_band_fallback:
                consensus_df = combined_df
                result["consensus_fallback_used"] = True
                result["fallback_notes"].append(
                    f"consensus fallback enabled: no points met min_band_support={min_support}"
                )
            else:
                result["error_message"] = f"No tie points meet min_band_support={min_support} and fallback is disabled"
                return result
        result["stage_counts"]["03_consensus_filter"] = int(len(consensus_df))
        scored_df = _compute_quality_score(consensus_df)
        result["stage_counts"]["04_quality_score"] = int(len(scored_df))
        result["visualization_df"] = scored_df.copy()
        strat = _apply_spatial_stratification(
            scored_df,
            grid_rows=max(1, int(grid_rows)),
            grid_cols=max(1, int(grid_cols)),
            max_points_per_cell=max(1, int(max_points_per_cell)),
            min_points_required=max(1, int(min_points_required)),
            min_distance=float(spatial_fallback_min_distance),
            stratification_extent=stratification_extent,
        )
        selected_df = strat.get("selected_df", scored_df)
        result["fallback_notes"].extend(strat.get("fallback_notes", []))
        result["occupied_cells_selected"] = int(strat.get("occupied_cells_selected", 0))
        result["occupied_cells_total"] = int(strat.get("occupied_cells_total", 0))
        result["stage_counts"]["05_spatial_stratification"] = int(len(selected_df))
        selected_df = selected_df.sort_values("QUALITY_SCORE", ascending=False).drop_duplicates(subset="CONSENSUS_GROUP_ID", keep="first")
        result["stage_counts"]["06_select_best_per_group_cell"] = int(len(selected_df))
        if "RELIABILITY" in selected_df.columns:
            thresholds = [float(min_reliability), 65.0, 55.0, 45.0, 30.0, 20.0]
            for threshold in thresholds:
                filtered = selected_df[selected_df["RELIABILITY"] >= threshold]
                if len(filtered) >= max(1, int(min_points_required)):
                    selected_df = filtered
                    result["reliability_threshold_used"] = threshold
                    break
                if threshold == thresholds[-1] and len(filtered) > 0:
                    selected_df = filtered
                    result["reliability_threshold_used"] = threshold
                    result["fallback_notes"].append(
                        f"reliability threshold relaxed to {threshold:.1f} due to low retained points"
                    )
        result["n_after_reliability_filter"] = int(len(selected_df))
        result["stage_counts"]["07_final_reliability_filter"] = int(len(selected_df))

        if trim_by_residual and len(selected_df) > max(3, int(min_points_required)):
            residuals = None
            model_res = _tiepoint_model_residuals(selected_df, order=1)
            if model_res.get("ok", False):
                residuals = np.asarray(model_res.get("residuals"), dtype=float)
                result["residual_basis"] = "model_residual_affine_m"
            elif "X_SHIFT_M" in selected_df.columns and "Y_SHIFT_M" in selected_df.columns:
                residuals = np.sqrt(
                    np.asarray(selected_df["X_SHIFT_M"], dtype=float) ** 2
                    + np.asarray(selected_df["Y_SHIFT_M"], dtype=float) ** 2
                )
                result["residual_basis"] = "shift_magnitude_fallback_m"
                result["fallback_notes"].append(
                    f"model residual trim unavailable; using shift magnitude ({model_res.get('error')})"
                )
            elif "ABS_SHIFT_M" in selected_df.columns:
                residuals = np.asarray(selected_df["ABS_SHIFT_M"], dtype=float)
                result["residual_basis"] = "abs_shift_fallback_m"
                result["fallback_notes"].append(
                    f"model residual trim unavailable; using ABS_SHIFT_M ({model_res.get('error')})"
                )

            if residuals is not None:
                finite_mask = np.isfinite(residuals)
                if np.any(finite_mask):
                    selected_df = selected_df.copy()
                    selected_df["_RESIDUAL_M"] = residuals
                    finite_residuals = residuals[finite_mask]
                    median_res = float(np.median(finite_residuals))
                    mad = float(np.median(np.abs(finite_residuals - median_res)))
                    if mad > 0:
                        threshold_m = median_res + float(residual_mad_factor) * 1.4826 * mad
                        result["residual_threshold_m"] = float(threshold_m)
                        trimmed_df = selected_df[selected_df["_RESIDUAL_M"] <= threshold_m]
                        if len(trimmed_df) < int(max(1, min_points_required)):
                            if hasattr(selected_df, "nsmallest"):
                                trimmed_df = selected_df.nsmallest(int(max(1, min_points_required)), "_RESIDUAL_M")
                            else:
                                trimmed_df = selected_df.sort_values("_RESIDUAL_M", ascending=True, kind="mergesort").head(
                                    int(max(1, min_points_required))
                                )
                        selected_df = trimmed_df
                if "_RESIDUAL_M" in selected_df.columns:
                    selected_df = selected_df.drop(columns=["_RESIDUAL_M"])

        result["n_after_residual_trim"] = int(len(selected_df))
        result["stage_counts"]["08_optional_residual_trim"] = int(len(selected_df))
        result["merged_df"] = selected_df
        result["success"] = len(selected_df) >= int(max(1, min_points_required))
        if not result["success"]:
            result["error_message"] = f"Insufficient tiepoints after merge: {len(selected_df)} < {int(max(1, min_points_required))}"
    except Exception as exc:
        result["error_message"] = f"Merge failed: {exc}"
    return result


def _build_gcps_from_tiepoints(merged_df):
    result = {"success": False, "gcps": [], "n_gcps": 0, "error_message": None}
    if merged_df is None or len(merged_df) == 0:
        result["error_message"] = "No tie points provided"
        return result
    try:
        from osgeo import gdal

        required = ["X_IM", "Y_IM", "X_MAP", "Y_MAP", "X_SHIFT_M", "Y_SHIFT_M"]
        missing = [c for c in required if c not in merged_df.columns]
        if missing:
            result["error_message"] = f"Missing columns: {missing}"
            return result
        gcps = []
        for _, row in merged_df.iterrows():
            pixel = float(row["X_IM"])
            line = float(row["Y_IM"])
            x_map_corrected = float(row["X_MAP"]) + float(row["X_SHIFT_M"])
            y_map_corrected = float(row["Y_MAP"]) + float(row["Y_SHIFT_M"])
            gcps.append(gdal.GCP(x_map_corrected, y_map_corrected, 0.0, pixel, line))
        result["gcps"] = gcps
        result["n_gcps"] = len(gcps)
        result["success"] = True
    except Exception as exc:
        result["error_message"] = f"GCP construction failed: {exc}"
    return result


__all__ = [
    "_apply_spatial_stratification",
    "_assess_gcp_geometry_for_order2",
    "_build_consensus_group_ids",
    "_build_gcps_from_tiepoints",
    "_compute_quality_score",
    "_compute_srf_weights",
    "_compute_tiepoint_residuals",
    "_count_occupied_cells",
    "_decide_polynomial_order",
    "_derive_scene_rmse_metrics",
    "_estimate_translation_phasecorr",
    "_extract_finite_xy",
    "_merge_tiepoints",
    "_minimum_gcps_for_polynomial_order",
    "_parse_arosics_output",
    "_quality_tier_from_score",
    "_resolve_band_indices_for_matching",
    "_resolve_fixed_band_pair",
    "_resolve_sensor_matcher_profile",
    "_run_postwarp_phasecorr_qa",
    "_safe_optional_float",
    "_select_transform_model_cv",
    "_summarize_scene_tiepoint_quality",
    "_tiepoint_model_residuals",
    "_validate_coreg_hybrid",
]
