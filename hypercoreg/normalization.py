"""
Raster normalization helpers for HyperCoreg.

This module provides memory-safe, windowed normalization utilities that avoid
loading full raster bands in memory. Percentile mode uses deterministic
reservoir sampling per band.
"""

import os
import re
import tempfile
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from hypercoreg.pipeline.raster_sanitize import _sanitize_raster_nonfinite_inplace

ALLOWED_NORMALIZATION_MODES = {"none", "minmax", "percentile"}
STALE_DERIVED_RASTER_BAND_METADATA_PREFIXES = ("STATISTICS_",)


@dataclass
class NormalizationParams:
    """Normalization controls."""

    mode: str = "none"
    p_low: float = 2.0
    p_high: float = 98.0
    clip: bool = True
    eps: float = 1e-6
    min_valid_pixels: int = 1024
    reservoir_size: int = 8192
    seed: int = 1337
    tile_size: int = 256


def normalize_mode(mode: Any, default: str = "none") -> str:
    """Normalize mode string to one of: none, minmax, percentile."""
    mode_s = str(mode).strip().lower() if mode is not None else ""
    if mode_s in ALLOWED_NORMALIZATION_MODES:
        return mode_s
    return str(default).strip().lower()


def sanitize_normalization_params(params: NormalizationParams) -> Tuple[NormalizationParams, List[str]]:
    """Clamp normalization params to safe ranges and return warning messages."""
    warnings: List[str] = []
    out = NormalizationParams(**vars(params))

    out.mode = normalize_mode(out.mode, default="none")

    if not np.isfinite(float(out.p_low)):
        warnings.append("Invalid norm_p_low; using 2.0.")
        out.p_low = 2.0
    if not np.isfinite(float(out.p_high)):
        warnings.append("Invalid norm_p_high; using 98.0.")
        out.p_high = 98.0
    out.p_low = float(np.clip(float(out.p_low), 0.0, 100.0))
    out.p_high = float(np.clip(float(out.p_high), 0.0, 100.0))
    if out.p_high <= out.p_low:
        warnings.append("norm_p_high must be > norm_p_low; using 2/98.")
        out.p_low = 2.0
        out.p_high = 98.0

    if not np.isfinite(float(out.eps)) or float(out.eps) <= 0.0:
        warnings.append("Invalid norm_eps; using 1e-6.")
        out.eps = 1e-6
    else:
        out.eps = float(out.eps)

    try:
        out.min_valid_pixels = int(out.min_valid_pixels)
    except Exception:
        warnings.append("Invalid norm_min_valid_pixels; using 1024.")
        out.min_valid_pixels = 1024
    out.min_valid_pixels = max(1, out.min_valid_pixels)

    try:
        out.reservoir_size = int(out.reservoir_size)
    except Exception:
        warnings.append("Invalid norm_reservoir_size; using 8192.")
        out.reservoir_size = 8192
    out.reservoir_size = max(64, out.reservoir_size)

    try:
        out.seed = int(out.seed)
    except Exception:
        warnings.append("Invalid norm_seed; using 1337.")
        out.seed = 1337

    try:
        out.tile_size = int(out.tile_size)
    except Exception:
        warnings.append("Invalid norm_tile_size; using 256.")
        out.tile_size = 256
    out.tile_size = min(2048, max(64, out.tile_size))

    out.clip = bool(out.clip)
    return out, warnings


def _iter_windows_rect(width: int, height: int, tile_width: int, tile_height: int) -> Iterator[Window]:
    """Yield rectangular fixed-size windows covering a raster extent."""
    w = int(width)
    h = int(height)
    step_w = int(max(1, tile_width))
    step_h = int(max(1, tile_height))
    for row_off in range(0, h, step_h):
        win_h = min(step_h, h - row_off)
        for col_off in range(0, w, step_w):
            win_w = min(step_w, w - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def _iter_windows(width: int, height: int, tile_size: int) -> Iterator[Window]:
    """Yield square fixed-size windows covering a raster extent."""
    step = int(max(1, tile_size))
    return _iter_windows_rect(width, height, step, step)


def _iter_windows_for_dataset(
    src: rasterio.io.DatasetReader,
    fallback_tile_size: int,
) -> Tuple[Iterator[Window], str]:
    """
    Return a window iterator aligned to dataset internals when possible.

    Strategy order:
    1) block_windows(1) for tiled rasters
    2) rectangular windows from block_shapes[0]
    3) square fixed windows from fallback tile size
    """
    try:
        iterator = src.block_windows(1)
        try:
            _, first_window = next(iterator)
        except StopIteration:
            first_window = None
        if first_window is not None:
            def _gen() -> Iterator[Window]:
                yield first_window
                for _, window in iterator:
                    yield window

            return _gen(), "block_windows"
    except Exception:
        pass

    try:
        if getattr(src, "block_shapes", None):
            block_h, block_w = src.block_shapes[0]
            if int(block_w) > 0 and int(block_h) > 0:
                return _iter_windows_rect(src.width, src.height, int(block_w), int(block_h)), "block_shape"
    except Exception:
        pass

    return _iter_windows(src.width, src.height, fallback_tile_size), "fixed_tile"


def _prepare_output_profile_for_stream_write(
    src: rasterio.io.DatasetReader,
    out_profile: Dict[str, Any],
    result_warnings: List[str],
    output_profile_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ensure streamed outputs use a concrete writable driver.

    VRT sources expose a virtual profile (driver=VRT) that cannot be used as a
    writable destination for pixel streaming.
    """
    src_driver = str(getattr(src, "driver", "") or out_profile.get("driver", "")).upper()
    if src_driver == "VRT":
        out_profile["driver"] = "GTiff"
        result_warnings.append("Source driver is VRT; forcing GTiff output writer.")
    if str(out_profile.get("driver", "")).upper() == "GTIFF":
        # Avoid emitting BigTIFF unless the dataset size actually requires it.
        out_profile.pop("bigtiff", None)
        out_profile["BIGTIFF"] = "IF_NEEDED"
    if output_profile_overrides:
        out_profile = _apply_output_profile_overrides(out_profile, output_profile_overrides)
    return out_profile


def _normalize_output_compression(value: Any) -> Optional[str]:
    """Normalize output compression token for rasterio profile writes."""
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    return token.upper()


def _apply_output_profile_overrides(
    out_profile: Dict[str, Any],
    output_profile_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply explicit writer-profile overrides for streamed output compatibility."""
    overrides = dict(output_profile_overrides or {})
    if not overrides:
        return out_profile

    if overrides.get("driver") is not None:
        out_profile["driver"] = str(overrides["driver"])

    if overrides.get("interleave") is not None:
        out_profile["interleave"] = str(overrides["interleave"]).strip().lower()

    if overrides.get("tiled") is not None:
        tiled = bool(overrides["tiled"])
        out_profile["tiled"] = tiled
        if not tiled:
            out_profile.pop("blockxsize", None)
            out_profile.pop("blockysize", None)

    if "compress" in overrides:
        compression = _normalize_output_compression(overrides.get("compress"))
        if compression is None:
            out_profile.pop("compress", None)
        else:
            out_profile["compress"] = compression
            if compression == "NONE":
                out_profile.pop("predictor", None)

    if str(out_profile.get("driver", "")).upper() == "GTIFF":
        out_profile.pop("bigtiff", None)
        if overrides.get("BIGTIFF") is not None:
            out_profile["BIGTIFF"] = str(overrides["BIGTIFF"]).strip().upper()

    return out_profile


NODATA_METADATA_KEYS = {
    "nodata",
    "no_data",
    "no_data_value",
    "data_ignore_value",
    "dataignorevalue",
    "background_value",
    "backgroundvalue",
    "fill_value",
    "_fillvalue",
}


def _normalize_metadata_key(key: Any) -> str:
    return str(key or "").strip().lower().replace(" ", "_").replace("-", "_")


def _finite_float_from_metadata(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        text = str(value or "").strip().strip("{}[]()").strip().strip("'\"")
        try:
            val = float(text)
        except Exception:
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
            if match is None:
                return None
            try:
                val = float(match.group(0))
            except Exception:
                return None
    return float(val) if np.isfinite(float(val)) else None


def _append_unique_float(values: List[float], value: Any) -> None:
    val = _finite_float_from_metadata(value)
    if val is None:
        return
    if not any(abs(float(existing) - float(val)) <= 1e-9 for existing in values):
        values.append(float(val))


def resolve_raster_nodata_values(
    src: rasterio.io.DatasetReader,
    nodata_fallback: Optional[float] = None,
    extra_values: Optional[Sequence[Any]] = None,
) -> Tuple[float, ...]:
    """Resolve primary and alias no-data values from raster metadata.

    Some EnMAP rasters carry both a processing no-data value and a native
    background/data-ignore value. Treat all finite aliases as invalid during
    streaming reads, while preserving the first value as the output no-data.
    """
    values: List[float] = []
    for value in list(extra_values or []):
        _append_unique_float(values, value)

    try:
        _append_unique_float(values, src.nodata)
    except Exception:
        pass

    tag_sets: List[Dict[str, Any]] = []
    for namespace in (None, "ENVI"):
        try:
            tag_sets.append(src.tags() if namespace is None else src.tags(ns=namespace))
        except Exception:
            pass
    for tags in tag_sets:
        for key, value in dict(tags or {}).items():
            if _normalize_metadata_key(key) in NODATA_METADATA_KEYS:
                _append_unique_float(values, value)

    if not values:
        _append_unique_float(values, nodata_fallback)
    return tuple(values)


def _is_valid_mask(block: np.ndarray, nodata_value: Optional[Any]) -> np.ndarray:
    """Return validity mask (finite and not equal to any no-data marker)."""
    valid = np.isfinite(block)
    if nodata_value is None:
        return valid
    if isinstance(nodata_value, (list, tuple, set, np.ndarray)):
        nodata_values = list(nodata_value)
    else:
        nodata_values = [nodata_value]
    for value in nodata_values:
        parsed = _finite_float_from_metadata(value)
        if parsed is not None:
            valid &= block != float(parsed)
    return valid


def _sanitize_block_for_write(block: np.ndarray, valid_mask: np.ndarray, nodata_value: float) -> None:
    """Replace invalid pixels in a write block with the raster nodata marker."""
    if not np.any(~valid_mask):
        return
    block[~valid_mask] = float(nodata_value)


def _update_reservoir_vectorized(
    reservoir: np.ndarray,
    fill_count: int,
    seen_count: int,
    values: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    """
    Vectorized reservoir update for a stream chunk.

    This is equivalent to algorithm-R decisions performed in batch:
    each incoming item t is selected with probability k/t, and if selected,
    replaces a random reservoir slot.
    """
    n_vals = int(values.size)
    if n_vals <= 0:
        return int(fill_count), int(seen_count)

    k = int(reservoir.size)
    idx = 0
    cur_fill = int(fill_count)
    cur_seen = int(seen_count)

    if cur_fill < k:
        n_take = min(k - cur_fill, n_vals)
        reservoir[cur_fill:cur_fill + n_take] = values[:n_take]
        cur_fill += n_take
        cur_seen += n_take
        idx = n_take

    if idx >= n_vals:
        return cur_fill, cur_seen

    rem = values[idx:]
    m = int(rem.size)
    positions = np.arange(cur_seen + 1, cur_seen + m + 1, dtype=np.float64)
    selected = rng.random(m) < (float(k) / positions)
    if np.any(selected):
        selected_vals = rem[selected]
        replace_idx = rng.integers(0, k, size=int(selected_vals.size), endpoint=False)
        reservoir[replace_idx] = selected_vals
    cur_seen += m
    return cur_fill, cur_seen


def _build_band_bounds(
    params: NormalizationParams,
    band_mins: np.ndarray,
    band_maxs: np.ndarray,
    valid_counts: np.ndarray,
    reservoir: Optional[np.ndarray],
    reservoir_fill: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Resolve per-band normalization bounds and fallback flags."""
    n_bands = int(valid_counts.size)
    lows = np.zeros(n_bands, dtype=np.float64)
    highs = np.zeros(n_bands, dtype=np.float64)
    constant_flags = np.ones(n_bands, dtype=bool)
    warnings: List[str] = []

    for b in range(n_bands):
        vcount = int(valid_counts[b])
        if vcount < int(params.min_valid_pixels):
            warnings.append(
                f"Band {b + 1}: valid pixels ({vcount}) below minimum "
                f"({params.min_valid_pixels}); using constant fallback."
            )
            continue

        if params.mode == "minmax":
            low = float(band_mins[b])
            high = float(band_maxs[b])
        else:
            if reservoir is None or reservoir_fill is None:
                warnings.append(f"Band {b + 1}: missing percentile reservoir; using constant fallback.")
                continue
            filled = int(reservoir_fill[b])
            if filled <= 0:
                warnings.append(f"Band {b + 1}: empty percentile reservoir; using constant fallback.")
                continue
            sample = reservoir[b, :filled]
            q = np.percentile(sample, [float(params.p_low), float(params.p_high)])
            low = float(q[0])
            high = float(q[1])

        if not np.isfinite(low) or not np.isfinite(high):
            warnings.append(f"Band {b + 1}: non-finite normalization bounds; using constant fallback.")
            continue
        if (high - low) < float(params.eps):
            warnings.append(
                f"Band {b + 1}: normalization range below eps ({params.eps}); "
                "using constant fallback."
            )
            continue

        lows[b] = low
        highs[b] = high
        constant_flags[b] = False

    return lows, highs, constant_flags, warnings


def _empty_stats(n_bands: int, total_px_per_band: int) -> List[Dict[str, Any]]:
    """Create empty statistics payload for all bands."""
    out: List[Dict[str, Any]] = []
    for _ in range(int(n_bands)):
        out.append(
            {
                "valid_count": 0,
                "invalid_count": int(total_px_per_band),
                "minimum": None,
                "maximum": None,
                "mean": None,
                "stddev": None,
                "valid_percent": 0.0,
            }
        )
    return out


def _resolve_keep_band_count(src_count: int, expected_band_count: Optional[int]) -> int:
    """Resolve output band count, allowing optional drop of one trailing aux band."""
    src_n = int(src_count)
    if expected_band_count is None:
        return src_n
    exp_n = int(expected_band_count)
    if exp_n <= 0:
        raise ValueError(f"Invalid expected band count: {expected_band_count}")
    if src_n == exp_n:
        return exp_n
    if src_n == exp_n + 1:
        return exp_n
    raise ValueError(f"Unexpected source band count {src_n}; expected {exp_n} or {exp_n + 1}.")


def _finalize_stats_payload(
    n_bands: int,
    total_px_per_band: int,
    out_valid_counts: np.ndarray,
    out_min: np.ndarray,
    out_max: np.ndarray,
    out_sum: np.ndarray,
    out_sum_sq: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build per-band stats and validation payload from streaming accumulators."""
    band_stats = _empty_stats(n_bands, total_px_per_band)
    for b in range(n_bands):
        vcount = int(out_valid_counts[b])
        if vcount <= 0:
            continue
        mean = float(out_sum[b] / float(vcount))
        variance = max(0.0, float(out_sum_sq[b] / float(vcount)) - (mean * mean))
        band_stats[b] = {
            "valid_count": vcount,
            "invalid_count": int(max(0, total_px_per_band - vcount)),
            "minimum": float(out_min[b]),
            "maximum": float(out_max[b]),
            "mean": float(mean),
            "stddev": float(np.sqrt(variance)),
            "valid_percent": float(100.0 * vcount / max(1, total_px_per_band)),
        }

    total_valid = int(np.sum(out_valid_counts))
    total_pixels = int(total_px_per_band * n_bands)
    validation = {
        "ok": bool(total_valid > 0),
        "error": None if total_valid > 0 else "Raster contains no finite non-nodata pixels.",
        "path": None,
        "valid_pixels": int(total_valid),
        "total_pixels": int(total_pixels),
        "valid_fraction": float(total_valid / float(total_pixels)) if total_pixels > 0 else 0.0,
        "bands_checked": int(n_bands),
    }
    return band_stats, validation


def _is_stale_derived_raster_metadata_key(key: Any) -> bool:
    key_upper = str(key or "").strip().upper()
    return any(key_upper.startswith(prefix) for prefix in STALE_DERIVED_RASTER_BAND_METADATA_PREFIXES)


def filter_dataset_tags_for_raster_copy(
    tags: Optional[Dict[str, Any]],
    *,
    output_band_count: Optional[int] = None,
    source_band_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Return dataset tags safe to carry onto a derived raster."""
    safe_tags = dict(tags or {})
    output_count = int(output_band_count) if output_band_count is not None else None
    source_count = int(source_band_count) if source_band_count is not None else None
    band_count_changed = (
        output_count is not None and source_count is not None and int(output_count) != int(source_count)
    )
    for key in list(safe_tags):
        key_text = str(key).strip()
        match = re.fullmatch(r"Band_(\d+)", key_text, flags=re.IGNORECASE)
        if match is not None and output_count is not None and int(match.group(1)) > output_count:
            safe_tags.pop(key, None)
            continue
        if band_count_changed and _is_stale_derived_raster_metadata_key(key):
            safe_tags.pop(key, None)
    return safe_tags


def filter_band_tags_for_raster_copy(tags: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return per-band tags safe to copy after resampling, subsetting, or rewriting."""
    safe_tags = dict(tags or {})
    for key in list(safe_tags):
        if _is_stale_derived_raster_metadata_key(key):
            safe_tags.pop(key, None)
    return safe_tags


def scrub_incomplete_raster_band_metadata_inplace(path: str) -> Dict[str, Any]:
    """
    Remove band metadata that GeoArray cannot read safely.

    GeoArray expects every per-band metadata key to be present on every band.
    QGIS/SNAP GeoTIFFs can carry STATISTICS_* tags on only inspected bands,
    which makes GeoArray raise before AROSICS can start matching.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "path": path,
        "removed_keys": [],
        "removed_items": 0,
        "warnings": [],
        "error": None,
    }
    if not path or not os.path.exists(path):
        result["error"] = f"Raster not found: {path}"
        return result

    try:
        from osgeo import gdal

        gdal.UseExceptions()
        ds = gdal.Open(str(path), gdal.GA_Update)
        if ds is None:
            result["error"] = f"GDAL could not open raster for metadata update: {path}"
            return result

        try:
            band_count = int(ds.RasterCount)
            key_counts: Dict[str, int] = {}
            for bidx in range(1, band_count + 1):
                band = ds.GetRasterBand(bidx)
                metadata = dict(band.GetMetadata() or {})
                for key in metadata:
                    key_counts[str(key)] = int(key_counts.get(str(key), 0)) + 1

            keys_to_remove = {
                key
                for key, count in key_counts.items()
                if _is_stale_derived_raster_metadata_key(key) or int(count) != band_count
            }

            removed_items = 0
            for bidx in range(1, band_count + 1):
                band = ds.GetRasterBand(bidx)
                metadata = dict(band.GetMetadata() or {})
                for key in keys_to_remove:
                    if key in metadata:
                        band.SetMetadataItem(key, None)
                        removed_items += 1
                band = None

            ds.FlushCache()
            result["ok"] = True
            result["removed_keys"] = sorted(keys_to_remove)
            result["removed_items"] = int(removed_items)
            return result
        finally:
            ds = None
    except Exception as exc:
        result["error"] = str(exc)
        return result


def _copy_dataset_metadata(
    src: rasterio.io.DatasetReader,
    dst: rasterio.io.DatasetWriter,
    source_band_indexes: Sequence[int],
) -> None:
    """Copy dataset-level tags and per-band descriptions/tags using a source-band mapping."""
    src_tags = filter_dataset_tags_for_raster_copy(
        src.tags(),
        output_band_count=int(dst.count),
        source_band_count=int(src.count),
    )
    src_tags["n_rows"] = str(int(dst.height))
    src_tags["n_cols"] = str(int(dst.width))
    src_tags["n_bands"] = str(int(dst.count))
    if dst.nodata is not None and np.isfinite(float(dst.nodata)):
        nodata_text = f"{float(dst.nodata):.10g}"
        src_tags["data_ignore_value"] = nodata_text
        if "background_value" in src_tags:
            src_tags["background_value"] = nodata_text
    if src_tags:
        dst.update_tags(**src_tags)
    for out_bidx, src_bidx in enumerate(source_band_indexes, start=1):
        src_bidx = int(src_bidx)
        desc = src.descriptions[src_bidx - 1]
        if desc:
            dst.set_band_description(out_bidx, desc)
        band_tags = filter_band_tags_for_raster_copy(src.tags(src_bidx))
        if band_tags:
            dst.update_tags(out_bidx, **band_tags)


def stream_copy_raster_to_path(
    source_path: str,
    output_path: str,
    nodata_fallback: float,
    expected_band_count: Optional[int] = None,
    source_bands_1based: Optional[Sequence[int]] = None,
    tile_size: int = 256,
    collect_band_stats: bool = True,
    output_profile_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Copy raster to output in one streaming pass.

    If expected_band_count is provided and source has one extra trailing band,
    the trailing band is dropped during the same pass.

    If source_bands_1based is provided, those source bands are read in the
    given order (supports reordering and sparse selection) in the same pass.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "warnings": [],
        "errors": [],
        "band_statistics": [],
        "output_validation": {
            "ok": False,
            "error": None,
            "path": output_path,
            "valid_pixels": 0,
            "total_pixels": 0,
            "valid_fraction": 0.0,
            "bands_checked": 0,
        },
        "source_bands": None,
        "output_bands": None,
        "window_strategy": None,
        "window_count": 0,
        "raster_passes": 1,
        "timings": {
            "copy_s": 0.0,
            "total_s": 0.0,
        },
    }

    t0_total = perf_counter()
    temp_output_path: Optional[str] = None
    try:
        if not source_path or not os.path.exists(source_path):
            result["errors"].append(f"Missing source raster: {source_path}")
            return result

        with rasterio.open(source_path) as src:
            src_bands = int(src.count)
            if source_bands_1based is not None:
                try:
                    read_indexes = tuple(int(b) for b in source_bands_1based)
                except Exception:
                    result["errors"].append(
                        f"Invalid source_bands_1based value: {source_bands_1based}"
                    )
                    return result
                if not read_indexes:
                    result["errors"].append("source_bands_1based cannot be empty.")
                    return result
                invalid = [b for b in read_indexes if b < 1 or b > src_bands]
                if invalid:
                    result["errors"].append(
                        f"source_bands_1based contains out-of-range indexes for source count {src_bands}: "
                        f"{invalid[:5]}"
                    )
                    return result
                keep_bands = int(len(read_indexes))
                if expected_band_count is not None and int(expected_band_count) != keep_bands:
                    result["errors"].append(
                        f"source_bands_1based length ({keep_bands}) does not match expected band count "
                        f"({int(expected_band_count)})."
                    )
                    return result
            else:
                keep_bands = _resolve_keep_band_count(src_bands, expected_band_count)
                read_indexes = tuple(range(1, keep_bands + 1))
            result["source_bands"] = src_bands
            result["output_bands"] = keep_bands

            nodata_values = resolve_raster_nodata_values(src, nodata_fallback=nodata_fallback)
            nodata_val = float(nodata_values[0]) if nodata_values else float(nodata_fallback)

            width = int(src.width)
            height = int(src.height)
            total_px_per_band = int(width * height)

            out_min = np.full(keep_bands, np.inf, dtype=np.float64)
            out_max = np.full(keep_bands, -np.inf, dtype=np.float64)
            out_sum = np.zeros(keep_bands, dtype=np.float64)
            out_sum_sq = np.zeros(keep_bands, dtype=np.float64)
            out_valid_counts = np.zeros(keep_bands, dtype=np.int64)

            out_profile = src.profile.copy()
            out_profile = _prepare_output_profile_for_stream_write(
                src,
                out_profile,
                result["warnings"],
                output_profile_overrides=output_profile_overrides,
            )
            out_profile.update(count=int(keep_bands))
            if np.isfinite(nodata_val):
                out_profile["nodata"] = float(nodata_val)

            out_dir = os.path.dirname(output_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            fd, temp_output_path = tempfile.mkstemp(
                prefix="coreg_copy_",
                suffix=".tif",
                dir=out_dir,
            )
            os.close(fd)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

            t0_copy = perf_counter()
            with rasterio.open(temp_output_path, "w", **out_profile) as dst:
                _copy_dataset_metadata(src, dst, read_indexes)
                window_iter, window_strategy = _iter_windows_for_dataset(src, tile_size)
                result["window_strategy"] = window_strategy
                window_count = 0
                for window in window_iter:
                    window_count += 1
                    block = src.read(indexes=read_indexes, window=window).astype(np.float32, copy=False)
                    valid = _is_valid_mask(block, nodata_values)
                    _sanitize_block_for_write(block, valid, nodata_val)
                    dst.write(block, window=window)
                    for b in range(keep_bands):
                        vb = valid[b]
                        if not np.any(vb):
                            continue
                        vals = block[b][vb].astype(np.float64, copy=False)
                        out_valid_counts[b] += int(vals.size)
                        if not collect_band_stats:
                            continue
                        local_min = float(np.min(vals))
                        local_max = float(np.max(vals))
                        if local_min < out_min[b]:
                            out_min[b] = local_min
                        if local_max > out_max[b]:
                            out_max[b] = local_max
                        out_sum[b] += float(np.sum(vals, dtype=np.float64))
                        out_sum_sq[b] += float(np.sum(vals * vals, dtype=np.float64))
                result["window_count"] = int(window_count)

            result["timings"]["copy_s"] = perf_counter() - t0_copy

        os.replace(temp_output_path, output_path)
        temp_output_path = None
        if collect_band_stats:
            band_stats, validation = _finalize_stats_payload(
                keep_bands,
                total_px_per_band,
                out_valid_counts,
                out_min,
                out_max,
                out_sum,
                out_sum_sq,
            )
            validation["path"] = output_path
            result["band_statistics"] = band_stats
            result["output_validation"] = validation
        else:
            total_valid = int(np.sum(out_valid_counts))
            total_pixels = int(total_px_per_band * keep_bands)
            result["output_validation"] = {
                "ok": bool(total_valid > 0),
                "error": None if total_valid > 0 else "Raster contains no finite non-nodata pixels.",
                "path": output_path,
                "valid_pixels": int(total_valid),
                "total_pixels": int(total_pixels),
                "valid_fraction": float(total_valid / float(total_pixels)) if total_pixels > 0 else 0.0,
                "bands_checked": int(keep_bands),
            }

        result["ok"] = True
        return result
    except Exception as e:
        result["errors"].append(str(e))
        return result
    finally:
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except Exception:
                pass
        result["timings"]["total_s"] = perf_counter() - t0_total


def normalize_raster_to_path(
    source_path: str,
    output_path: str,
    params: NormalizationParams,
    nodata_fallback: float,
    expected_band_count: Optional[int] = None,
    source_bands_1based: Optional[Sequence[int]] = None,
    output_profile_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Normalize raster from source to output using a two-pass total-I/O design.

    Pass 1: estimate per-band normalization bounds (minmax or percentile).
    Pass 2: apply scaling and collect exact output statistics.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "warnings": [],
        "errors": [],
        "mode": normalize_mode(params.mode, default="none"),
        "band_bounds": [],
        "band_statistics": [],
        "output_validation": {
            "ok": False,
            "error": None,
            "path": output_path,
            "valid_pixels": 0,
            "total_pixels": 0,
            "valid_fraction": 0.0,
            "bands_checked": 0,
        },
        "timings": {
            "estimate_s": 0.0,
            "apply_s": 0.0,
            "total_s": 0.0,
        },
        "source_bands": None,
        "output_bands": None,
        "window_strategy": None,
        "window_count_estimate_pass1": 0,
        "window_count_estimate_pass2": 0,
        "raster_passes": 2,
    }

    t0_total = perf_counter()
    params, sanitize_warnings = sanitize_normalization_params(params)
    result["mode"] = params.mode
    result["warnings"].extend(sanitize_warnings)

    if params.mode == "none":
        copy_result = stream_copy_raster_to_path(
            source_path=source_path,
            output_path=output_path,
            nodata_fallback=nodata_fallback,
            expected_band_count=expected_band_count,
            source_bands_1based=source_bands_1based,
            output_profile_overrides=output_profile_overrides,
            collect_band_stats=True,
        )
        copy_result["mode"] = params.mode
        copy_result.setdefault("warnings", [])
        copy_result["warnings"] = (
            list(result["warnings"])
            + ["Normalization mode is 'none'; copied source raster without radiometric normalization."]
            + list(copy_result.get("warnings", []))
        )
        copy_result.setdefault("timings", {})
        copy_result["timings"]["total_s"] = perf_counter() - t0_total
        return copy_result

    temp_output_path: Optional[str] = None
    try:
        if not source_path or not os.path.exists(source_path):
            result["errors"].append(f"Missing source raster: {source_path}")
            return result

        with rasterio.open(source_path) as src:
            src_bands = int(src.count)
            if source_bands_1based is not None:
                try:
                    read_indexes = tuple(int(b) for b in source_bands_1based)
                except Exception:
                    result["errors"].append(
                        f"Invalid source_bands_1based value: {source_bands_1based}"
                    )
                    return result
                if not read_indexes:
                    result["errors"].append("source_bands_1based cannot be empty.")
                    return result
                invalid = [b for b in read_indexes if b < 1 or b > src_bands]
                if invalid:
                    result["errors"].append(
                        f"source_bands_1based contains out-of-range indexes for source count {src_bands}: "
                        f"{invalid[:5]}"
                    )
                    return result
                n_bands = int(len(read_indexes))
                if expected_band_count is not None and int(expected_band_count) != n_bands:
                    result["errors"].append(
                        f"source_bands_1based length ({n_bands}) does not match expected band count "
                        f"({int(expected_band_count)})."
                    )
                    return result
            else:
                n_bands = _resolve_keep_band_count(src_bands, expected_band_count)
                read_indexes = tuple(range(1, n_bands + 1))
            result["source_bands"] = src_bands
            result["output_bands"] = n_bands
            if n_bands < 1:
                result["errors"].append("Source raster has no bands.")
                return result

            nodata_values = resolve_raster_nodata_values(src, nodata_fallback=nodata_fallback)
            nodata_val = float(nodata_values[0]) if nodata_values else float(nodata_fallback)

            width = int(src.width)
            height = int(src.height)
            total_px_per_band = int(width * height)

            band_mins = np.full(n_bands, np.inf, dtype=np.float64)
            band_maxs = np.full(n_bands, -np.inf, dtype=np.float64)
            valid_counts = np.zeros(n_bands, dtype=np.int64)
            seen_counts = np.zeros(n_bands, dtype=np.int64)

            reservoir = None
            reservoir_fill = None
            rngs: Sequence[np.random.Generator] = ()
            if params.mode == "percentile":
                reservoir = np.empty((n_bands, int(params.reservoir_size)), dtype=np.float32)
                reservoir_fill = np.zeros(n_bands, dtype=np.int32)
                rngs = tuple(
                    np.random.default_rng(int(params.seed) + int(bidx) + 1)
                    for bidx in range(n_bands)
                )

            t0_est = perf_counter()
            estimate_windows, estimate_strategy = _iter_windows_for_dataset(src, params.tile_size)
            result["window_strategy"] = estimate_strategy
            estimate_count = 0
            for window in estimate_windows:
                estimate_count += 1
                block = src.read(indexes=read_indexes, window=window).astype(np.float32, copy=False)
                valid = _is_valid_mask(block, nodata_values)

                for b in range(n_bands):
                    vb = valid[b]
                    if not np.any(vb):
                        continue
                    vals = block[b][vb].astype(np.float64, copy=False)
                    local_min = float(np.min(vals))
                    local_max = float(np.max(vals))
                    if local_min < band_mins[b]:
                        band_mins[b] = local_min
                    if local_max > band_maxs[b]:
                        band_maxs[b] = local_max
                    valid_counts[b] += int(vals.size)

                    if params.mode == "percentile" and reservoir is not None and reservoir_fill is not None:
                        fill, seen = _update_reservoir_vectorized(
                            reservoir[b],
                            int(reservoir_fill[b]),
                            int(seen_counts[b]),
                            vals.astype(np.float32, copy=False),
                            rngs[b],
                        )
                        reservoir_fill[b] = int(fill)
                        seen_counts[b] = int(seen)
            result["window_count_estimate_pass1"] = int(estimate_count)
            result["timings"]["estimate_s"] = perf_counter() - t0_est

            lows, highs, constant_flags, bound_warnings = _build_band_bounds(
                params=params,
                band_mins=band_mins,
                band_maxs=band_maxs,
                valid_counts=valid_counts,
                reservoir=reservoir,
                reservoir_fill=reservoir_fill,
            )
            result["warnings"].extend(bound_warnings)

            band_bounds: List[Dict[str, Any]] = []
            for b in range(n_bands):
                band_bounds.append(
                    {
                        "band": int(b + 1),
                        "valid_count": int(valid_counts[b]),
                        "low": None if bool(constant_flags[b]) else float(lows[b]),
                        "high": None if bool(constant_flags[b]) else float(highs[b]),
                        "constant_fallback": bool(constant_flags[b]),
                    }
                )
            result["band_bounds"] = band_bounds

            out_min = np.full(n_bands, np.inf, dtype=np.float64)
            out_max = np.full(n_bands, -np.inf, dtype=np.float64)
            out_sum = np.zeros(n_bands, dtype=np.float64)
            out_sum_sq = np.zeros(n_bands, dtype=np.float64)
            out_valid_counts = np.zeros(n_bands, dtype=np.int64)

            out_profile = src.profile.copy()
            out_profile = _prepare_output_profile_for_stream_write(
                src,
                out_profile,
                result["warnings"],
                output_profile_overrides=output_profile_overrides,
            )
            out_profile.update(dtype=np.float32, count=int(n_bands))
            if np.isfinite(nodata_val):
                out_profile["nodata"] = float(nodata_val)

            out_dir = os.path.dirname(output_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            fd, temp_output_path = tempfile.mkstemp(
                prefix="coreg_norm_",
                suffix=".tif",
                dir=out_dir,
            )
            os.close(fd)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

            t0_apply = perf_counter()
            with rasterio.open(temp_output_path, "w", **out_profile) as dst:
                _copy_dataset_metadata(src, dst, read_indexes)

                apply_windows, apply_strategy = _iter_windows_for_dataset(src, params.tile_size)
                if result.get("window_strategy") is None:
                    result["window_strategy"] = apply_strategy
                elif result.get("window_strategy") != apply_strategy:
                    result["warnings"].append(
                        f"Window strategy changed between passes: {result['window_strategy']} -> {apply_strategy}."
                    )
                apply_count = 0
                for window in apply_windows:
                    apply_count += 1
                    block = src.read(indexes=read_indexes, window=window).astype(np.float32, copy=False)
                    valid = _is_valid_mask(block, nodata_values)

                    for b in range(n_bands):
                        vb = valid[b]
                        if not np.any(vb):
                            continue

                        if bool(constant_flags[b]):
                            block[b][vb] = 0.0
                        else:
                            scale = float(highs[b] - lows[b])
                            scaled = (block[b][vb].astype(np.float64, copy=False) - float(lows[b])) / scale
                            if params.clip:
                                np.clip(scaled, 0.0, 1.0, out=scaled)
                            block[b][vb] = scaled.astype(np.float32, copy=False)

                        vals_out = block[b][vb].astype(np.float64, copy=False)
                        out_valid_counts[b] += int(vals_out.size)
                        local_min = float(np.min(vals_out))
                        local_max = float(np.max(vals_out))
                        if local_min < out_min[b]:
                            out_min[b] = local_min
                        if local_max > out_max[b]:
                            out_max[b] = local_max
                        out_sum[b] += float(np.sum(vals_out, dtype=np.float64))
                        out_sum_sq[b] += float(np.sum(vals_out * vals_out, dtype=np.float64))

                    _sanitize_block_for_write(block, valid, nodata_val)
                    dst.write(block, window=window)
                result["window_count_estimate_pass2"] = int(apply_count)
            result["timings"]["apply_s"] = perf_counter() - t0_apply

        os.replace(temp_output_path, output_path)
        temp_output_path = None

        band_stats, validation = _finalize_stats_payload(
            n_bands,
            total_px_per_band,
            out_valid_counts,
            out_min,
            out_max,
            out_sum,
            out_sum_sq,
        )
        validation["path"] = output_path
        result["band_statistics"] = band_stats
        result["output_validation"] = validation
        result["ok"] = True
        return result
    except Exception as e:
        result["errors"].append(str(e))
        return result
    finally:
        if temp_output_path and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
            except Exception:
                pass
        result["timings"]["total_s"] = perf_counter() - t0_total
