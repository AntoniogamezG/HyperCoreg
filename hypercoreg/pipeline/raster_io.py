"""Raster and file-backed coregistration helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from affine import Affine

from hypercoreg.config import (
    CPUS_FOR_AROSICS,
    LOCAL_GRID_RES_M,
    MIN_RELIABILITY_THRESHOLD,
    MULTIBAND_S2_WAVELENGTHS,
    S2_L2A_ANCILLARY_BANDS,
    S2_L2A_OUTPUT_BANDS,
    S2_L2A_REFERENCE_STACK_BANDS,
    PROCESSING_NODATA,
)
from hypercoreg.normalization import (
    filter_band_tags_for_raster_copy,
    filter_dataset_tags_for_raster_copy,
    resolve_raster_nodata_values,
)
from hypercoreg.pipeline import coreg_math
from hypercoreg.pipeline import runtime as _runtime
from hypercoreg.pipeline.raster_sanitize import _sanitize_raster_nonfinite_inplace
from hypercoreg.utils import resolve_gdalwarp_exe

logger = logging.getLogger("COREG_PROCESSING")

ARTIFACT_ROLE_QUALITY_MASK = "quality_mask"
ARTIFACT_ROLE_NO_PIPELINE_SIDECAR = "no_pipeline_sidecar"
ARTIFACT_ROLE_MAIN_COREG_SPECTRAL = "main_coreg_spectral"
ARTIFACT_ROLE_PRE_COREG_SPECTRAL = "pre_coreg_spectral"
PIPELINE_HDR_ARTIFACT_ROLES = {
    ARTIFACT_ROLE_MAIN_COREG_SPECTRAL,
    ARTIFACT_ROLE_PRE_COREG_SPECTRAL,
    ARTIFACT_ROLE_QUALITY_MASK,
}


def _fmt_issue(scope: str, message: str) -> str:
    return f"[{scope}] {message}"


def _compat_attr(name: str) -> Any:
    if name in globals():
        return globals()[name]
    value = getattr(_runtime, name)
    globals()[name] = value
    return value


def _supports_constructor_kwarg(target: Any, kwarg_name: str) -> bool:
    import inspect

    try:
        sig = inspect.signature(target.__init__)
    except Exception:
        try:
            sig = inspect.signature(target)
        except Exception:
            return False
    if kwarg_name in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _make_gdal_temp_path(output_raster: str, suffix: str) -> str:
    out_path = Path(output_raster)
    out_dir = out_path.parent if str(out_path.parent) else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{out_path.stem}.", suffix=suffix, dir=str(out_dir))
    os.close(fd)
    try:
        os.remove(temp_path)
    except FileNotFoundError:
        pass
    return temp_path


def _cleanup_raster_temp_outputs(path: Optional[str]) -> None:
    if not path:
        return
    candidates = [
        path,
        f"{path}.aux.xml",
        f"{path}.ovr",
        f"{path}.msk",
    ]
    for candidate in candidates:
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except Exception:
            pass


def _validate_warp_output(path: str) -> None:
    import rasterio

    with rasterio.open(path) as ds:
        if int(ds.width) <= 0 or int(ds.height) <= 0 or int(ds.count) <= 0:
            raise RuntimeError("Warped raster is invalid")


def _remove_sidecar_if_exists(path: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": True, "error": None, "removed": False}
    try:
        if os.path.exists(path):
            os.remove(path)
            result["removed"] = True
        return result
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
        return result


def _finalize_pipeline_sidecars(
    tif_path: str,
    sensor_type: str,
    artifact_role: str,
    wl,
    fwhm,
    band_names,
    band_detectors,
    strict_metadata: bool = True,
    label_precision: int = 2,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False, "warnings": [], "errors": []}
    hdr_path = str(Path(tif_path).with_suffix(".hdr"))
    aux_path = str(Path(tif_path).with_suffix(".aux.xml"))

    if str(artifact_role) in PIPELINE_HDR_ARTIFACT_ROLES:
        hdr_result = _write_envi_header(
            tif_path=tif_path,
            sensor_type=sensor_type,
            wl=wl,
            fwhm=fwhm,
            band_names=band_names,
            band_detectors=band_detectors,
            label_precision=label_precision,
            artifact_role=artifact_role,
        )
        result["warnings"].extend(list(hdr_result.get("warnings", [])))
        if not hdr_result.get("ok", False):
            errors = list(hdr_result.get("errors", []) or ["Unknown ENVI metadata error"])
            if strict_metadata:
                result["errors"].extend(errors)
                return result
            result["warnings"].append("; ".join(str(msg) for msg in errors))
    else:
        hdr_cleanup = _remove_sidecar_if_exists(hdr_path)
        if not hdr_cleanup.get("ok", False):
            msg = f"Failed to remove stale ENVI header: {hdr_cleanup.get('error', 'unknown error')}"
            if strict_metadata:
                result["errors"].append(msg)
                return result
            result["warnings"].append(msg)

    aux_cleanup = _remove_sidecar_if_exists(aux_path)
    if not aux_cleanup.get("ok", False):
        msg = f"Failed to remove stale PAM metadata sidecar: {aux_cleanup.get('error', 'unknown error')}"
        if strict_metadata:
            result["errors"].append(msg)
            return result
        result["warnings"].append(msg)

    result["ok"] = True
    return result


def _write_envi_header(
    tif_path: str,
    sensor_type: str,
    wl,
    fwhm,
    band_names,
    band_detectors,
    metadata_schema_version: int = 1,
    remove_detector_overlap: bool = False,
    label_precision: int = 2,
    metadata_extension_level: str = "stats",
    metadata_histogram_buckets: int = 64,
    artifact_role: str = ARTIFACT_ROLE_MAIN_COREG_SPECTRAL,
) -> Dict[str, Any]:
    """Write a minimal ENVI-compatible .hdr sidecar for GeoTIFF artifacts."""
    del metadata_schema_version
    del remove_detector_overlap
    del metadata_extension_level
    del metadata_histogram_buckets
    del artifact_role

    result: Dict[str, Any] = {"ok": False, "warnings": [], "errors": []}
    hdr_path = str(Path(tif_path).with_suffix(".hdr"))
    try:
        import rasterio

        with rasterio.open(tif_path) as src:
            lines = ["ENVI", f"description = {{HyperCoreg output: {sensor_type}}}"]
            lines.append(f"samples = {int(src.width)}")
            lines.append(f"lines = {int(src.height)}")
            lines.append(f"bands = {int(src.count)}")
            lines.append("header offset = 0")
            if Path(tif_path).suffix.lower() in {".tif", ".tiff"}:
                lines.append("file type = TIFF")
            else:
                lines.append("file type = ENVI Standard")

            dtype_map = {
                "uint8": 1,
                "int16": 2,
                "int32": 3,
                "float32": 4,
                "float64": 5,
                "complex64": 6,
                "complex128": 9,
                "uint16": 12,
                "uint32": 13,
                "int64": 14,
                "uint64": 15,
            }
            dtype_key = str(src.dtypes[0]).lower()
            lines.append(f"data type = {dtype_map.get(dtype_key, 4)}")
            lines.append("interleave = bsq")
            lines.append("byte order = 0")
            if src.nodata is not None and np.isfinite(float(src.nodata)):
                lines.append(f"data ignore value = {float(src.nodata):.10g}")

            if src.transform is not None:
                tx = src.transform
                # ENVI map info is 1-indexed for reference pixel.
                map_info = (
                    f"Arbitrary, 1, 1, {float(tx.c):.10g}, {float(tx.f):.10g}, "
                    f"{abs(float(tx.a)):.10g}, {abs(float(tx.e)):.10g}"
                )
                lines.append(f"map info = {{{map_info}}}")
            if getattr(src, "crs", None):
                lines.append(f"coordinate system string = {{{src.crs.to_wkt()}}}")

            if wl is not None:
                wl_vals = np.asarray(wl).reshape(-1)
                if int(wl_vals.size) == int(src.count):
                    wl_str = ", ".join([f"{float(v):.{int(label_precision)}f}" for v in wl_vals])
                    lines.append(f"wavelength = {{{wl_str}}}")
                    lines.append("wavelength units = Nanometers")
            if fwhm is not None:
                fwhm_vals = np.asarray(fwhm).reshape(-1)
                if int(fwhm_vals.size) == int(src.count):
                    fwhm_str = ", ".join(
                        [f"{float(v):.{int(label_precision)}f}" for v in fwhm_vals]
                    )
                    lines.append(f"fwhm = {{{fwhm_str}}}")

            names: Optional[list[str]] = None
            if band_names is not None:
                arr = np.asarray(band_names).reshape(-1)
                if int(arr.size) == int(src.count):
                    names = [str(v) for v in arr]
            elif band_detectors is not None and wl is not None:
                det = np.asarray(band_detectors).reshape(-1)
                wl_vals = np.asarray(wl).reshape(-1)
                if int(det.size) == int(src.count) and int(wl_vals.size) == int(src.count):
                    names = [
                        f"{str(det[i]).upper()} | {float(wl_vals[i]):.{int(label_precision)}f} nm"
                        for i in range(int(src.count))
                    ]
            if names:
                lines.append(f"band names = {{{', '.join(names)}}}")

            lines.append(f"sensor type = {str(sensor_type)}")

        Path(hdr_path).write_text("\n".join(lines), encoding="utf-8")
        result["ok"] = True
        return result
    except Exception as exc:
        result["errors"].append(str(exc))
        return result


def _normalize_gdalwarp_num_threads(raw_value: Any, default: str = "ALL_CPUS") -> str:
    token = str(raw_value if raw_value is not None else default).strip().upper()
    if not token:
        token = str(default).strip().upper() or "ALL_CPUS"
    if token == "ALL_CPUS":
        return "ALL_CPUS"
    try:
        parsed = int(token)
    except Exception:
        logger.warning("Invalid gdalwarp_num_threads '%s'; using ALL_CPUS.", raw_value)
        return "ALL_CPUS"
    if parsed < 1:
        logger.warning("Invalid gdalwarp_num_threads '%s'; using ALL_CPUS.", raw_value)
        return "ALL_CPUS"
    return str(parsed)


def _crs_equivalent(lhs: Any, rhs: Any) -> bool:
    if lhs is None or rhs is None:
        return False
    try:
        from rasterio.crs import CRS

        return CRS.from_user_input(lhs) == CRS.from_user_input(rhs)
    except Exception:
        return str(lhs) == str(rhs)


def _transforms_equivalent(lhs: Any, rhs: Any, tol: float = 1e-9) -> bool:
    try:
        lvals = tuple(float(v) for v in lhs)
        rvals = tuple(float(v) for v in rhs)
        if len(lvals) != len(rvals):
            return False
        return bool(np.allclose(np.asarray(lvals), np.asarray(rvals), atol=float(tol), rtol=0.0))
    except Exception:
        return str(lhs) == str(rhs)


def _summarize_raster_grid(path: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": path,
        "exists": False,
        "width": None,
        "height": None,
        "count": None,
        "crs": None,
        "transform": None,
        "bounds": None,
        "res": None,
        "nodata": None,
        "error": None,
    }
    try:
        import rasterio

        if not path or not os.path.exists(path):
            out["error"] = f"Raster missing: {path}"
            return out
        with rasterio.open(path) as src:
            out["exists"] = True
            out["width"] = int(src.width)
            out["height"] = int(src.height)
            out["count"] = int(src.count)
            out["crs"] = str(src.crs) if src.crs is not None else None
            out["transform"] = tuple(float(v) for v in src.transform)
            out["bounds"] = (
                float(src.bounds.left),
                float(src.bounds.bottom),
                float(src.bounds.right),
                float(src.bounds.top),
            )
            out["res"] = (float(abs(src.res[0])), float(abs(src.res[1])))
            nodata_val = src.nodata
            out["nodata"] = (
                float(nodata_val)
                if nodata_val is not None and np.isfinite(float(nodata_val))
                else None
            )
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _normalize_s2_band_label(value: Any) -> Optional[str]:
    text = str(value or "").upper().strip()
    if not text:
        return None
    for ancillary_label in S2_L2A_ANCILLARY_BANDS:
        if ancillary_label in text:
            return ancillary_label
    if "B8A" in text:
        return "B8A"
    match = re.search(r"\bB0?([1-9])\b|\bB(1[12])\b", text)
    if not match:
        return None
    number = match.group(1) or match.group(2)
    if number is None:
        return None
    try:
        return f"B{int(number):02d}"
    except Exception:
        return None


def _probe_raster_valid_pixels(
    path: Optional[str],
    bands_1based: Sequence[int],
    nodata_fallback: float = PROCESSING_NODATA,
    sample_max_dim: int = 512,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "path": path,
        "nodata": None,
        "band_count": 0,
        "sampled_bands": [],
        "band_valid_pixels": {},
        "sample_shape": None,
    }
    try:
        import rasterio
        from rasterio.enums import Resampling

        if not path:
            out["error"] = "No raster path provided."
            return out
        if not os.path.exists(path):
            out["error"] = f"Raster missing: {path}"
            return out

        with rasterio.open(path) as src:
            out["band_count"] = int(src.count)
            if int(src.count) < 1:
                out["error"] = "Raster has no bands."
                return out

            nodata_values = resolve_raster_nodata_values(src, nodata_fallback=nodata_fallback)
            nodata_val = float(nodata_values[0]) if nodata_values else float(nodata_fallback)
            out["nodata"] = nodata_val

            sampled_bands: list[int] = []
            for b in bands_1based:
                try:
                    bidx = int(b)
                except Exception:
                    continue
                if bidx < 1 or bidx > int(src.count):
                    continue
                if bidx not in sampled_bands:
                    sampled_bands.append(bidx)
            if not sampled_bands:
                out["error"] = "No valid probe bands."
                return out
            out["sampled_bands"] = list(sampled_bands)

            src_w = int(src.width)
            src_h = int(src.height)
            max_dim = int(max(1, int(sample_max_dim)))
            longest = max(src_w, src_h)
            if longest > max_dim:
                scale = float(max_dim) / float(longest)
                out_w = max(1, int(round(src_w * scale)))
                out_h = max(1, int(round(src_h * scale)))
            else:
                out_w = src_w
                out_h = src_h
            out["sample_shape"] = (int(out_h), int(out_w))

            any_valid = False
            for bidx in sampled_bands:
                band = src.read(
                    bidx,
                    out_shape=(out_h, out_w),
                    resampling=Resampling.nearest,
                ).astype(np.float32, copy=False)
                valid = np.isfinite(band)
                for nodata_candidate in nodata_values:
                    valid &= band != float(nodata_candidate)
                valid_count = int(np.count_nonzero(valid))
                out["band_valid_pixels"][int(bidx)] = valid_count
                if valid_count > 0:
                    any_valid = True

            out["ok"] = bool(any_valid)
            if not out["ok"]:
                out["error"] = "No finite non-nodata pixels in sampled probe bands."
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _resolve_selected_band_values(
    values: Any,
    source_band_count: int,
    selected_bands_1based: Sequence[int],
    default_value: float,
    label: str,
) -> np.ndarray:
    selected = [int(b) for b in selected_bands_1based]
    if values is None:
        return np.full(len(selected), float(default_value), dtype=np.float32)

    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.full(len(selected), float(default_value), dtype=np.float32)
    if arr.size == 1:
        return np.full(len(selected), float(arr[0]), dtype=np.float32)
    if arr.size == int(source_band_count):
        return np.asarray([arr[int(b) - 1] for b in selected], dtype=np.float32)
    if arr.size == len(selected):
        return arr.astype(np.float32, copy=False)

    raise ValueError(
        f"{label} length mismatch: got {int(arr.size)}, expected 1, "
        f"{int(source_band_count)}, or {len(selected)}."
    )


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def _stream_copy_raster_with_band_order(
    source_path: str,
    output_path: str,
    source_bands_1based: Sequence[int],
    out_dtype: Any = np.float32,
    compress: Optional[str] = "NONE",
    tile_size: int = 512,
    band_gains: Optional[Sequence[float]] = None,
    band_offsets: Optional[Sequence[float]] = None,
    source_nodata: Optional[float] = None,
    output_nodata: Optional[float] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "source_bands": None,
        "output_bands": 0,
        "window_strategy": None,
        "window_count": 0,
        "radiometric_scaling_applied": False,
        "source_nodata": None,
        "output_nodata": None,
    }
    try:
        import rasterio
        from rasterio.errors import RasterioIOError
        from rasterio.windows import Window

        src_bands = [int(b) for b in source_bands_1based]
    except Exception:
        out["error"] = f"Invalid band-order specification: {source_bands_1based}"
        return out
    if not src_bands:
        out["error"] = "Band-order specification is empty."
        return out

    try:
        with rasterio.open(source_path) as src:
            out["source_bands"] = int(src.count)
            invalid = [b for b in src_bands if b < 1 or b > int(src.count)]
            if invalid:
                out["error"] = (
                    f"Band-order contains out-of-range indices for source count {int(src.count)}: {invalid[:5]}"
                )
                return out

            gain_values = _resolve_selected_band_values(
                band_gains,
                int(src.count),
                src_bands,
                1.0,
                "EnMAP data gain values",
            )
            offset_values = _resolve_selected_band_values(
                band_offsets,
                int(src.count),
                src_bands,
                0.0,
                "EnMAP data offset values",
            )
            apply_radiometry = bool(
                (band_gains is not None or band_offsets is not None)
                and (
                    np.any(np.abs(gain_values.astype(float) - 1.0) > 1e-12)
                    or np.any(np.abs(offset_values.astype(float)) > 1e-12)
                )
            )

            requested_dtype = np.dtype(out_dtype)
            out_dtype_name = (
                np.dtype(np.float32).name
                if apply_radiometry and not np.issubdtype(requested_dtype, np.floating)
                else requested_dtype.name
            )

            src_nodata_values = resolve_raster_nodata_values(
                src,
                extra_values=[source_nodata],
            )
            src_nodata_value = src_nodata_values[0] if src_nodata_values else None

            dst_nodata_value = _finite_float_or_none(output_nodata)
            if dst_nodata_value is None:
                dst_nodata_value = (
                    float(src_nodata_value)
                    if src_nodata_value is not None
                    else float(PROCESSING_NODATA)
                )

            out["radiometric_scaling_applied"] = bool(apply_radiometry)
            out["source_nodata"] = src_nodata_value
            out["output_nodata"] = float(dst_nodata_value)

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                count=len(src_bands),
                dtype=out_dtype_name,
                nodata=float(dst_nodata_value),
                tiled=True,
                interleave="pixel",
                BIGTIFF="YES",
            )
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            if compress is not None:
                comp = str(compress).strip().upper()
                profile["compress"] = comp
                if comp == "NONE":
                    profile.pop("predictor", None)

            read_indexes = tuple(src_bands)

            windows_iter = None
            window_strategy = "fixed_tile"
            try:
                bw_iter = src.block_windows(1)
                first_item = next(bw_iter, None)
                if first_item is not None:
                    window_strategy = "block_windows"
                    first_window = first_item[1]

                    def _iter_block_windows():
                        yield first_window
                        for _, win in bw_iter:
                            yield win

                    windows_iter = _iter_block_windows()
            except Exception:
                windows_iter = None
            if windows_iter is None:
                step = int(max(64, int(tile_size)))

                def _iter_fixed_windows():
                    for row_off in range(0, int(src.height), step):
                        win_h = min(step, int(src.height) - row_off)
                        for col_off in range(0, int(src.width), step):
                            win_w = min(step, int(src.width) - col_off)
                            yield Window(
                                col_off=int(col_off),
                                row_off=int(row_off),
                                width=int(win_w),
                                height=int(win_h),
                            )

                windows_iter = _iter_fixed_windows()
            out["window_strategy"] = window_strategy

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with rasterio.open(output_path, "w", **profile) as dst:
                src_tags = filter_dataset_tags_for_raster_copy(
                    src.tags(),
                    output_band_count=int(dst.count),
                    source_band_count=int(src.count),
                )
                if src_tags:
                    dst.update_tags(**src_tags)

                for out_bidx, src_bidx in enumerate(src_bands, start=1):
                    desc = src.descriptions[src_bidx - 1]
                    if desc:
                        dst.set_band_description(out_bidx, desc)
                    band_tags = filter_band_tags_for_raster_copy(src.tags(src_bidx))
                    if band_tags:
                        dst.update_tags(out_bidx, **band_tags)

                window_count = 0
                for window in windows_iter:
                    window_count += 1
                    window_txt = (
                        f"row_off={int(window.row_off)}, col_off={int(window.col_off)}, "
                        f"height={int(window.height)}, width={int(window.width)}"
                    )
                    try:
                        block = src.read(indexes=read_indexes, window=window, out_dtype=out_dtype_name)
                    except (RasterioIOError, OSError) as io_err:
                        raise RuntimeError(
                            "Window read failed "
                            f"(source={source_path}, output={output_path}, source_driver={src.driver}, "
                            f"output_driver={profile.get('driver')}, window={window_txt}): {io_err}"
                        ) from io_err

                    valid = np.ones(block.shape, dtype=bool)
                    if np.issubdtype(block.dtype, np.floating):
                        valid &= np.isfinite(block)
                    for nodata_candidate in src_nodata_values:
                        valid &= block != float(nodata_candidate)

                    invalid = ~valid
                    if apply_radiometry:
                        block = block.astype(np.float32, copy=True)
                        for band_pos in range(block.shape[0]):
                            band_valid = valid[band_pos]
                            if np.any(band_valid):
                                block[band_pos][band_valid] = (
                                    block[band_pos][band_valid] * gain_values[band_pos]
                                    + offset_values[band_pos]
                                )
                        if np.any(invalid):
                            block[invalid] = float(dst_nodata_value)
                    elif np.any(invalid):
                        block = block.copy()
                        block[invalid] = float(dst_nodata_value)

                    try:
                        dst.write(block, window=window)
                    except (RasterioIOError, OSError) as io_err:
                        raise RuntimeError(
                            "Window write failed "
                            f"(source={source_path}, output={output_path}, source_driver={src.driver}, "
                            f"output_driver={profile.get('driver')}, window={window_txt}): {io_err}"
                        ) from io_err
                out["window_count"] = int(window_count)

        out["output_bands"] = int(len(src_bands))
        out["ok"] = True
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _validate_local_s2_stack_override(
    local_s2_stack_path: Optional[str],
    min_band_count: Optional[int] = None,
) -> Dict[str, Any]:
    expected_count = int(min_band_count or len(S2_L2A_OUTPUT_BANDS))
    expected_bands = tuple(
        S2_L2A_REFERENCE_STACK_BANDS[: min(expected_count, len(S2_L2A_REFERENCE_STACK_BANDS))]
    )
    out: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "path": None,
        "crs": None,
        "band_count": 0,
        "warnings": [],
    }
    path_raw = str(local_s2_stack_path or "").strip()
    if not path_raw:
        out["error"] = "No local S2 override path provided."
        return out

    path_abs = os.path.abspath(path_raw)
    out["path"] = path_abs
    if not os.path.isfile(path_abs):
        out["error"] = f"Local S2 stack override not found: {path_abs}"
        return out

    try:
        import rasterio

        with rasterio.open(path_abs) as src:
            out["band_count"] = int(src.count)
            out["crs"] = src.crs
            if int(src.count) < int(max(1, int(expected_count))):
                out["error"] = (
                    f"Local S2 stack has too few bands ({int(src.count)} < {int(expected_count)})."
                )
                return out
            if src.crs is None:
                out["error"] = "Local S2 stack CRS is missing."
                return out
            descriptions = [str(v or "").strip() for v in src.descriptions[:expected_count]]
            if not any(descriptions):
                out["warnings"].append(
                    "Local S2 stack has no band descriptions; assuming canonical L2A order "
                    f"{', '.join(expected_bands)}."
                )
            else:
                for bidx, expected_label in enumerate(expected_bands, start=1):
                    desc = descriptions[bidx - 1] if bidx - 1 < len(descriptions) else ""
                    if not desc:
                        out["warnings"].append(
                            f"Local S2 stack band {bidx} has no description; assuming {expected_label}."
                        )
                        continue
                    found_label = _normalize_s2_band_label(desc)
                    if found_label != expected_label:
                        out["error"] = (
                            f"Local S2 stack band {bidx} should be {expected_label}, "
                            f"but description is {desc!r}."
                        )
                        return out
            if int(src.count) > expected_count:
                out["warnings"].append(
                    f"Local S2 stack has {int(src.count)} bands; coregistration will use the first "
                    f"{expected_count} canonical L2A reflectance bands."
                )
    except Exception as exc:
        out["error"] = f"Failed to open local S2 stack override: {exc}"
        return out

    out["ok"] = True
    return out


def _infer_detector_from_band_name(band_name: Optional[str], sensor_type: str) -> Optional[str]:
    if band_name is None:
        return None

    name = str(band_name).strip()
    if not name:
        return None

    name_lower = name.lower()
    if "swir" in name_lower:
        return "SWIR"
    if "vnir" in name_lower:
        return "VNIR"

    if str(sensor_type).upper() == "ENMAP":
        match = re.search(r"\d+", name)
        if match:
            slot = int(match.group(0))
            if slot >= 92:
                return "SWIR"
            if slot > 0:
                return "VNIR"

    return None


def _fallback_rgb_band_indices(
    band_count: int,
    targets_nm: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    if band_count <= 1:
        return 1, 1, 1
    if band_count == 2:
        return 2, 1, 1
    min_wl = 400.0
    max_wl = 2500.0
    raw: list[int] = []
    for target in targets_nm:
        frac = (float(target) - min_wl) / (max_wl - min_wl)
        idx = 1 + int(round(frac * (band_count - 1)))
        raw.append(max(1, min(band_count, idx)))
    used = set()
    uniq: list[int] = []
    for idx in raw:
        if idx not in used:
            uniq.append(idx)
            used.add(idx)
            continue
        for radius in range(1, band_count):
            for candidate in (idx - radius, idx + radius):
                if 1 <= candidate <= band_count and candidate not in used:
                    uniq.append(candidate)
                    used.add(candidate)
                    break
            if len(uniq) == len(raw):
                break
    while len(uniq) < 3:
        uniq.append(min(band_count, len(uniq) + 1))
    return int(uniq[0]), int(uniq[1]), int(uniq[2])


def _pick_unique_band_index_for_wavelength(
    target_nm: float,
    wavelengths_nm: np.ndarray,
    used_zero_based: set,
) -> int:
    arr = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    if arr.size < 1:
        return 1
    finite_idx = np.where(np.isfinite(arr))[0]
    if finite_idx.size < 1:
        return 1

    order = finite_idx[np.argsort(np.abs(arr[finite_idx] - float(target_nm)))]
    for idx0 in order:
        if int(idx0) not in used_zero_based:
            used_zero_based.add(int(idx0))
            return int(idx0) + 1

    idx0 = int(order[0])
    used_zero_based.add(idx0)
    return idx0 + 1


def _resolve_quicklook_rgb_bands(
    wavelengths_nm: Optional[np.ndarray],
    band_count: int,
    targets_nm: Tuple[float, float, float],
) -> Tuple[Tuple[int, int, int], list[Optional[float]], str]:
    wl_arr = (
        np.asarray(wavelengths_nm, dtype=float).reshape(-1)
        if wavelengths_nm is not None
        else np.array([])
    )
    if wl_arr.size >= int(band_count) and np.any(np.isfinite(wl_arr[:band_count])):
        wl_match = np.asarray(wl_arr[:band_count], dtype=float)
        finite = wl_match[np.isfinite(wl_match)]
        wl_scale = 1.0
        rgb_source = "wavelength_metadata"
        if finite.size > 0:
            finite_abs_med = float(np.nanmedian(np.abs(finite)))
            if finite_abs_med < 10.0:
                wl_scale = 1000.0
                rgb_source = "wavelength_metadata_um_to_nm"
        wl_match_nm = wl_match * float(wl_scale)

        used: set[int] = set()
        r_idx = _pick_unique_band_index_for_wavelength(targets_nm[0], wl_match_nm, used)
        g_idx = _pick_unique_band_index_for_wavelength(targets_nm[1], wl_match_nm, used)
        b_idx = _pick_unique_band_index_for_wavelength(targets_nm[2], wl_match_nm, used)
        indices = (int(r_idx), int(g_idx), int(b_idx))
        selected_wl: list[Optional[float]] = [
            float(wl_arr[r_idx - 1]) if np.isfinite(wl_arr[r_idx - 1]) else None,
            float(wl_arr[g_idx - 1]) if np.isfinite(wl_arr[g_idx - 1]) else None,
            float(wl_arr[b_idx - 1]) if np.isfinite(wl_arr[b_idx - 1]) else None,
        ]
        return indices, selected_wl, rgb_source

    indices = _fallback_rgb_band_indices(int(max(1, int(band_count))), targets_nm)
    logger.warning(
        _fmt_issue(
            "QUICKLOOK",
            "Wavelength metadata unavailable/incomplete; using deterministic fallback RGB band indices.",
        )
    )
    return indices, [None, None, None], "fallback_assumed_400_2500nm"


def _iter_valid_band_values(src: Any, bidx: int, nodata_value: Optional[Any]):
    if isinstance(nodata_value, (list, tuple, set, np.ndarray)):
        nodata_values = tuple(float(v) for v in nodata_value if v is not None and np.isfinite(float(v)))
    else:
        nodata_values = resolve_raster_nodata_values(
            src,
            nodata_fallback=nodata_value,
            extra_values=[nodata_value],
        )
    for _, window in src.block_windows(bidx):
        band = src.read(bidx, window=window).astype(np.float64, copy=False)
        valid = np.isfinite(band)
        for nodata_candidate in nodata_values:
            valid &= band != float(nodata_candidate)
        if np.any(valid):
            yield band[valid]


def _compute_band_statistics(
    src: Any,
    bidx: int,
    nodata_value: Optional[float],
) -> Dict[str, Any]:
    total_px = int(src.height) * int(src.width)
    valid_count = 0
    data_min = np.inf
    data_max = -np.inf
    sum_val = 0.0
    sum_sq = 0.0

    for vals in _iter_valid_band_values(src, bidx, nodata_value):
        valid_count += int(vals.size)
        local_min = float(np.min(vals))
        local_max = float(np.max(vals))
        if local_min < data_min:
            data_min = local_min
        if local_max > data_max:
            data_max = local_max
        sum_val += float(np.sum(vals, dtype=np.float64))
        sum_sq += float(np.sum(vals * vals, dtype=np.float64))

    if valid_count <= 0:
        return {
            "valid_count": 0,
            "invalid_count": total_px,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "stddev": None,
            "valid_percent": 0.0,
        }

    mean = sum_val / float(valid_count)
    variance = max(0.0, (sum_sq / float(valid_count)) - (mean * mean))
    return {
        "valid_count": valid_count,
        "invalid_count": max(0, total_px - valid_count),
        "minimum": float(data_min),
        "maximum": float(data_max),
        "mean": float(mean),
        "stddev": float(np.sqrt(variance)),
        "valid_percent": float(100.0 * valid_count / max(1, total_px)),
    }


def _resolve_arosics_cpu_count(raw_value: Any, default: int = 1) -> int:
    fallback = max(1, int(default))
    try:
        parsed = int(raw_value)
    except Exception:
        parsed = fallback
    if parsed > 0:
        return max(1, parsed)
    cpu_total = os.cpu_count() or fallback
    return max(1, int(cpu_total) - 1)


def _estimate_prisma_geotransform_safe(
    lon: np.ndarray,
    lat: np.ndarray,
    rows: int,
    cols: int,
    target_crs,
    use_geolocation_mesh: bool = False,
    geolocation_mesh_stride: int = 32,
):
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    def _solve_affine(
        pixel: np.ndarray,
        line: np.ndarray,
        lon_vals: np.ndarray,
        lat_vals: np.ndarray,
    ) -> Affine:
        x_map, y_map = tr.transform(lon_vals, lat_vals)
        system = np.column_stack([pixel.astype(float), line.astype(float), np.ones_like(pixel, dtype=float)])
        sol_x = np.linalg.lstsq(system, np.asarray(x_map, dtype=float), rcond=None)[0]
        sol_y = np.linalg.lstsq(system, np.asarray(y_map, dtype=float), rcond=None)[0]
        a, b, c = sol_x
        d, e, f = sol_y
        c -= (a * 0.5 + b * 0.5)
        f -= (d * 0.5 + e * 0.5)
        return Affine(a, b, c, d, e, f)

    if use_geolocation_mesh:
        try:
            stride = max(1, int(geolocation_mesh_stride))
            row_idx = np.arange(0, int(rows), stride, dtype=int)
            col_idx = np.arange(0, int(cols), stride, dtype=int)
            if row_idx[-1] != int(rows) - 1:
                row_idx = np.append(row_idx, int(rows) - 1)
            if col_idx[-1] != int(cols) - 1:
                col_idx = np.append(col_idx, int(cols) - 1)

            lon_mesh = np.asarray(lon[np.ix_(row_idx, col_idx)], dtype=float).ravel()
            lat_mesh = np.asarray(lat[np.ix_(row_idx, col_idx)], dtype=float).ravel()
            rr, cc = np.meshgrid(row_idx, col_idx, indexing="ij")
            line_mesh = rr.ravel().astype(float)
            pixel_mesh = cc.ravel().astype(float)

            valid = np.isfinite(lon_mesh) & np.isfinite(lat_mesh)
            if int(np.count_nonzero(valid)) >= 6:
                return _solve_affine(
                    pixel_mesh[valid],
                    line_mesh[valid],
                    lon_mesh[valid],
                    lat_mesh[valid],
                )
            logger.warning(
                "Geolocation mesh affine fallback to corners: only %d valid mesh samples.",
                int(np.count_nonzero(valid)),
            )
        except Exception as exc:
            logger.warning("Geolocation mesh affine fallback to corners due to error: %s", exc)

    corners = [
        (lon[0, 0], lat[0, 0], 0.0, 0.0),
        (lon[0, -1], lat[0, -1], float(cols - 1), 0.0),
        (lon[-1, 0], lat[-1, 0], 0.0, float(rows - 1)),
        (lon[-1, -1], lat[-1, -1], float(cols - 1), float(rows - 1)),
    ]
    lon_vals = np.asarray([c[0] for c in corners], dtype=float)
    lat_vals = np.asarray([c[1] for c in corners], dtype=float)
    pixel = np.asarray([c[2] for c in corners], dtype=float)
    line = np.asarray([c[3] for c in corners], dtype=float)
    return _solve_affine(pixel, line, lon_vals, lat_vals)


def _apply_polynomial_warp(
    input_raster,
    output_raster,
    gcps,
    target_crs,
    polynomial_order: int = 2,
    output_resolution: float = 30.0,
    nodata: float = -9999.0,
    resampling: str = "cubic",
    s2_bounds=None,
    gdalwarp_multi: bool = True,
    gdalwarp_num_threads: str = "ALL_CPUS",
) -> Dict[str, Any]:
    """Apply polynomial warp using GDAL with configurable order."""
    from hypercoreg.pipeline import coreg_math

    order = 1 if int(polynomial_order) <= 1 else 2
    result = {
        "success": False,
        "output_path": None,
        "error_message": None,
        "order_used": order,
    }

    min_needed = coreg_math._minimum_gcps_for_polynomial_order(order)
    if not gcps or len(gcps) < min_needed:
        result["error_message"] = (
            f"Insufficient GCPs for order {order}: {len(gcps) if gcps else 0} < {min_needed}"
        )
        return result

    temp_vrt = None
    tmp_output = None
    try:
        from osgeo import gdal

        gdal.UseExceptions()
        temp_vrt = _make_gdal_temp_path(output_raster, ".gcps.vrt")
        tmp_output = _make_gdal_temp_path(output_raster, ".tmp.tif")
        ds_in = gdal.Open(input_raster, gdal.GA_ReadOnly)
        if ds_in is None:
            result["error_message"] = f"Failed to open input: {input_raster}"
            return result

        driver_vrt = gdal.GetDriverByName("VRT")
        ds_vrt = driver_vrt.CreateCopy(temp_vrt, ds_in)
        crs_wkt = target_crs.to_wkt() if hasattr(target_crs, "to_wkt") else str(target_crs)
        ds_vrt.SetGCPs(gcps, crs_wkt)
        ds_vrt.FlushCache()
        ds_vrt = None
        ds_in = None

        gdalwarp_exe = resolve_gdalwarp_exe()
        resampling_alg = str(resampling or "cubic").strip().lower()
        if resampling_alg not in {"near", "bilinear", "cubic", "cubicspline", "lanczos"}:
            resampling_alg = "cubic"
        threads_token = _normalize_gdalwarp_num_threads(gdalwarp_num_threads)

        cmd = [
            gdalwarp_exe,
            "-order",
            str(order),
            "-t_srs",
            crs_wkt,
            "-tr",
            str(output_resolution),
            str(output_resolution),
            "-r",
            str(resampling_alg),
            "-of",
            "GTiff",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "BIGTIFF=YES",
            "-co",
            "TILED=YES",
            "-srcnodata",
            str(nodata),
            "-dstnodata",
            str(nodata),
        ]
        if bool(gdalwarp_multi):
            cmd.append("-multi")
        if threads_token:
            cmd.extend(["-wo", f"NUM_THREADS={threads_token}"])
        if s2_bounds is not None:
            minx, miny, maxx, maxy = s2_bounds
            cmd.extend(["-te", str(minx), str(miny), str(maxx), str(maxy)])
        cmd.extend([temp_vrt, tmp_output])

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(tmp_output):
            _validate_warp_output(tmp_output)
            os.replace(tmp_output, output_raster)
            result["success"] = True
            result["output_path"] = output_raster
        else:
            result["error_message"] = "Output file not created"
    except subprocess.CalledProcessError as exc:
        result["error_message"] = f"gdalwarp failed: {exc.stderr}"
    except Exception as exc:
        result["error_message"] = f"Polynomial warp failed: {exc}"
    finally:
        if temp_vrt and os.path.exists(temp_vrt):
            try:
                os.remove(temp_vrt)
            except Exception:
                pass
        _cleanup_raster_temp_outputs(tmp_output)
    return result


def _estimate_transform_from_corner_coords(corner_info: Dict[str, Any], target_crs) -> Affine:
    """Estimate affine transform from UL/UR/LL/LR lon/lat corners."""
    required_keys = (
        "ul_lon",
        "ul_lat",
        "ur_lon",
        "ur_lat",
        "ll_lon",
        "ll_lat",
        "lr_lon",
        "lr_lat",
        "rows",
        "cols",
    )
    missing = [k for k in required_keys if corner_info.get(k) is None]
    if missing:
        raise RuntimeError(f"Missing corner georeference keys: {missing}")

    rows = int(corner_info["rows"])
    cols = int(corner_info["cols"])
    if rows <= 1 or cols <= 1:
        raise RuntimeError(f"Invalid raster shape for corner transform ({rows}, {cols}).")

    from pyproj import CRS, Transformer

    target = CRS.from_user_input(target_crs)
    corner_lonlat = [
        (float(corner_info["ul_lon"]), float(corner_info["ul_lat"])),
        (float(corner_info["ur_lon"]), float(corner_info["ur_lat"])),
        (float(corner_info["ll_lon"]), float(corner_info["ll_lat"])),
        (float(corner_info["lr_lon"]), float(corner_info["lr_lat"])),
    ]
    transformer = Transformer.from_crs("EPSG:4326", target, always_xy=True)
    x_map = [transformer.transform(x, y)[0] for x, y in corner_lonlat]
    y_map = [transformer.transform(x, y)[1] for x, y in corner_lonlat]

    # Solve affine from image-corner pixel coordinates.
    system = np.array(
        [
            [0, 0, 1],
            [cols - 1, 0, 1],
            [0, rows - 1, 1],
            [cols - 1, rows - 1, 1],
        ],
        dtype=float,
    )
    sol_x = np.linalg.lstsq(system, np.array(x_map, dtype=float), rcond=None)[0]
    sol_y = np.linalg.lstsq(system, np.array(y_map, dtype=float), rcond=None)[0]
    a, b, c = sol_x
    d, e, f = sol_y

    # Convert from pixel-center fit to pixel-corner geotransform convention.
    c -= (a * 0.5 + b * 0.5)
    f -= (d * 0.5 + e * 0.5)
    return Affine(a, b, c, d, e, f)


def _prepare_bands_first(data: np.ndarray, rows: int, cols: int, label: str) -> np.ndarray:
    """Convert 2D/3D raster array to (bands, rows, cols)."""
    def _msg(message: str) -> str:
        return _fmt_issue("ANCILLARY", message)

    arr = np.asarray(data)
    if arr.ndim == 2:
        if arr.shape != (rows, cols):
            raise RuntimeError(_msg(f"{label} shape mismatch: expected {(rows, cols)}, got {tuple(arr.shape)}."))
        return arr[np.newaxis, :, :]

    if arr.ndim != 3:
        raise RuntimeError(_msg(f"{label} array must be 2D/3D, got ndim={arr.ndim}."))

    for row_ax in range(3):
        for col_ax in range(3):
            if row_ax == col_ax:
                continue
            if (arr.shape[row_ax], arr.shape[col_ax]) == (rows, cols):
                band_ax = [ax for ax in range(3) if ax not in (row_ax, col_ax)][0]
                return np.moveaxis(arr, (band_ax, row_ax, col_ax), (0, 1, 2))

    raise RuntimeError(
        _msg(f"Could not map {label} dimensions {tuple(arr.shape)} to geolocation shape {(rows, cols)}.")
    )


def _write_georeferenced_raster(
    output_path: str,
    data_bands_first: np.ndarray,
    crs,
    transform: Affine,
    dtype: str,
    nodata: Optional[float] = None,
) -> None:
    """Write bands-first data to GeoTIFF with georeference."""
    import rasterio

    data = np.asarray(data_bands_first)
    if data.ndim != 3:
        raise RuntimeError(_fmt_issue("ANCILLARY", "Expected bands-first 3D array."))
    bands, rows, cols = data.shape
    profile: Dict[str, Any] = {
        "driver": "GTiff",
        "height": int(rows),
        "width": int(cols),
        "count": int(bands),
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "BIGTIFF": "YES",
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)


def _infer_raster_native_resolution(raster_path: str, fallback: float = 30.0) -> Tuple[float, float]:
    """Estimate raster native resolution from affine transform."""
    import rasterio

    with rasterio.open(raster_path) as src:
        tr = src.transform
        x_res = float(np.hypot(tr.a, tr.b))
        y_res = float(np.hypot(tr.d, tr.e))
    if not np.isfinite(x_res) or x_res <= 0:
        x_res = float(fallback)
    if not np.isfinite(y_res) or y_res <= 0:
        y_res = float(fallback)
    return x_res, y_res


def _resolve_pan_window_size_for_raster(
    raster_path: str,
    requested_window_size: Tuple[int, int] = (512, 512),
) -> Tuple[int, int]:
    """Clamp PAN matcher window to raster dimensions."""
    import rasterio

    req = requested_window_size
    if isinstance(req, str):
        cleaned = req.replace("x", ",")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) == 2:
            try:
                req = (int(parts[0]), int(parts[1]))
            except Exception:
                req = (512, 512)
        else:
            req = (512, 512)
    if not isinstance(req, (tuple, list)) or len(req) != 2:
        req = (512, 512)
    try:
        req_w, req_h = int(req[0]), int(req[1])
    except Exception:
        req_w, req_h = 512, 512
    req_w = max(512, req_w)
    req_h = max(512, req_h)
    with rasterio.open(raster_path) as src:
        width = int(src.width)
        height = int(src.height)
    return (int(max(1, min(req_w, width))), int(max(1, min(req_h, height))))


def _create_synthetic_s2_pan(
    s2_stack_path: str,
    output_path: str,
    s2_band_indices: Sequence[int] = (1, 2, 3, 4),
) -> Dict[str, Any]:
    """Create synthetic S2 PAN proxy as nodata-safe mean of selected S2 bands."""
    import warnings
    import rasterio

    out: Dict[str, Any] = {
        "ok": False,
        "path": output_path,
        "error": None,
        "dtype": None,
        "nodata": None,
        "window_count": 0,
    }
    try:
        if not s2_stack_path or not os.path.exists(s2_stack_path):
            out["error"] = f"S2 stack not found for synthetic PAN: {s2_stack_path}"
            return out

        band_ids = [int(b) for b in list(s2_band_indices)]
        if not band_ids:
            out["error"] = "Synthetic PAN requires at least one Sentinel-2 band index."
            return out

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with rasterio.open(s2_stack_path) as src:
            invalid = [b for b in band_ids if b < 1 or b > int(src.count)]
            if invalid:
                out["error"] = (
                    f"Synthetic PAN band index out of range for stack count={int(src.count)}: {invalid}"
                )
                return out

            out_dtype_name = str(src.dtypes[band_ids[0] - 1])
            out_dtype = np.dtype(out_dtype_name)

            src_nodata = src.nodata
            if src_nodata is not None and np.isfinite(float(src_nodata)):
                nodata_value = float(src_nodata)
            else:
                nodata_value = 0.0 if np.issubdtype(out_dtype, np.integer) else float(PROCESSING_NODATA)

            if np.issubdtype(out_dtype, np.integer):
                info = np.iinfo(out_dtype)
                nodata_value = float(np.clip(np.rint(nodata_value), info.min, info.max))

            profile = src.profile.copy()
            profile.update(
                count=1,
                dtype=out_dtype_name,
                nodata=nodata_value,
                compress="LZW",
                BIGTIFF="YES",
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                src_tags = filter_dataset_tags_for_raster_copy(
                    src.tags(),
                    output_band_count=int(dst.count),
                    source_band_count=int(src.count),
                )
                if src_tags:
                    dst.update_tags(**src_tags)
                dst.set_band_description(1, "S2_SYNTHETIC_PAN_B02_B03_B04_B08")

                window_count = 0
                for _, window in src.block_windows(1):
                    window_count += 1
                    block = src.read(band_ids, window=window).astype(np.float64, copy=False)
                    valid = np.isfinite(block)
                    if src_nodata is not None and np.isfinite(float(src_nodata)):
                        valid &= block != float(src_nodata)

                    all_invalid = ~np.any(valid, axis=0)
                    block = np.where(valid, block, np.nan)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            mean_block = np.nanmean(block, axis=0)

                    if np.issubdtype(out_dtype, np.integer):
                        info = np.iinfo(out_dtype)
                        out_block_f = np.full(mean_block.shape, float(nodata_value), dtype=np.float64)
                        valid_mean = (~all_invalid) & np.isfinite(mean_block)
                        if np.any(valid_mean):
                            out_block_f[valid_mean] = np.rint(mean_block[valid_mean])
                        out_block_f = np.clip(out_block_f, info.min, info.max)
                        out_block = out_block_f.astype(out_dtype, copy=False)
                    else:
                        out_block = np.where(
                            (~all_invalid) & np.isfinite(mean_block),
                            mean_block,
                            float(nodata_value),
                        ).astype(out_dtype, copy=False)
                    dst.write(out_block, 1, window=window)

            out["window_count"] = int(window_count)
            out["dtype"] = out_dtype_name
            out["nodata"] = float(nodata_value)

        out["ok"] = True
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _collect_pan_tiepoints_with_synthetic_reference(
    synthetic_s2_pan_path: str,
    pan_source_path: str,
    pan_global_path: str,
    pan_local_path: str,
    ws: Tuple[int, int],
    grid_res: int,
    max_shift: float,
    tiep_filter_level: int,
    local_max_iter: Optional[int],
    reference_nodata: float,
    source_nodata: float,
    arosics_cpus: int = 1,
) -> Dict[str, Any]:
    """Run PAN-specific global+local AROSICS against synthetic S2 PAN reference."""
    import contextlib
    import io

    out: Dict[str, Any] = {
        "ok": False,
        "warnings": [],
        "global_path": pan_source_path,
        "local_path": None,
        "tiepoints_df": None,
        "n_tiepoints_raw": 0,
    }
    effective_arosics_cpus = max(1, int(arosics_cpus))
    supports_kwarg = _supports_constructor_kwarg
    coreg_cls = _compat_attr("COREG")
    coreg_local_cls = _compat_attr("COREG_LOCAL")

    try:
        global_kwargs = {
            "r_b4match": 1,
            "s_b4match": 1,
            "path_out": pan_global_path,
            "fmt_out": "GTiff",
            "out_crea_options": ["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
            "max_shift": float(max(5.0, max_shift)),
            "ws": tuple(ws),
            "resamp_alg_deshift": "cubic",
            "nodata": (float(reference_nodata), float(source_nodata)),
            "progress": False,
            "ignore_errors": True,
            "v": False,
            "q": True,
        }
        if supports_kwarg(coreg_cls, "CPUs"):
            global_kwargs["CPUs"] = int(effective_arosics_cpus)
        with contextlib.redirect_stdout(io.StringIO()):
            crg = coreg_cls(synthetic_s2_pan_path, pan_source_path, **global_kwargs)
            crg.correct_shifts()
        if os.path.exists(pan_global_path):
            out["global_path"] = pan_global_path
    except Exception as exc:
        out["warnings"].append(f"PAN global synthetic-reference alignment failed: {exc}")

    if not os.path.exists(out["global_path"]):
        copy_to_path = _compat_attr("stream_copy_raster_to_path")
        copy_res = copy_to_path(
            source_path=pan_source_path,
            output_path=pan_global_path,
            nodata_fallback=float(source_nodata),
            collect_band_stats=False,
        )
        if not copy_res.get("ok", False):
            out["warnings"].append(
                f"PAN fallback copy after global failure also failed: {copy_res.get('errors', ['unknown'])}"
            )
            return out
        out["global_path"] = pan_global_path

    supports_local_max_iter = supports_kwarg(coreg_local_cls, "max_iter")
    local_kwargs = {
        "grid_res": int(max(30, int(grid_res))),
        "window_size": tuple(ws),
        "path_out": pan_local_path,
        "fmt_out": "GTiff",
        "out_crea_options": ["COMPRESS=LZW", "BIGTIFF=YES", "TILED=YES"],
        "r_b4match": 1,
        "s_b4match": 1,
        "max_shift": float(max(5.0, max_shift)),
        "resamp_alg_deshift": "cubic",
        "tieP_filter_level": int(max(0, int(tiep_filter_level))),
        "outFillVal": float(source_nodata),
        "CPUs": int(effective_arosics_cpus),
        "nodata": (float(reference_nodata), float(source_nodata)),
        "progress": False,
        "v": False,
        "q": True,
        "ignore_errors": True,
    }
    if local_max_iter is not None and supports_local_max_iter:
        local_kwargs["max_iter"] = int(max(1, int(local_max_iter)))

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            crl = coreg_local_cls(
                synthetic_s2_pan_path,
                out["global_path"],
                **local_kwargs,
            )
            crl.correct_shifts()
        tie_points_df = getattr(crl, "CoRegPoints_table", None)
        if tie_points_df is not None and len(tie_points_df) > 0:
            tie_points_df = tie_points_df.copy()
            tie_points_df["BAND_LABEL"] = "S2_SYNTH_PAN"
            if "RELIABILITY" in tie_points_df.columns:
                import pandas as pd

                tie_points_df["RELIABILITY"] = pd.to_numeric(tie_points_df["RELIABILITY"], errors="coerce")
            out["tiepoints_df"] = tie_points_df
            out["n_tiepoints_raw"] = int(len(tie_points_df))
            out["local_path"] = pan_local_path if os.path.exists(pan_local_path) else None
            out["ok"] = True
            return out
        out["warnings"].append("PAN local synthetic-reference matching returned zero tie points.")
        return out
    except Exception as exc:
        out["warnings"].append(f"PAN local synthetic-reference matching failed: {exc}")
        return out


def _normalize_pan_gcp_mode(mode: Any) -> str:
    text = str(mode or "").strip().lower()
    if text in {"map_inverse", "scaled_image"}:
        return text
    return "map_inverse"


def _normalize_pan_dxdy_source(source: Any) -> str:
    text = str(source or "").strip().lower()
    if text in {"auto", "xy_shift_m", "zero"}:
        return text
    return "auto"


def _validate_ancillary_raster(
    path: str,
    target_crs,
    max_windows: int = 64,
    stop_on_first_valid: bool = True,
) -> Dict[str, Any]:
    """Lightweight validation of ancillary output raster georeference and content."""
    import rasterio
    from pyproj import CRS

    out: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "valid_pixels": 0,
        "total_pixels": 0,
        "windows_scanned": 0,
        "truncated": False,
    }
    try:
        if not os.path.exists(path):
            out["error"] = f"Output missing: {path}"
            return out

        expected_crs = CRS.from_user_input(target_crs) if target_crs is not None else None
        with rasterio.open(path) as src:
            if src.count < 1 or src.width < 1 or src.height < 1:
                out["error"] = "Invalid raster dimensions/count."
                return out
            if src.crs is None:
                out["error"] = "Output CRS is missing."
                return out
            if expected_crs is not None and src.crs != expected_crs:
                out["error"] = f"CRS mismatch (expected {expected_crs}, got {src.crs})."
                return out

            nodata_values = resolve_raster_nodata_values(src)

            max_wins = int(max(0, int(max_windows)))
            windows_scanned = 0
            valid_pixels = 0
            total_pixels = 0
            found_valid = False
            for _, window in src.block_windows(1):
                if max_wins > 0 and windows_scanned >= max_wins:
                    out["truncated"] = True
                    break
                band = src.read(1, window=window).astype(np.float32, copy=False)
                total_pixels += int(band.size)
                valid = np.isfinite(band)
                for nodata_candidate in nodata_values:
                    valid &= band != float(nodata_candidate)
                vcount = int(np.count_nonzero(valid))
                valid_pixels += vcount
                windows_scanned += 1
                if bool(stop_on_first_valid) and vcount > 0:
                    out["truncated"] = True
                    found_valid = True
                    break

            out["windows_scanned"] = int(windows_scanned)
            out["valid_pixels"] = int(valid_pixels)
            out["total_pixels"] = int(total_pixels)
            out["ok"] = bool(valid_pixels > 0)
            if not out["ok"]:
                out["error"] = "No valid pixels found in output."
            elif found_valid:
                out["error"] = None
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def _apply_tps_warp_from_gcps(
    input_raster: str,
    output_raster: str,
    gcps,
    target_crs,
    x_res: float,
    y_res: float,
    resampling: str = "near",
    nodata: Optional[float] = None,
    target_aligned_pixels: bool = False,
    target_extent: Optional[Tuple[float, float, float, float]] = None,
    gdalwarp_multi: bool = True,
    gdalwarp_num_threads: str = "ALL_CPUS",
) -> Dict[str, Any]:
    """Apply thin-plate-spline warp from GCPs to raster using GDAL."""
    result = {"success": False, "output_path": None, "error_message": None}
    if not gcps:
        result["error_message"] = "No GCPs provided for TPS warp."
        return result

    temp_vrt = None
    tmp_output = None
    try:
        from osgeo import gdal

        gdal.UseExceptions()
        temp_vrt = _make_gdal_temp_path(output_raster, ".gcps.vrt")
        tmp_output = _make_gdal_temp_path(output_raster, ".tmp.tif")
        ds_in = gdal.Open(input_raster, gdal.GA_ReadOnly)
        if ds_in is None:
            result["error_message"] = f"Failed to open input raster: {input_raster}"
            return result

        driver_vrt = gdal.GetDriverByName("VRT")
        ds_vrt = driver_vrt.CreateCopy(temp_vrt, ds_in)
        crs_wkt = target_crs.to_wkt() if hasattr(target_crs, "to_wkt") else str(target_crs)
        ds_vrt.SetGCPs(gcps, crs_wkt)
        ds_vrt.FlushCache()
        ds_vrt = None
        ds_in = None

        gdalwarp_exe = resolve_gdalwarp_exe()
        build_gdalwarp_tps_command = _compat_attr("_build_gdalwarp_tps_command")
        cmd = build_gdalwarp_tps_command(
            gdalwarp_exe=gdalwarp_exe,
            temp_vrt=temp_vrt,
            output_raster=tmp_output,
            crs_wkt=crs_wkt,
            x_res=x_res,
            y_res=y_res,
            resampling=resampling,
            nodata=nodata,
            target_aligned_pixels=target_aligned_pixels,
            target_extent=target_extent,
            gdalwarp_multi=bool(gdalwarp_multi),
            gdalwarp_num_threads=_normalize_gdalwarp_num_threads(gdalwarp_num_threads),
        )
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        if os.path.exists(tmp_output):
            _validate_warp_output(tmp_output)
            os.replace(tmp_output, output_raster)
            result["success"] = True
            result["output_path"] = output_raster
        else:
            result["error_message"] = "TPS warp completed but output was not created."
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        msg = stderr if stderr else stdout
        result["error_message"] = f"gdalwarp TPS failed: {msg}"
    except Exception as exc:
        result["error_message"] = f"TPS warp failed: {exc}"
    finally:
        if temp_vrt and os.path.exists(temp_vrt):
            try:
                os.remove(temp_vrt)
            except Exception:
                pass
        _cleanup_raster_temp_outputs(tmp_output)

    return result


def _build_tps_gcps_for_source_raster(
    tie_points_df,
    source_raster_path: str,
    min_gcps: int = 10,
) -> Dict[str, Any]:
    """Build raster-specific TPS GCPs from map-space tiepoint shifts."""
    out = {"success": False, "gcps": [], "n_gcps": 0, "error_message": None}
    if tie_points_df is None or len(tie_points_df) == 0:
        out["error_message"] = "No tie points available."
        return out

    required = ["X_MAP", "Y_MAP", "X_SHIFT_M", "Y_SHIFT_M"]
    missing = [c for c in required if c not in tie_points_df.columns]
    if missing:
        out["error_message"] = f"Missing tiepoint columns for TPS: {missing}"
        return out

    try:
        import rasterio
        from osgeo import gdal

        with rasterio.open(source_raster_path) as src:
            inv = ~src.transform
            width = int(src.width)
            height = int(src.height)

        gcps = []
        for _, row in tie_points_df.iterrows():
            try:
                x_map = float(row["X_MAP"])
                y_map = float(row["Y_MAP"])
                dx = float(row["X_SHIFT_M"])
                dy = float(row["Y_SHIFT_M"])
                if not (np.isfinite(x_map) and np.isfinite(y_map) and np.isfinite(dx) and np.isfinite(dy)):
                    continue
                pixel, line = inv * (x_map, y_map)
                if not (np.isfinite(pixel) and np.isfinite(line)):
                    continue
                if pixel < -1 or line < -1 or pixel > (width + 1) or line > (height + 1):
                    continue
                gcps.append(gdal.GCP(x_map + dx, y_map + dy, 0.0, float(pixel), float(line)))
            except Exception:
                continue

        out["gcps"] = gcps
        out["n_gcps"] = len(gcps)
        if len(gcps) >= max(3, int(min_gcps)):
            out["success"] = True
            return out

        out["error_message"] = f"Insufficient TPS GCPs after reprojection: {len(gcps)}"
        return out
    except Exception as exc:
        out["error_message"] = f"Failed to build TPS GCPs for source raster: {exc}"
        return out


def _coregister_enmap_auxiliary_outputs(
    scene_name: str,
    enmap_spectral_image: str,
    folder_struct: Dict[str, str],
    s2_crs,
    best_candidate: Optional[Dict[str, Any]],
    hs_reference_raster_path: Optional[str] = None,
    gdalwarp_multi: bool = True,
    gdalwarp_num_threads: str = "ALL_CPUS",
    coregister_ql: bool = True,
) -> Dict[str, Any]:
    """Generate coregistered EnMAP auxiliary outputs from QL rasters and sidecars."""
    result: Dict[str, Any] = {
        "status": "not_requested",
        "warnings": [],
        "outputs": {
            "pan": None,
            "quality_vnir": None,
            "quality_swir": None,
            "enmap_ql": {},
            "enmap_sidecars": {},
        },
        "warp_method": None,
        "tiepoints_used": 0,
        "ql_coregistration_requested": bool(coregister_ql),
    }
    gdal_threads = _normalize_gdalwarp_num_threads(gdalwarp_num_threads)

    discover_enmap_auxiliary_inputs = _compat_attr("_discover_enmap_auxiliary_inputs")
    aux_inputs = discover_enmap_auxiliary_inputs(enmap_spectral_image)
    ql_rasters_discovered = list(aux_inputs.get("ql_rasters", []) or [])
    ql_rasters = ql_rasters_discovered if coregister_ql else []
    sidecars = dict(aux_inputs.get("sidecars", {}) or {})
    scene_prefix = str(aux_inputs.get("scene_prefix") or "").strip()

    fmt_issue = _fmt_issue
    if not ql_rasters_discovered and not sidecars:
        result["status"] = "degraded"
        result["warnings"].append(
            fmt_issue(
                "ANCILLARY",
                f"No EnMAP auxiliary inputs found for scene prefix derived from: {enmap_spectral_image}",
            )
        )
        return result

    result["status"] = "ok"
    if not sidecars:
        result["status"] = "degraded"
        result["warnings"].append(
            fmt_issue(
                "ANCILLARY",
                "No EnMAP XML/HDR sidecars were discovered for provenance copy.",
            )
        )
    if coregister_ql and not ql_rasters:
        result["status"] = "degraded"
        result["warnings"].append(
            fmt_issue(
                "ANCILLARY",
                "No EnMAP QL rasters were discovered; only sidecar copy will be attempted.",
            )
        )

    coreg_aux_dir = os.path.join(folder_struct["coreg"], "enmap_aux")
    reports_dir = folder_struct.get("reports")
    metadata_dir = (
        os.path.join(reports_dir, "provenance", "enmap_metadata")
        if reports_dir
        else os.path.join(coreg_aux_dir, "metadata")
    )
    os.makedirs(coreg_aux_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    for sidecar_key, sidecar_src in sidecars.items():
        try:
            if not sidecar_src or not os.path.isfile(sidecar_src):
                raise RuntimeError(f"Missing sidecar source: {sidecar_src}")
            sidecar_dst = os.path.join(metadata_dir, os.path.basename(sidecar_src))
            shutil.copy2(sidecar_src, sidecar_dst)
            result["outputs"]["enmap_sidecars"][sidecar_key] = sidecar_dst
        except Exception as exc:
            result["status"] = "degraded"
            result["warnings"].append(
                fmt_issue("ANCILLARY", f"Failed to copy EnMAP sidecar '{sidecar_key}': {exc}")
            )

    if not coregister_ql:
        result["warp_method"] = "not_requested"
        return result

    base_tie_points_df = None if best_candidate is None else best_candidate.get("local_tiepoints_df")
    if base_tie_points_df is None or len(base_tie_points_df) == 0:
        if ql_rasters:
            result["status"] = "degraded"
            result["warnings"].append(
                fmt_issue(
                    "ANCILLARY",
                    "No accepted base-scene tie-point table available; skipped EnMAP QL ancillary coregistration.",
                )
            )
        return result

    target_crs = s2_crs
    target_extent = None
    target_x_res: Optional[float] = None
    target_y_res: Optional[float] = None
    if hs_reference_raster_path and os.path.exists(hs_reference_raster_path):
        try:
            import rasterio

            with rasterio.open(hs_reference_raster_path) as ref_src:
                if ref_src.crs is not None:
                    target_crs = ref_src.crs
                bounds = ref_src.bounds
                target_extent = (
                    float(bounds.left),
                    float(bounds.bottom),
                    float(bounds.right),
                    float(bounds.top),
                )
            target_x_res, target_y_res = _infer_raster_native_resolution(
                hs_reference_raster_path,
                fallback=30.0,
            )
        except Exception as exc:
            result["status"] = "degraded"
            result["warnings"].append(
                fmt_issue(
                    "ANCILLARY",
                    f"Failed to derive EnMAP ancillary target grid from coreg reference raster: {exc}",
                )
            )

    for src_path in ql_rasters:
        src_name = os.path.basename(src_path)
        src_stem = Path(src_name).stem
        if scene_prefix and src_stem.lower().startswith(f"{scene_prefix.lower()}-"):
            suffix_token = src_stem[len(scene_prefix) + 1 :]
        else:
            suffix_token = src_stem
        out_name = f"{scene_name}_{suffix_token}_coreg.tif"
        out_path = os.path.join(coreg_aux_dir, out_name)

        try:
            gcp_result = _build_tps_gcps_for_source_raster(
                base_tie_points_df,
                src_path,
                min_gcps=10,
            )
            if not gcp_result.get("success", False):
                raise RuntimeError(gcp_result.get("error_message", "failed to derive TPS GCPs"))

            result["tiepoints_used"] = max(
                int(result.get("tiepoints_used", 0)),
                int(gcp_result.get("n_gcps", 0) or 0),
            )

            nodata_val = None
            if target_x_res is None or target_y_res is None:
                src_x_res, src_y_res = _infer_raster_native_resolution(src_path, fallback=30.0)
            else:
                src_x_res, src_y_res = target_x_res, target_y_res
            import rasterio

            with rasterio.open(src_path) as src_ds:
                src_nodata = src_ds.nodata
                if src_nodata is not None and np.isfinite(float(src_nodata)):
                    nodata_val = float(src_nodata)

            warp_result = _apply_tps_warp_from_gcps(
                input_raster=src_path,
                output_raster=out_path,
                gcps=gcp_result.get("gcps", []),
                target_crs=target_crs,
                x_res=float(src_x_res),
                y_res=float(src_y_res),
                resampling="near",
                nodata=nodata_val,
                target_aligned_pixels=bool(target_extent is not None),
                target_extent=target_extent,
                gdalwarp_multi=bool(gdalwarp_multi),
                gdalwarp_num_threads=gdal_threads,
            )
            if not warp_result.get("success", False):
                raise RuntimeError(warp_result.get("error_message", "unknown EnMAP auxiliary warp error"))

            aux_check = _validate_ancillary_raster(out_path, target_crs)
            if not aux_check.get("ok", False):
                raise RuntimeError(aux_check.get("error", "EnMAP ancillary output validation failed"))

            artifact_role = (
                ARTIFACT_ROLE_QUALITY_MASK
                if "PIXELMASK" in str(src_name).upper()
                else ARTIFACT_ROLE_NO_PIPELINE_SIDECAR
            )
            sidecar_result = _finalize_pipeline_sidecars(
                tif_path=out_path,
                sensor_type="ENMAP",
                artifact_role=artifact_role,
                wl=None,
                fwhm=None,
                band_names=None,
                band_detectors=None,
                strict_metadata=True,
            )
            if not sidecar_result.get("ok", False):
                err = "; ".join(
                    list(sidecar_result.get("errors", []) or ["unknown ancillary metadata error"])
                )
                raise RuntimeError(err)

            result["outputs"]["enmap_ql"][src_name] = out_path
            result["warp_method"] = "tps_from_base_tiepoints_near"
        except Exception as exc:
            result["status"] = "degraded"
            result["warnings"].append(fmt_issue("ANCILLARY", f"EnMAP auxiliary '{src_name}' failed: {exc}"))

    if ql_rasters and not result["outputs"]["enmap_ql"]:
        result["status"] = "degraded"
        result["warnings"].append(
            fmt_issue("ANCILLARY", "No EnMAP QL ancillary outputs were successfully generated.")
        )

    return result


def _coregister_prisma_ancillary_outputs(
    scene_name: str,
    folder_struct: Dict[str, str],
    s2_crs,
    save_pan: bool,
    save_quality_mask: bool,
    pan_data: Optional[np.ndarray],
    pan_geo_info: Optional[Dict[str, Any]],
    vnir_quality_data: Optional[np.ndarray],
    swir_quality_data: Optional[np.ndarray],
    lat_qm: Optional[np.ndarray],
    lon_qm: Optional[np.ndarray],
    best_candidate: Optional[Dict[str, Any]],
    hs_reference_raster_path: Optional[str] = None,
    s2_reference_raster_path: Optional[str] = None,
    matcher_profile: Optional[Dict[str, Any]] = None,
    pan_gcp_mode: str = "map_inverse",
    pan_map_dxdy_source: str = "auto",
    pan_target_aligned_pixels: bool = False,
    pan_use_synthetic_reference: bool = True,
    pan_min_points_for_poly2: int = 20,
    pan_local_window_size: Tuple[int, int] = (512, 512),
    pan_local_grid_res: int = 30,
    pan_local_max_shift: float = 220.0,
    pan_local_tieP_filter_level: int = 1,
    pan_local_max_iter: Optional[int] = None,
    pan_residual_check: bool = False,
    pan_residual_threshold_px: float = 0.5,
    pan_residual_max_dim: int = 1024,
    use_geolocation_mesh_affine: bool = False,
    geolocation_mesh_stride: int = 32,
    arosics_cpus: int = 1,
    gdalwarp_multi: bool = True,
    gdalwarp_num_threads: str = "ALL_CPUS",
    radiometric_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate coregistered PRISMA ancillary outputs (PAN and quality masks)."""
    fmt_issue = _fmt_issue
    norm_pan_mode = _normalize_pan_gcp_mode
    norm_pan_dxdy_fn = _normalize_pan_dxdy_source

    result: Dict[str, Any] = {
        "status": "not_requested",
        "warnings": [],
        "outputs": {
            "pan": None,
            "quality_vnir": None,
            "quality_swir": None,
            "enmap_ql": {},
            "enmap_sidecars": {},
        },
        "warp_method": None,
        "tiepoints_used": 0,
        "pan_gcp_mode": norm_pan_mode(pan_gcp_mode),
        "pan_target_aligned_pixels": bool(pan_target_aligned_pixels),
        "pan_residual_check": {"enabled": bool(pan_residual_check), "ok": None},
    }
    if not (save_pan or save_quality_mask):
        return result

    result["status"] = "ok"
    base_tie_points_df = None if best_candidate is None else best_candidate.get("local_tiepoints_df")
    norm_pan_gcp_mode = norm_pan_mode(pan_gcp_mode)
    norm_pan_dxdy = norm_pan_dxdy_fn(pan_map_dxdy_source)
    min_pan_points_for_poly2 = int(max(6, int(pan_min_points_for_poly2)))
    default_arosics_cpus = int(CPUS_FOR_AROSICS)
    local_grid_res_m = int(LOCAL_GRID_RES_M)
    processing_nodata = float(PROCESSING_NODATA)
    effective_arosics_cpus = _resolve_arosics_cpu_count(
        arosics_cpus,
        default=default_arosics_cpus,
    )
    gdal_threads = _normalize_gdalwarp_num_threads(gdalwarp_num_threads)

    try:
        pan_grid_res = int(max(30, int(pan_local_grid_res)))
    except Exception:
        pan_grid_res = int(max(30, local_grid_res_m))
    try:
        pan_shift = float(max(5.0, float(pan_local_max_shift)))
    except Exception:
        pan_shift = 220.0
    try:
        pan_tiep_filter = int(max(0, int(pan_local_tieP_filter_level)))
    except Exception:
        pan_tiep_filter = 1
    pan_max_iter = pan_local_max_iter
    if pan_max_iter is None and isinstance(matcher_profile, dict):
        pan_max_iter = matcher_profile.get("local_max_iter")

    pan_dir = os.path.join(folder_struct["coreg"], "pan")
    quality_dir = os.path.join(folder_struct["coreg"], "quality")
    os.makedirs(pan_dir, exist_ok=True)
    os.makedirs(quality_dir, exist_ok=True)

    if save_pan:
        pan_out = os.path.join(pan_dir, f"{scene_name}_pan_coreg.tif")
        pan_src = os.path.join(folder_struct["temp"], f"{scene_name}_PAN_src.tif")
        synthetic_s2_pan_path = os.path.join(folder_struct["temp"], f"{scene_name}_S2_SYN_PAN.tif")
        pan_global_path = os.path.join(folder_struct["temp"], f"{scene_name}_PAN_GLOBAL_SYN.tif")
        pan_local_path = os.path.join(folder_struct["temp"], f"{scene_name}_PAN_LOCAL_SYN.tif")
        pan_temp_paths = [synthetic_s2_pan_path, pan_global_path, pan_local_path, pan_src]
        try:
            if pan_data is None or pan_geo_info is None:
                raise RuntimeError("PAN source data unavailable.")
            pan_arr = np.asarray(np.squeeze(pan_data), dtype=np.float32)
            if pan_arr.ndim != 2:
                raise RuntimeError(f"PAN array must be 2D; got shape {tuple(pan_arr.shape)}.")
            rows, cols = pan_arr.shape
            transform_pan = _estimate_transform_from_corner_coords(
                {**dict(pan_geo_info), "rows": rows, "cols": cols},
                s2_crs,
            )
            pan_clean = np.where(np.isfinite(pan_arr), pan_arr, processing_nodata).astype(np.float32)
            _write_georeferenced_raster(
                pan_src,
                pan_clean[np.newaxis, :, :],
                s2_crs,
                transform_pan,
                dtype="float32",
                nodata=processing_nodata,
            )
            import rasterio

            with rasterio.open(pan_src, "r+") as pan_src_dst:
                _runtime._apply_prisma_radiometric_tags(
                    pan_src_dst,
                    radiometric_contract,
                    ["PAN"],
                )
            pan_xres, pan_yres = _infer_raster_native_resolution(
                pan_src,
                fallback=float(pan_geo_info.get("pixel_size_m", 5.0) or 5.0),
            )
            pan_tie_points_df = None
            pan_warp_source = pan_src

            if bool(pan_use_synthetic_reference):
                if not s2_reference_raster_path or not os.path.exists(s2_reference_raster_path):
                    raise RuntimeError("Sentinel-2 reference stack unavailable for synthetic PAN branch.")

                synth_result = _create_synthetic_s2_pan(
                    s2_stack_path=s2_reference_raster_path,
                    output_path=synthetic_s2_pan_path,
                    s2_band_indices=(
                        int(MULTIBAND_S2_WAVELENGTHS["B02"]["stack_idx"]),
                        int(MULTIBAND_S2_WAVELENGTHS["B03"]["stack_idx"]),
                        int(MULTIBAND_S2_WAVELENGTHS["B04"]["stack_idx"]),
                        int(MULTIBAND_S2_WAVELENGTHS["B08"]["stack_idx"]),
                    ),
                )
                if not synth_result.get("ok", False):
                    raise RuntimeError(synth_result.get("error", "failed to create synthetic S2 PAN"))

                import rasterio

                with rasterio.open(synthetic_s2_pan_path) as s2_pan_ref:
                    ref_nodata = s2_pan_ref.nodata
                    if ref_nodata is None or not np.isfinite(float(ref_nodata)):
                        ref_nodata = 0.0
                    else:
                        ref_nodata = float(ref_nodata)

                pan_ws = _resolve_pan_window_size_for_raster(
                    pan_src,
                    requested_window_size=pan_local_window_size,
                )
                pan_tp_result = _collect_pan_tiepoints_with_synthetic_reference(
                    synthetic_s2_pan_path=synthetic_s2_pan_path,
                    pan_source_path=pan_src,
                    pan_global_path=pan_global_path,
                    pan_local_path=pan_local_path,
                    ws=pan_ws,
                    grid_res=pan_grid_res,
                    max_shift=pan_shift,
                    tiep_filter_level=pan_tiep_filter,
                    local_max_iter=pan_max_iter,
                    reference_nodata=float(ref_nodata),
                    source_nodata=float(processing_nodata),
                    arosics_cpus=int(effective_arosics_cpus),
                )
                for note in pan_tp_result.get("warnings", []):
                    result["warnings"].append(fmt_issue("ANCILLARY", str(note)))
                pan_warp_source = str(pan_tp_result.get("global_path") or pan_src)
                pan_tie_points_df = pan_tp_result.get("tiepoints_df")

            if pan_tie_points_df is None or len(pan_tie_points_df) == 0:
                pan_tie_points_df = base_tie_points_df
                if pan_tie_points_df is not None and len(pan_tie_points_df) > 0:
                    result["warnings"].append(
                        fmt_issue(
                            "ANCILLARY",
                            "PAN synthetic-reference tie points unavailable; falling back to base-scene tie points.",
                        )
                    )

            if pan_tie_points_df is None or len(pan_tie_points_df) == 0:
                raise RuntimeError("No tie-point table available for PAN ancillary warp.")

            min_order_fn = coreg_math._minimum_gcps_for_polynomial_order
            merged_pan = coreg_math._merge_tiepoints(
                [pan_tie_points_df],
                min_reliability=MIN_RELIABILITY_THRESHOLD,
                trim_by_residual=True,
                residual_mad_factor=3.0,
                min_band_support=1,
                allow_single_band_fallback=True,
                consensus_group_rounding_px=1.0,
                grid_rows=4,
                grid_cols=4,
                max_points_per_cell=4,
                min_points_required=max(min_order_fn(1), min_pan_points_for_poly2),
                spatial_fallback_min_distance=60.0,
            )
            pan_selected_df = merged_pan.get("merged_df")
            if pan_selected_df is None or len(pan_selected_df) == 0:
                pan_selected_df = pan_tie_points_df

            result["tiepoints_used"] = int(len(pan_selected_df))
            if "fallback_notes" in merged_pan:
                for note in merged_pan.get("fallback_notes", []):
                    logger.info("PAN tie-point merge fallback: %s", note)

            pan_res = float(0.5 * (pan_xres + pan_yres))
            if norm_pan_gcp_mode == "scaled_image":
                if hs_reference_raster_path and os.path.exists(hs_reference_raster_path):
                    hs_xres, hs_yres = _infer_raster_native_resolution(
                        hs_reference_raster_path,
                        fallback=30.0,
                    )
                    hs_res = float(0.5 * (hs_xres + hs_yres))
                else:
                    hs_res = 30.0
                    result["warnings"].append(
                        fmt_issue(
                            "ANCILLARY",
                            "HS reference path missing for PAN scaled-image GCP mode; using 30m fallback.",
                        )
                    )
                build_pan_gcps = _compat_attr("build_pan_gcps_from_tiepoints")
                pan_gcps = build_pan_gcps(
                    tiepoints_df=pan_selected_df,
                    hs_pixel_size_m=hs_res,
                    pan_pixel_size_m=pan_res,
                    map_dx_dy_source=norm_pan_dxdy,
                    nodata=processing_nodata,
                    crs_wkt_or_epsg=s2_crs.to_wkt() if hasattr(s2_crs, "to_wkt") else str(s2_crs),
                    pan_shape=(rows, cols),
                )
            else:
                pan_gcp_result = _build_tps_gcps_for_source_raster(
                    pan_selected_df,
                    pan_warp_source,
                    min_gcps=min_order_fn(1),
                )
                if not pan_gcp_result.get("success", False):
                    raise RuntimeError(
                        pan_gcp_result.get("error_message", "failed to derive PAN polynomial GCPs")
                    )
                pan_gcps = pan_gcp_result.get("gcps", [])

            if len(pan_gcps) < min_order_fn(1):
                raise RuntimeError(
                    f"Insufficient PAN GCPs for affine warp: {len(pan_gcps)} < {min_order_fn(1)}"
                )

            poly_decision = coreg_math._decide_polynomial_order(
                merged_df=pan_selected_df,
                preferred_order=2,
                auto_downgrade=True,
                min_gcps_order2=max(min_order_fn(2), min_pan_points_for_poly2),
                min_cells_order2=4,
                grid_rows=4,
                grid_cols=4,
            )
            pan_order = int(max(1, int(poly_decision.get("order_used", 2))))
            if len(pan_gcps) < int(min_pan_points_for_poly2):
                pan_order = 1
                logger.warning(
                    "PAN GCP density low (%d < %d). Forcing order-1 affine warp (disabling order-2/TPS).",
                    int(len(pan_gcps)),
                    int(min_pan_points_for_poly2),
                )

            pan_warp = _apply_polynomial_warp(
                input_raster=pan_warp_source,
                output_raster=pan_out,
                gcps=pan_gcps,
                target_crs=s2_crs,
                polynomial_order=pan_order,
                output_resolution=pan_res,
                nodata=processing_nodata,
                resampling="cubic",
                gdalwarp_multi=bool(gdalwarp_multi),
                gdalwarp_num_threads=gdal_threads,
            )
            if not pan_warp.get("success", False) and pan_order > 1:
                logger.warning(
                    "PAN order-%d warp failed (%s). Retrying with order-1 affine.",
                    int(pan_order),
                    pan_warp.get("error_message", "unknown error"),
                )
                pan_warp = _apply_polynomial_warp(
                    input_raster=pan_warp_source,
                    output_raster=pan_out,
                    gcps=pan_gcps,
                    target_crs=s2_crs,
                    polynomial_order=1,
                    output_resolution=pan_res,
                    nodata=processing_nodata,
                    resampling="cubic",
                    gdalwarp_multi=bool(gdalwarp_multi),
                    gdalwarp_num_threads=gdal_threads,
                )
                pan_order = 1
            if not pan_warp.get("success", False):
                raise RuntimeError(pan_warp.get("error_message", "unknown PAN warp error"))

            with rasterio.open(pan_out, "r+") as pan_out_dst:
                _runtime._apply_prisma_radiometric_tags(
                    pan_out_dst,
                    radiometric_contract,
                    ["PAN"],
                )

            pan_check = _validate_ancillary_raster(pan_out, s2_crs)
            if not pan_check.get("ok", False):
                raise RuntimeError(pan_check.get("error", "PAN output validation failed"))
            pan_sidecar_result = _finalize_pipeline_sidecars(
                tif_path=pan_out,
                sensor_type="PRISMA",
                artifact_role=ARTIFACT_ROLE_NO_PIPELINE_SIDECAR,
                wl=None,
                fwhm=None,
                band_names=None,
                band_detectors=None,
                strict_metadata=True,
            )
            if not pan_sidecar_result.get("ok", False):
                err = "; ".join(list(pan_sidecar_result.get("errors", []) or ["unknown PAN metadata error"]))
                raise RuntimeError(err)
            result["outputs"]["pan"] = pan_out
            result["warp_method"] = (
                f"poly_order_{int(pan_order)}_synthetic_s2_pan"
                if bool(pan_use_synthetic_reference)
                else f"poly_order_{int(pan_order)}_base_tiepoints"
            )
            result["pan_gcp_mode"] = norm_pan_gcp_mode

            if bool(pan_residual_check) and hs_reference_raster_path and os.path.exists(hs_reference_raster_path):
                residual = coreg_math._estimate_translation_phasecorr(
                    reference_raster_path=hs_reference_raster_path,
                    candidate_raster_path=pan_out,
                    max_dim=int(max(128, int(pan_residual_max_dim))),
                    reference_band=1,
                    candidate_band=1,
                    reference_nodata=processing_nodata,
                    candidate_nodata=processing_nodata,
                )
                result["pan_residual_check"] = {"enabled": True, **residual}
                if residual.get("ok", False):
                    shift_mag = float(residual.get("shift_magnitude_px", 0.0) or 0.0)
                    if shift_mag > float(max(0.0, pan_residual_threshold_px)):
                        result["warnings"].append(
                            fmt_issue(
                                "ANCILLARY",
                                (
                                    f"PAN residual shift estimate is high ({shift_mag:.2f}px > "
                                    f"{float(pan_residual_threshold_px):.2f}px). "
                                    "Try pan_gcp_mode='scaled_image' and pan_target_aligned_pixels=True."
                                ),
                            )
                        )
                else:
                    result["warnings"].append(
                        fmt_issue(
                            "ANCILLARY",
                            f"PAN residual check unavailable: {residual.get('error', 'unknown error')}",
                        )
                    )
        except Exception as exc:
            result["status"] = "degraded"
            result["warnings"].append(fmt_issue("ANCILLARY", f"PAN ancillary output failed: {exc}"))
        finally:
            for tmp_path in pan_temp_paths:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    if save_quality_mask:
        if base_tie_points_df is None or len(base_tie_points_df) == 0:
            result["status"] = "degraded"
            result["warnings"].append(
                fmt_issue(
                    "ANCILLARY",
                    "No accepted base-scene tie-point table available; skipped quality-mask ancillary coregistration.",
                )
            )
            return result
        quality_items = [
            ("VNIR", vnir_quality_data, "quality_vnir", f"{scene_name}_quality_vnir_coreg.tif"),
            ("SWIR", swir_quality_data, "quality_swir", f"{scene_name}_quality_swir_coreg.tif"),
        ]
        for label, mask_data, key, out_name in quality_items:
            src_path = os.path.join(folder_struct["temp"], f"{scene_name}_{label}_quality_src.tif")
            out_path = os.path.join(quality_dir, out_name)
            try:
                if mask_data is None:
                    raise RuntimeError(f"{label} quality matrix unavailable.")
                if lat_qm is None or lon_qm is None:
                    raise RuntimeError("Quality geolocation arrays unavailable.")
                lat_arr = np.asarray(lat_qm)
                lon_arr = np.asarray(lon_qm)
                if lat_arr.shape != lon_arr.shape or lat_arr.ndim != 2:
                    raise RuntimeError(
                        f"Invalid quality geolocation shape lat={lat_arr.shape}, lon={lon_arr.shape}."
                    )
                rows, cols = lat_arr.shape
                quality_bands = _prepare_bands_first(mask_data, rows, cols, f"{label} quality")
                quality_bands = np.where(np.isfinite(quality_bands), quality_bands, 255)
                quality_bands = np.clip(quality_bands, 0, 255).astype(np.uint8)

                transform_qm = _estimate_prisma_geotransform_safe(
                    lon_arr,
                    lat_arr,
                    rows,
                    cols,
                    s2_crs,
                    use_geolocation_mesh=use_geolocation_mesh_affine,
                    geolocation_mesh_stride=geolocation_mesh_stride,
                )
                _write_georeferenced_raster(
                    src_path,
                    quality_bands,
                    s2_crs,
                    transform_qm,
                    dtype="uint8",
                    nodata=255,
                )
                qm_gcp_result = _build_tps_gcps_for_source_raster(base_tie_points_df, src_path, min_gcps=10)
                if not qm_gcp_result.get("success", False):
                    raise RuntimeError(qm_gcp_result.get("error_message", f"failed to derive {label} TPS GCPs"))
                qm_xres, qm_yres = _infer_raster_native_resolution(src_path, fallback=30.0)
                qm_warp = _apply_tps_warp_from_gcps(
                    src_path,
                    out_path,
                    qm_gcp_result.get("gcps", []),
                    s2_crs,
                    x_res=qm_xres,
                    y_res=qm_yres,
                    resampling="near",
                    nodata=255,
                    gdalwarp_multi=bool(gdalwarp_multi),
                    gdalwarp_num_threads=gdal_threads,
                )
                if not qm_warp.get("success", False):
                    raise RuntimeError(qm_warp.get("error_message", f"unknown {label} quality warp error"))
                qm_check = _validate_ancillary_raster(out_path, s2_crs)
                if not qm_check.get("ok", False):
                    raise RuntimeError(qm_check.get("error", f"{label} quality output validation failed"))
                sidecar_result = _finalize_pipeline_sidecars(
                    tif_path=out_path,
                    sensor_type="PRISMA",
                    artifact_role=ARTIFACT_ROLE_QUALITY_MASK,
                    wl=None,
                    fwhm=None,
                    band_names=None,
                    band_detectors=None,
                    strict_metadata=True,
                )
                if not sidecar_result.get("ok", False):
                    err = "; ".join(
                        list(sidecar_result.get("errors", []) or ["unknown quality metadata error"])
                    )
                    raise RuntimeError(err)
                result["outputs"][key] = out_path
            except Exception as exc:
                result["status"] = "degraded"
                result["warnings"].append(fmt_issue("ANCILLARY", f"{label} quality ancillary output failed: {exc}"))

    return result


__all__ = [
    "_apply_polynomial_warp",
    "_apply_tps_warp_from_gcps",
    "_build_band_display_labels",
    "_build_detector_branch_plan",
    "_build_gdalwarp_tps_command",
    "_build_internal_overviews",
    "_build_tiepoint_legend_handles",
    "_build_tps_gcps_for_source_raster",
    "_build_vrt_with_band_order",
    "_build_s2_stack",
    "_coregister_enmap_auxiliary_outputs",
    "_coregister_prisma_ancillary_outputs",
    "_collect_pan_tiepoints_with_synthetic_reference",
    "_create_synthetic_s2_pan",
    "_crs_equivalent",
    "_discover_enmap_auxiliary_inputs",
    "_ensure_abs_shift_column",
    "_estimate_transform_from_corner_coords",
    "_export_displacement_shapefile_from_df",
    "_finalize_coreg_output",
    "_finalize_pipeline_sidecars",
    "_remove_sidecar_if_exists",
    "_generate_mandatory_quicklooks",
    "_harmonize_detector_branch_grids",
    "_infer_detector_from_band_name",
    "_prepare_enmap_processing_source",
    "_probe_raster_valid_pixels",
    "_promote_s2_stack",
    "_prepare_bands_first",
    "_recombine_detector_branches_windowed",
    "_reproject_reference_stack_to_target_crs",
    "_resolve_enmap_band_selection",
    "_resolve_pan_window_size_for_raster",
    "_resolve_quicklook_rgb_bands",
    "_resample_raster_to_shared_grid",
    "_sanitize_raster_nonfinite_inplace",
    "_save_precoreg_output",
    "_stream_copy_raster_with_band_order",
    "_summarize_raster_grid",
    "_transforms_equivalent",
    "_validate_ancillary_raster",
    "_validate_coreg_raster_content",
    "_validate_local_s2_stack_override",
    "_write_branch_raster_windowed",
    "_write_displacement_vector_cartography_png",
    "_write_envi_header",
    "_write_georeferenced_raster",
    "_write_geotiff_band_metadata",
    "_write_pam_aux_xml",
    "_write_tiepoint_quicklook_png",
    "_compute_band_statistics",
    "build_pan_gcps_from_tiepoints",
]

_LOCAL_EXPORTS = {
    "_apply_polynomial_warp",
    "_apply_tps_warp_from_gcps",
    "_build_tps_gcps_for_source_raster",
    "_collect_pan_tiepoints_with_synthetic_reference",
    "_coregister_enmap_auxiliary_outputs",
    # Keep the exported PRISMA ancillary path bound to runtime's canonical
    # implementation, which owns radiometric tags and active scale/offsets.
    "_crs_equivalent",
    "_compute_band_statistics",
    "_create_synthetic_s2_pan",
    "_estimate_transform_from_corner_coords",
    "_infer_detector_from_band_name",
    "_infer_raster_native_resolution",
    "_normalize_gdalwarp_num_threads",
    "_resolve_arosics_cpu_count",
    "_prepare_bands_first",
    "_resolve_pan_window_size_for_raster",
    "_resolve_quicklook_rgb_bands",
    "_validate_ancillary_raster",
    "_validate_local_s2_stack_override",
    "_write_georeferenced_raster",
    "_normalize_pan_gcp_mode",
    "_normalize_pan_dxdy_source",
    "_finalize_pipeline_sidecars",
    "_probe_raster_valid_pixels",
    "_remove_sidecar_if_exists",
    "_sanitize_raster_nonfinite_inplace",
    "_summarize_raster_grid",
    "_transforms_equivalent",
}

for _name in __all__:
    if _name in _LOCAL_EXPORTS:
        continue
    try:
        globals()[_name] = getattr(_runtime, _name)
    except AttributeError:
        continue
