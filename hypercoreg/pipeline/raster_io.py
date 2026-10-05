"""Raster and file-backed coregistration helpers."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

import numpy as np
from affine import Affine

__all__ = [
    "ARTIFACT_ROLE_MAIN_COREG_SPECTRAL",
    "ARTIFACT_ROLE_NO_PIPELINE_SIDECAR",
    "ARTIFACT_ROLE_PRE_COREG_SPECTRAL",
    "ARTIFACT_ROLE_QUALITY_MASK",
    "PIPELINE_HDR_ARTIFACT_ROLES",
    "_apply_polynomial_warp",
    "_apply_tps_warp_from_gcps",
    "_build_band_display_labels",
    "_build_detector_branch_plan",
    "_build_gdalwarp_tps_command",
    "_build_internal_overviews",
    "_build_s2_stack",
    "_build_tiepoint_legend_handles",
    "_build_tps_gcps_for_source_raster",
    "_build_vrt_with_band_order",
    "_cleanup_raster_temp_outputs",
    "_collect_pan_tiepoints_with_synthetic_reference",
    "_compute_band_statistics",
    "_coregister_enmap_auxiliary_outputs",
    "_coregister_prisma_ancillary_outputs",
    "_create_synthetic_s2_pan",
    "_crs_equivalent",
    "_discover_enmap_auxiliary_inputs",
    "_ensure_abs_shift_column",
    "_estimate_prisma_geotransform_safe",
    "_estimate_transform_from_corner_coords",
    "_export_displacement_shapefile_from_df",
    "_fallback_rgb_band_indices",
    "_finalize_coreg_output",
    "_finalize_pipeline_sidecars",
    "_finite_float_or_none",
    "_fmt_issue",
    "_generate_mandatory_quicklooks",
    "_harmonize_detector_branch_grids",
    "_infer_detector_from_band_name",
    "_infer_raster_native_resolution",
    "_iter_valid_band_values",
    "_make_gdal_temp_path",
    "_normalize_gdalwarp_num_threads",
    "_normalize_pan_dxdy_source",
    "_normalize_pan_gcp_mode",
    "_normalize_s2_band_label",
    "_pick_unique_band_index_for_wavelength",
    "_prepare_bands_first",
    "_prepare_enmap_processing_source",
    "_probe_raster_valid_pixels",
    "_promote_s2_stack",
    "_recombine_detector_branches_windowed",
    "_remove_sidecar_if_exists",
    "_reproject_reference_stack_to_target_crs",
    "_resample_raster_to_shared_grid",
    "_resolve_arosics_cpu_count",
    "_resolve_enmap_band_selection",
    "_resolve_pan_window_size_for_raster",
    "_resolve_quicklook_rgb_bands",
    "_resolve_selected_band_values",
    "_sanitize_raster_nonfinite_inplace",
    "_save_precoreg_output",
    "_stream_copy_raster_with_band_order",
    "_summarize_raster_grid",
    "_supports_constructor_kwarg",
    "_transforms_equivalent",
    "_validate_ancillary_raster",
    "_validate_coreg_raster_content",
    "_validate_local_s2_stack_override",
    "_validate_warp_output",
    "_write_branch_raster_windowed",
    "_write_displacement_vector_cartography_png",
    "_write_envi_header",
    "_write_georeferenced_raster",
    "_write_geotiff_band_metadata",
    "_write_pam_aux_xml",
    "_write_tiepoint_quicklook_png",
    "build_pan_gcps_from_tiepoints",
]

# Defined only in this module (no runtime.py equivalent); everything else
# in __all__ resolves to hypercoreg.pipeline.runtime on attribute access.
_LOCAL_NAMES = frozenset({
    "_estimate_prisma_geotransform_safe",
})

logger = logging.getLogger("COREG_PROCESSING")


def _runtime():
    # Imported lazily: runtime pulls in AROSICS/geoarray and is edited independently,
    # so names are looked up on every access instead of being copied at import time.
    return import_module("hypercoreg.pipeline.runtime")


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


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_runtime(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
