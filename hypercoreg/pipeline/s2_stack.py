"""Sentinel-2 stack discovery, caching, and preparation helpers."""

from __future__ import annotations

from hypercoreg.pipeline import runtime as _runtime

__all__ = [
    "_acquire_cache_lock",
    "_apply_s2_band_subset_overrides",
    "_build_s2_cache_key",
    "_build_s2_stack",
    "_build_s2_stack_vrt",
    "_coerce_scl_exclude_classes",
    "_copy_s2_stack_with_mask",
    "_default_cache_root",
    "_destination_window_source_read",
    "_expected_s2_l2a_output_band_count",
    "_format_expected_s2_l2a_bands",
    "_normalise_s2_band_labels",
    "_promote_s2_stack",
    "_release_cache_lock",
    "_reproject_reference_stack_to_target_crs",
    "_reproject_source_window_to_destination",
    "_require_s2_l2a_band_paths",
    "_require_s2_l2a_output_band_count",
    "_resolve_cache_dir",
    "_resolve_s2_zip_band_paths",
    "_s2_platform_from_product_name",
    "_s2_valid_mask_path",
    "_s2_vrt_zip_dependency_info",
    "_validate_local_s2_stack_override",
    "_vsizip_path",
]

for _name in __all__:
    globals()[_name] = getattr(_runtime, _name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
