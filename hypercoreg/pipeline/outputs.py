"""Scene output, metadata, quicklook, and cleanup helpers."""

from __future__ import annotations

from hypercoreg.pipeline import runtime as _runtime

__all__ = [
    "_build_internal_overviews",
    "_build_scene_output_dir",
    "_build_scene_output_root",
    "_cleanup_temp_folder",
    "_create_scene_folder_structure",
    "_export_displacement_shapefile_from_df",
    "_finalize_coreg_output",
    "_finalize_pipeline_sidecars",
    "_generate_mandatory_quicklooks",
    "_prepare_output_metadata",
    "_record_batch_scene_result",
    "_record_stage_timing",
    "_save_precoreg_output",
    "_validate_coreg_raster_content",
    "_write_displacement_vector_cartography_png",
    "_write_geotiff_band_metadata",
    "_write_pam_aux_xml",
    "_write_tiepoint_quicklook_png",
]

for _name in __all__:
    globals()[_name] = getattr(_runtime, _name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
