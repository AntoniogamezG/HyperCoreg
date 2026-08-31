"""
Data readers for hyperspectral sensors.

This package provides readers for PRISMA and EnMAP hyperspectral data.
"""

from hypercoreg.readers.prisma import (
    read_prisma_cube_and_meta,
    read_prisma_pan_and_geo,
    read_prisma_pan_quality_mask,
    read_prisma_quality_mask,
    extract_prisma_extended_metadata,
    estimate_prisma_geotransform,
    check_cloud_threshold,
)

from hypercoreg.readers.enmap import (
    read_enmap_metadata,
    derive_enmap_bbox_from_raster,
    extract_enmap_extended_metadata,
    find_enmap_metadata_for_spectral_image,
    copy_enmap_auxiliary_tifs,
    check_enmap_crs_compatibility,
    reproject_enmap_to_s2_crs,
)

__all__ = [
    # PRISMA
    "read_prisma_cube_and_meta",
    "read_prisma_pan_and_geo",
    "read_prisma_pan_quality_mask",
    "read_prisma_quality_mask",
    "extract_prisma_extended_metadata",
    "estimate_prisma_geotransform",
    "check_cloud_threshold",
    # EnMAP
    "read_enmap_metadata",
    "derive_enmap_bbox_from_raster",
    "extract_enmap_extended_metadata",
    "find_enmap_metadata_for_spectral_image",
    "copy_enmap_auxiliary_tifs",
    "check_enmap_crs_compatibility",
    "reproject_enmap_to_s2_crs",
]
