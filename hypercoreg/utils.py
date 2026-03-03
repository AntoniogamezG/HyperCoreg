"""
Utility functions for HyperCoreg.

This module provides common utility functions used throughout the package.
"""

import os
import shutil
import logging
from typing import Optional, Tuple, Dict, Any

import numpy as np

from hypercoreg.config import (
    DEFAULT_GDALWARP_NAME,
    S2_BANDS,
    MULTIBAND_S2_WAVELENGTHS,
)

logger = logging.getLogger("COREG_PROCESSING")


class SentinelNotFoundError(Exception):
    """Exception raised when no suitable Sentinel-2 scene is found for a product."""
    pass


def detect_hyp_type(file_path: str) -> str:
    """
    Auto-detect hyperspectral sensor type from file path.

    Args:
        file_path: Path to hyperspectral file

    Returns:
        str: "PRISMA" or "ENMAP"

    Raises:
        ValueError: If sensor type cannot be determined
    """
    basename = os.path.basename(file_path).upper()

    if basename.endswith(".HE5"):
        return "PRISMA"

    if "SPECTRAL_IMAGE" in basename or basename.startswith("ENMAP"):
        return "ENMAP"

    # Check file extension and content hints
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".he5":
        return "PRISMA"

    if ext in (".tif", ".tiff", ".bsq"):
        # Could be EnMAP - check for metadata files
        parent_dir = os.path.dirname(file_path)
        xml_files = [f for f in os.listdir(parent_dir) if f.lower().endswith("-metadata.xml")]
        if xml_files:
            return "ENMAP"

    raise ValueError(
        f"Cannot determine sensor type for: {file_path}. "
        "Expected PRISMA (.he5) or EnMAP "
        "(SPECTRAL_IMAGE.TIF/.TIFF/.BSQ with -METADATA.XML)"
    )


def resolve_gdalwarp_exe() -> str:
    """
    Find the gdalwarp executable.

    Returns:
        str: Path to gdalwarp executable

    Raises:
        FileNotFoundError: If gdalwarp cannot be found
    """
    # Check environment variable first
    env_path = os.getenv("GDAL_WARP_EXE")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Check PATH
    which_path = shutil.which(DEFAULT_GDALWARP_NAME)
    if which_path:
        return which_path

    raise FileNotFoundError(
        "Could not find gdalwarp. Ensure GDAL is on PATH or set GDAL_WARP_EXE."
    )


def compute_wave_min_max(
    wl: np.ndarray,
    fwhm: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute wavelength min/max bounds from center wavelength and FWHM.

    Args:
        wl: Center wavelength array
        fwhm: FWHM array

    Returns:
        tuple: (wl_min, wl_max) arrays, or (None, None) if invalid
    """
    if wl is None or fwhm is None:
        return None, None

    wl = np.asarray(wl).astype(float)
    fwhm = np.asarray(fwhm).astype(float)

    if wl.shape != fwhm.shape:
        return None, None

    half = fwhm / 2.0
    return wl - half, wl + half


def create_narrowband_average(
    cube: np.ndarray,
    wl: np.ndarray,
    center_wl: float = 842.0,
    bandwidth: float = 20.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create pseudo-panchromatic band by averaging narrow spectral window.

    Args:
        cube: HSI cube (rows, cols, bands)
        wl: Wavelength array (nm)
        center_wl: Target center wavelength (default: 842nm for S2 B08)
        bandwidth: Total bandwidth for averaging (default: +/-10nm = 20nm total)

    Returns:
        tuple: (averaged_band, band_indices)
    """
    wl = np.asarray(wl).reshape(-1)
    if wl.size == 0:
        raise ValueError("Cannot build narrowband average: empty wavelength array.")

    lower = center_wl - bandwidth / 2
    upper = center_wl + bandwidth / 2
    mask = (wl >= lower) & (wl <= upper)
    band_indices = np.where(mask)[0]

    if len(band_indices) < 2:
        # Fall back to closest bands
        diffs = np.abs(wl - center_wl)
        n_pick = min(3, wl.size)
        if n_pick >= wl.size:
            band_indices = np.argsort(diffs)[:n_pick]
        else:
            band_indices = np.argpartition(diffs, n_pick - 1)[:n_pick]
        band_indices = np.sort(band_indices)

    selected_bands = cube[:, :, band_indices]
    averaged = np.mean(selected_bands, axis=2)
    return averaged, band_indices


def get_s2_band_for_wavelength(target_wl: float) -> int:
    """
    Get the S2 band index (in multi-band stack) for a target wavelength.

    The S2 stack contains: B02(1), B03(2), B04(3), B08(4), B11(5), B12(6)

    Args:
        target_wl: Target wavelength in nm

    Returns:
        int: Band index (1-based) in the stack, or 4 (B08) as default
    """
    # Map target wavelength to available bands
    stack_bands = {
        490.0: 1,   # B02 Blue
        560.0: 2,   # B03 Green
        665.0: 3,   # B04 Red
        842.0: 4,   # B08 NIR
        1610.0: 5,  # B11 SWIR1
        2190.0: 6,  # B12 SWIR2
    }

    # Find closest match
    best_band = 4  # Default to B08
    best_diff = float('inf')

    for wl, band_idx in stack_bands.items():
        diff = abs(wl - target_wl)
        if diff < best_diff:
            best_diff = diff
            best_band = band_idx

    return best_band


def diagnose_array(arr: np.ndarray, name: str):
    """
    Log diagnostic information about a numpy array.

    Args:
        arr: Array to diagnose
        name: Name/label for logging
    """
    logger.debug(f"Array Diagnostic: {name}")
    logger.debug(f"  Shape: {arr.shape}, Dtype: {arr.dtype}")
    logger.debug(
        f"  Min: {np.nanmin(arr):.6f}, Max: {np.nanmax(arr):.6f}, "
        f"Mean: {np.nanmean(arr):.6f}"
    )
    logger.debug(
        f"  NaN: {np.isnan(arr).sum()}, Inf: {np.isinf(arr).sum()}, "
        f"Zero: {(arr == 0).sum()}"
    )


def diagnose_raster(file_path: str, stage_name: str, band_idx: Optional[int] = None) -> bool:
    """
    Comprehensive diagnostic check for raster outputs.

    Args:
        file_path: Path to raster file
        stage_name: Description of processing stage
        band_idx: Optional specific band to check

    Returns:
        bool: True if file exists and is valid
    """
    import rasterio

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"DIAGNOSTIC: {stage_name}")
    logger.info("=" * 60)

    if not os.path.exists(file_path):
        logger.error(f"FILE MISSING: {file_path}")
        return False

    logger.info(f"File exists: {file_path}")
    logger.info(f"  File size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")

    try:
        with rasterio.open(file_path) as src:
            logger.info(f"  Bands: {src.count}")
            logger.info(f"  Size: {src.width} x {src.height}")
            logger.info(f"  CRS: {src.crs}")
            logger.info(f"  Dtype: {src.dtypes[0]}")
            logger.info(f"  NoData: {src.nodata}")

            # Check a band
            check_band = band_idx if band_idx else 1
            if check_band <= src.count:
                data = src.read(check_band)
                valid_mask = ~np.isnan(data) if np.issubdtype(data.dtype, np.floating) else data != src.nodata
                logger.info(f"  Band {check_band} valid pixels: {valid_mask.sum()} / {data.size}")

        return True

    except Exception as e:
        logger.error(f"  Failed to read: {e}")
        return False


def verify_crs_match(crs1, crs2) -> bool:
    """
    Verify that two CRS objects are equivalent.

    Args:
        crs1: First CRS (rasterio CRS or pyproj CRS)
        crs2: Second CRS

    Returns:
        bool: True if CRS match
    """
    from pyproj import CRS

    try:
        c1 = CRS.from_user_input(crs1)
        c2 = CRS.from_user_input(crs2)
        return c1.equals(c2)
    except Exception as e:
        logger.warning(f"CRS comparison failed: {e}")
        return False


def bbox_to_wkt(bbox: Tuple[float, float, float, float]) -> str:
    """
    Convert bounding box to WKT polygon string.

    Args:
        bbox: (minx, miny, maxx, maxy)

    Returns:
        str: WKT POLYGON string
    """
    minx, miny, maxx, maxy = bbox
    return f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"


def normalize_path(path: str) -> str:
    """
    Normalize a file path for cross-platform compatibility.

    Args:
        path: Input path string

    Returns:
        str: Normalized absolute path
    """
    return os.path.normpath(os.path.abspath(path))


def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        str: Absolute path to directory
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path
