"""
PRISMA hyperspectral data reader.

This module provides functions for reading PRISMA L2D HDF5 files,
including spectral cubes, panchromatic data, quality masks, and cloud masks.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any

import numpy as np
import h5py
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer

from hypercoreg.spectral import build_prisma_band_table, SpectralBandTable
from hypercoreg.logging_config import log_section_header

logger = logging.getLogger("COREG_PROCESSING")


def estimate_prisma_geotransform(
    lon: np.ndarray,
    lat: np.ndarray,
    rows: int,
    cols: int,
    target_crs: str,
    use_geolocation_mesh: bool = False,
    geolocation_mesh_stride: int = 32,
) -> Affine:
    """
    Estimate affine geotransform from PRISMA lat/lon corner coordinates.

    Args:
        lon: Longitude array (2D grid)
        lat: Latitude array (2D grid)
        rows: Number of rows
        cols: Number of columns
        target_crs: Target CRS (e.g., "EPSG:32632")
        use_geolocation_mesh: Fit affine from a subsampled geolocation mesh.
        geolocation_mesh_stride: Sampling stride in pixels for mesh fitting.

    Returns:
        rasterio.Affine: Estimated geotransform
    """
    tr = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    def _solve_affine(pixel: np.ndarray, line: np.ndarray, lon_vals: np.ndarray, lat_vals: np.ndarray) -> Affine:
        x_map, y_map = tr.transform(lon_vals, lat_vals)
        A = np.column_stack([pixel.astype(float), line.astype(float), np.ones_like(pixel, dtype=float)])
        sol_x = np.linalg.lstsq(A, np.asarray(x_map, dtype=float), rcond=None)[0]
        sol_y = np.linalg.lstsq(A, np.asarray(y_map, dtype=float), rcond=None)[0]
        a, b, c = sol_x
        d, e, f = sol_y
        # Adjust from center-fit to corner-based geotransform convention.
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


def read_prisma_cube_and_meta(
    prisma_file: str,
    remove_detector_overlap: bool = False
) -> Tuple[
    np.ndarray, np.ndarray, datetime, Tuple[float, float, float, float],
    np.ndarray, np.ndarray, Optional[np.ndarray], list, list
]:
    """
    Read PRISMA L2D hyperspectral cube and metadata from HDF5 file.

    Args:
        prisma_file: Path to PRISMA L2D .he5 file
        remove_detector_overlap: If True, remove overlapping VNIR/SWIR bands.

    Returns:
        tuple: (cube, wl, prisma_time, bbox, lat, lon, fwhm, band_names, detectors)
            - cube: 3D array (rows, cols, bands) reordered to ascending wavelength
            - wl: Wavelength array (nm) in ascending order
            - prisma_time: Acquisition datetime (UTC)
            - bbox: Geographic bounds (minx, miny, maxx, maxy)
            - lat: Latitude array
            - lon: Longitude array
            - fwhm: FWHM array (or None)
            - band_names: List of band names
            - detectors: List of detector labels (VNIR/SWIR) aligned with bands
    """
    log_section_header("READING PRISMA HDF5 FILE")
    logger.debug(f"File: {prisma_file}")

    with h5py.File(prisma_file, "r") as f:
        VNIR = f["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"][:]
        SWIR = f["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"][:]

        logger.info(f"VNIR: {VNIR.shape}")
        logger.info(f"SWIR: {SWIR.shape}")

        # Extract wavelengths
        attrs = f.attrs
        wl_vnir = np.array(attrs["List_Cw_Vnir"])
        wl_swir = np.array(attrs["List_Cw_Swir"])

        # Validate spectral dimensions
        vnir_cube_bands = VNIR.shape[1]
        if vnir_cube_bands != len(wl_vnir):
            raise ValueError(
                f"PRISMA VNIR spectral mismatch: cube has {vnir_cube_bands} bands "
                f"but wavelength array has {len(wl_vnir)} entries"
            )

        swir_cube_bands = SWIR.shape[1]
        if swir_cube_bands != len(wl_swir):
            raise ValueError(
                f"PRISMA SWIR spectral mismatch: cube has {swir_cube_bands} bands "
                f"but wavelength array has {len(wl_swir)} entries"
            )

        logger.debug(f"VNIR bands validated: {vnir_cube_bands} bands = {len(wl_vnir)} wavelengths")
        logger.debug(f"SWIR bands validated: {swir_cube_bands} bands = {len(wl_swir)} wavelengths")

        # Extract FWHM arrays
        fwhm_vnir = None
        fwhm_swir = None
        for key in ["List_Fwhm_Vnir", "List_FWHM_Vnir", "List_Fwhm_VNIR"]:
            if key in attrs:
                fwhm_vnir = np.array(attrs[key])
                break
        for key in ["List_Fwhm_Swir", "List_FWHM_Swir", "List_Fwhm_SWIR"]:
            if key in attrs:
                fwhm_swir = np.array(attrs[key])
                break

        # Build SpectralBandTable (handles sorting to ascending order)
        band_table, sort_idx = build_prisma_band_table(
            wl_vnir,
            wl_swir,
            fwhm_vnir,
            fwhm_swir,
            remove_detector_overlap=remove_detector_overlap
        )

        # Validate the band table
        is_valid, issues = band_table.validate()
        if not is_valid:
            for issue in issues:
                if "overlap" in issue.lower():
                    logger.info(f"PRISMA spectral note: {issue} (this is expected)")
                else:
                    logger.warning(f"PRISMA spectral validation warning: {issue}")

        logger.info(band_table.summary())

        # Merge cubes
        merged = np.concatenate([VNIR, SWIR], axis=1)
        cube = np.transpose(merged, (0, 2, 1))
        logger.info(f"Merged cube (raw): {cube.shape}")

        # Apply band filtering/reordering if needed
        if sort_idx is not None:
            original_bands = cube.shape[2]
            cube = cube[:, :, sort_idx]
            if len(sort_idx) < original_bands:
                logger.info(
                    f"Filtered & reordered cube: {original_bands} -> {cube.shape[2]} bands "
                    "(removed invalid wavelengths)"
                )
            else:
                logger.info("Reordered cube bands to ascending wavelength order")
            logger.debug(f"Cube shape after processing: {cube.shape}")

        # Get wavelength array from band table (guaranteed ascending order)
        wl = band_table.wavelengths
        fwhm = band_table.fwhm
        band_names = band_table.band_names
        detectors = band_table.detectors

        logger.info(f"Wavelengths: {len(wl)} bands from {wl.min():.1f} to {wl.max():.1f} nm")

        # Extract acquisition time
        t = attrs["Product_StartTime"].decode()
        prisma_time = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        logger.info(f"Acquisition time: {prisma_time}")

        # Extract geolocation
        lat = f["HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude"][:]
        lon = f["HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude"][:]

        bbox = (
            float(np.nanmin(lon)), float(np.nanmin(lat)),
            float(np.nanmax(lon)), float(np.nanmax(lat))
        )
        logger.info(f"Geographic bounds: {bbox}")

    return cube, wl, prisma_time, bbox, lat, lon, fwhm, band_names, detectors


def read_prisma_pan_and_geo(prisma_file: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Read PRISMA L2D Panchromatic (PAN) image and corner coordinates from HDF5 file.

    Args:
        prisma_file: Path to PRISMA L2D HE5 file

    Returns:
        tuple: (pan_data, pan_geo_info) or (None, None) if not available
            - pan_data: 2D numpy array with PAN image (cleaned, correctly oriented)
            - pan_geo_info: dict with georeferencing info from product attributes
    """
    try:
        with h5py.File(prisma_file, "r") as f:
            # Check if PAN data exists (PRS_L2D_PCO = Panchromatic Camera Output)
            pco_path = "HDFEOS/SWATHS/PRS_L2D_PCO"
            if pco_path not in f:
                logger.warning(f"No PAN data found in PRISMA file (no {pco_path})")
                return None, None

            # Read PAN cube
            pan_cube = f[f"{pco_path}/Data Fields/Cube"][:]
            logger.info(f"PAN Cube raw shape: {pan_cube.shape}, dtype: {pan_cube.dtype}")

            # Squeeze if needed to get 2D
            if pan_cube.ndim == 3:
                pan_data = np.squeeze(pan_cube)
            else:
                pan_data = pan_cube

            # Convert to float for processing
            pan_data = pan_data.astype(np.float32)

            # Log statistics before cleaning
            logger.info(
                f"PAN raw stats: min={np.nanmin(pan_data):.2f}, max={np.nanmax(pan_data):.2f}, "
                f"mean={np.nanmean(pan_data):.2f}"
            )

            # Handle nodata values (typically 0 or negative values in PRISMA)
            pan_data[pan_data <= 0] = np.nan

            # Log statistics after cleaning
            valid_mask = ~np.isnan(pan_data)
            if np.any(valid_mask):
                logger.info(
                    f"PAN cleaned stats: min={np.nanmin(pan_data):.2f}, max={np.nanmax(pan_data):.2f}, "
                    f"mean={np.nanmean(pan_data):.2f}, valid pixels={np.sum(valid_mask)}"
                )

            logger.info(f"PAN image shape: {pan_data.shape}")

            # Extract corner coordinates from product attributes
            attrs = f.attrs

            pan_geo_info = {
                'rows': pan_data.shape[0],
                'cols': pan_data.shape[1],
                'ul_lon': None, 'ul_lat': None,
                'ur_lon': None, 'ur_lat': None,
                'll_lon': None, 'll_lat': None,
                'lr_lon': None, 'lr_lat': None,
                'pixel_size_m': 5.0  # PRISMA PAN is ~5m
            }

            # Try to get corner coordinates from PCO-specific attributes first
            pco_attrs = f[pco_path].attrs if pco_path in f else {}

            # Corner naming conventions in PRISMA products
            corner_keys = [
                ('Product_ULcorner_Lat', 'Product_ULcorner_Long'),
                ('Product_URcorner_Lat', 'Product_URcorner_Long'),
                ('Product_LLcorner_Lat', 'Product_LLcorner_Long'),
                ('Product_LRcorner_Lat', 'Product_LRcorner_Long')
            ]
            corner_names = ['ul', 'ur', 'll', 'lr']

            for (lat_key, lon_key), name in zip(corner_keys, corner_names):
                for attr_src in [pco_attrs, attrs]:
                    if lat_key in attr_src and lon_key in attr_src:
                        lat_val = attr_src[lat_key]
                        lon_val = attr_src[lon_key]
                        if isinstance(lat_val, bytes):
                            lat_val = float(lat_val.decode())
                        if isinstance(lon_val, bytes):
                            lon_val = float(lon_val.decode())
                        pan_geo_info[f'{name}_lat'] = float(lat_val)
                        pan_geo_info[f'{name}_lon'] = float(lon_val)
                        break

            # Fallback: use lat/lon grids if corner attrs not available
            if pan_geo_info['ul_lat'] is None:
                logger.debug("Corner attributes not found, falling back to lat/lon grids")
                lat_pan = f[f"{pco_path}/Geolocation Fields/Latitude"][:]
                lon_pan = f[f"{pco_path}/Geolocation Fields/Longitude"][:]

                pan_geo_info['ul_lat'] = float(lat_pan[0, 0])
                pan_geo_info['ul_lon'] = float(lon_pan[0, 0])
                pan_geo_info['ur_lat'] = float(lat_pan[0, -1])
                pan_geo_info['ur_lon'] = float(lon_pan[0, -1])
                pan_geo_info['ll_lat'] = float(lat_pan[-1, 0])
                pan_geo_info['ll_lon'] = float(lon_pan[-1, 0])
                pan_geo_info['lr_lat'] = float(lat_pan[-1, -1])
                pan_geo_info['lr_lon'] = float(lon_pan[-1, -1])

            logger.info(
                f"PAN geo corners: UL=({pan_geo_info['ul_lon']:.4f}, {pan_geo_info['ul_lat']:.4f}), "
                f"LR=({pan_geo_info['lr_lon']:.4f}, {pan_geo_info['lr_lat']:.4f})"
            )

            return pan_data, pan_geo_info

    except Exception as e:
        logger.warning(f"Could not read PRISMA PAN data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None, None


def read_prisma_quality_mask(prisma_file: str) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
]:
    """
    Read PRISMA pixel quality matrices separately for VNIR and SWIR detectors.

    Args:
        prisma_file: Path to PRISMA L2D HE5 file

    Returns:
        tuple: (vnir_quality, swir_quality, lat, lon) or (None, None, None, None) if not available
            - vnir_quality: 3D numpy array with uint8 flag values
            - swir_quality: 3D numpy array with uint8 flag values
            - lat: 2D latitude array
            - lon: 2D longitude array
    """
    try:
        with h5py.File(prisma_file, "r") as f:
            hco_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields"

            # Check if error matrices exist
            vnir_err_path = f"{hco_path}/VNIR_PIXEL_L2_ERR_MATRIX"
            swir_err_path = f"{hco_path}/SWIR_PIXEL_L2_ERR_MATRIX"

            vnir_err = None
            swir_err = None

            if vnir_err_path in f:
                vnir_err = f[vnir_err_path][:].astype(np.uint8)
                logger.info(f"VNIR pixel quality matrix shape: {vnir_err.shape}, dtype: {vnir_err.dtype}")
                # Log flag statistics
                for flag_val, flag_name in [(0, "OK"), (1, "KDP"), (2, "Saturation"),
                                             (3, "Low confidence"), (4, "NaN/Inf")]:
                    count = np.sum(vnir_err == flag_val)
                    pct = (count / vnir_err.size) * 100
                    logger.debug(f"  VNIR flag {flag_val} ({flag_name}): {count:,} ({pct:.2f}%)")
            else:
                logger.debug(f"VNIR error matrix not found at {vnir_err_path}")

            if swir_err_path in f:
                swir_err = f[swir_err_path][:].astype(np.uint8)
                logger.info(f"SWIR pixel quality matrix shape: {swir_err.shape}, dtype: {swir_err.dtype}")
                for flag_val, flag_name in [(0, "OK"), (1, "KDP"), (2, "Saturation"),
                                             (3, "Low confidence"), (4, "NaN/Inf")]:
                    count = np.sum(swir_err == flag_val)
                    pct = (count / swir_err.size) * 100
                    logger.debug(f"  SWIR flag {flag_val} ({flag_name}): {count:,} ({pct:.2f}%)")
            else:
                logger.debug(f"SWIR error matrix not found at {swir_err_path}")

            if vnir_err is None and swir_err is None:
                logger.warning("No PRISMA quality/error matrices found")
                return None, None, None, None

            # Read geolocation
            lat = f["HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude"][:]
            lon = f["HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude"][:]

            return vnir_err, swir_err, lat, lon

    except Exception as e:
        logger.warning(f"Could not read PRISMA quality mask: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None, None, None, None


def check_cloud_threshold(
    extended_meta: Dict[str, Any],
    threshold: float,
    hyp_type: str
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Check if image cloud coverage exceeds the threshold.

    Args:
        extended_meta: Dictionary with extended metadata
        threshold: Maximum acceptable cloud cover percentage (0-100)
        hyp_type: Sensor type ('PRISMA' or 'ENMAP')

    Returns:
        tuple: (passed, cloud_pct, reason)
            - passed: True if image passes threshold check
            - cloud_pct: Computed cloud coverage percentage, or None if unavailable
            - reason: Rejection reason string, or None if passed
    """
    cloud_pct = None

    if hyp_type == "PRISMA":
        cloud_pct = extended_meta.get('prisma_cloud_pct')
        if cloud_pct is None:
            logger.warning("Cloud metadata unavailable for PRISMA image - proceeding with processing")
            return (True, None, None)

    else:  # EnMAP
        enmap_cloud = extended_meta.get('prisma_cloud_pct')  # Mapped from enmap_cloud_pct
        enmap_haze = extended_meta.get('enmap_haze_pct', 0.0) or 0.0
        enmap_cirrus = extended_meta.get('enmap_cirrus_pct', 0.0) or 0.0

        if enmap_cloud is None:
            logger.warning("Cloud metadata unavailable for EnMAP image - proceeding with processing")
            return (True, None, None)

        cloud_pct = enmap_cloud + enmap_haze + enmap_cirrus
        logger.info(
            f"EnMAP total cloud coverage: {cloud_pct:.1f}% "
            f"(cloud={enmap_cloud:.1f}%, haze={enmap_haze:.1f}%, cirrus={enmap_cirrus:.1f}%)"
        )

    # Check against threshold
    if cloud_pct > threshold:
        reason = f"Cloud cover {cloud_pct:.1f}% exceeds threshold {threshold:.1f}%"
        logger.warning(f"Image rejected: {reason}")
        return (False, cloud_pct, reason)

    logger.info(f"Cloud cover check passed: {cloud_pct:.1f}% <= {threshold:.1f}%")
    return (True, cloud_pct, None)


def extract_prisma_extended_metadata(prisma_file: str) -> Dict[str, Any]:
    """
    Extract extended PRISMA metadata for metrics reporting.

    Args:
        prisma_file: Path to PRISMA HE5 file

    Returns:
        dict: Extended metadata including angles, cloud cover, etc.
    """
    result = {
        'prisma_id': None,
        'prisma_date': None,
        'prisma_cloud_pct': None,
        'prisma_sea_pct': None,
        'observation_angle': None,
        'rel_azimuth_angle': None,
        'sun_azimuth_angle': None,
        'solar_zenith_angle': None,
        'view_zenith_angle': None
    }

    try:
        with h5py.File(prisma_file, 'r') as f:
            attrs = f.attrs
            file_attrs = f.get('HDFEOS/ADDITIONAL/FILE_ATTRIBUTES')

            def _get_attr(name):
                """Fetch attribute, decoding bytes and casting to float when possible."""
                val = None
                if name in attrs:
                    val = attrs[name]
                elif file_attrs is not None and name in file_attrs.attrs:
                    val = file_attrs.attrs[name]
                if val is None:
                    return None
                if isinstance(val, bytes):
                    try:
                        return float(val.decode())
                    except Exception:
                        pass
                try:
                    return float(val)
                except Exception:
                    return None

            # Extract product ID
            for key in ['Product_Name', 'ProductName', 'product_name']:
                if key in attrs:
                    result['prisma_id'] = attrs[key].decode() if isinstance(attrs[key], bytes) else str(attrs[key])
                    break

            # Extract date from filename
            name = os.path.basename(prisma_file)
            if 'PRS_L2D' in name:
                parts = name.split('_')
                for part in parts:
                    if len(part) == 8 and part.isdigit():
                        try:
                            result['prisma_date'] = datetime.strptime(part, '%Y%m%d').isoformat()
                            break
                        except Exception:
                            pass

            # Cloud cover percentage
            for key in ['Cloudy_pixels_percentage', 'CloudCover', 'Cloud_Cover', 'CloudPercentage']:
                if key in attrs:
                    try:
                        result['prisma_cloud_pct'] = float(attrs[key])
                        break
                    except Exception:
                        pass

            # Sea pixels percentage
            for key in ['Sea_pixels_percentage', 'Sea_pixels_pct', 'SeaPixels']:
                if key in attrs:
                    try:
                        result['prisma_sea_pct'] = float(attrs[key])
                        break
                    except Exception:
                        pass

            # View Zenith Angle
            try:
                vza = _get_attr('Atm_LutGeomInfo_ViewZenith')
                if vza is not None:
                    result['view_zenith_angle'] = vza
            except Exception:
                pass

            # Geometric angles from HDF5 fields
            geo_fields_path = "HDFEOS/SWATHS/PRS_L2D_HCO/Geometric Fields"

            # Observing Angle
            try:
                obs_path = f"{geo_fields_path}/Observing_Angle"
                if obs_path in f:
                    obs_data = f[obs_path][()]
                    obs_valid = obs_data[(~np.isnan(obs_data)) & (obs_data != 0)]
                    if obs_valid.size > 0:
                        result['observation_angle'] = float(np.median(obs_valid))
            except Exception:
                pass

            # Relative Azimuth Angle
            try:
                raz_path = f"{geo_fields_path}/Rel_Azimuth_Angle"
                if raz_path in f:
                    raz_data = f[raz_path][()]
                    raz_valid = raz_data[(~np.isnan(raz_data)) & (raz_data != 0)]
                    if raz_valid.size > 0:
                        result['rel_azimuth_angle'] = float(np.median(raz_valid))
            except Exception:
                pass

            # Sun Azimuth Angle
            try:
                if 'Sun_azimuth_angle' in attrs:
                    result['sun_azimuth_angle'] = float(attrs['Sun_azimuth_angle'])
            except Exception:
                pass

            # Solar Zenith Angle
            try:
                sza_path = f"{geo_fields_path}/Solar_Zenith_Angle"
                if sza_path in f:
                    sza_data = f[sza_path][()]
                    sza_valid = sza_data[(~np.isnan(sza_data)) & (sza_data != 0)]
                    if sza_valid.size > 0:
                        result['solar_zenith_angle'] = float(np.median(sza_valid))
            except Exception:
                pass

    except Exception as e:
        logger.debug(f"Could not extract extended PRISMA metadata: {e}")

    # Fallback: extract date from filename if not found
    if not result.get('prisma_date') and prisma_file:
        try:
            name = os.path.basename(prisma_file)
            if 'PRS_L2D' in name:
                parts = name.split('_')
                for part in parts:
                    if len(part) >= 8 and part[:8].isdigit():
                        try:
                            if len(part) == 14:
                                dt = datetime.strptime(part[:14], '%Y%m%d%H%M%S')
                            else:
                                dt = datetime.strptime(part[:8], '%Y%m%d')
                            result['prisma_date'] = dt.isoformat()
                            break
                        except Exception:
                            pass
        except Exception:
            pass

    return result
