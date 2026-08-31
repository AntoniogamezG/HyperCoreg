"""
Configuration constants for HyperCoreg.

This module contains default parameters and shared constants
for the coregistration pipeline.
"""

import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Sequence

import numpy as np

logger = logging.getLogger("COREG_PROCESSING")

# =============================================================================
# Platform Detection
# =============================================================================

IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
IS_MACOS = sys.platform == "darwin"
EXE_EXT = ".exe" if IS_WINDOWS else ""

# =============================================================================
# Sentinel-2 Band Information
# =============================================================================

S2_BANDS: Dict[int, Dict[str, Any]] = {
    1: {"name": "B01 (Coastal)", "wavelength": 443.0, "resolution": 60},
    2: {"name": "B02 (Blue)", "wavelength": 490.0, "resolution": 10},
    3: {"name": "B03 (Green)", "wavelength": 560.0, "resolution": 10},
    4: {"name": "B04 (Red)", "wavelength": 665.0, "resolution": 10},
    5: {"name": "B05 (Red Edge 1)", "wavelength": 705.0, "resolution": 20},
    6: {"name": "B06 (Red Edge 2)", "wavelength": 740.0, "resolution": 20},
    7: {"name": "B07 (Red Edge 3)", "wavelength": 783.0, "resolution": 20},
    8: {"name": "B08 (NIR)", "wavelength": 842.0, "resolution": 10},
    9: {"name": "B8A (NIR Narrow)", "wavelength": 865.0, "resolution": 20},
    10: {"name": "B09 (Water Vapor)", "wavelength": 945.0, "resolution": 60},
    11: {"name": "B10 (Cirrus)", "wavelength": 1375.0, "resolution": 60},
    12: {"name": "B11 (SWIR 1)", "wavelength": 1610.0, "resolution": 20},
    13: {"name": "B12 (SWIR 2)", "wavelength": 2190.0, "resolution": 20},
}

# Approximate MSI band FWHM values used only as a deterministic fallback when
# bundled/tabulated SRF curves do not overlap a hyperspectral band table.
S2_APPROX_FWHM_NM: Dict[str, float] = {
    "B01": 20.0,
    "B02": 65.0,
    "B03": 35.0,
    "B04": 30.0,
    "B05": 15.0,
    "B06": 15.0,
    "B07": 20.0,
    "B08": 115.0,
    "B8A": 20.0,
    "B09": 20.0,
    "B10": 30.0,
    "B11": 90.0,
    "B12": 180.0,
}

# Sentinel-2 L2A surface-reflectance bands available in CDSE S2MSI2A products.
# B10 is intentionally absent from L2A because it is not a BOA reflectance band.
S2_L2A_OUTPUT_BANDS: Tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)
S2_L2A_ANCILLARY_BANDS: Tuple[str, ...] = (
    "SCL",
    "AOT",
    "WVP",
)
S2_L2A_REFERENCE_STACK_BANDS: Tuple[str, ...] = (
    *S2_L2A_OUTPUT_BANDS,
    *S2_L2A_ANCILLARY_BANDS,
)
S2_L2A_OUTPUT_BAND_INDEX: Dict[str, int] = {
    band: idx for idx, band in enumerate(S2_L2A_OUTPUT_BANDS, start=1)
}

# Multi-band S2 wavelengths used in coregistration stack
MULTIBAND_S2_WAVELENGTHS: Dict[str, Dict[str, Any]] = {
    'B02': {
        'wavelength': 490.0,
        'stack_idx': S2_L2A_OUTPUT_BAND_INDEX['B02'],
        'name': 'Blue',
    },
    'B03': {
        'wavelength': 560.0,
        'stack_idx': S2_L2A_OUTPUT_BAND_INDEX['B03'],
        'name': 'Green',
    },
    'B04': {
        'wavelength': 665.0,
        'stack_idx': S2_L2A_OUTPUT_BAND_INDEX['B04'],
        'name': 'Red',
    },
    'B08': {
        'wavelength': 842.0,
        'stack_idx': S2_L2A_OUTPUT_BAND_INDEX['B08'],
        'name': 'NIR',
    },
    'B11': {
        'wavelength': 1610.0,
        'stack_idx': S2_L2A_OUTPUT_BAND_INDEX['B11'],
        'name': 'SWIR1',
    },
    'B12': {
        'wavelength': 2190.0,
        'stack_idx': S2_L2A_OUTPUT_BAND_INDEX['B12'],
        'name': 'SWIR2',
    },
}

# Curated 1-based PRISMA band indices for Sentinel-2 matching bands.
# These are used as deterministic defaults when prefer_fixed_band_pairs=True.
PRISMA_FIXED_BAND_PAIRS: Dict[str, int] = {
    'B02': 10,
    'B03': 17,
    'B04': 28,
    'B08': 46,
    'B11': 123,
    'B12': 171,
}

# Sensor-aware default COREG attempt ladders and local max shifts (in pixels at working resolution).
DEFAULT_GLOBAL_COREG_PROFILES_BY_SENSOR: Dict[str, List[Dict[str, Any]]] = {
    'PRISMA': [
        {'window_size': (256, 256), 'max_shift': 180},
        {'window_size': (384, 384), 'max_shift': 220},
        {'window_size': (512, 512), 'max_shift': 260},
    ],
    'ENMAP': [
        {'window_size': (256, 256), 'max_shift': 90},
        {'window_size': (320, 320), 'max_shift': 120},
        {'window_size': (384, 384), 'max_shift': 150},
    ],
    'DEFAULT': [
        {'window_size': (256, 256), 'max_shift': 150},
        {'window_size': (384, 384), 'max_shift': 175},
        {'window_size': (512, 512), 'max_shift': 200},
    ],
}

DEFAULT_LOCAL_MAX_SHIFT_BY_SENSOR: Dict[str, int] = {
    'PRISMA': 70,
    'ENMAP': 40,
    'DEFAULT': 50,
}

# =============================================================================
# Default Parameters
# =============================================================================

# Sentinel-2 reference band
DEFAULT_S2_BAND = 8
S2_BAND08_CENTER_WL_NM = 842.0

# Coregistration grid parameters
GLOBAL_GRID_RES_M = 200
LOCAL_GRID_RES_M = 150

# Sentinel-2 query defaults
DEFAULT_DAYS_WINDOW = 30
DEFAULT_MAX_CLOUD_COVER = 20
DEFAULT_MIN_OVERLAP = 0.5

# Coregistration quality thresholds
DEFAULT_RESIDUAL_THRESHOLD_M = 25.0
DEFAULT_MIN_TIE_POINTS = 10
DEFAULT_MAX_S2_CANDIDATES = 3

# Scene Classification Layer (SCL) classes to exclude
SCL_EXCLUDE_CLASSES: Dict[int, str] = {
    0: "No Data",
    1: "Saturated/Defective",
    3: "Cloud Shadows",
    8: "Cloud Medium Probability",
    9: "Cloud High Probability",
    10: "Thin Cirrus"
}

# Polynomial warp configuration
POLYNOMIAL_ORDER = 2
MIN_TIE_POINTS_FOR_POLYNOMIAL = 10
MIN_RELIABILITY_THRESHOLD = 75.0
POLYNOMIAL_FALLBACK_TO_AROSICS = True

# Output resolution
PRISMA_OUTPUT_RESOLUTION = 30.0

# Processing configuration
PROCESSING_DTYPE = np.float32
PROCESSING_NODATA = -9999.0
CPUS_FOR_AROSICS = 1

# File/path configuration
TEMP_DIR_PREFIX = "tmp_coreg_"
DEFAULT_GDALWARP_NAME = "gdalwarp"

# CDSE API endpoint
CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# =============================================================================
# Default Configuration Dictionary
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # Sentinel-2 search parameters
    'days_window': DEFAULT_DAYS_WINDOW,
    'min_overlap': DEFAULT_MIN_OVERLAP,
    'max_cloud': DEFAULT_MAX_CLOUD_COVER,
    'max_input_cloud_cover': 70.0,
    'local_s2_stack_path': None,
    's2_stack_cache': True,
    's2_cache_dir': None,
    'scl_exclude_classes': list(SCL_EXCLUDE_CLASSES.keys()),
    's2_stack_mode': "materialized",

    # Coregistration parameters
    'residual_threshold': DEFAULT_RESIDUAL_THRESHOLD_M,
    'min_tie_points': DEFAULT_MIN_TIE_POINTS,
    'max_s2_candidates': DEFAULT_MAX_S2_CANDIDATES,
    'residual_mad_factor': 3.0,
    's2_ref_band': DEFAULT_S2_BAND,
    'prefer_fixed_band_pairs': True,
    'synthetic_s2_band_mode': "fixed_pair",
    'fixed_band_pairs_by_sensor': {
        'PRISMA': dict(PRISMA_FIXED_BAND_PAIRS),
    },
    'bandpair_wavelength_window_nm': 20.0,
    's2_band_subset_by_branch': {
        'VNIR': ['B03', 'B04', 'B08'],
        'SWIR': ['B11', 'B12'],
    },
    'local_tiepoint_early_stop': True,
    'local_tiepoint_early_stop_min_points': None,
    'local_tiepoint_early_stop_min_cells': None,
    'local_tiepoint_early_stop_reliability': MIN_RELIABILITY_THRESHOLD,
    'cache_hs_narrowbands': True,
    'hs_narrowband_cache_dir': None,
    'min_band_support': 2,
    'allow_single_band_fallback': True,
    'consensus_group_rounding_px': 1.0,
    'spatial_stratification_grid_rows': 4,
    'spatial_stratification_grid_cols': 4,
    'max_points_per_cell': 3,
    'preferred_polynomial_order': 2,
    'auto_downgrade_polynomial_order': True,
    'min_gcps_order2': 12,
    'min_cells_order2': 6,
    'transform_model_selection': "rule_based",
    'transform_cv_folds': 5,
    'transform_cv_repeats': 3,
    'transform_cv_holdout_fraction': 0.25,
    'transform_cv_seed': 1337,
    'transform_cv_min_tps_gcps': 20,
    'transform_cv_min_tps_cells': 8,
    'transform_cv_tps_min_p90_improvement_m': 1.0,
    'transform_cv_edge_instability_factor': 2.5,
    'local_coreg_grid_res': LOCAL_GRID_RES_M,
    'local_coreg_window_size': (256, 256),
    'local_coreg_tieP_filter_level': 1,
    'local_coreg_max_iter': None,
    'global_coreg_profiles_by_sensor': {
        key: [dict(item) for item in val]
        for key, val in DEFAULT_GLOBAL_COREG_PROFILES_BY_SENSOR.items()
    },
    'local_max_shift_by_sensor': dict(DEFAULT_LOCAL_MAX_SHIFT_BY_SENSOR),
    'global_coreg_attempt_ladder': None,
    'postwarp_phasecorr_check': False,
    'postwarp_phasecorr_warn_threshold_px': 1.0,
    'postwarp_phasecorr_reject_threshold_px': 3.0,
    'postwarp_phasecorr_reject_bad': False,
    'postwarp_phasecorr_max_dim': 1024,
    'use_geolocation_mesh_affine': False,
    'geolocation_mesh_stride': 32,

    # Quality parameters
    'min_accuracy': 70.0,
    'max_displacement': 350.0,

    # Output options
    'save_pre': False,
    'gen_tiepoint_pngs': False,
    'save_displacement_vectors': False,
    'use_inmemory': True,
    'keep_temp_files': False,
    'save_pan': False,
    'save_quality_mask': False,
    'arosics_cpus': 0,
    'gdalwarp_multi': True,
    'gdalwarp_num_threads': 'ALL_CPUS',
    'batch_workers': 1,
    'allow_gui_prompt': False,
    'remove_detector_overlap_bands': False,
    # Product-defined PRISMA L2 decoding. This is independent of optional
    # scene-dependent normalization below.
    'prisma_radiometric_mode': "reflectance",
    'normalization_mode': "none",
    'norm_p_low': 2.0,
    'norm_p_high': 98.0,
    'norm_clip': True,
    'norm_eps': 1e-6,
    'norm_min_valid_pixels': 1024,
    'norm_reservoir_size': 8192,
    'norm_seed': 1337,
    'norm_tile_size': 256,
    'build_overviews': False,
    'strict_metadata': True,
    'metadata_extension_level': "stats",
    'metadata_stats_mode': "approx",
    'enmap_metadata_stats_mode': "none",
    'metadata_stats_sample_windows': 32,
    'metadata_stats_seed': 1337,
    'metadata_histogram_buckets': 64,
    'metadata_label_precision': 2,
    'validation_max_windows': 64,
    'quicklook_max_dim': 1200,
    'quicklook_rgb_targets_nm': (660.0, 550.0, 480.0),
    'quicklook_percentiles': (2.0, 98.0),
    'quicklook_gamma': 1.0,
    'quicklook_dpi': 140,
    'quicklook_crop_to_valid': False,
    'quicklook_scalebar': True,
    'quicklook_rgb_source_path': None,
    'pan_gcp_mode': "map_inverse",
    'pan_map_dxdy_source': "auto",
    'pan_target_aligned_pixels': False,
    'pan_use_synthetic_reference': True,
    'pan_min_points_for_poly2': 20,
    'pan_local_window_size': (512, 512),
    'pan_local_grid_res': LOCAL_GRID_RES_M,
    'pan_local_max_shift': 220.0,
    'pan_local_tieP_filter_level': 1,
    'pan_local_max_iter': None,
    'pan_residual_check': False,
    'pan_residual_threshold_px': 0.5,
    'pan_residual_max_dim': 1024,
    'defer_temp_cleanup_gui': False,
    'timing_logs': True,
}

FAST_CONFIG: Dict[str, Any] = {
    'max_s2_candidates': 2,
    'synthetic_s2_band_mode': "fixed_pair",
    's2_band_subset_by_branch': {
        'VNIR': ['B04', 'B08'],
        'SWIR': ['B11', 'B12'],
    },
    'local_tiepoint_early_stop': True,
    'local_tiepoint_early_stop_min_points': 10,
    'local_tiepoint_early_stop_min_cells': 3,
    'metadata_extension_level': "none",
    'metadata_stats_mode': "none",
    'enmap_metadata_stats_mode': "none",
    'validation_max_windows': 16,
    'quicklook_max_dim': 900,
    'quicklook_scalebar': False,
    'save_pre': False,
    'gen_tiepoint_pngs': False,
    'save_displacement_vectors': False,
    'save_pan': False,
    'save_quality_mask': False,
    'build_overviews': False,
    'postwarp_phasecorr_check': False,
    'pan_residual_check': False,
    'timing_logs': False,
}

ACCURACY_CONFIG: Dict[str, Any] = {
    'max_s2_candidates': 5,
    'max_cloud': 5,
    'days_window': 15,
    'min_overlap': 0.7,
    'synthetic_s2_band_mode': "srf_weighted",
    'min_band_support': 2,
    'allow_single_band_fallback': False,
    'min_tie_points': 25,
    'spatial_stratification_grid_rows': 6,
    'spatial_stratification_grid_cols': 6,
    'max_points_per_cell': 2,
    'local_tiepoint_early_stop': False,
    'preferred_polynomial_order': 2,
    'auto_downgrade_polynomial_order': True,
    'min_gcps_order2': 24,
    'min_cells_order2': 10,
    'transform_model_selection': "cv",
    'transform_cv_folds': 5,
    'transform_cv_repeats': 5,
    'transform_cv_holdout_fraction': 0.25,
    'transform_cv_seed': 1337,
    'transform_cv_min_tps_gcps': 20,
    'transform_cv_min_tps_cells': 8,
    'transform_cv_tps_min_p90_improvement_m': 1.0,
    'transform_cv_edge_instability_factor': 2.5,
    'postwarp_phasecorr_check': True,
    'postwarp_phasecorr_warn_threshold_px': 0.5,
    'postwarp_phasecorr_reject_threshold_px': 1.5,
    'postwarp_phasecorr_reject_bad': True,
    'use_geolocation_mesh_affine': True,
    'pan_gcp_mode': "scaled_image",
    'pan_target_aligned_pixels': True,
    'pan_residual_check': True,
    'pan_residual_threshold_px': 0.5,
    'metadata_stats_mode': "approx",
    'validation_max_windows': 0,
}

PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "fast": FAST_CONFIG,
    "accuracy": ACCURACY_CONFIG,
}


def apply_cpu_oversubscription_guard(
    config: Dict[str, Any],
    explicit_keys: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Cap nested worker thread settings when batch processing uses processes.

    The guard is intentionally conservative: if callers explicitly provide
    ``arosics_cpus`` or ``gdalwarp_num_threads`` those values are preserved.
    """
    guarded = dict(config or {})
    explicit = {str(k) for k in (explicit_keys or [])}
    warnings_out: List[str] = []
    try:
        workers = int(guarded.get("batch_workers", DEFAULT_CONFIG.get("batch_workers", 1)) or 1)
    except Exception:
        workers = 1
    if workers <= 1:
        guarded["_cpu_guard_applied"] = False
        return guarded, warnings_out

    if "arosics_cpus" not in explicit:
        raw_arosics = guarded.get("arosics_cpus", DEFAULT_CONFIG.get("arosics_cpus", 0))
        try:
            arosics_cpus = int(raw_arosics)
        except Exception:
            arosics_cpus = 0
        if arosics_cpus != 1:
            guarded["arosics_cpus"] = 1
            warnings_out.append(
                "batch_workers > 1: auto-capped arosics_cpus to 1 to avoid CPU oversubscription."
            )

    if "gdalwarp_num_threads" not in explicit:
        raw_threads = guarded.get("gdalwarp_num_threads", DEFAULT_CONFIG.get("gdalwarp_num_threads", "ALL_CPUS"))
        token = str(raw_threads if raw_threads is not None else "").strip().upper()
        should_cap = token in {"", "0", "ALL_CPUS"}
        if not should_cap:
            try:
                should_cap = int(token) > 1
            except Exception:
                should_cap = True
        if should_cap:
            guarded["gdalwarp_num_threads"] = "1"
            warnings_out.append(
                "batch_workers > 1: auto-capped gdalwarp_num_threads to 1 to avoid CPU oversubscription."
            )

    guarded["_cpu_guard_applied"] = bool(warnings_out)
    if warnings_out:
        guarded["_cpu_guard_warnings"] = list(warnings_out)
    return guarded, warnings_out


@dataclass
class CoregConfig:
    """Configuration dataclass for coregistration parameters."""

    # Sentinel-2 search
    days_window: int = DEFAULT_DAYS_WINDOW
    min_overlap: float = DEFAULT_MIN_OVERLAP
    max_cloud: float = DEFAULT_MAX_CLOUD_COVER
    max_input_cloud_cover: float = 70.0
    local_s2_stack_path: Optional[str] = None
    s2_stack_cache: bool = True
    s2_cache_dir: Optional[str] = None
    scl_exclude_classes: List[int] = field(
        default_factory=lambda: list(SCL_EXCLUDE_CLASSES.keys())
    )
    s2_stack_mode: str = "materialized"

    # Coregistration
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD_M
    min_tie_points: int = DEFAULT_MIN_TIE_POINTS
    max_s2_candidates: int = DEFAULT_MAX_S2_CANDIDATES
    residual_mad_factor: float = 3.0
    s2_ref_band: int = DEFAULT_S2_BAND
    prefer_fixed_band_pairs: bool = True
    synthetic_s2_band_mode: str = "fixed_pair"
    fixed_band_pairs_by_sensor: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {'PRISMA': dict(PRISMA_FIXED_BAND_PAIRS)}
    )
    bandpair_wavelength_window_nm: float = 20.0
    s2_band_subset_by_branch: Dict[str, List[str]] = field(
        default_factory=lambda: {
            'VNIR': ['B03', 'B04', 'B08'],
            'SWIR': ['B11', 'B12'],
        }
    )
    local_tiepoint_early_stop: bool = True
    local_tiepoint_early_stop_min_points: Optional[int] = None
    local_tiepoint_early_stop_min_cells: Optional[int] = None
    local_tiepoint_early_stop_reliability: float = MIN_RELIABILITY_THRESHOLD
    cache_hs_narrowbands: bool = True
    hs_narrowband_cache_dir: Optional[str] = None
    min_band_support: int = 2
    allow_single_band_fallback: bool = True
    consensus_group_rounding_px: float = 1.0
    spatial_stratification_grid_rows: int = 4
    spatial_stratification_grid_cols: int = 4
    max_points_per_cell: int = 3
    preferred_polynomial_order: int = 2
    auto_downgrade_polynomial_order: bool = True
    min_gcps_order2: int = 12
    min_cells_order2: int = 6
    transform_model_selection: str = "rule_based"
    transform_cv_folds: int = 5
    transform_cv_repeats: int = 3
    transform_cv_holdout_fraction: float = 0.25
    transform_cv_seed: int = 1337
    transform_cv_min_tps_gcps: int = 20
    transform_cv_min_tps_cells: int = 8
    transform_cv_tps_min_p90_improvement_m: float = 1.0
    transform_cv_edge_instability_factor: float = 2.5
    local_coreg_grid_res: int = LOCAL_GRID_RES_M
    local_coreg_window_size: Tuple[int, int] = (256, 256)
    local_coreg_tieP_filter_level: int = 1
    local_coreg_max_iter: Optional[int] = None
    global_coreg_profiles_by_sensor: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {
            key: [dict(item) for item in val]
            for key, val in DEFAULT_GLOBAL_COREG_PROFILES_BY_SENSOR.items()
        }
    )
    local_max_shift_by_sensor: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_LOCAL_MAX_SHIFT_BY_SENSOR)
    )
    global_coreg_attempt_ladder: Optional[List[Dict[str, Any]]] = None
    postwarp_phasecorr_check: bool = False
    postwarp_phasecorr_warn_threshold_px: float = 1.0
    postwarp_phasecorr_reject_threshold_px: float = 3.0
    postwarp_phasecorr_reject_bad: bool = False
    postwarp_phasecorr_max_dim: int = 1024
    use_geolocation_mesh_affine: bool = False
    geolocation_mesh_stride: int = 32

    # Quality
    min_accuracy: float = 70.0
    max_displacement: float = 350.0

    # Output options
    save_pre: bool = False
    gen_tiepoint_pngs: bool = False
    save_displacement_vectors: bool = False
    use_inmemory: bool = True
    keep_temp_files: bool = False
    save_pan: bool = False
    save_quality_mask: bool = False
    arosics_cpus: int = 0
    gdalwarp_multi: bool = True
    gdalwarp_num_threads: str = 'ALL_CPUS'
    batch_workers: int = 1
    allow_gui_prompt: bool = False
    remove_detector_overlap_bands: bool = False
    prisma_radiometric_mode: str = "reflectance"
    normalization_mode: str = "none"
    norm_p_low: float = 2.0
    norm_p_high: float = 98.0
    norm_clip: bool = True
    norm_eps: float = 1e-6
    norm_min_valid_pixels: int = 1024
    norm_reservoir_size: int = 8192
    norm_seed: int = 1337
    norm_tile_size: int = 256
    build_overviews: bool = False
    strict_metadata: bool = True
    metadata_extension_level: str = "stats"
    metadata_stats_mode: str = "approx"
    enmap_metadata_stats_mode: str = "none"
    metadata_stats_sample_windows: int = 32
    metadata_stats_seed: int = 1337
    metadata_histogram_buckets: int = 64
    metadata_label_precision: int = 2
    validation_max_windows: int = 64
    quicklook_max_dim: int = 1200
    quicklook_rgb_targets_nm: Tuple[float, float, float] = (660.0, 550.0, 480.0)
    quicklook_percentiles: Tuple[float, float] = (2.0, 98.0)
    quicklook_gamma: float = 1.0
    quicklook_dpi: int = 140
    quicklook_crop_to_valid: bool = False
    quicklook_scalebar: bool = True
    quicklook_rgb_source_path: Optional[str] = None
    pan_gcp_mode: str = "map_inverse"
    pan_map_dxdy_source: str = "auto"
    pan_target_aligned_pixels: bool = False
    pan_use_synthetic_reference: bool = True
    pan_min_points_for_poly2: int = 20
    pan_local_window_size: Tuple[int, int] = (512, 512)
    pan_local_grid_res: int = LOCAL_GRID_RES_M
    pan_local_max_shift: float = 220.0
    pan_local_tieP_filter_level: int = 1
    pan_local_max_iter: Optional[int] = None
    pan_residual_check: bool = False
    pan_residual_threshold_px: float = 0.5
    pan_residual_max_dim: int = 1024
    defer_temp_cleanup_gui: bool = False
    timing_logs: bool = True
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'days_window': self.days_window,
            'min_overlap': self.min_overlap,
            'max_cloud': self.max_cloud,
            'max_input_cloud_cover': self.max_input_cloud_cover,
            'local_s2_stack_path': self.local_s2_stack_path,
            's2_stack_cache': self.s2_stack_cache,
            's2_cache_dir': self.s2_cache_dir,
            'scl_exclude_classes': [int(v) for v in self.scl_exclude_classes],
            's2_stack_mode': self.s2_stack_mode,
            'residual_threshold': self.residual_threshold,
            'min_tie_points': self.min_tie_points,
            'max_s2_candidates': self.max_s2_candidates,
            'residual_mad_factor': self.residual_mad_factor,
            's2_ref_band': self.s2_ref_band,
            'prefer_fixed_band_pairs': self.prefer_fixed_band_pairs,
            'synthetic_s2_band_mode': self.synthetic_s2_band_mode,
            'fixed_band_pairs_by_sensor': {
                str(sensor): {str(k): int(v) for k, v in pairs.items()}
                for sensor, pairs in self.fixed_band_pairs_by_sensor.items()
            },
            'bandpair_wavelength_window_nm': self.bandpair_wavelength_window_nm,
            's2_band_subset_by_branch': {
                str(branch): [str(band) for band in bands]
                for branch, bands in self.s2_band_subset_by_branch.items()
            },
            'local_tiepoint_early_stop': self.local_tiepoint_early_stop,
            'local_tiepoint_early_stop_min_points': self.local_tiepoint_early_stop_min_points,
            'local_tiepoint_early_stop_min_cells': self.local_tiepoint_early_stop_min_cells,
            'local_tiepoint_early_stop_reliability': self.local_tiepoint_early_stop_reliability,
            'cache_hs_narrowbands': self.cache_hs_narrowbands,
            'hs_narrowband_cache_dir': self.hs_narrowband_cache_dir,
            'min_band_support': self.min_band_support,
            'allow_single_band_fallback': self.allow_single_band_fallback,
            'consensus_group_rounding_px': self.consensus_group_rounding_px,
            'spatial_stratification_grid_rows': self.spatial_stratification_grid_rows,
            'spatial_stratification_grid_cols': self.spatial_stratification_grid_cols,
            'max_points_per_cell': self.max_points_per_cell,
            'preferred_polynomial_order': self.preferred_polynomial_order,
            'auto_downgrade_polynomial_order': self.auto_downgrade_polynomial_order,
            'min_gcps_order2': self.min_gcps_order2,
            'min_cells_order2': self.min_cells_order2,
            'transform_model_selection': self.transform_model_selection,
            'transform_cv_folds': self.transform_cv_folds,
            'transform_cv_repeats': self.transform_cv_repeats,
            'transform_cv_holdout_fraction': self.transform_cv_holdout_fraction,
            'transform_cv_seed': self.transform_cv_seed,
            'transform_cv_min_tps_gcps': self.transform_cv_min_tps_gcps,
            'transform_cv_min_tps_cells': self.transform_cv_min_tps_cells,
            'transform_cv_tps_min_p90_improvement_m': self.transform_cv_tps_min_p90_improvement_m,
            'transform_cv_edge_instability_factor': self.transform_cv_edge_instability_factor,
            'local_coreg_grid_res': self.local_coreg_grid_res,
            'local_coreg_window_size': self.local_coreg_window_size,
            'local_coreg_tieP_filter_level': self.local_coreg_tieP_filter_level,
            'local_coreg_max_iter': self.local_coreg_max_iter,
            'global_coreg_profiles_by_sensor': {
                str(sensor): [dict(item) for item in attempts]
                for sensor, attempts in self.global_coreg_profiles_by_sensor.items()
            },
            'local_max_shift_by_sensor': {
                str(sensor): int(shift) for sensor, shift in self.local_max_shift_by_sensor.items()
            },
            'global_coreg_attempt_ladder': (
                None
                if self.global_coreg_attempt_ladder is None
                else [dict(item) for item in self.global_coreg_attempt_ladder]
            ),
            'postwarp_phasecorr_check': self.postwarp_phasecorr_check,
            'postwarp_phasecorr_warn_threshold_px': self.postwarp_phasecorr_warn_threshold_px,
            'postwarp_phasecorr_reject_threshold_px': self.postwarp_phasecorr_reject_threshold_px,
            'postwarp_phasecorr_reject_bad': self.postwarp_phasecorr_reject_bad,
            'postwarp_phasecorr_max_dim': self.postwarp_phasecorr_max_dim,
            'use_geolocation_mesh_affine': self.use_geolocation_mesh_affine,
            'geolocation_mesh_stride': self.geolocation_mesh_stride,
            'min_accuracy': self.min_accuracy,
            'max_displacement': self.max_displacement,
            'save_pre': self.save_pre,
            'gen_tiepoint_pngs': self.gen_tiepoint_pngs,
            'save_displacement_vectors': self.save_displacement_vectors,
            'use_inmemory': self.use_inmemory,
            'keep_temp_files': self.keep_temp_files,
            'save_pan': self.save_pan,
            'save_quality_mask': self.save_quality_mask,
            'arosics_cpus': self.arosics_cpus,
            'gdalwarp_multi': self.gdalwarp_multi,
            'gdalwarp_num_threads': self.gdalwarp_num_threads,
            'batch_workers': self.batch_workers,
            'allow_gui_prompt': self.allow_gui_prompt,
            'remove_detector_overlap_bands': self.remove_detector_overlap_bands,
            'prisma_radiometric_mode': self.prisma_radiometric_mode,
            'normalization_mode': self.normalization_mode,
            'norm_p_low': self.norm_p_low,
            'norm_p_high': self.norm_p_high,
            'norm_clip': self.norm_clip,
            'norm_eps': self.norm_eps,
            'norm_min_valid_pixels': self.norm_min_valid_pixels,
            'norm_reservoir_size': self.norm_reservoir_size,
            'norm_seed': self.norm_seed,
            'norm_tile_size': self.norm_tile_size,
            'build_overviews': self.build_overviews,
            'strict_metadata': self.strict_metadata,
            'metadata_extension_level': self.metadata_extension_level,
            'metadata_stats_mode': self.metadata_stats_mode,
            'enmap_metadata_stats_mode': self.enmap_metadata_stats_mode,
            'metadata_stats_sample_windows': self.metadata_stats_sample_windows,
            'metadata_stats_seed': self.metadata_stats_seed,
            'metadata_histogram_buckets': self.metadata_histogram_buckets,
            'metadata_label_precision': self.metadata_label_precision,
            'validation_max_windows': self.validation_max_windows,
            'quicklook_max_dim': self.quicklook_max_dim,
            'quicklook_rgb_targets_nm': self.quicklook_rgb_targets_nm,
            'quicklook_percentiles': self.quicklook_percentiles,
            'quicklook_gamma': self.quicklook_gamma,
            'quicklook_dpi': self.quicklook_dpi,
            'quicklook_crop_to_valid': self.quicklook_crop_to_valid,
            'quicklook_scalebar': self.quicklook_scalebar,
            'quicklook_rgb_source_path': self.quicklook_rgb_source_path,
            'pan_gcp_mode': self.pan_gcp_mode,
            'pan_map_dxdy_source': self.pan_map_dxdy_source,
            'pan_target_aligned_pixels': self.pan_target_aligned_pixels,
            'pan_use_synthetic_reference': self.pan_use_synthetic_reference,
            'pan_min_points_for_poly2': self.pan_min_points_for_poly2,
            'pan_local_window_size': self.pan_local_window_size,
            'pan_local_grid_res': self.pan_local_grid_res,
            'pan_local_max_shift': self.pan_local_max_shift,
            'pan_local_tieP_filter_level': self.pan_local_tieP_filter_level,
            'pan_local_max_iter': self.pan_local_max_iter,
            'pan_residual_check': self.pan_residual_check,
            'pan_residual_threshold_px': self.pan_residual_threshold_px,
            'pan_residual_max_dim': self.pan_residual_max_dim,
            'defer_temp_cleanup_gui': self.defer_temp_cleanup_gui,
            'timing_logs': self.timing_logs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CoregConfig':
        """Create from dictionary, using defaults for missing keys."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


