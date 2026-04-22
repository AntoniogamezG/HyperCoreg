"""
Command-line interface for HyperCoreg.

This module provides the CLI entry point for running coregistration
from the command line, enabling automation and scripting.

Usage:
    hypercoreg single --input file.he5 --output ./results
    hypercoreg batch --input ./folder --output ./results
"""

import sys
import argparse
import logging
import json
from typing import Optional

from hypercoreg._version import __version__
from hypercoreg.config import DEFAULT_CONFIG
from hypercoreg.logging_config import setup_logging, log_section_header


def _parse_int_pair(value: str, field_name: str) -> tuple[int, int]:
    """Parse a comma-separated pair of positive ints."""
    text = str(value).strip()
    parts = [p.strip() for p in text.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{field_name} must be in 'A,B' form.")
    try:
        a = int(parts[0])
        b = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{field_name} must contain integers.") from exc
    if a <= 0 or b <= 0:
        raise argparse.ArgumentTypeError(f"{field_name} values must be > 0.")
    return a, b


def _parse_json_dict(value: Optional[str], field_name: str) -> Optional[dict]:
    """Parse optional JSON dictionary CLI fields."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{field_name} must decode to a JSON object.")
    return parsed


def _parse_json_list(value: Optional[str], field_name: str) -> Optional[list]:
    """Parse optional JSON list CLI fields."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError(f"{field_name} must decode to a JSON list.")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="hypercoreg",
        description="HyperCoreg - Automated hyperspectral PRISMA/EnMAP to Sentinel-2 coregistration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file processing
  hypercoreg single -i image.he5 -o ./output

  # Batch processing
  hypercoreg batch -i ./input_folder -o ./output

  # With custom parameters
  hypercoreg single -i image.he5 -o ./output --days-window 60 --max-cloud 30

For more information, visit: https://github.com/AntoniogamezG/HyperCoreg-An-optimized-hyperspectral-to-Sentinel-2-co-registration-pipeline-for-PRISMA-and-EnMAP
        """
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    # Create subparsers for modes
    subparsers = parser.add_subparsers(
        dest="mode",
        title="commands",
        description="Processing modes",
        required=True
    )

    # Single file mode
    single_parser = subparsers.add_parser(
        "single",
        help="Process a single hyperspectral file",
        description="Process a single PRISMA or EnMAP file"
    )
    _add_common_arguments(single_parser)
    single_parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Input hyperspectral file (PRISMA .he5 or EnMAP SPECTRAL_IMAGE.TIF/.BSQ)"
    )

    # Batch mode
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process a directory of hyperspectral files",
        description="Batch process multiple PRISMA or EnMAP files"
    )
    _add_common_arguments(batch_parser)
    batch_parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Input directory containing hyperspectral files"
    )

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser):
    """Add common arguments to a subparser."""

    # Required output
    parser.add_argument(
        "-o", "--output",
        required=True,
        type=str,
        help="Output directory for coregistered products"
    )

    # Sentinel-2 search options
    s2_group = parser.add_argument_group("Sentinel-2 Reference Options")
    s2_group.add_argument(
        "--days-window",
        type=int,
        default=DEFAULT_CONFIG['days_window'],
        metavar="DAYS",
        help=f"Temporal search window for S2 reference (default: {DEFAULT_CONFIG['days_window']})"
    )
    s2_group.add_argument(
        "--min-overlap",
        type=float,
        default=DEFAULT_CONFIG['min_overlap'],
        metavar="FRAC",
        help=f"Minimum spatial overlap 0-1 (default: {DEFAULT_CONFIG['min_overlap']})"
    )
    s2_group.add_argument(
        "--max-cloud",
        type=float,
        default=DEFAULT_CONFIG['max_cloud'],
        metavar="PCT",
        help=f"Maximum S2 cloud cover %% (default: {DEFAULT_CONFIG['max_cloud']})"
    )
    s2_group.add_argument(
        "--max-input-cloud",
        type=float,
        default=DEFAULT_CONFIG['max_input_cloud_cover'],
        metavar="PCT",
        help=f"Skip input if cloud > this %% (default: {DEFAULT_CONFIG['max_input_cloud_cover']})"
    )
    s2_group.add_argument(
        "--local-s2-stack",
        type=str,
        default=DEFAULT_CONFIG.get('local_s2_stack_path'),
        metavar="PATH",
        help="Use a local prebuilt Sentinel-2 stack and skip CDSE search/download."
    )

    # Coregistration options
    coreg_group = parser.add_argument_group("Coregistration Options")
    coreg_group.add_argument(
        "--residual-threshold",
        type=float,
        default=DEFAULT_CONFIG['residual_threshold'],
        metavar="M",
        help=f"Residual threshold in meters (default: {DEFAULT_CONFIG['residual_threshold']})"
    )
    coreg_group.add_argument(
        "--min-tie-points",
        type=int,
        default=DEFAULT_CONFIG['min_tie_points'],
        metavar="N",
        help=f"Minimum tie points required (default: {DEFAULT_CONFIG['min_tie_points']})"
    )
    coreg_group.add_argument(
        "--max-s2-candidates",
        type=int,
        default=DEFAULT_CONFIG['max_s2_candidates'],
        metavar="N",
        help=f"Max S2 candidates to evaluate (default: {DEFAULT_CONFIG['max_s2_candidates']})"
    )
    coreg_group.add_argument(
        "--mad-factor",
        type=float,
        default=DEFAULT_CONFIG['residual_mad_factor'],
        metavar="F",
        help=f"MAD factor for outlier removal (default: {DEFAULT_CONFIG['residual_mad_factor']})"
    )
    coreg_group.add_argument(
        "--s2-band",
        type=int,
        choices=[2, 3, 4, 8],
        default=DEFAULT_CONFIG['s2_ref_band'],
        help=f"S2 reference band: 2=Blue, 3=Green, 4=Red, 8=NIR (default: {DEFAULT_CONFIG['s2_ref_band']})"
    )
    coreg_group.add_argument(
        "--prefer-fixed-band-pairs",
        action="store_true",
        default=DEFAULT_CONFIG['prefer_fixed_band_pairs'],
        help=(
            "Prefer curated fixed PRISMA<->S2 band mappings before wavelength-window averaging "
            f"(default: {DEFAULT_CONFIG['prefer_fixed_band_pairs']})"
        ),
    )
    coreg_group.add_argument(
        "--no-prefer-fixed-band-pairs",
        action="store_false",
        dest="prefer_fixed_band_pairs",
        help="Disable curated fixed PRISMA<->S2 band mappings",
    )
    coreg_group.add_argument(
        "--bandpair-window-nm",
        type=float,
        default=DEFAULT_CONFIG['bandpair_wavelength_window_nm'],
        metavar="NM",
        help=(
            "Wavelength averaging window (nm) for non-fixed band matching fallback "
            f"(default: {DEFAULT_CONFIG['bandpair_wavelength_window_nm']})"
        ),
    )
    coreg_group.add_argument(
        "--min-band-support",
        type=int,
        default=DEFAULT_CONFIG['min_band_support'],
        metavar="N",
        help=(
            "Minimum distinct band matches required for consensus-preferred tie points "
            f"(default: {DEFAULT_CONFIG['min_band_support']})"
        ),
    )
    coreg_group.add_argument(
        "--allow-single-band-fallback",
        action="store_true",
        default=DEFAULT_CONFIG['allow_single_band_fallback'],
        help=(
            "Allow single-band points when consensus filtering yields too few candidates "
            f"(default: {DEFAULT_CONFIG['allow_single_band_fallback']})"
        ),
    )
    coreg_group.add_argument(
        "--no-allow-single-band-fallback",
        action="store_false",
        dest="allow_single_band_fallback",
        help="Disallow single-band fallback when consensus filtering is too restrictive",
    )
    coreg_group.add_argument(
        "--consensus-group-rounding-px",
        type=float,
        default=DEFAULT_CONFIG['consensus_group_rounding_px'],
        metavar="PX",
        help=(
            "Image-space rounding (pixels) for deterministic consensus grouping when POINT_ID is absent "
            f"(default: {DEFAULT_CONFIG['consensus_group_rounding_px']})"
        ),
    )
    coreg_group.add_argument(
        "--strat-grid-rows",
        type=int,
        default=DEFAULT_CONFIG['spatial_stratification_grid_rows'],
        metavar="N",
        help=(
            "Rows in image-space stratification grid "
            f"(default: {DEFAULT_CONFIG['spatial_stratification_grid_rows']})"
        ),
    )
    coreg_group.add_argument(
        "--strat-grid-cols",
        type=int,
        default=DEFAULT_CONFIG['spatial_stratification_grid_cols'],
        metavar="N",
        help=(
            "Columns in image-space stratification grid "
            f"(default: {DEFAULT_CONFIG['spatial_stratification_grid_cols']})"
        ),
    )
    coreg_group.add_argument(
        "--max-points-per-cell",
        type=int,
        default=DEFAULT_CONFIG['max_points_per_cell'],
        metavar="N",
        help=(
            "Maximum selected tie points per spatial grid cell "
            f"(default: {DEFAULT_CONFIG['max_points_per_cell']})"
        ),
    )
    coreg_group.add_argument(
        "--preferred-polynomial-order",
        type=int,
        choices=[1, 2],
        default=DEFAULT_CONFIG['preferred_polynomial_order'],
        help=(
            "Preferred polynomial warp order (auto-downgrade may still apply) "
            f"(default: {DEFAULT_CONFIG['preferred_polynomial_order']})"
        ),
    )
    coreg_group.add_argument(
        "--auto-downgrade-polynomial-order",
        action="store_true",
        default=DEFAULT_CONFIG['auto_downgrade_polynomial_order'],
        help=(
            "Auto-downgrade polynomial order when order-2 geometry is weak "
            f"(default: {DEFAULT_CONFIG['auto_downgrade_polynomial_order']})"
        ),
    )
    coreg_group.add_argument(
        "--no-auto-downgrade-polynomial-order",
        action="store_false",
        dest="auto_downgrade_polynomial_order",
        help="Disable polynomial order auto-downgrade safeguards",
    )
    coreg_group.add_argument(
        "--min-gcps-order2",
        type=int,
        default=DEFAULT_CONFIG['min_gcps_order2'],
        metavar="N",
        help=f"Minimum GCPs to keep order-2 warp (default: {DEFAULT_CONFIG['min_gcps_order2']})",
    )
    coreg_group.add_argument(
        "--min-cells-order2",
        type=int,
        default=DEFAULT_CONFIG['min_cells_order2'],
        metavar="N",
        help=(
            "Minimum occupied stratification cells to keep order-2 warp "
            f"(default: {DEFAULT_CONFIG['min_cells_order2']})"
        ),
    )
    coreg_group.add_argument(
        "--local-grid-res",
        type=int,
        default=DEFAULT_CONFIG['local_coreg_grid_res'],
        metavar="M",
        help=f"COREG_LOCAL grid resolution in meters (default: {DEFAULT_CONFIG['local_coreg_grid_res']})",
    )
    coreg_group.add_argument(
        "--local-window-size",
        type=lambda v: _parse_int_pair(v, "--local-window-size"),
        default=tuple(DEFAULT_CONFIG['local_coreg_window_size']),
        metavar="W,H",
        help=(
            "COREG_LOCAL window size in pixels as 'W,H' "
            f"(default: {DEFAULT_CONFIG['local_coreg_window_size'][0]},{DEFAULT_CONFIG['local_coreg_window_size'][1]})"
        ),
    )
    coreg_group.add_argument(
        "--local-tiep-filter-level",
        type=int,
        default=DEFAULT_CONFIG['local_coreg_tieP_filter_level'],
        metavar="N",
        help=(
            "COREG_LOCAL tie-point filtering level "
            f"(default: {DEFAULT_CONFIG['local_coreg_tieP_filter_level']})"
        ),
    )
    coreg_group.add_argument(
        "--local-max-iter",
        type=int,
        default=DEFAULT_CONFIG['local_coreg_max_iter'],
        metavar="N",
        help="Optional COREG_LOCAL max_iter override (applied only when supported by installed AROSICS)",
    )
    coreg_group.add_argument(
        "--global-coreg-profiles-json",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "Optional JSON object overriding sensor-specific global COREG profiles "
            "(expects {sensor:[{window_size:[w,h],max_shift:n}, ...]})"
        ),
    )
    coreg_group.add_argument(
        "--global-coreg-attempt-ladder-json",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "Optional JSON list overriding global COREG attempt ladder "
            "(expects [{window_size:[w,h],max_shift:n}, ...])"
        ),
    )
    coreg_group.add_argument(
        "--local-max-shift-by-sensor-json",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "Optional JSON object overriding local max_shift by sensor "
            "(expects {sensor:max_shift_px})"
        ),
    )
    coreg_group.add_argument(
        "--postwarp-phasecorr-check",
        action="store_true",
        default=DEFAULT_CONFIG['postwarp_phasecorr_check'],
        help=(
            "Run optional post-warp phase-correlation QA on final HS candidate "
            f"(default: {DEFAULT_CONFIG['postwarp_phasecorr_check']})"
        ),
    )
    coreg_group.add_argument(
        "--no-postwarp-phasecorr-check",
        action="store_false",
        dest="postwarp_phasecorr_check",
        help="Disable post-warp phase-correlation QA",
    )
    coreg_group.add_argument(
        "--postwarp-phasecorr-warn-threshold-px",
        type=float,
        default=DEFAULT_CONFIG['postwarp_phasecorr_warn_threshold_px'],
        metavar="PX",
        help=(
            "Warning threshold in pixels for post-warp phase-correlation QA "
            f"(default: {DEFAULT_CONFIG['postwarp_phasecorr_warn_threshold_px']})"
        ),
    )
    coreg_group.add_argument(
        "--postwarp-phasecorr-reject-threshold-px",
        type=float,
        default=DEFAULT_CONFIG['postwarp_phasecorr_reject_threshold_px'],
        metavar="PX",
        help=(
            "Rejection threshold in pixels for post-warp phase-correlation QA "
            f"(default: {DEFAULT_CONFIG['postwarp_phasecorr_reject_threshold_px']})"
        ),
    )
    coreg_group.add_argument(
        "--postwarp-phasecorr-reject-bad",
        action="store_true",
        default=DEFAULT_CONFIG['postwarp_phasecorr_reject_bad'],
        help=(
            "Reject candidate outputs that exceed post-warp phase-correlation rejection threshold "
            f"(default: {DEFAULT_CONFIG['postwarp_phasecorr_reject_bad']})"
        ),
    )
    coreg_group.add_argument(
        "--no-postwarp-phasecorr-reject-bad",
        action="store_false",
        dest="postwarp_phasecorr_reject_bad",
        help="Do not reject candidates based on post-warp phase-correlation QA",
    )
    coreg_group.add_argument(
        "--postwarp-phasecorr-max-dim",
        type=int,
        default=DEFAULT_CONFIG['postwarp_phasecorr_max_dim'],
        metavar="N",
        help=(
            "Maximum dimension used in post-warp phase-correlation QA downsampling "
            f"(default: {DEFAULT_CONFIG['postwarp_phasecorr_max_dim']})"
        ),
    )
    coreg_group.add_argument(
        "--use-geolocation-mesh-affine",
        action="store_true",
        default=DEFAULT_CONFIG['use_geolocation_mesh_affine'],
        help=(
            "Use optional subsampled PRISMA geolocation mesh for affine estimation "
            f"(default: {DEFAULT_CONFIG['use_geolocation_mesh_affine']})"
        ),
    )
    coreg_group.add_argument(
        "--no-use-geolocation-mesh-affine",
        action="store_false",
        dest="use_geolocation_mesh_affine",
        help="Disable subsampled geolocation-mesh affine estimation",
    )
    coreg_group.add_argument(
        "--geolocation-mesh-stride",
        type=int,
        default=DEFAULT_CONFIG['geolocation_mesh_stride'],
        metavar="N",
        help=(
            "Sampling stride (pixels) for optional geolocation-mesh affine estimation "
            f"(default: {DEFAULT_CONFIG['geolocation_mesh_stride']})"
        ),
    )

    # Quality options
    quality_group = parser.add_argument_group("Quality Options")
    quality_group.add_argument(
        "--min-accuracy",
        type=float,
        default=DEFAULT_CONFIG['min_accuracy'],
        metavar="PCT",
        help=f"Minimum accuracy %% (default: {DEFAULT_CONFIG['min_accuracy']})"
    )
    quality_group.add_argument(
        "--max-displacement",
        type=float,
        default=DEFAULT_CONFIG['max_displacement'],
        metavar="M",
        help=f"Maximum displacement in meters (default: {DEFAULT_CONFIG['max_displacement']})"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--save-pre",
        action="store_true",
        help="Save pre-coregistration image"
    )
    output_group.add_argument(
        "--save-pan",
        action="store_true",
        default=DEFAULT_CONFIG["save_pan"],
        help="Save panchromatic band (PRISMA only, default: False)"
    )
    output_group.add_argument(
        "--no-save-pan",
        action="store_false",
        dest="save_pan",
        help="Do not save panchromatic band"
    )
    output_group.add_argument(
        "--save-quality-mask",
        action="store_true",
        default=DEFAULT_CONFIG["save_quality_mask"],
        help="Save ancillary quality outputs (PRISMA masks + EnMAP QL auxiliaries, default: False)"
    )
    output_group.add_argument(
        "--no-save-quality-mask",
        action="store_false",
        dest="save_quality_mask",
        help="Do not save ancillary quality outputs"
    )
    output_group.add_argument(
        "--pan-gcp-mode",
        type=str,
        choices=["map_inverse", "scaled_image"],
        default=DEFAULT_CONFIG['pan_gcp_mode'],
        help=(
            "PAN GCP construction mode: map_inverse (legacy inverse-transform from map coords) or "
            f"scaled_image (X_IM/Y_IM scaled to PAN grid) (default: {DEFAULT_CONFIG['pan_gcp_mode']})"
        ),
    )
    output_group.add_argument(
        "--pan-map-dxdy-source",
        type=str,
        choices=["auto", "xy_shift_m", "zero"],
        default=DEFAULT_CONFIG['pan_map_dxdy_source'],
        help=(
            "Source of map-space dx/dy for PAN scaled-image GCPs: auto, xy_shift_m, or zero "
            f"(default: {DEFAULT_CONFIG['pan_map_dxdy_source']})"
        ),
    )
    output_group.add_argument(
        "--pan-target-aligned-pixels",
        action="store_true",
        default=DEFAULT_CONFIG['pan_target_aligned_pixels'],
        help=(
            "Use gdalwarp target-aligned pixels (-tap) for PAN warp "
            f"(default: {DEFAULT_CONFIG['pan_target_aligned_pixels']})"
        ),
    )
    output_group.add_argument(
        "--no-pan-target-aligned-pixels",
        action="store_false",
        dest="pan_target_aligned_pixels",
        help="Disable gdalwarp target-aligned pixels (-tap) for PAN warp",
    )
    output_group.add_argument(
        "--pan-residual-check",
        action="store_true",
        default=DEFAULT_CONFIG['pan_residual_check'],
        help=(
            "Run optional post-warp PAN residual translation estimate (phase correlation) "
            f"(default: {DEFAULT_CONFIG['pan_residual_check']})"
        ),
    )
    output_group.add_argument(
        "--no-pan-residual-check",
        action="store_false",
        dest="pan_residual_check",
        help="Disable post-warp PAN residual translation estimate",
    )
    output_group.add_argument(
        "--pan-residual-threshold-px",
        type=float,
        default=DEFAULT_CONFIG['pan_residual_threshold_px'],
        metavar="PX",
        help=(
            "Warn when PAN residual shift estimate exceeds this pixel threshold "
            f"(default: {DEFAULT_CONFIG['pan_residual_threshold_px']})"
        ),
    )
    output_group.add_argument(
        "--pan-residual-max-dim",
        type=int,
        default=DEFAULT_CONFIG['pan_residual_max_dim'],
        metavar="N",
        help=(
            "Max display dimension used by PAN residual check phase-correlation "
            f"(default: {DEFAULT_CONFIG['pan_residual_max_dim']})"
        ),
    )
    output_group.add_argument(
        "--gen-tiepoint-pngs",
        action="store_true",
        default=DEFAULT_CONFIG['gen_tiepoint_pngs'],
        help=f"Generate tie point visualizations (default: {DEFAULT_CONFIG['gen_tiepoint_pngs']})"
    )
    output_group.add_argument(
        "--no-gen-tiepoint-pngs",
        action="store_false",
        dest="gen_tiepoint_pngs",
        help="Do not generate tie point visualizations"
    )
    output_group.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files"
    )
    output_group.add_argument(
        "--remove-overlap-bands",
        action="store_true",
        default=DEFAULT_CONFIG['remove_detector_overlap_bands'],
        help="Remove VNIR/SWIR overlap bands (keeps SWIR in overlap region)"
    )
    output_group.add_argument(
        "--no-remove-overlap-bands",
        action="store_false",
        dest="remove_overlap_bands",
        help="Keep all VNIR/SWIR bands (including overlap region)"
    )
    output_group.add_argument(
        "--normalization-mode",
        type=str,
        choices=["none", "minmax", "percentile"],
        default=DEFAULT_CONFIG['normalization_mode'],
        help=(
            "Normalization mode: none (disabled), minmax (per-band min/max), "
            f"percentile (robust per-band percentile scaling) (default: {DEFAULT_CONFIG['normalization_mode']})"
        ),
    )
    output_group.add_argument(
        "--build-overviews",
        action="store_true",
        default=DEFAULT_CONFIG['build_overviews'],
        help="Build internal GeoTIFF overviews (2,4,8,16,32) for faster ENVI/QGIS display"
    )
    output_group.add_argument(
        "--no-build-overviews",
        action="store_false",
        dest="build_overviews",
        help="Do not build internal GeoTIFF overviews"
    )
    output_group.add_argument(
        "--strict-metadata",
        action="store_true",
        default=DEFAULT_CONFIG['strict_metadata'],
        help="Fail scene when output metadata writing is incomplete (default: True)"
    )
    output_group.add_argument(
        "--no-strict-metadata",
        action="store_false",
        dest="strict_metadata",
        help="Allow degraded metadata output with warnings when non-critical writes fail"
    )
    output_group.add_argument(
        "--metadata-extension",
        type=str,
        choices=["none", "stats", "full"],
        default=DEFAULT_CONFIG['metadata_extension_level'],
        help=(
            "Metadata sidecar extension level: none (disabled), stats (PAM statistics), "
            "full (statistics + histogram)"
        ),
    )
    output_group.add_argument(
        "--metadata-stats-mode",
        type=str,
        choices=["exact", "approx", "none"],
        default=DEFAULT_CONFIG['metadata_stats_mode'],
        help=(
            "Metadata statistics mode: exact (full scan), approx (deterministic sampled windows), "
            "none (skip STATISTICS_* fields)"
        ),
    )
    output_group.add_argument(
        "--enmap-metadata-stats-mode",
        type=str,
        choices=["exact", "approx", "none"],
        default=DEFAULT_CONFIG['enmap_metadata_stats_mode'],
        help=(
            "EnMAP-only metadata statistics mode override. Defaults to none for faster EnMAP output generation; "
            "PRISMA still uses --metadata-stats-mode."
        ),
    )
    output_group.add_argument(
        "--metadata-stats-sample-windows",
        type=int,
        default=DEFAULT_CONFIG['metadata_stats_sample_windows'],
        metavar="N",
        help=(
            "Target sampled windows per band when metadata-stats-mode=approx "
            f"(default: {DEFAULT_CONFIG['metadata_stats_sample_windows']})"
        ),
    )
    output_group.add_argument(
        "--metadata-stats-seed",
        type=int,
        default=DEFAULT_CONFIG['metadata_stats_seed'],
        metavar="N",
        help=f"Deterministic seed for approximate metadata stats (default: {DEFAULT_CONFIG['metadata_stats_seed']})",
    )
    output_group.add_argument(
        "--metadata-histogram-buckets",
        type=int,
        default=DEFAULT_CONFIG['metadata_histogram_buckets'],
        metavar="N",
        help=f"Histogram bucket count for full metadata extension (default: {DEFAULT_CONFIG['metadata_histogram_buckets']})",
    )
    output_group.add_argument(
        "--metadata-label-precision",
        type=int,
        default=DEFAULT_CONFIG['metadata_label_precision'],
        metavar="N",
        help=f"Decimal precision for wavelength labels (default: {DEFAULT_CONFIG['metadata_label_precision']})",
    )
    output_group.add_argument(
        "--validation-max-windows",
        type=int,
        default=DEFAULT_CONFIG['validation_max_windows'],
        metavar="N",
        help=(
            "Optional cap on scanned windows for final validation (0 = full scan; "
            f"default: {DEFAULT_CONFIG['validation_max_windows']})"
        ),
    )
    # General options
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Increase verbosity (DEBUG level)"
    )
    general_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output (WARNING level only)"
    )
    general_group.add_argument(
        "--use-pipeline-native",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Enable pipeline-native backend path during migration rollout. "
            "When omitted, HYPERCOREG_USE_PIPELINE_NATIVE may supply the value."
        ),
    )
    general_group.add_argument(
        "--disable-legacy-fallback",
        action="store_false",
        dest="enable_legacy_fallback",
        default=argparse.SUPPRESS,
        help=(
            "Disable fallback to legacy backend when native backend fails. "
            "When omitted, HYPERCOREG_ENABLE_LEGACY_FALLBACK may supply the value."
        ),
    )
    general_group.add_argument(
        "--assert-legacy-parity",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Run optional parity shadow-check against legacy backend for selected output fields. "
            "When omitted, HYPERCOREG_ASSERT_LEGACY_PARITY may supply the value."
        ),
    )


def build_config_from_args(args: argparse.Namespace) -> dict:
    """
    Build configuration dictionary from parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        dict: Configuration dictionary for coregistration
    """
    global_coreg_profiles_json = _parse_json_dict(
        getattr(args, "global_coreg_profiles_json", None),
        "--global-coreg-profiles-json",
    )
    global_coreg_attempt_ladder_json = _parse_json_list(
        getattr(args, "global_coreg_attempt_ladder_json", None),
        "--global-coreg-attempt-ladder-json",
    )
    local_max_shift_by_sensor_json = _parse_json_dict(
        getattr(args, "local_max_shift_by_sensor_json", None),
        "--local-max-shift-by-sensor-json",
    )
    local_window_size = getattr(args, "local_window_size", DEFAULT_CONFIG['local_coreg_window_size'])
    if isinstance(local_window_size, list):
        local_window_size = tuple(local_window_size)

    config = {
        'input_path': args.input,
        'output_dir': args.output,
        'batch_mode': args.mode == 'batch',

        # S2 search
        'days_window': args.days_window,
        'min_overlap': args.min_overlap,
        'max_cloud': args.max_cloud,
        'max_input_cloud_cover': args.max_input_cloud,
        'local_s2_stack_path': getattr(args, "local_s2_stack", DEFAULT_CONFIG.get('local_s2_stack_path')),

        # Coregistration
        'residual_threshold': args.residual_threshold,
        'min_tie_points': args.min_tie_points,
        'max_s2_candidates': args.max_s2_candidates,
        'residual_mad_factor': args.mad_factor,
        's2_ref_band': args.s2_band,
        'prefer_fixed_band_pairs': getattr(
            args,
            "prefer_fixed_band_pairs",
            DEFAULT_CONFIG['prefer_fixed_band_pairs'],
        ),
        'bandpair_wavelength_window_nm': getattr(
            args,
            "bandpair_window_nm",
            DEFAULT_CONFIG['bandpair_wavelength_window_nm'],
        ),
        'min_band_support': getattr(args, "min_band_support", DEFAULT_CONFIG['min_band_support']),
        'allow_single_band_fallback': getattr(
            args,
            "allow_single_band_fallback",
            DEFAULT_CONFIG['allow_single_band_fallback'],
        ),
        'consensus_group_rounding_px': getattr(
            args,
            "consensus_group_rounding_px",
            DEFAULT_CONFIG['consensus_group_rounding_px'],
        ),
        'spatial_stratification_grid_rows': getattr(
            args,
            "strat_grid_rows",
            DEFAULT_CONFIG['spatial_stratification_grid_rows'],
        ),
        'spatial_stratification_grid_cols': getattr(
            args,
            "strat_grid_cols",
            DEFAULT_CONFIG['spatial_stratification_grid_cols'],
        ),
        'max_points_per_cell': getattr(
            args,
            "max_points_per_cell",
            DEFAULT_CONFIG['max_points_per_cell'],
        ),
        'preferred_polynomial_order': getattr(
            args,
            "preferred_polynomial_order",
            DEFAULT_CONFIG['preferred_polynomial_order'],
        ),
        'auto_downgrade_polynomial_order': getattr(
            args,
            "auto_downgrade_polynomial_order",
            DEFAULT_CONFIG['auto_downgrade_polynomial_order'],
        ),
        'min_gcps_order2': getattr(args, "min_gcps_order2", DEFAULT_CONFIG['min_gcps_order2']),
        'min_cells_order2': getattr(args, "min_cells_order2", DEFAULT_CONFIG['min_cells_order2']),
        'local_coreg_grid_res': getattr(
            args,
            "local_grid_res",
            DEFAULT_CONFIG['local_coreg_grid_res'],
        ),
        'local_coreg_window_size': tuple(local_window_size),
        'local_coreg_tieP_filter_level': getattr(
            args,
            "local_tiep_filter_level",
            DEFAULT_CONFIG['local_coreg_tieP_filter_level'],
        ),
        'local_coreg_max_iter': getattr(args, "local_max_iter", DEFAULT_CONFIG['local_coreg_max_iter']),
        'global_coreg_profiles_by_sensor': (
            global_coreg_profiles_json
            if global_coreg_profiles_json is not None
            else DEFAULT_CONFIG['global_coreg_profiles_by_sensor']
        ),
        'global_coreg_attempt_ladder': global_coreg_attempt_ladder_json,
        'local_max_shift_by_sensor': (
            local_max_shift_by_sensor_json
            if local_max_shift_by_sensor_json is not None
            else DEFAULT_CONFIG['local_max_shift_by_sensor']
        ),
        'postwarp_phasecorr_check': getattr(
            args,
            "postwarp_phasecorr_check",
            DEFAULT_CONFIG['postwarp_phasecorr_check'],
        ),
        'postwarp_phasecorr_warn_threshold_px': getattr(
            args,
            "postwarp_phasecorr_warn_threshold_px",
            DEFAULT_CONFIG['postwarp_phasecorr_warn_threshold_px'],
        ),
        'postwarp_phasecorr_reject_threshold_px': getattr(
            args,
            "postwarp_phasecorr_reject_threshold_px",
            DEFAULT_CONFIG['postwarp_phasecorr_reject_threshold_px'],
        ),
        'postwarp_phasecorr_reject_bad': getattr(
            args,
            "postwarp_phasecorr_reject_bad",
            DEFAULT_CONFIG['postwarp_phasecorr_reject_bad'],
        ),
        'postwarp_phasecorr_max_dim': getattr(
            args,
            "postwarp_phasecorr_max_dim",
            DEFAULT_CONFIG['postwarp_phasecorr_max_dim'],
        ),
        'use_geolocation_mesh_affine': getattr(
            args,
            "use_geolocation_mesh_affine",
            DEFAULT_CONFIG['use_geolocation_mesh_affine'],
        ),
        'geolocation_mesh_stride': getattr(
            args,
            "geolocation_mesh_stride",
            DEFAULT_CONFIG['geolocation_mesh_stride'],
        ),
        'fixed_band_pairs_by_sensor': DEFAULT_CONFIG.get('fixed_band_pairs_by_sensor', {}),

        # Quality
        'min_accuracy': args.min_accuracy,
        'max_displacement': args.max_displacement,

        # Output
        'save_pre': args.save_pre,
        'save_pan': args.save_pan,
        'save_quality_mask': args.save_quality_mask,
        'pan_gcp_mode': getattr(args, "pan_gcp_mode", DEFAULT_CONFIG['pan_gcp_mode']),
        'pan_map_dxdy_source': getattr(args, "pan_map_dxdy_source", DEFAULT_CONFIG['pan_map_dxdy_source']),
        'pan_target_aligned_pixels': getattr(
            args,
            "pan_target_aligned_pixels",
            DEFAULT_CONFIG['pan_target_aligned_pixels'],
        ),
        'pan_residual_check': getattr(args, "pan_residual_check", DEFAULT_CONFIG['pan_residual_check']),
        'pan_residual_threshold_px': getattr(
            args,
            "pan_residual_threshold_px",
            DEFAULT_CONFIG['pan_residual_threshold_px'],
        ),
        'pan_residual_max_dim': getattr(
            args,
            "pan_residual_max_dim",
            DEFAULT_CONFIG['pan_residual_max_dim'],
        ),
        'gen_tiepoint_pngs': args.gen_tiepoint_pngs,
        'use_inmemory': True,
        'keep_temp_files': args.keep_temp,
        'allow_gui_prompt': False,
        'remove_detector_overlap_bands': args.remove_overlap_bands,
        'normalization_mode': getattr(args, "normalization_mode", DEFAULT_CONFIG['normalization_mode']),
        'norm_p_low': DEFAULT_CONFIG['norm_p_low'],
        'norm_p_high': DEFAULT_CONFIG['norm_p_high'],
        'norm_clip': DEFAULT_CONFIG['norm_clip'],
        'norm_eps': DEFAULT_CONFIG['norm_eps'],
        'norm_min_valid_pixels': DEFAULT_CONFIG['norm_min_valid_pixels'],
        'norm_reservoir_size': DEFAULT_CONFIG['norm_reservoir_size'],
        'norm_seed': DEFAULT_CONFIG['norm_seed'],
        'norm_tile_size': DEFAULT_CONFIG['norm_tile_size'],
        'build_overviews': getattr(args, "build_overviews", DEFAULT_CONFIG['build_overviews']),
        'strict_metadata': args.strict_metadata,
        'metadata_extension_level': getattr(
            args, "metadata_extension", DEFAULT_CONFIG['metadata_extension_level']
        ),
        'metadata_stats_mode': getattr(
            args, "metadata_stats_mode", DEFAULT_CONFIG['metadata_stats_mode']
        ),
        'enmap_metadata_stats_mode': getattr(
            args,
            "enmap_metadata_stats_mode",
            DEFAULT_CONFIG['enmap_metadata_stats_mode'],
        ),
        'metadata_stats_sample_windows': getattr(
            args, "metadata_stats_sample_windows", DEFAULT_CONFIG['metadata_stats_sample_windows']
        ),
        'metadata_stats_seed': getattr(
            args, "metadata_stats_seed", DEFAULT_CONFIG['metadata_stats_seed']
        ),
        'metadata_histogram_buckets': getattr(
            args, "metadata_histogram_buckets", DEFAULT_CONFIG['metadata_histogram_buckets']
        ),
        'metadata_label_precision': getattr(
            args, "metadata_label_precision", DEFAULT_CONFIG['metadata_label_precision']
        ),
        'validation_max_windows': getattr(
            args, "validation_max_windows", DEFAULT_CONFIG['validation_max_windows']
        ),
        'defer_temp_cleanup_gui': False,
        'timing_logs': True,
    }

    if hasattr(args, "use_pipeline_native"):
        config["use_pipeline_native"] = bool(getattr(args, "use_pipeline_native"))
    if hasattr(args, "enable_legacy_fallback"):
        config["enable_legacy_fallback"] = bool(getattr(args, "enable_legacy_fallback"))
    if hasattr(args, "assert_legacy_parity"):
        config["assert_legacy_parity"] = bool(getattr(args, "assert_legacy_parity"))

    return config


def main(args: Optional[list] = None) -> int:
    """
    Main CLI entry point.

    Args:
        args: Command-line arguments (defaults to sys.argv)

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # Determine log level
    if parsed_args.quiet:
        log_level = logging.WARNING
    elif parsed_args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # Build config
    config = build_config_from_args(parsed_args)

    # Setup logging
    setup_logging(
        output_dir=config['output_dir'],
        log_level=log_level,
        verbose=parsed_args.verbose
    )

    log_section_header("HYPERCOREG CLI")

    logger = logging.getLogger("COREG_PROCESSING")
    logger.info(f"HyperCoreg version {__version__}")
    logger.info(f"Mode: {parsed_args.mode}")
    logger.info(f"Input: {config['input_path']}")
    logger.info(f"Output: {config['output_dir']}")

    try:
        # Import here to avoid circular imports and speed up --help
        from hypercoreg.coregistration import run_coregistration, run_batch_coregistration
        from hypercoreg.utils import detect_hyp_type

        if parsed_args.mode == "single":
            hyp_type = detect_hyp_type(config['input_path'])
            logger.info(f"Detected sensor type: {hyp_type}")
            run_coregistration(
                config['input_path'],
                hyp_type,
                config['output_dir'],
                config
            )
        else:  # batch
            run_batch_coregistration(
                config['input_path'],
                config['output_dir'],
                config
            )

        logger.info("Processing completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
