"""Radiometric contracts for PRISMA Level-2 products.

PRISMA Level-2 detector arrays are uint16-encoded samples.  The scale
coefficients stored in each product are part of the product definition and
must be applied independently from optional, scene-dependent normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


PRISMA_DN_DENOMINATOR = 65535
PRISMA_RADIOMETRIC_MODES = ("reflectance", "native-dn")


class PrismaRadiometricMetadataError(ValueError):
    """Raised when a PRISMA product has unusable radiometric metadata."""


@dataclass(frozen=True)
class PrismaScalePair:
    """Validated minimum/maximum scale coefficients for one detector."""

    minimum: float
    maximum: float

    @property
    def gain(self) -> float:
        return (self.maximum - self.minimum) / float(PRISMA_DN_DENOMINATOR)

    @property
    def offset(self) -> float:
        return self.minimum


def normalize_prisma_radiometric_mode(value: Any, default: str = "reflectance") -> str:
    """Return a validated PRISMA radiometric mode."""
    mode = str(default if value is None else value).strip().lower().replace("_", "-")
    if mode not in PRISMA_RADIOMETRIC_MODES:
        choices = ", ".join(PRISMA_RADIOMETRIC_MODES)
        raise ValueError(f"Invalid PRISMA radiometric mode {value!r}; expected one of: {choices}")
    return mode


def validate_prisma_scale_pair(scale_min: Any, scale_max: Any, detector: str) -> PrismaScalePair:
    """Validate and return one detector's scalar, finite scale pair."""
    label = str(detector).upper()

    def _scalar(value: Any, name: str) -> float:
        arr = np.asarray(value)
        if arr.ndim != 0:
            raise PrismaRadiometricMetadataError(
                f"PRISMA {label} {name} must be scalar; got shape {arr.shape}"
            )
        try:
            result = float(arr.item())
        except (TypeError, ValueError, OverflowError) as exc:
            raise PrismaRadiometricMetadataError(
                f"PRISMA {label} {name} is not numeric: {value!r}"
            ) from exc
        if not np.isfinite(result):
            raise PrismaRadiometricMetadataError(
                f"PRISMA {label} {name} must be finite; got {result!r}"
            )
        return result

    minimum = _scalar(scale_min, "scale minimum")
    maximum = _scalar(scale_max, "scale maximum")
    if maximum <= minimum:
        raise PrismaRadiometricMetadataError(
            f"PRISMA {label} scale maximum must be greater than minimum; "
            f"got minimum={minimum!r}, maximum={maximum!r}"
        )
    return PrismaScalePair(minimum=minimum, maximum=maximum)


def validate_prisma_native_samples(
    samples: Optional[np.ndarray], detector: str
) -> Optional[np.ndarray]:
    """Return native PRISMA samples after validating their encoded state.

    PRISMA Level-2 detector samples are stored as unsigned 16-bit integers.
    Checking ``kind`` and ``itemsize`` deliberately accepts either byte order
    while rejecting float arrays that may already have been radiometrically
    decoded.  ``None`` is preserved for optional detector inputs such as PAN.
    """
    if samples is None:
        return None
    values = np.asarray(samples)
    if values.dtype.kind != "u" or values.dtype.itemsize != 2:
        label = str(detector).upper()
        raise TypeError(
            f"PRISMA {label} native samples must use uint16 storage; got {values.dtype}. "
            "Refusing possible double decoding."
        )
    return values


def decode_prisma_l2(samples: np.ndarray, scale: PrismaScalePair) -> np.ndarray:
    """Decode native PRISMA uint16 samples to unitless float32 reflectance.

    Requiring uint16 is an intentional state guard: already-decoded float data
    cannot silently pass through the Level-2 affine transform a second time.
    """
    values = validate_prisma_native_samples(samples, "Level-2")
    if values is None:
        raise TypeError("PRISMA Level-2 decoding requires native uint16 samples; got None")
    gain = np.float32(scale.gain)
    offset = np.float32(scale.offset)
    return values.astype(np.float32) * gain + offset


def _scale_from_metadata(metadata: Mapping[str, Any], detector: str) -> Optional[PrismaScalePair]:
    key = str(detector).lower()
    minimum = metadata.get(f"prisma_l2_scale_{key}_min")
    maximum = metadata.get(f"prisma_l2_scale_{key}_max")
    recorded_error = metadata.get(f"prisma_l2_scale_{key}_error")
    if recorded_error:
        raise PrismaRadiometricMetadataError(str(recorded_error))
    if minimum is None and maximum is None:
        return None
    if minimum is None or maximum is None:
        missing = "minimum" if minimum is None else "maximum"
        raise PrismaRadiometricMetadataError(
            f"PRISMA {str(detector).upper()} scale {missing} is missing"
        )
    return validate_prisma_scale_pair(minimum, maximum, detector)


def build_prisma_radiometric_contract(
    metadata: Mapping[str, Any],
    mode: str,
    normalization_mode: str = "none",
    normalization_clip: Optional[bool] = None,
    spatial_resampling_kernel: Optional[str] = None,
    require_pan: bool = False,
) -> Dict[str, Any]:
    """Build a JSON-safe radiometric state/provenance contract."""
    resolved_mode = normalize_prisma_radiometric_mode(mode)
    norm_mode = str(normalization_mode or "none").strip().lower()
    scales: Dict[str, Dict[str, float]] = {}
    optional_scale_errors: Dict[str, str] = {}
    for detector in ("VNIR", "SWIR", "PAN"):
        try:
            scale = _scale_from_metadata(metadata, detector)
        except PrismaRadiometricMetadataError as exc:
            if detector == "PAN" and not require_pan:
                optional_scale_errors[detector] = str(exc)
                continue
            if detector == "PAN":
                raise PrismaRadiometricMetadataError(
                    f"Invalid required PRISMA PAN scale metadata: {exc}"
                ) from exc
            raise
        if scale is None:
            if detector in {"VNIR", "SWIR"} or (detector == "PAN" and require_pan):
                raise PrismaRadiometricMetadataError(
                    f"Missing PRISMA L2 scale coefficients for {detector}"
                )
            continue
        scales[detector] = {
            "minimum": float(scale.minimum),
            "maximum": float(scale.maximum),
            "gain": float(scale.gain),
            "offset": float(scale.offset),
        }

    if norm_mode == "none":
        quantity = "surface_reflectance" if resolved_mode == "reflectance" else "native_encoded_dn"
        units = "1" if resolved_mode == "reflectance" else "DN"
    else:
        quantity = "normalized_unitless"
        units = "1"

    if norm_mode == "none":
        overshoot_policy = "preserve_and_flag_not_clip"
    else:
        clipping_state = (
            "with_clipping"
            if normalization_clip is not False
            else "without_clipping"
        )
        overshoot_policy = (
            f"transformed_by_{norm_mode}_normalization_{clipping_state}"
        )

    return {
        "mode": resolved_mode,
        "normalization_mode": norm_mode,
        "quantity": quantity,
        "units": units,
        "l2_scaling_applied": resolved_mode == "reflectance",
        "dn_denominator": PRISMA_DN_DENOMINATOR,
        "source_product_name": metadata.get("source_product_name") or metadata.get("prisma_id"),
        "source_processing_level": metadata.get("source_processing_level"),
        "source_processor_version": metadata.get("source_processor_version"),
        "source_processing_time": metadata.get("source_processing_time"),
        "spatial_resampling_kernel": spatial_resampling_kernel,
        "resampling_overshoot_policy": overshoot_policy,
        "scales": scales,
        "optional_scale_errors": optional_scale_errors,
    }


def prisma_radiometric_report_fields(
    contract: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Flatten contract fields used by dataset and batch report rows."""
    radiometric = dict(contract or {})
    scales = radiometric.get("scales")
    if not isinstance(scales, Mapping):
        scales = {}
    fields: Dict[str, Any] = {
        "prisma_radiometric_mode": radiometric.get("mode"),
        "radiometric_quantity": radiometric.get("quantity"),
        "prisma_l2_scaling_applied": radiometric.get("l2_scaling_applied"),
        "radiometric": radiometric,
    }
    for detector in ("VNIR", "SWIR", "PAN"):
        scale = scales.get(detector)
        if not isinstance(scale, Mapping):
            scale = {}
        key = detector.lower()
        fields[f"prisma_l2_scale_{key}_min"] = scale.get("minimum")
        fields[f"prisma_l2_scale_{key}_max"] = scale.get("maximum")
    return fields


def prisma_radiometric_dataset_tags(contract: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Return stable GeoTIFF/PAM dataset tags for a radiometric contract."""
    if not contract:
        return {}
    tags: Dict[str, str] = {
        "RADIOMETRIC_QUANTITY": str(contract.get("quantity", "unknown")),
        "RADIOMETRIC_UNITS": str(contract.get("units", "unknown")),
        "NORMALIZATION_MODE": str(contract.get("normalization_mode", "none")),
        "PRISMA_RADIOMETRIC_MODE": str(contract.get("mode", "unknown")),
        "PRISMA_L2_SCALING_APPLIED": str(bool(contract.get("l2_scaling_applied", False))).lower(),
        "PRISMA_DN_DENOMINATOR": str(contract.get("dn_denominator", PRISMA_DN_DENOMINATOR)),
        "RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY": str(
            contract.get("resampling_overshoot_policy", "preserve_and_flag_not_clip")
        ),
    }
    range_validation = contract.get("range_validation") or {}
    for source_key, tag_key in (
        ("status", "RADIOMETRIC_RANGE_VALIDATION_STATUS"),
        ("sampling_basis", "RADIOMETRIC_RANGE_VALIDATION_BASIS"),
        (
            "sampled_below_range_band_pixels",
            "RADIOMETRIC_RANGE_SAMPLED_BELOW_SOURCE_RANGE_BAND_PIXELS",
        ),
        (
            "sampled_above_range_band_pixels",
            "RADIOMETRIC_RANGE_SAMPLED_ABOVE_SOURCE_RANGE_BAND_PIXELS",
        ),
        (
            "sampled_nonfinite_band_pixels",
            "RADIOMETRIC_RANGE_SAMPLED_NONFINITE_BAND_PIXELS",
        ),
        (
            "sampled_valid_band_pixels",
            "RADIOMETRIC_RANGE_SAMPLED_VALID_BAND_PIXELS",
        ),
        ("sampled_band_pixels_read", "RADIOMETRIC_RANGE_SAMPLED_BAND_PIXELS_READ"),
        ("blocks_scanned", "RADIOMETRIC_RANGE_BLOCKS_SCANNED"),
        ("blocks_total", "RADIOMETRIC_RANGE_BLOCKS_TOTAL"),
        ("block_coverage_fraction", "RADIOMETRIC_RANGE_BLOCK_COVERAGE_FRACTION"),
        ("spatial_pixels_scanned", "RADIOMETRIC_RANGE_SPATIAL_PIXELS_SCANNED"),
        ("spatial_pixels_total", "RADIOMETRIC_RANGE_SPATIAL_PIXELS_TOTAL"),
        ("spatial_coverage_fraction", "RADIOMETRIC_RANGE_SPATIAL_COVERAGE_FRACTION"),
        ("truncated", "RADIOMETRIC_RANGE_SCAN_TRUNCATED"),
    ):
        if source_key not in range_validation:
            continue
        if source_key != "status" and range_validation.get("status") == "not_applicable":
            continue
        value = range_validation[source_key]
        if value is None:
            continue
        tags[tag_key] = str(value).lower() if isinstance(value, bool) else str(value)
    for key, tag in (
        ("source_product_name", "SOURCE_PRODUCT_NAME"),
        ("source_processing_level", "SOURCE_PROCESSING_LEVEL"),
        ("source_processor_version", "SOURCE_PROCESSOR_VERSION"),
        ("source_processing_time", "SOURCE_PROCESSING_TIME"),
        ("spatial_resampling_kernel", "SPATIAL_RESAMPLING_KERNEL"),
    ):
        value = contract.get(key)
        if value is not None:
            tags[tag] = str(value)

    scales = contract.get("scales", {}) or {}
    for detector, values in scales.items():
        detector_key = str(detector).upper()
        for field in ("minimum", "maximum"):
            if field in values:
                suffix = "MIN" if field == "minimum" else "MAX"
                tags[f"PRISMA_L2_SCALE_{detector_key}_{suffix}"] = str(values[field])
    return tags


def prisma_radiometric_band_tags(
    contract: Optional[Mapping[str, Any]], detector: str
) -> Dict[str, str]:
    """Return per-band provenance tags for the given detector."""
    if not contract:
        return {}
    detector_key = str(detector).upper()
    tags = {
        "RADIOMETRIC_QUANTITY": str(contract.get("quantity", "unknown")),
        "RADIOMETRIC_UNITS": str(contract.get("units", "unknown")),
    }
    scale = (contract.get("scales", {}) or {}).get(detector_key)
    if scale:
        tags["SOURCE_DN_GAIN"] = str(scale["gain"])
        tags["SOURCE_DN_OFFSET"] = str(scale["offset"])
        if contract.get("quantity") == "native_encoded_dn":
            tags["SOURCE_VALID_RANGE_MIN"] = "0"
            tags["SOURCE_VALID_RANGE_MAX"] = str(PRISMA_DN_DENOMINATOR)
        elif contract.get("quantity") == "surface_reflectance":
            tags["SOURCE_VALID_RANGE_MIN"] = str(scale["minimum"])
            tags["SOURCE_VALID_RANGE_MAX"] = str(scale["maximum"])
        if contract.get("quantity") in {"native_encoded_dn", "surface_reflectance"}:
            tags["INTERPOLATION_OVERSHOOTS_PRESERVED"] = "true"
    return tags


def prisma_active_scale_offset(
    contract: Optional[Mapping[str, Any]], detector: str
) -> Tuple[float, float]:
    """Return the active GDAL scale/offset for output pixels."""
    if not contract or contract.get("quantity") != "native_encoded_dn":
        return 1.0, 0.0
    scale = (contract.get("scales", {}) or {}).get(str(detector).upper())
    if not scale:
        return 1.0, 0.0
    return float(scale["gain"]), float(scale["offset"])
