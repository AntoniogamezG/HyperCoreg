#!/usr/bin/env python3
"""Validate PRISMA radiometry without loading complete rasters into memory.

The validator treats the source HE5 file as the authority for detector scale
coefficients and spectral order.  It can inspect any subset of a legacy
(``--old``), explicit native-DN (``--native-dn``), and decoded reflectance
(``--reflectance``) GeoTIFF.  All pixel comparisons use spatial windows and
small band batches.

Examples
--------
Validate a new reflectance result and write the same JSON report to disk::

    python scripts/validate_prisma_radiometry.py \
      --source-he5 scene.he5 --reflectance scene_coreg.tif \
      --json-output radiometry_validation.json

Compare a legacy result with both compatibility and corrected outputs::

    python scripts/validate_prisma_radiometry.py \
      --source-he5 scene.he5 --old old_coreg.tif \
      --native-dn native_coreg.tif --reflectance reflectance_coreg.tif
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window


SCHEMA_VERSION = 1
DN_DENOMINATOR = 65535.0
HCO_DATA_ROOT = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields"
VNIR_CUBE_PATH = f"{HCO_DATA_ROOT}/VNIR_Cube"
SWIR_CUBE_PATH = f"{HCO_DATA_ROOT}/SWIR_Cube"
DETECTORS = ("VNIR", "SWIR")
SCALE_ATTRIBUTES = {
    "VNIR": ("L2ScaleVnirMin", "L2ScaleVnirMax"),
    "SWIR": ("L2ScaleSwirMin", "L2ScaleSwirMax"),
    "PAN": ("L2ScalePanMin", "L2ScalePanMax"),
}
KEY_DATASET_TAGS = (
    "PRISMA_RADIOMETRIC_MODE",
    "PRISMA_L2_SCALING_APPLIED",
    "PRISMA_DN_DENOMINATOR",
    "RADIOMETRIC_QUANTITY",
    "RADIOMETRIC_UNITS",
    "NORMALIZATION_MODE",
    "NORM_CLIP",
    "REMOVE_OVERLAPPING_BANDS",
    "PRISMA_L2_SCALE_VNIR_MIN",
    "PRISMA_L2_SCALE_VNIR_MAX",
    "PRISMA_L2_SCALE_SWIR_MIN",
    "PRISMA_L2_SCALE_SWIR_MAX",
    "PRISMA_L2_SCALE_PAN_MIN",
    "PRISMA_L2_SCALE_PAN_MAX",
    "RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY",
    "SOURCE_PRODUCT_NAME",
    "SOURCE_PROCESSING_LEVEL",
    "SOURCE_PROCESSOR_VERSION",
    "SOURCE_PROCESSING_TIME",
)
SOURCE_PROVENANCE_TAGS = (
    ("SOURCE_PRODUCT_NAME", "product_name"),
    ("SOURCE_PROCESSING_LEVEL", "processing_level"),
    ("SOURCE_PROCESSOR_VERSION", "processor_version"),
    ("SOURCE_PROCESSING_TIME", "processing_time"),
)
DESCRIPTION_RE = re.compile(
    r"\b(VNIR|SWIR)\b\s*(?:\||:|-)?\s*([-+]?\d+(?:\.\d+)?)\s*nm\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScalePair:
    minimum: float
    maximum: float

    @property
    def gain(self) -> float:
        return (self.maximum - self.minimum) / DN_DENOMINATOR

    @property
    def offset(self) -> float:
        return self.minimum

    def report(self) -> Dict[str, float]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "gain": self.gain,
            "offset": self.offset,
        }


@dataclass(frozen=True)
class SourceBand:
    detector: str
    detector_index: int
    wavelength_nm: float
    fwhm_nm: Optional[float]

    def report(self, index: int) -> Dict[str, Any]:
        return {
            "index": index,
            "detector": self.detector,
            "detector_index": self.detector_index,
            "wavelength_nm": self.wavelength_nm,
            "fwhm_nm": self.fwhm_nm,
        }


@dataclass
class SourceProduct:
    path: Path
    rows: int
    columns: int
    scales: Dict[str, ScalePair]
    bands_all: List[SourceBand]
    bands_overlap_removed: List[SourceBand]
    product_metadata: Dict[str, Optional[str]]

    def bands(self, remove_overlap: bool) -> List[SourceBand]:
        return self.bands_overlap_removed if remove_overlap else self.bands_all

    def report(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "rows": self.rows,
            "columns": self.columns,
            "source_band_count": len(self.bands_all),
            "overlap_removed_band_count": len(self.bands_overlap_removed),
            "removed_overlap_bands": len(self.bands_all) - len(self.bands_overlap_removed),
            "scales": {key: value.report() for key, value in self.scales.items()},
            "product_metadata": self.product_metadata,
            "band_order": {
                "all": _band_sequence_summary(self.bands_all),
                "overlap_removed": _band_sequence_summary(self.bands_overlap_removed),
            },
        }


@dataclass(frozen=True)
class RasterBand:
    detector: Optional[str]
    wavelength_nm: Optional[float]
    description: Optional[str]
    tags: Dict[str, str]


@dataclass
class RasterInfo:
    role: str
    path: Path
    width: int
    height: int
    count: int
    crs: Optional[str]
    transform: Tuple[float, ...]
    dtypes: Tuple[str, ...]
    nodata: Optional[float]
    dataset_tags: Dict[str, str]
    bands: List[RasterBand]
    active_scales: Tuple[float, ...]
    active_offsets: Tuple[float, ...]
    block_shapes: Tuple[Tuple[int, int], ...]
    expected_bands: List[SourceBand] = field(default_factory=list)
    remove_overlap: Optional[bool] = None
    band_order_validation: Dict[str, Any] = field(default_factory=dict)
    detector_mapping_available: bool = False
    detector_mapping_reason: Optional[str] = None

    @property
    def normalization_mode(self) -> str:
        return str(self.dataset_tags.get("NORMALIZATION_MODE", "none") or "none").lower()

    def report(self) -> Dict[str, Any]:
        dtype_counts: Dict[str, int] = {}
        for value in self.dtypes:
            dtype_counts[value] = dtype_counts.get(value, 0) + 1
        block_shape_counts: Dict[Tuple[int, int], int] = {}
        for shape in self.block_shapes:
            block_shape_counts[shape] = block_shape_counts.get(shape, 0) + 1
        return {
            "path": str(self.path),
            "role": self.role,
            "grid": {
                "width": self.width,
                "height": self.height,
                "bands": self.count,
                "crs": self.crs,
                "transform_gdal": list(self.transform),
                "dtypes": dtype_counts,
                "nodata": self.nodata,
                "block_shapes": [
                    {"shape": list(shape), "bands": count}
                    for shape, count in sorted(block_shape_counts.items())
                ],
            },
            "key_dataset_tags": {
                key: self.dataset_tags.get(key) for key in KEY_DATASET_TAGS
            },
            "active_scale_offset_summary": _active_scale_offset_summary(
                self.active_scales, self.active_offsets
            ),
            "band_order_validation": self.band_order_validation,
        }


class Findings:
    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(self, severity: str, code: str, message: str, **context: Any) -> None:
        item: Dict[str, Any] = {
            "severity": severity,
            "code": code,
            "message": message,
        }
        if context:
            item["context"] = context
        self.items.append(item)

    def summary(self) -> Dict[str, Any]:
        counts = {
            severity: sum(1 for item in self.items if item["severity"] == severity)
            for severity in ("error", "warning", "info")
        }
        if counts["error"]:
            status = "fail"
        elif counts["warning"]:
            status = "warning"
        else:
            status = "pass"
        return {"status": status, "finding_counts": counts}


def _attr_text(attrs: Mapping[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key not in attrs:
            continue
        value = np.asarray(attrs[key])
        if value.ndim != 0:
            continue
        scalar = value.item()
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8", errors="replace")
        return str(scalar)
    return None


def _attr_text_from_sources(
    candidates: Sequence[str],
    *attribute_sources: Mapping[str, Any],
) -> Optional[str]:
    """Resolve aliases in order, preferring the root value for the same alias."""
    for key in candidates:
        for attrs in attribute_sources:
            value = _attr_text(attrs, (key,))
            if value is not None:
                return value
    return None


def _finite_scalar_attribute(attrs: Mapping[str, Any], key: str) -> float:
    if key not in attrs:
        raise ValueError(f"Missing source HE5 attribute {key}")
    value = np.asarray(attrs[key])
    if value.ndim != 0:
        raise ValueError(f"Source HE5 attribute {key} must be scalar; got {value.shape}")
    try:
        result = float(value.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Source HE5 attribute {key} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"Source HE5 attribute {key} is not finite: {result!r}")
    return result


def _is_uint16_dtype(dtype: Any) -> bool:
    parsed = np.dtype(dtype)
    return parsed.kind == "u" and parsed.itemsize == 2


def _optional_fwhm(attrs: Mapping[str, Any], candidates: Sequence[str], count: int) -> List[Optional[float]]:
    for key in candidates:
        if key not in attrs:
            continue
        values = np.asarray(attrs[key], dtype=float).reshape(-1)
        result: List[Optional[float]] = []
        for index in range(count):
            value = float(values[index]) if index < values.size else float("nan")
            result.append(value if math.isfinite(value) else None)
        return result
    return [None] * count


def _bands_overlap(left: SourceBand, right: SourceBand) -> bool:
    left_half = 0.5 * left.fwhm_nm if left.fwhm_nm is not None and left.fwhm_nm > 0 else 0.0
    right_half = 0.5 * right.fwhm_nm if right.fwhm_nm is not None and right.fwhm_nm > 0 else 0.0
    return (
        left.wavelength_nm - left_half <= right.wavelength_nm + right_half
        and right.wavelength_nm - right_half <= left.wavelength_nm + left_half
    )


def _remove_detector_overlap(bands: Sequence[SourceBand]) -> List[SourceBand]:
    """Mirror HyperCoreg's SWIR-priority overlap policy independently."""
    keep = np.ones(len(bands), dtype=bool)
    vnir_indices = [index for index, band in enumerate(bands) if band.detector == "VNIR"]
    swir_indices = [index for index, band in enumerate(bands) if band.detector == "SWIR"]

    if vnir_indices and swir_indices:
        vnir_wavelengths = [bands[index].wavelength_nm for index in vnir_indices]
        swir_wavelengths = [bands[index].wavelength_nm for index in swir_indices]
        overlap_min = max(min(vnir_wavelengths), min(swir_wavelengths))
        overlap_max = min(max(vnir_wavelengths), max(swir_wavelengths))
        if overlap_min <= overlap_max:
            for index in vnir_indices:
                band = bands[index]
                if band.fwhm_nm is None or band.fwhm_nm <= 0:
                    if overlap_min <= band.wavelength_nm <= overlap_max:
                        keep[index] = False

    for vnir_index in vnir_indices:
        if not keep[vnir_index]:
            continue
        for swir_index in swir_indices:
            if keep[swir_index] and _bands_overlap(bands[vnir_index], bands[swir_index]):
                keep[vnir_index] = False
                break
    return [band for index, band in enumerate(bands) if keep[index]]


def _load_source(path: Path) -> SourceProduct:
    with h5py.File(path, "r") as source:
        if VNIR_CUBE_PATH not in source or SWIR_CUBE_PATH not in source:
            raise ValueError("Source HE5 does not contain PRISMA L2D VNIR/SWIR cubes")
        vnir_cube = source[VNIR_CUBE_PATH]
        swir_cube = source[SWIR_CUBE_PATH]
        if len(vnir_cube.shape) != 3 or len(swir_cube.shape) != 3:
            raise ValueError("Source PRISMA cubes must be three-dimensional")
        if vnir_cube.shape[0] != swir_cube.shape[0] or vnir_cube.shape[2] != swir_cube.shape[2]:
            raise ValueError(
                f"VNIR/SWIR spatial dimensions differ: {vnir_cube.shape} vs {swir_cube.shape}"
            )
        if not _is_uint16_dtype(vnir_cube.dtype) or not _is_uint16_dtype(swir_cube.dtype):
            raise ValueError(
                "Source PRISMA Level-2 cubes must be uint16 for native-DN validation; "
                f"got {vnir_cube.dtype} and {swir_cube.dtype}"
            )

        attrs = source.attrs
        file_attributes = source.get("HDFEOS/ADDITIONAL/FILE_ATTRIBUTES")
        additional_attrs = file_attributes.attrs if file_attributes is not None else {}
        vnir_wavelengths = np.asarray(attrs["List_Cw_Vnir"], dtype=float).reshape(-1)
        swir_wavelengths = np.asarray(attrs["List_Cw_Swir"], dtype=float).reshape(-1)
        if vnir_wavelengths.size != vnir_cube.shape[1]:
            raise ValueError(
                f"VNIR wavelength count {vnir_wavelengths.size} != cube bands {vnir_cube.shape[1]}"
            )
        if swir_wavelengths.size != swir_cube.shape[1]:
            raise ValueError(
                f"SWIR wavelength count {swir_wavelengths.size} != cube bands {swir_cube.shape[1]}"
            )

        vnir_fwhm = _optional_fwhm(
            attrs,
            ("List_Fwhm_Vnir", "List_FWHM_Vnir", "List_Fwhm_VNIR"),
            vnir_wavelengths.size,
        )
        swir_fwhm = _optional_fwhm(
            attrs,
            ("List_Fwhm_Swir", "List_FWHM_Swir", "List_Fwhm_SWIR"),
            swir_wavelengths.size,
        )

        unsorted_bands: List[SourceBand] = []
        for detector, wavelengths, fwhm_values in (
            ("VNIR", vnir_wavelengths, vnir_fwhm),
            ("SWIR", swir_wavelengths, swir_fwhm),
        ):
            for index, wavelength in enumerate(wavelengths):
                value = float(wavelength)
                if not math.isfinite(value) or value <= 0:
                    continue
                unsorted_bands.append(
                    SourceBand(
                        detector=detector,
                        detector_index=index,
                        wavelength_nm=value,
                        fwhm_nm=fwhm_values[index],
                    )
                )

        overlap_removed = _remove_detector_overlap(unsorted_bands)
        bands_all = sorted(unsorted_bands, key=lambda item: item.wavelength_nm)
        bands_overlap_removed = sorted(overlap_removed, key=lambda item: item.wavelength_nm)
        scales: Dict[str, ScalePair] = {}
        for detector in DETECTORS:
            minimum_key, maximum_key = SCALE_ATTRIBUTES[detector]
            minimum = _finite_scalar_attribute(attrs, minimum_key)
            maximum = _finite_scalar_attribute(attrs, maximum_key)
            if maximum <= minimum:
                raise ValueError(
                    f"Invalid {detector} scale range: minimum={minimum}, maximum={maximum}"
                )
            scales[detector] = ScalePair(minimum, maximum)

        pan_scale_status = "available"
        pan_scale_reason: Optional[str] = None
        pan_minimum_key, pan_maximum_key = SCALE_ATTRIBUTES["PAN"]
        try:
            pan_minimum = _finite_scalar_attribute(attrs, pan_minimum_key)
            pan_maximum = _finite_scalar_attribute(attrs, pan_maximum_key)
            if pan_maximum <= pan_minimum:
                raise ValueError(
                    f"Invalid PAN scale range: minimum={pan_minimum}, maximum={pan_maximum}"
                )
            scales["PAN"] = ScalePair(pan_minimum, pan_maximum)
        except ValueError as exc:
            # This tool accepts only spectral GeoTIFF inputs.  PAN metadata is
            # therefore informative when present, but cannot block spectral
            # radiometry validation when absent or malformed.
            pan_scale_status = "not_applicable"
            pan_scale_reason = str(exc)

        metadata = {
            "product_name": _attr_text_from_sources(
                ("Product_Name", "ProductName", "product_name"), attrs, additional_attrs
            ),
            "processing_level": _attr_text_from_sources(
                ("Processing_Level", "ProcessingLevel"), attrs, additional_attrs
            ),
            "processor_version": _attr_text_from_sources(
                ("Processor_Version", "ProcessorVersion"), attrs, additional_attrs
            ),
            "processing_time": _attr_text_from_sources(
                ("Processing_Time", "ProcessingTime"), attrs, additional_attrs
            ),
            "product_start_time": _attr_text_from_sources(
                ("Product_StartTime",), attrs, additional_attrs
            ),
            "pan_scale_status": pan_scale_status,
            "pan_scale_reason": pan_scale_reason,
        }

        return SourceProduct(
            path=path,
            rows=int(vnir_cube.shape[0]),
            columns=int(vnir_cube.shape[2]),
            scales=scales,
            bands_all=bands_all,
            bands_overlap_removed=bands_overlap_removed,
            product_metadata=metadata,
        )


def _band_sequence_summary(bands: Sequence[SourceBand]) -> Dict[str, Any]:
    detectors = [band.detector for band in bands]
    wavelengths = [band.wavelength_nm for band in bands]
    return {
        "count": len(bands),
        "vnir_count": detectors.count("VNIR"),
        "swir_count": detectors.count("SWIR"),
        "ascending": all(left < right for left, right in zip(wavelengths, wavelengths[1:])),
        "first": bands[0].report(1) if bands else None,
        "last": bands[-1].report(len(bands)) if bands else None,
    }


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _parse_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _extract_raster_band(
    tags: Mapping[str, str], description: Optional[str]
) -> RasterBand:
    detector_raw = tags.get("DETECTOR")
    detector = str(detector_raw).strip().upper() if detector_raw is not None else None
    if detector not in DETECTORS:
        detector = None
    wavelength = _parse_float(tags.get("WAVELENGTH_NM"))
    if wavelength is None:
        wavelength = _parse_float(tags.get("wavelength"))
    if description:
        match = DESCRIPTION_RE.search(description)
        if match:
            if detector is None:
                detector = match.group(1).upper()
            if wavelength is None:
                wavelength = float(match.group(2))
    return RasterBand(detector, wavelength, description, dict(tags))


def _load_raster(role: str, path: Path) -> RasterInfo:
    with rasterio.open(path) as raster:
        tags = {str(key): str(value) for key, value in raster.tags().items()}
        bands = [
            _extract_raster_band(raster.tags(index), raster.descriptions[index - 1])
            for index in range(1, raster.count + 1)
        ]
        nodata = _parse_float(raster.nodata)
        return RasterInfo(
            role=role,
            path=path,
            width=int(raster.width),
            height=int(raster.height),
            count=int(raster.count),
            crs=str(raster.crs) if raster.crs is not None else None,
            transform=tuple(float(value) for value in raster.transform.to_gdal()),
            dtypes=tuple(str(value) for value in raster.dtypes),
            nodata=nodata,
            dataset_tags=tags,
            bands=bands,
            active_scales=tuple(float(value) for value in raster.scales),
            active_offsets=tuple(float(value) for value in raster.offsets),
            block_shapes=tuple(tuple(int(value) for value in shape) for shape in raster.block_shapes),
        )


def _active_scale_offset_summary(
    scales: Sequence[float], offsets: Sequence[float]
) -> Dict[str, Any]:
    pairs: Dict[Tuple[float, float], int] = {}
    for scale, offset in zip(scales, offsets):
        key = (float(scale), float(offset))
        pairs[key] = pairs.get(key, 0) + 1
    return {
        "band_count": min(len(scales), len(offsets)),
        "unique_pairs": [
            {"scale": key[0], "offset": key[1], "bands": count}
            for key, count in sorted(pairs.items())
        ],
    }


def _sequence_from_complete_band_metadata(
    raster: RasterInfo,
    source: SourceProduct,
) -> Tuple[Optional[List[SourceBand]], Optional[str]]:
    """Resolve a non-canonical raster sequence without guessing detectors."""
    if len(raster.bands) != raster.count:
        return None, (
            f"raster exposes {len(raster.bands)} band metadata records for "
            f"{raster.count} raster bands"
        )
    matched: List[SourceBand] = []
    used_source_indices: set[int] = set()
    previous_source_index = -1
    for raster_index, actual in enumerate(raster.bands):
        if actual.detector not in DETECTORS or actual.wavelength_nm is None:
            return None, f"band {raster_index + 1} lacks detector/wavelength metadata"
        candidates = [
            (source_index, band)
            for source_index, band in enumerate(source.bands_all)
            if source_index not in used_source_indices
            and band.detector == actual.detector
            and abs(band.wavelength_nm - actual.wavelength_nm)
            <= _ARGS.wavelength_tolerance_nm
        ]
        if not candidates:
            return None, (
                f"band {raster_index + 1} does not match a source detector/wavelength"
            )
        if len(candidates) != 1:
            return None, (
                f"band {raster_index + 1} ambiguously matches multiple source bands"
            )
        source_index, band = candidates[0]
        if source_index <= previous_source_index:
            return None, "band metadata does not preserve source spectral order"
        matched.append(band)
        used_source_indices.add(source_index)
        previous_source_index = source_index
    return matched, None


def _select_expected_bands(
    raster: RasterInfo,
    source: SourceProduct,
    findings: Findings,
) -> None:
    remove_tag = raster.dataset_tags.get("REMOVE_OVERLAPPING_BANDS")
    parsed_remove = _parse_bool(remove_tag)
    if remove_tag is not None and parsed_remove is None:
        findings.add(
            "warning",
            "invalid_overlap_tag",
            f"{raster.role} has an unrecognized REMOVE_OVERLAPPING_BANDS value",
            role=raster.role,
            actual=remove_tag,
        )

    all_count = len(source.bands_all)
    removed_count = len(source.bands_overlap_removed)
    if raster.count not in {all_count, removed_count}:
        metadata_sequence, sequence_error = _sequence_from_complete_band_metadata(
            raster, source
        )
        raster.remove_overlap = None
        if metadata_sequence is not None:
            raster.expected_bands = metadata_sequence
            raster.detector_mapping_available = True
            raster.detector_mapping_reason = (
                "non-canonical band count resolved from complete source-matched band metadata"
            )
            raster.band_order_validation = {
                "remove_detector_overlap": None,
                "tagged_remove_detector_overlap": parsed_remove,
                "selection_basis": "complete_band_metadata",
                "expected_count": len(metadata_sequence),
                "actual_count": raster.count,
                "metadata_complete_bands": raster.count,
                "missing_band_metadata": 0,
                "mismatch_count": 0,
                "matches_source": True,
                "detector_mapping_available": True,
                "detector_mapping_reason": raster.detector_mapping_reason,
                "mismatches": [],
            }
            findings.add(
                "info",
                "detector_sequence_from_band_metadata",
                (
                    f"Resolved non-canonical {raster.role} detector sequence from "
                    "complete band metadata"
                ),
                role=raster.role,
                actual_band_count=raster.count,
                source_all_band_count=all_count,
                source_overlap_removed_band_count=removed_count,
            )
            return

        raster.expected_bands = []
        raster.detector_mapping_available = False
        raster.detector_mapping_reason = (
            "band count matches neither supported source sequence and "
            f"band metadata cannot resolve it: {sequence_error}"
        )
        metadata_complete = sum(
            1
            for band in raster.bands
            if band.detector in DETECTORS and band.wavelength_nm is not None
        )
        raster.band_order_validation = {
            "remove_detector_overlap": None,
            "tagged_remove_detector_overlap": parsed_remove,
            "selection_basis": "unavailable",
            "expected_count": None,
            "actual_count": raster.count,
            "metadata_complete_bands": metadata_complete,
            "missing_band_metadata": raster.count - metadata_complete,
            "mismatch_count": 1,
            "matches_source": False,
            "detector_mapping_available": False,
            "detector_mapping_reason": raster.detector_mapping_reason,
            "mismatches": [
                {
                    "kind": "unsupported_band_count_and_metadata",
                    "source_all_band_count": all_count,
                    "source_overlap_removed_band_count": removed_count,
                    "reason": sequence_error,
                }
            ],
        }
        findings.add(
            "error",
            "detector_sequence_unavailable",
            (
                f"{raster.role} detector sequence cannot be established safely; "
                "detector-dependent validation will be skipped"
            ),
            role=raster.role,
            actual_band_count=raster.count,
            source_all_band_count=all_count,
            source_overlap_removed_band_count=removed_count,
            reason=sequence_error,
        )
        return

    exact_count_remove: Optional[bool] = None
    if all_count != removed_count:
        if raster.count == removed_count:
            exact_count_remove = True
        elif raster.count == all_count:
            exact_count_remove = False

    if (
        parsed_remove is not None
        and exact_count_remove is not None
        and parsed_remove != exact_count_remove
    ):
        findings.add(
            "warning" if raster.role == "old" else "error",
            "overlap_tag_band_count_mismatch",
            (
                f"{raster.role} REMOVE_OVERLAPPING_BANDS contradicts the "
                "unique overlap mode implied by its band count"
            ),
            role=raster.role,
            actual_tag=remove_tag,
            tagged_remove_detector_overlap=parsed_remove,
            count_inferred_remove_detector_overlap=exact_count_remove,
            actual_band_count=raster.count,
            tagged_expected_band_count=(removed_count if parsed_remove else all_count),
            count_inferred_expected_band_count=(
                removed_count if exact_count_remove else all_count
            ),
        )
        # The exact raster count is stronger evidence for the effective band
        # sequence.  Continue all order, affine, and range checks with that
        # sequence while retaining the metadata defect as a finding above.
        parsed_remove = exact_count_remove
    elif parsed_remove is None:
        if exact_count_remove is not None:
            parsed_remove = exact_count_remove
        elif raster.count == all_count:
            parsed_remove = False
        else:
            parsed_remove = abs(raster.count - removed_count) < abs(raster.count - all_count)
        findings.add(
            "info",
            "overlap_mode_inferred",
            f"Inferred detector-overlap mode for {raster.role} from its band count",
            role=raster.role,
            inferred=parsed_remove,
        )

    raster.remove_overlap = parsed_remove
    expected = source.bands(bool(parsed_remove))
    raster.expected_bands = expected
    raster.detector_mapping_available = True
    raster.detector_mapping_reason = "band count selects a supported source sequence"
    mismatches: List[Dict[str, Any]] = []
    missing_metadata = 0
    if raster.count != len(expected):
        mismatches.append(
            {
                "kind": "band_count",
                "expected": len(expected),
                "actual": raster.count,
            }
        )

    compare_count = min(raster.count, len(expected))
    for index in range(compare_count):
        actual = raster.bands[index]
        wanted = expected[index]
        if actual.detector is None or actual.wavelength_nm is None:
            missing_metadata += 1
            continue
        detector_ok = actual.detector == wanted.detector
        wavelength_error = abs(actual.wavelength_nm - wanted.wavelength_nm)
        if not detector_ok or wavelength_error > _ARGS.wavelength_tolerance_nm:
            if len(mismatches) < _ARGS.max_reported_mismatches:
                mismatches.append(
                    {
                        "kind": "band_metadata",
                        "band": index + 1,
                        "expected_detector": wanted.detector,
                        "actual_detector": actual.detector,
                        "expected_wavelength_nm": wanted.wavelength_nm,
                        "actual_wavelength_nm": actual.wavelength_nm,
                        "wavelength_error_nm": wavelength_error,
                    }
                )

    mismatch_count = (
        (1 if raster.count != len(expected) else 0)
        + sum(
            1
            for index in range(compare_count)
            if raster.bands[index].detector is not None
            and raster.bands[index].wavelength_nm is not None
            and (
                raster.bands[index].detector != expected[index].detector
                or abs(raster.bands[index].wavelength_nm - expected[index].wavelength_nm)
                > _ARGS.wavelength_tolerance_nm
            )
        )
    )
    raster.band_order_validation = {
        "remove_detector_overlap": parsed_remove,
        "expected_count": len(expected),
        "actual_count": raster.count,
        "metadata_complete_bands": compare_count - missing_metadata,
        "missing_band_metadata": missing_metadata,
        "mismatch_count": mismatch_count,
        "matches_source": mismatch_count == 0 and missing_metadata == 0,
        "detector_mapping_available": True,
        "detector_mapping_reason": raster.detector_mapping_reason,
        "mismatches": mismatches,
    }
    if mismatch_count:
        findings.add(
            "error",
            "source_band_order_mismatch",
            f"{raster.role} band order does not match the source product",
            role=raster.role,
            mismatch_count=mismatch_count,
        )
    elif missing_metadata:
        findings.add(
            "error" if raster.role != "old" else "warning",
            "band_metadata_missing",
            f"{raster.role} cannot prove band order because band metadata is missing",
            role=raster.role,
            missing_bands=missing_metadata,
        )


def _numeric_close(actual: Optional[float], expected: float, atol: float = 1e-9) -> bool:
    return actual is not None and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=atol)


def _normalize_provenance_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().split())
    return normalized or None


def _validate_contract_tags(
    raster: RasterInfo,
    source: SourceProduct,
    findings: Findings,
) -> Dict[str, Any]:
    if raster.role == "old":
        mode = raster.dataset_tags.get("PRISMA_RADIOMETRIC_MODE")
        if mode is None:
            findings.add(
                "warning",
                "legacy_radiometric_tags_absent",
                "Legacy output has no explicit radiometric-state tags (expected for pre-fix output)",
                role=raster.role,
            )
        return {
            "contract_expected": False,
            "mode_tag": mode,
            "note": "Legacy output is reported but is not required to carry the new contract.",
        }

    expected_mode = "native-dn" if raster.role == "native_dn" else "reflectance"
    normalization = raster.normalization_mode
    normalization_valid = normalization in {"none", "minmax", "percentile"}
    if not normalization_valid:
        findings.add(
            "error",
            "invalid_normalization_mode",
            f"{raster.role} declares unsupported NORMALIZATION_MODE={normalization!r}",
            role=raster.role,
            normalization_mode=normalization,
        )
    normalized = normalization != "none"
    expected_quantity = (
        "normalized_unitless"
        if normalized
        else ("native_encoded_dn" if expected_mode == "native-dn" else "surface_reflectance")
    )
    expected_units = "1" if normalized or expected_mode == "reflectance" else "DN"
    norm_clip_text = str(raster.dataset_tags.get("NORM_CLIP", "")).strip().lower()
    normalization_clip = (
        True if norm_clip_text == "true" else False if norm_clip_text == "false" else None
    )
    if not normalized:
        expected_policy_values = ("preserve_and_flag_not_clip",)
    elif normalization_clip is True:
        expected_policy_values = (
            f"transformed_by_{normalization}_normalization_with_clipping",
        )
    elif normalization_clip is False:
        expected_policy_values = (
            f"transformed_by_{normalization}_normalization_without_clipping",
        )
    else:
        expected_policy_values = (
            f"transformed_by_{normalization}_normalization_with_clipping",
            f"transformed_by_{normalization}_normalization_without_clipping",
        )
    expected_tags = {
        "PRISMA_RADIOMETRIC_MODE": (expected_mode,),
        "PRISMA_L2_SCALING_APPLIED": (
            "false" if expected_mode == "native-dn" else "true",
        ),
        "PRISMA_DN_DENOMINATOR": ("65535",),
        "RADIOMETRIC_QUANTITY": (expected_quantity,),
        "RADIOMETRIC_UNITS": (expected_units,),
        "NORMALIZATION_MODE": (normalization,),
        "RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY": expected_policy_values,
    }
    if normalized:
        expected_tags["NORM_CLIP"] = ("true", "false")
    tag_checks: List[Dict[str, Any]] = []
    for key, expected_values in expected_tags.items():
        actual = raster.dataset_tags.get(key)
        actual_normalized = None if actual is None else str(actual).strip().lower()
        ok = actual_normalized in {value.lower() for value in expected_values}
        expected: Any = (
            expected_values[0] if len(expected_values) == 1 else list(expected_values)
        )
        tag_checks.append({"tag": key, "expected": expected, "actual": actual, "ok": ok})
        if not ok:
            findings.add(
                "error",
                "radiometric_dataset_tag_mismatch",
                f"{raster.role} dataset tag {key} is missing or incorrect",
                role=raster.role,
                tag=key,
                expected=expected,
                actual=actual,
            )

    source_provenance_checks: List[Dict[str, Any]] = []
    for tag, metadata_key in SOURCE_PROVENANCE_TAGS:
        source_value = source.product_metadata.get(metadata_key)
        expected = _normalize_provenance_text(source_value)
        actual_text = raster.dataset_tags.get(tag)
        actual = _normalize_provenance_text(actual_text)
        if expected is None:
            source_provenance_checks.append(
                {
                    "tag": tag,
                    "source_metadata_key": metadata_key,
                    "expected": None,
                    "actual": actual_text,
                    "status": "not_applicable",
                    "ok": None,
                    "reason": "source product metadata value is unavailable",
                }
            )
            continue

        ok = actual == expected
        source_provenance_checks.append(
            {
                "tag": tag,
                "source_metadata_key": metadata_key,
                "expected": expected,
                "actual": actual_text,
                "normalized_actual": actual,
                "status": "pass" if ok else "fail",
                "ok": ok,
            }
        )
        if not ok:
            findings.add(
                "error",
                "source_provenance_tag_mismatch",
                f"{raster.role} source provenance tag {tag} is missing or incorrect",
                role=raster.role,
                tag=tag,
                source_metadata_key=metadata_key,
                expected=expected,
                actual=actual_text,
            )

    source_scale_checks: List[Dict[str, Any]] = []
    for detector, scale in source.scales.items():
        for suffix, expected in (("MIN", scale.minimum), ("MAX", scale.maximum)):
            key = f"PRISMA_L2_SCALE_{detector}_{suffix}"
            actual_text = raster.dataset_tags.get(key)
            actual = _parse_float(actual_text)
            ok = _numeric_close(actual, expected)
            source_scale_checks.append(
                {"tag": key, "expected": expected, "actual": actual_text, "ok": ok}
            )
            if not ok:
                findings.add(
                    "error",
                    "radiometric_source_scale_tag_mismatch",
                    f"{raster.role} source scale tag {key} is missing or incorrect",
                    role=raster.role,
                    tag=key,
                    expected=expected,
                    actual=actual_text,
                )

    active_mismatches: List[Dict[str, Any]] = []
    band_tag_mismatches: List[Dict[str, Any]] = []
    for index in range(min(raster.count, len(raster.expected_bands))):
        band = raster.expected_bands[index]
        source_scale = source.scales[band.detector]
        if normalized or expected_mode == "reflectance":
            wanted_scale, wanted_offset = 1.0, 0.0
        else:
            wanted_scale, wanted_offset = source_scale.gain, source_scale.offset
        actual_scale = raster.active_scales[index]
        actual_offset = raster.active_offsets[index]
        if not (
            _numeric_close(actual_scale, wanted_scale, atol=1e-12)
            and _numeric_close(actual_offset, wanted_offset, atol=1e-9)
        ):
            if len(active_mismatches) < _ARGS.max_reported_mismatches:
                active_mismatches.append(
                    {
                        "band": index + 1,
                        "detector": band.detector,
                        "expected_scale": wanted_scale,
                        "actual_scale": actual_scale,
                        "expected_offset": wanted_offset,
                        "actual_offset": actual_offset,
                    }
                )

        actual_band = raster.bands[index]
        gain_tag = _parse_float(actual_band.tags.get("SOURCE_DN_GAIN"))
        offset_tag = _parse_float(actual_band.tags.get("SOURCE_DN_OFFSET"))
        quantity_tag = actual_band.tags.get("RADIOMETRIC_QUANTITY")
        units_tag = actual_band.tags.get("RADIOMETRIC_UNITS")
        if not (
            _numeric_close(gain_tag, source_scale.gain, atol=1e-12)
            and _numeric_close(offset_tag, source_scale.offset, atol=1e-9)
            and quantity_tag == expected_quantity
            and units_tag == expected_units
        ):
            if len(band_tag_mismatches) < _ARGS.max_reported_mismatches:
                band_tag_mismatches.append(
                    {
                        "band": index + 1,
                        "detector": band.detector,
                        "source_dn_gain": actual_band.tags.get("SOURCE_DN_GAIN"),
                        "source_dn_offset": actual_band.tags.get("SOURCE_DN_OFFSET"),
                        "radiometric_quantity": quantity_tag,
                        "radiometric_units": units_tag,
                    }
                )

    if active_mismatches:
        findings.add(
            "error",
            "active_scale_offset_mismatch",
            f"{raster.role} active GDAL scale/offset state is incorrect",
            role=raster.role,
            reported_mismatches=len(active_mismatches),
        )
    if band_tag_mismatches:
        findings.add(
            "error",
            "radiometric_band_tag_mismatch",
            f"{raster.role} per-band radiometric provenance is missing or incorrect",
            role=raster.role,
            reported_mismatches=len(band_tag_mismatches),
        )

    return {
        "contract_expected": True,
        "normalization_mode": normalization,
        "dataset_tag_checks": tag_checks,
        "source_provenance_tag_checks": source_provenance_checks,
        "source_scale_tag_checks": source_scale_checks,
        "detector_dependent_band_checks": {
            "status": "complete" if raster.detector_mapping_available else "skipped",
            "reason": (
                None
                if raster.detector_mapping_available
                else "detector sequence unavailable for per-band radiometric validation"
            ),
            "detector_mapping_reason": raster.detector_mapping_reason,
        },
        "active_scale_offset_mismatches": active_mismatches,
        "band_tag_mismatches": band_tag_mismatches,
        "ok": not any(not check["ok"] for check in tag_checks + source_scale_checks)
        and not any(check["ok"] is False for check in source_provenance_checks)
        and normalization_valid
        and raster.detector_mapping_available
        and not active_mismatches
        and not band_tag_mismatches,
    }


def _compare_grids(left: RasterInfo, right: RasterInfo, findings: Findings) -> Dict[str, Any]:
    checks = {
        "width": {"left": left.width, "right": right.width, "match": left.width == right.width},
        "height": {"left": left.height, "right": right.height, "match": left.height == right.height},
        "bands": {"left": left.count, "right": right.count, "match": left.count == right.count},
        "crs": {"left": left.crs, "right": right.crs, "match": left.crs == right.crs},
        "transform": {
            "left": list(left.transform),
            "right": list(right.transform),
            "match": bool(
                np.allclose(
                    np.asarray(left.transform),
                    np.asarray(right.transform),
                    rtol=0.0,
                    atol=_ARGS.grid_tolerance,
                )
            ),
        },
    }
    match = all(item["match"] for item in checks.values())
    band_sequence_match: Optional[bool] = None
    if left.expected_bands and right.expected_bands:
        band_sequence_match = len(left.expected_bands) == len(right.expected_bands) and all(
            lband.detector == rband.detector
            and abs(lband.wavelength_nm - rband.wavelength_nm) <= _ARGS.wavelength_tolerance_nm
            for lband, rband in zip(left.expected_bands, right.expected_bands)
        )
    if not match:
        findings.add(
            "error",
            "raster_grid_mismatch",
            f"{left.role} and {right.role} are not on the same grid",
            left=left.role,
            right=right.role,
        )
    if band_sequence_match is False:
        findings.add(
            "error",
            "raster_band_sequence_mismatch",
            f"{left.role} and {right.role} use different spectral sequences",
            left=left.role,
            right=right.role,
        )
    return {
        "left": left.role,
        "right": right.role,
        "grid_match": match,
        "band_sequence_match": band_sequence_match,
        "checks": checks,
    }


def _windows(width: int, height: int, block_size: int) -> Iterable[Window]:
    for row in range(0, height, block_size):
        window_height = min(block_size, height - row)
        for column in range(0, width, block_size):
            window_width = min(block_size, width - column)
            yield Window(column, row, window_width, window_height)


def _metric_state() -> Dict[str, Any]:
    return {
        "count": 0,
        "sum_error": 0.0,
        "sum_squared_error": 0.0,
        "max_absolute_error": 0.0,
        "tolerance_violations": 0,
    }


def _update_metric(state: Dict[str, Any], errors: np.ndarray, tolerances: np.ndarray) -> None:
    if errors.size == 0:
        return
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    tolerance_values = np.asarray(tolerances, dtype=np.float64).reshape(-1)
    absolute = np.abs(values)
    state["count"] += int(values.size)
    state["sum_error"] += float(np.sum(values, dtype=np.float64))
    state["sum_squared_error"] += float(np.sum(values * values, dtype=np.float64))
    state["max_absolute_error"] = max(state["max_absolute_error"], float(np.max(absolute)))
    state["tolerance_violations"] += int(np.count_nonzero(absolute > tolerance_values))


def _final_metric(state: Mapping[str, Any]) -> Dict[str, Any]:
    count = int(state["count"])
    if count == 0:
        return {
            "compared_pixels": 0,
            "rmse": None,
            "mean_error": None,
            "max_absolute_error": None,
            "tolerance_violations": 0,
            "tolerance_violation_fraction": None,
        }
    return {
        "compared_pixels": count,
        "rmse": math.sqrt(float(state["sum_squared_error"]) / count),
        "mean_error": float(state["sum_error"]) / count,
        "max_absolute_error": float(state["max_absolute_error"]),
        "tolerance_violations": int(state["tolerance_violations"]),
        "tolerance_violation_fraction": int(state["tolerance_violations"]) / count,
    }


def _mask_state() -> Dict[str, int]:
    return {
        "both_valid": 0,
        "both_invalid": 0,
        "left_only_valid": 0,
        "right_only_valid": 0,
    }


def _update_masks(state: Dict[str, int], left_valid: np.ndarray, right_valid: np.ndarray) -> None:
    state["both_valid"] += int(np.count_nonzero(left_valid & right_valid))
    state["both_invalid"] += int(np.count_nonzero(~left_valid & ~right_valid))
    state["left_only_valid"] += int(np.count_nonzero(left_valid & ~right_valid))
    state["right_only_valid"] += int(np.count_nonzero(~left_valid & right_valid))


def _final_masks(state: Mapping[str, int]) -> Dict[str, Any]:
    total = sum(int(value) for value in state.values())
    disagreement = int(state["left_only_valid"]) + int(state["right_only_valid"])
    return {
        **{key: int(value) for key, value in state.items()},
        "total_pixels": total,
        "disagreement_pixels": disagreement,
        "disagreement_fraction": disagreement / total if total else None,
    }


def _valid_data(masked: np.ma.MaskedArray) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(masked.data)
    valid = ~np.ma.getmaskarray(masked)
    if np.issubdtype(values.dtype, np.number):
        valid &= np.isfinite(values)
    return values, valid


def _float_tolerance(
    expected: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
    ulps: int,
) -> np.ndarray:
    expected32 = np.asarray(expected, dtype=np.float32)
    spacing = np.abs(np.spacing(expected32)).astype(np.float64)
    expected64 = np.abs(expected32.astype(np.float64))
    tolerance = np.maximum(absolute_tolerance, relative_tolerance * expected64)
    return np.maximum(tolerance, float(ulps) * spacing)


def _detector_for_band(info: RasterInfo, zero_based_index: int) -> str:
    if zero_based_index < len(info.expected_bands):
        return info.expected_bands[zero_based_index].detector
    if zero_based_index < len(info.bands) and info.bands[zero_based_index].detector:
        return str(info.bands[zero_based_index].detector)
    return "UNKNOWN"


def _compare_raster_values(
    left: RasterInfo,
    right: RasterInfo,
    source: SourceProduct,
    affine: bool,
    findings: Findings,
) -> Dict[str, Any]:
    comparison_name = f"{left.role}_vs_{right.role}"
    if not (
        left.width == right.width
        and left.height == right.height
        and left.count == right.count
        and left.crs == right.crs
        and np.allclose(left.transform, right.transform, rtol=0.0, atol=_ARGS.grid_tolerance)
    ):
        return {
            "comparison": comparison_name,
            "status": "skipped",
            "reason": "rasters are not on the same grid",
        }
    if affine and (
        not left.detector_mapping_available or not right.detector_mapping_available
    ):
        unavailable = [
            {
                "role": raster.role,
                "reason": raster.detector_mapping_reason,
            }
            for raster in (left, right)
            if not raster.detector_mapping_available
        ]
        return {
            "comparison": comparison_name,
            "status": "skipped",
            "reason": "detector sequence unavailable for affine radiometric comparison",
            "unavailable_detector_mappings": unavailable,
        }
    if affine and (left.normalization_mode != "none" or right.normalization_mode != "none"):
        findings.add(
            "warning",
            "affine_check_skipped_normalized",
            "Affine radiometric validation is not applicable after scene normalization",
            left=left.role,
            right=right.role,
        )
        return {
            "comparison": comparison_name,
            "status": "skipped",
            "reason": "scene normalization is not 'none'",
        }
    if affine:
        unknown_bands = [
            index + 1
            for index in range(left.count)
            if _detector_for_band(left, index) not in source.scales
        ]
        if unknown_bands:
            findings.add(
                "error",
                "affine_detector_mapping_missing",
                "Cannot apply source coefficients because one or more output bands lack a detector mapping",
                comparison=comparison_name,
                bands=unknown_bands[: _ARGS.max_reported_mismatches],
            )
            return {
                "comparison": comparison_name,
                "status": "fail",
                "reason": "one or more bands lack a VNIR/SWIR detector mapping",
                "bands": unknown_bands[: _ARGS.max_reported_mismatches],
            }

    metric_states = {"TOTAL": _metric_state(), "VNIR": _metric_state(), "SWIR": _metric_state()}
    mask_states = {"TOTAL": _mask_state(), "VNIR": _mask_state(), "SWIR": _mask_state()}
    per_band = [_metric_state() for _ in range(left.count)]

    with rasterio.open(left.path) as left_raster, rasterio.open(right.path) as right_raster:
        for window in _windows(left.width, left.height, _ARGS.block_size):
            for band_start in range(0, left.count, _ARGS.band_batch_size):
                band_stop = min(left.count, band_start + _ARGS.band_batch_size)
                indexes = list(range(band_start + 1, band_stop + 1))
                left_values, left_valid = _valid_data(
                    left_raster.read(indexes=indexes, window=window, masked=True)
                )
                right_values, right_valid = _valid_data(
                    right_raster.read(indexes=indexes, window=window, masked=True)
                )

                if affine:
                    gains = np.asarray(
                        [source.scales[_detector_for_band(left, index)].gain for index in range(band_start, band_stop)],
                        dtype=np.float32,
                    )[:, None, None]
                    offsets = np.asarray(
                        [source.scales[_detector_for_band(left, index)].offset for index in range(band_start, band_stop)],
                        dtype=np.float32,
                    )[:, None, None]
                    expected = left_values.astype(np.float32) * gains + offsets
                    absolute_tolerance = _ARGS.reflectance_absolute_tolerance
                    relative_tolerance = _ARGS.reflectance_relative_tolerance
                else:
                    expected = left_values.astype(np.float32, copy=False)
                    absolute_tolerance = _ARGS.dn_absolute_tolerance
                    relative_tolerance = _ARGS.dn_relative_tolerance

                co_valid = left_valid & right_valid
                errors = right_values.astype(np.float64) - expected.astype(np.float64)
                tolerances = _float_tolerance(
                    expected,
                    absolute_tolerance,
                    relative_tolerance,
                    _ARGS.float32_ulps,
                )

                _update_masks(mask_states["TOTAL"], left_valid, right_valid)
                _update_metric(metric_states["TOTAL"], errors[co_valid], tolerances[co_valid])
                for local_index, global_index in enumerate(range(band_start, band_stop)):
                    detector = _detector_for_band(left, global_index)
                    if detector not in metric_states:
                        continue
                    _update_masks(
                        mask_states[detector], left_valid[local_index], right_valid[local_index]
                    )
                    valid = co_valid[local_index]
                    _update_metric(
                        metric_states[detector],
                        errors[local_index][valid],
                        tolerances[local_index][valid],
                    )
                    _update_metric(
                        per_band[global_index],
                        errors[local_index][valid],
                        tolerances[local_index][valid],
                    )

    metrics = {key: _final_metric(value) for key, value in metric_states.items()}
    masks = {key: _final_masks(value) for key, value in mask_states.items()}
    total_violations = metrics["TOTAL"]["tolerance_violations"]
    mask_disagreements = masks["TOTAL"]["disagreement_pixels"]
    if total_violations:
        findings.add(
            "error",
            "radiometric_value_mismatch" if affine else "native_dn_value_mismatch",
            f"{comparison_name} has pixels outside the float32-aware tolerance",
            comparison=comparison_name,
            violations=total_violations,
            compared=metrics["TOTAL"]["compared_pixels"],
        )
    if mask_disagreements:
        findings.add(
            "error",
            "nodata_mask_mismatch",
            f"{comparison_name} has different nodata/finite masks",
            comparison=comparison_name,
            disagreements=mask_disagreements,
        )

    finalized_bands = []
    for index, state in enumerate(per_band):
        metric = _final_metric(state)
        metric.update(
            {
                "band": index + 1,
                "detector": _detector_for_band(left, index),
                "wavelength_nm": (
                    left.expected_bands[index].wavelength_nm
                    if index < len(left.expected_bands)
                    else None
                ),
            }
        )
        finalized_bands.append(metric)
    worst_bands = sorted(
        finalized_bands,
        key=lambda item: (
            int(item["tolerance_violations"]),
            float(item["max_absolute_error"] or 0.0),
        ),
        reverse=True,
    )[: _ARGS.max_reported_mismatches]

    return {
        "comparison": comparison_name,
        "status": "pass" if not total_violations and not mask_disagreements else "fail",
        "equation": (
            "right = source_offset(detector) + left * source_gain(detector)"
            if affine
            else "right = left"
        ),
        "tolerance": {
            "absolute": absolute_tolerance,
            "relative": relative_tolerance,
            "float32_ulps": _ARGS.float32_ulps,
            "rule": "max(absolute, relative*abs(expected_float32), ulps*spacing(expected_float32))",
        },
        "metrics_by_detector": metrics,
        "nodata_masks_by_detector": masks,
        "worst_bands": worst_bands,
    }


def _range_state(minimum: float, maximum: float) -> Dict[str, Any]:
    return {
        "expected_minimum": minimum,
        "expected_maximum": maximum,
        "valid_pixels": 0,
        "invalid_pixels": 0,
        "below_minimum": 0,
        "above_maximum": 0,
        "observed_minimum": None,
        "observed_maximum": None,
    }


def _update_range(state: Dict[str, Any], values: np.ndarray, valid: np.ndarray) -> None:
    state["invalid_pixels"] += int(valid.size - np.count_nonzero(valid))
    selected = np.asarray(values[valid], dtype=np.float64)
    if selected.size == 0:
        return
    state["valid_pixels"] += int(selected.size)
    observed_minimum = float(np.min(selected))
    observed_maximum = float(np.max(selected))
    state["observed_minimum"] = (
        observed_minimum
        if state["observed_minimum"] is None
        else min(state["observed_minimum"], observed_minimum)
    )
    state["observed_maximum"] = (
        observed_maximum
        if state["observed_maximum"] is None
        else max(state["observed_maximum"], observed_maximum)
    )
    state["below_minimum"] += int(np.count_nonzero(selected < state["expected_minimum"]))
    state["above_maximum"] += int(np.count_nonzero(selected > state["expected_maximum"]))


def _scan_overshoots(
    raster: RasterInfo,
    source: SourceProduct,
    findings: Findings,
) -> Dict[str, Any]:
    if not raster.detector_mapping_available:
        return {
            "status": "skipped",
            "reason": "detector sequence unavailable for source-range validation",
            "detector_mapping_reason": raster.detector_mapping_reason,
        }
    if raster.normalization_mode != "none":
        return {
            "status": "skipped",
            "reason": "scene-normalized output has no detector source-range envelope",
        }
    domain = "reflectance" if raster.role == "reflectance" else "native_dn"
    states: Dict[str, Dict[str, Any]] = {}
    for detector in DETECTORS:
        if domain == "reflectance":
            scale = source.scales[detector]
            states[detector] = _range_state(scale.minimum, scale.maximum)
        else:
            states[detector] = _range_state(0.0, DN_DENOMINATOR)

    with rasterio.open(raster.path) as dataset:
        for window in _windows(raster.width, raster.height, _ARGS.block_size):
            for band_start in range(0, raster.count, _ARGS.band_batch_size):
                band_stop = min(raster.count, band_start + _ARGS.band_batch_size)
                indexes = list(range(band_start + 1, band_stop + 1))
                values, valid = _valid_data(dataset.read(indexes=indexes, window=window, masked=True))
                for local_index, global_index in enumerate(range(band_start, band_stop)):
                    detector = _detector_for_band(raster, global_index)
                    if detector in states:
                        _update_range(states[detector], values[local_index], valid[local_index])

    total_overshoots = 0
    results: Dict[str, Any] = {}
    for detector, state in states.items():
        overshoots = int(state["below_minimum"]) + int(state["above_maximum"])
        total_overshoots += overshoots
        valid_count = int(state["valid_pixels"])
        results[detector] = {
            **state,
            "overshoot_pixels": overshoots,
            "overshoot_fraction": overshoots / valid_count if valid_count else None,
        }
    if total_overshoots:
        findings.add(
            "warning",
            "spatial_resampling_overshoot",
            f"{raster.role} contains valid pixels outside its detector source range",
            role=raster.role,
            overshoot_pixels=total_overshoots,
        )
    return {
        "status": "complete",
        "domain": domain,
        "policy": "report_only_do_not_clip",
        "total_overshoot_pixels": total_overshoots,
        "by_detector": results,
    }


def _read_source_band_batch(
    source_h5: h5py.File,
    bands: Sequence[SourceBand],
    row_slice: slice,
    column_slice: slice,
) -> np.ndarray:
    height = row_slice.stop - row_slice.start
    width = column_slice.stop - column_slice.start
    result = np.empty((len(bands), height, width), dtype=np.uint16)
    for detector, dataset_path in (("VNIR", VNIR_CUBE_PATH), ("SWIR", SWIR_CUBE_PATH)):
        positions = [index for index, band in enumerate(bands) if band.detector == detector]
        if not positions:
            continue
        detector_indices = [bands[index].detector_index for index in positions]
        sorted_unique = sorted(set(detector_indices))
        values = source_h5[dataset_path][row_slice, sorted_unique, column_slice]
        index_lookup = {value: index for index, value in enumerate(sorted_unique)}
        values = np.transpose(values, (1, 0, 2))
        for output_position, detector_index in zip(positions, detector_indices):
            result[output_position] = values[index_lookup[detector_index]]
    return result


def _compare_source_pixels(
    raster: RasterInfo,
    source: SourceProduct,
    findings: Findings,
) -> Dict[str, Any]:
    name = f"source_he5_vs_{raster.role}"
    if not raster.detector_mapping_available:
        return {
            "comparison": name,
            "status": "skipped",
            "reason": "detector sequence unavailable for source-pixel validation",
            "detector_mapping_reason": raster.detector_mapping_reason,
        }
    if raster.normalization_mode != "none":
        return {"comparison": name, "status": "skipped", "reason": "output is scene-normalized"}
    if raster.width != source.columns or raster.height != source.rows:
        return {
            "comparison": name,
            "status": "skipped",
            "reason": "output is spatially warped; dimensions differ from the source cube",
            "source_dimensions": [source.columns, source.rows],
            "output_dimensions": [raster.width, raster.height],
        }
    if raster.count != len(raster.expected_bands):
        return {
            "comparison": name,
            "status": "skipped",
            "reason": "output band count does not match the selected source sequence",
        }

    metrics = {"TOTAL": _metric_state(), "VNIR": _metric_state(), "SWIR": _metric_state()}
    per_band = [_metric_state() for _ in range(raster.count)]
    invalid_pixels = 0
    with h5py.File(source.path, "r") as source_h5, rasterio.open(raster.path) as output:
        for window in _windows(raster.width, raster.height, _ARGS.block_size):
            row_slice = slice(int(window.row_off), int(window.row_off + window.height))
            column_slice = slice(int(window.col_off), int(window.col_off + window.width))
            for band_start in range(0, raster.count, _ARGS.band_batch_size):
                band_stop = min(raster.count, band_start + _ARGS.band_batch_size)
                band_sequence = raster.expected_bands[band_start:band_stop]
                source_dn = _read_source_band_batch(
                    source_h5, band_sequence, row_slice, column_slice
                ).astype(np.float32)
                if raster.role == "reflectance":
                    gains = np.asarray(
                        [source.scales[band.detector].gain for band in band_sequence],
                        dtype=np.float32,
                    )[:, None, None]
                    offsets = np.asarray(
                        [source.scales[band.detector].offset for band in band_sequence],
                        dtype=np.float32,
                    )[:, None, None]
                    expected = source_dn * gains + offsets
                    absolute_tolerance = _ARGS.reflectance_absolute_tolerance
                    relative_tolerance = _ARGS.reflectance_relative_tolerance
                else:
                    expected = source_dn
                    absolute_tolerance = _ARGS.dn_absolute_tolerance
                    relative_tolerance = _ARGS.dn_relative_tolerance
                indexes = list(range(band_start + 1, band_stop + 1))
                observed, valid = _valid_data(output.read(indexes=indexes, window=window, masked=True))
                invalid_pixels += int(valid.size - np.count_nonzero(valid))
                errors = observed.astype(np.float64) - expected.astype(np.float64)
                tolerances = _float_tolerance(
                    expected,
                    absolute_tolerance,
                    relative_tolerance,
                    _ARGS.float32_ulps,
                )
                _update_metric(metrics["TOTAL"], errors[valid], tolerances[valid])
                for local_index, global_index in enumerate(range(band_start, band_stop)):
                    detector = band_sequence[local_index].detector
                    band_valid = valid[local_index]
                    _update_metric(
                        metrics[detector],
                        errors[local_index][band_valid],
                        tolerances[local_index][band_valid],
                    )
                    _update_metric(
                        per_band[global_index],
                        errors[local_index][band_valid],
                        tolerances[local_index][band_valid],
                    )

    finalized = {key: _final_metric(value) for key, value in metrics.items()}
    violations = finalized["TOTAL"]["tolerance_violations"]
    if violations:
        findings.add(
            "error",
            "source_reflectance_mismatch" if raster.role == "reflectance" else "source_native_dn_mismatch",
            (
                f"{raster.role} does not reproduce detector-decoded source HE5 samples on the source grid"
                if raster.role == "reflectance"
                else f"{raster.role} does not reproduce source HE5 native samples on the source grid"
            ),
            role=raster.role,
            violations=violations,
        )
    if invalid_pixels:
        findings.add(
            "warning",
            "source_grid_output_nodata",
            f"{raster.role} has invalid pixels although the source uint16 cube has no nodata mask",
            role=raster.role,
            invalid_pixels=invalid_pixels,
        )
    worst_bands = []
    for index, state in enumerate(per_band):
        item = _final_metric(state)
        item.update(
            {
                "band": index + 1,
                "detector": raster.expected_bands[index].detector,
                "wavelength_nm": raster.expected_bands[index].wavelength_nm,
            }
        )
        worst_bands.append(item)
    worst_bands.sort(
        key=lambda item: (
            int(item["tolerance_violations"]),
            float(item["max_absolute_error"] or 0.0),
        ),
        reverse=True,
    )
    return {
        "comparison": name,
        "status": "pass" if not violations else "fail",
        "equation": (
            "output_reflectance = source_offset(detector) + source_uint16_dn * source_gain(detector)"
            if raster.role == "reflectance"
            else "output_native_dn = source_uint16_dn"
        ),
        "output_invalid_pixels": invalid_pixels,
        "metrics_by_detector": finalized,
        "worst_bands": worst_bands[: _ARGS.max_reported_mismatches],
    }


def _path_argument(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blockwise validation of legacy/native-DN/reflectance PRISMA GeoTIFFs "
            "against their source Level-2D HE5 product. JSON is always emitted to stdout."
        )
    )
    parser.add_argument("--source-he5", required=True, type=_path_argument, help="Source PRISMA L2D .he5 file")
    parser.add_argument("--old", type=_path_argument, help="Legacy HyperCoreg GeoTIFF (native DN without contract tags)")
    parser.add_argument("--native-dn", type=_path_argument, help="New GeoTIFF produced in native-dn compatibility mode")
    parser.add_argument("--reflectance", type=_path_argument, help="New GeoTIFF produced in reflectance mode")
    parser.add_argument("--json-output", type=Path, help="Also write the JSON report to this path")
    parser.add_argument("--block-size", type=_positive_int, default=128, help="Spatial window size (default: 128)")
    parser.add_argument("--band-batch-size", type=_positive_int, default=512, help="Bands read per window (default: 512; all PRISMA bands)")
    parser.add_argument(
        "--reflectance-absolute-tolerance",
        type=_nonnegative_float,
        default=1e-6,
        help="Absolute tolerance for decoded reflectance (default: 1e-6)",
    )
    parser.add_argument(
        "--reflectance-relative-tolerance",
        type=_nonnegative_float,
        default=1e-6,
        help="Relative tolerance for decoded reflectance (default: 1e-6)",
    )
    parser.add_argument(
        "--dn-absolute-tolerance",
        type=_nonnegative_float,
        default=1e-3,
        help="Absolute tolerance for native-DN identity checks (default: 1e-3)",
    )
    parser.add_argument(
        "--dn-relative-tolerance",
        type=_nonnegative_float,
        default=0.0,
        help="Relative tolerance for native-DN identity checks (default: 0)",
    )
    parser.add_argument(
        "--float32-ulps",
        type=_positive_int,
        default=8,
        help="Minimum tolerance in float32 units-in-last-place (default: 8)",
    )
    parser.add_argument(
        "--grid-tolerance",
        type=_nonnegative_float,
        default=1e-7,
        help="Absolute affine-transform comparison tolerance (default: 1e-7)",
    )
    parser.add_argument(
        "--wavelength-tolerance-nm",
        type=_nonnegative_float,
        default=0.02,
        help="Band-tag wavelength tolerance in nm (default: 0.02)",
    )
    parser.add_argument(
        "--max-reported-mismatches",
        type=_positive_int,
        default=20,
        help="Maximum detailed band mismatches/worst bands in JSON (default: 20)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact rather than indented JSON",
    )
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _emit(payload: Mapping[str, Any], output_path: Optional[Path], compact: bool) -> None:
    safe_payload = _json_safe(dict(payload))
    text = json.dumps(
        safe_payload,
        indent=None if compact else 2,
        sort_keys=True,
        allow_nan=False,
    )
    print(text)
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(text + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Tuple[Dict[str, Any], int]:
    findings = Findings()
    if args.json_output is not None:
        json_target = args.json_output.expanduser().resolve()
        input_paths = [
            path for path in (args.source_he5, args.old, args.native_dn, args.reflectance) if path is not None
        ]
        if json_target in input_paths:
            raise ValueError("--json-output must not overwrite an input HE5 or GeoTIFF")
    source = _load_source(args.source_he5)
    raster_paths = {
        "old": args.old,
        "native_dn": args.native_dn,
        "reflectance": args.reflectance,
    }
    rasters: Dict[str, RasterInfo] = {}
    for role, path in raster_paths.items():
        if path is None:
            continue
        raster = _load_raster(role, path)
        _select_expected_bands(raster, source, findings)
        rasters[role] = raster

    if not rasters:
        findings.add(
            "warning",
            "no_output_rasters",
            "No output GeoTIFF was supplied; only source metadata was validated",
        )

    contracts: Dict[str, Any] = {}
    ranges: Dict[str, Any] = {}
    for role, raster in rasters.items():
        contracts[role] = _validate_contract_tags(raster, source, findings)
        ranges[role] = _scan_overshoots(raster, source, findings)

    grids: Dict[str, Any] = {}
    for (left_role, left), (right_role, right) in itertools.combinations(rasters.items(), 2):
        key = f"{left_role}_vs_{right_role}"
        grids[key] = _compare_grids(left, right, findings)

    value_comparisons: Dict[str, Any] = {}
    if "old" in rasters and "native_dn" in rasters:
        result = _compare_raster_values(
            rasters["old"], rasters["native_dn"], source, affine=False, findings=findings
        )
        value_comparisons[result["comparison"]] = result

    dn_role: Optional[str] = None
    if "native_dn" in rasters:
        dn_role = "native_dn"
    elif "old" in rasters:
        dn_role = "old"
    if dn_role is not None and "reflectance" in rasters:
        result = _compare_raster_values(
            rasters[dn_role], rasters["reflectance"], source, affine=True, findings=findings
        )
        value_comparisons[result["comparison"]] = result

    source_comparisons: Dict[str, Any] = {}
    for role in ("old", "native_dn", "reflectance"):
        if role not in rasters:
            continue
        result = _compare_source_pixels(rasters[role], source, findings)
        source_comparisons[result["comparison"]] = result

    summary = findings.summary()
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "inputs": {
            "source_he5": str(args.source_he5),
            "old": str(args.old) if args.old else None,
            "native_dn": str(args.native_dn) if args.native_dn else None,
            "reflectance": str(args.reflectance) if args.reflectance else None,
        },
        "settings": {
            "block_size": args.block_size,
            "band_batch_size": args.band_batch_size,
            "reflectance_absolute_tolerance": args.reflectance_absolute_tolerance,
            "reflectance_relative_tolerance": args.reflectance_relative_tolerance,
            "dn_absolute_tolerance": args.dn_absolute_tolerance,
            "dn_relative_tolerance": args.dn_relative_tolerance,
            "float32_ulps": args.float32_ulps,
            "grid_tolerance": args.grid_tolerance,
            "wavelength_tolerance_nm": args.wavelength_tolerance_nm,
        },
        "source": source.report(),
        "rasters": {role: raster.report() for role, raster in rasters.items()},
        "radiometric_contract_validation": contracts,
        "grid_comparisons": grids,
        "range_and_overshoot_checks": ranges,
        "value_comparisons": value_comparisons,
        "source_grid_comparisons": source_comparisons,
        "findings": findings.items,
    }
    return payload, 1 if summary["status"] == "fail" else 0


_ARGS: argparse.Namespace


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _ARGS
    parser = _build_parser()
    _ARGS = parser.parse_args(argv)
    try:
        payload, exit_code = run(_ARGS)
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "summary": {"status": "fail", "finding_counts": {"error": 1, "warning": 0, "info": 0}},
            "findings": [
                {
                    "severity": "error",
                    "code": "validator_exception",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                }
            ],
        }
        exit_code = 2
    _emit(payload, _ARGS.json_output, _ARGS.compact)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
