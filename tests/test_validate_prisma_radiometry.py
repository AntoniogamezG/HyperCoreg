"""Focused tests for the standalone PRISMA radiometry validator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import h5py
import numpy as np
import pytest

from scripts import validate_prisma_radiometry as validator


def _band(detector: str, detector_index: int, wavelength_nm: float) -> validator.SourceBand:
    return validator.SourceBand(
        detector=detector,
        detector_index=detector_index,
        wavelength_nm=wavelength_nm,
        fwhm_nm=10.0,
    )


ALL_BANDS = [
    _band("VNIR", 0, 500.0),
    _band("VNIR", 1, 900.0),
    _band("SWIR", 0, 1000.0),
]
OVERLAP_REMOVED_BANDS = [ALL_BANDS[0], ALL_BANDS[2]]


def _source(
    *,
    all_bands: Iterable[validator.SourceBand] = ALL_BANDS,
    overlap_removed_bands: Iterable[validator.SourceBand] = OVERLAP_REMOVED_BANDS,
    product_metadata: Optional[dict[str, Optional[str]]] = None,
) -> validator.SourceProduct:
    return validator.SourceProduct(
        path=Path("source.he5"),
        rows=1,
        columns=1,
        scales={
            "VNIR": validator.ScalePair(0.1, 0.5),
            "SWIR": validator.ScalePair(-0.2, 0.8),
        },
        bands_all=list(all_bands),
        bands_overlap_removed=list(overlap_removed_bands),
        product_metadata=dict(product_metadata or {}),
    )


def _raster(
    role: str,
    overlap_tag: Optional[str],
    bands: Iterable[validator.SourceBand],
) -> validator.RasterInfo:
    source_bands = list(bands)
    dataset_tags = {}
    if overlap_tag is not None:
        dataset_tags["REMOVE_OVERLAPPING_BANDS"] = overlap_tag
    return validator.RasterInfo(
        role=role,
        path=Path(f"{role}.tif"),
        width=1,
        height=1,
        count=len(source_bands),
        crs="EPSG:32632",
        transform=(0.0, 30.0, 0.0, 0.0, 0.0, -30.0),
        dtypes=tuple("float32" for _ in source_bands),
        nodata=None,
        dataset_tags=dataset_tags,
        bands=[
            validator.RasterBand(
                detector=band.detector,
                wavelength_nm=band.wavelength_nm,
                description=None,
                tags={},
            )
            for band in source_bands
        ],
        active_scales=tuple(1.0 for _ in source_bands),
        active_offsets=tuple(0.0 for _ in source_bands),
        block_shapes=tuple((1, 1) for _ in source_bands),
    )


@pytest.fixture(autouse=True)
def _validator_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        validator,
        "_ARGS",
        SimpleNamespace(
            wavelength_tolerance_nm=0.01,
            max_reported_mismatches=10,
            grid_tolerance=1e-7,
        ),
        raising=False,
    )


def _findings_with_code(
    findings: validator.Findings, code: str
) -> list[dict[str, object]]:
    return [item for item in findings.items if item["code"] == code]


def _contract_raster(
    source: validator.SourceProduct,
    *,
    role: str = "reflectance",
) -> validator.RasterInfo:
    raster = _raster(role, "false", ALL_BANDS)
    mode = "native-dn" if role == "native_dn" else "reflectance"
    raster.dataset_tags.update(
        {
            "PRISMA_RADIOMETRIC_MODE": mode,
            "PRISMA_L2_SCALING_APPLIED": "false" if mode == "native-dn" else "true",
            "PRISMA_DN_DENOMINATOR": "65535",
            "RADIOMETRIC_QUANTITY": (
                "native_encoded_dn" if mode == "native-dn" else "surface_reflectance"
            ),
            "RADIOMETRIC_UNITS": "DN" if mode == "native-dn" else "1",
            "NORMALIZATION_MODE": "none",
            "RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY": "preserve_and_flag_not_clip",
        }
    )
    for detector, scale in source.scales.items():
        raster.dataset_tags[f"PRISMA_L2_SCALE_{detector}_MIN"] = str(scale.minimum)
        raster.dataset_tags[f"PRISMA_L2_SCALE_{detector}_MAX"] = str(scale.maximum)
    raster.detector_mapping_available = True
    raster.detector_mapping_reason = "test fixture contract mapping"
    return raster


def test_true_tag_cannot_override_exact_full_band_count_or_affine_mapping() -> None:
    source = _source()
    raster = _raster("reflectance", "true", ALL_BANDS)
    findings = validator.Findings()

    validator._select_expected_bands(raster, source, findings)

    mismatch = _findings_with_code(findings, "overlap_tag_band_count_mismatch")
    assert len(mismatch) == 1
    assert mismatch[0]["severity"] == "error"
    assert raster.remove_overlap is False
    assert raster.expected_bands == ALL_BANDS
    assert raster.band_order_validation["matches_source"] is True

    # The second full-sequence band is VNIR.  Selecting the tag-implied
    # overlap-removed sequence would incorrectly apply the SWIR affine pair.
    assert validator._detector_for_band(raster, 1) == "VNIR"
    native_dn = 1000.0
    actual_expected = source.scales[validator._detector_for_band(raster, 1)]
    assert actual_expected.offset + native_dn * actual_expected.gain == pytest.approx(
        source.scales["VNIR"].offset + native_dn * source.scales["VNIR"].gain
    )


def test_false_tag_cannot_override_exact_overlap_removed_count_or_affine_mapping() -> None:
    source = _source()
    raster = _raster("native_dn", "false", OVERLAP_REMOVED_BANDS)
    findings = validator.Findings()

    validator._select_expected_bands(raster, source, findings)

    mismatch = _findings_with_code(findings, "overlap_tag_band_count_mismatch")
    assert len(mismatch) == 1
    assert mismatch[0]["severity"] == "error"
    assert raster.remove_overlap is True
    assert raster.expected_bands == OVERLAP_REMOVED_BANDS
    assert raster.band_order_validation["matches_source"] is True

    # In the reduced sequence the second band is SWIR, not the second VNIR
    # band from the tag-implied full sequence.
    assert validator._detector_for_band(raster, 1) == "SWIR"


@pytest.mark.parametrize(
    ("tag", "bands", "expected_remove"),
    [
        ("false", ALL_BANDS, False),
        ("true", OVERLAP_REMOVED_BANDS, True),
    ],
)
def test_consistent_overlap_tags_preserve_normal_selection(
    tag: str,
    bands: list[validator.SourceBand],
    expected_remove: bool,
) -> None:
    source = _source()
    raster = _raster("reflectance", tag, bands)
    findings = validator.Findings()

    validator._select_expected_bands(raster, source, findings)

    assert raster.remove_overlap is expected_remove
    assert raster.expected_bands == source.bands(expected_remove)
    assert not _findings_with_code(findings, "overlap_tag_band_count_mismatch")
    assert not _findings_with_code(findings, "overlap_mode_inferred")


def test_equal_candidate_counts_do_not_create_a_false_contradiction() -> None:
    source = _source(overlap_removed_bands=ALL_BANDS)
    raster = _raster("reflectance", "true", ALL_BANDS)
    findings = validator.Findings()

    validator._select_expected_bands(raster, source, findings)

    assert raster.remove_overlap is True
    assert not _findings_with_code(findings, "overlap_tag_band_count_mismatch")


def test_legacy_tag_count_contradiction_is_a_warning() -> None:
    source = _source()
    raster = _raster("old", "true", ALL_BANDS)
    findings = validator.Findings()

    validator._select_expected_bands(raster, source, findings)

    mismatch = _findings_with_code(findings, "overlap_tag_band_count_mismatch")
    assert len(mismatch) == 1
    assert mismatch[0]["severity"] == "warning"
    assert raster.remove_overlap is False


PROVENANCE_METADATA = {
    "product_name": "PRS   L2D TEST",
    "processing_level": " L2D ",
    "processor_version": "9.8.7",
    "processing_time": "2025-01-02T05:06:07Z",
}


def test_new_output_source_provenance_matches_normalized_source_text() -> None:
    source = _source(product_metadata=PROVENANCE_METADATA)
    raster = _contract_raster(source)
    raster.dataset_tags.update(
        {
            "SOURCE_PRODUCT_NAME": "  PRS L2D TEST  ",
            "SOURCE_PROCESSING_LEVEL": "L2D",
            "SOURCE_PROCESSOR_VERSION": " 9.8.7 ",
            "SOURCE_PROCESSING_TIME": "2025-01-02T05:06:07Z",
        }
    )
    findings = validator.Findings()

    result = validator._validate_contract_tags(raster, source, findings)

    checks = result["source_provenance_tag_checks"]
    assert [check["status"] for check in checks] == ["pass"] * 4
    assert not _findings_with_code(findings, "source_provenance_tag_mismatch")
    assert result["ok"] is True


@pytest.mark.parametrize("clipping_state", ["with_clipping", "without_clipping"])
def test_normalized_output_accepts_its_transformed_overshoot_policy(
    clipping_state: str,
) -> None:
    source = _source()
    raster = _contract_raster(source)
    raster.dataset_tags.update(
        {
            "NORMALIZATION_MODE": "percentile",
            "NORM_CLIP": "true" if clipping_state == "with_clipping" else "false",
            "RADIOMETRIC_QUANTITY": "normalized_unitless",
            "RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY": (
                f"transformed_by_percentile_normalization_{clipping_state}"
            ),
        }
    )
    findings = validator.Findings()

    result = validator._validate_contract_tags(raster, source, findings)

    assert not _findings_with_code(findings, "radiometric_dataset_tag_mismatch")
    assert result["ok"] is True


def test_normalized_output_rejects_policy_inconsistent_with_clip_tag() -> None:
    source = _source()
    raster = _contract_raster(source)
    raster.dataset_tags.update(
        {
            "NORMALIZATION_MODE": "minmax",
            "NORM_CLIP": "true",
            "RADIOMETRIC_QUANTITY": "normalized_unitless",
            "RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY": (
                "transformed_by_minmax_normalization_without_clipping"
            ),
        }
    )
    findings = validator.Findings()

    result = validator._validate_contract_tags(raster, source, findings)

    mismatches = _findings_with_code(findings, "radiometric_dataset_tag_mismatch")
    assert any(
        item["context"]["tag"] == "RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY"
        for item in mismatches
    )
    assert result["ok"] is False


def test_new_output_missing_and_mismatched_source_provenance_are_errors() -> None:
    source = _source(product_metadata=PROVENANCE_METADATA)
    raster = _contract_raster(source)
    raster.dataset_tags.update(
        {
            # SOURCE_PRODUCT_NAME is deliberately absent.
            "SOURCE_PROCESSING_LEVEL": "L1",
            "SOURCE_PROCESSOR_VERSION": "9.8.7",
            "SOURCE_PROCESSING_TIME": "2025-01-02T05:06:07Z",
        }
    )
    findings = validator.Findings()

    result = validator._validate_contract_tags(raster, source, findings)

    failures = _findings_with_code(findings, "source_provenance_tag_mismatch")
    assert len(failures) == 2
    assert {item["context"]["tag"] for item in failures} == {
        "SOURCE_PRODUCT_NAME",
        "SOURCE_PROCESSING_LEVEL",
    }
    assert all(item["severity"] == "error" for item in failures)
    assert result["ok"] is False


def test_unavailable_source_provenance_is_reported_not_applicable() -> None:
    source = _source(
        product_metadata={
            "product_name": None,
            "processing_level": "  ",
            # The remaining source keys are deliberately absent.
        }
    )
    raster = _contract_raster(source)
    raster.dataset_tags["SOURCE_PRODUCT_NAME"] = "UNVERIFIABLE VALUE"
    findings = validator.Findings()

    result = validator._validate_contract_tags(raster, source, findings)

    checks = result["source_provenance_tag_checks"]
    assert [check["status"] for check in checks] == ["not_applicable"] * 4
    assert all(check["ok"] is None for check in checks)
    assert not _findings_with_code(findings, "source_provenance_tag_mismatch")
    assert result["ok"] is True


def test_legacy_output_is_exempt_from_source_provenance_contract() -> None:
    source = _source(product_metadata=PROVENANCE_METADATA)
    raster = _contract_raster(source, role="old")
    for tag, _metadata_key in validator.SOURCE_PROVENANCE_TAGS:
        raster.dataset_tags.pop(tag, None)
    findings = validator.Findings()

    result = validator._validate_contract_tags(raster, source, findings)

    assert result["contract_expected"] is False
    assert "source_provenance_tag_checks" not in result
    assert not _findings_with_code(findings, "source_provenance_tag_mismatch")


def _write_validator_source(
    path: Path,
    *,
    cube_dtype: str,
    pan_scale: str = "absent",
) -> Path:
    dtype = np.dtype(cube_dtype)
    with h5py.File(path, "w") as product:
        product.attrs["List_Cw_Vnir"] = np.array([500.0, 900.0])
        product.attrs["List_Cw_Swir"] = np.array([1000.0])
        product.attrs["L2ScaleVnirMin"] = 0.1
        product.attrs["L2ScaleVnirMax"] = 0.5
        product.attrs["L2ScaleSwirMin"] = -0.2
        product.attrs["L2ScaleSwirMax"] = 0.8
        if pan_scale == "valid":
            product.attrs["L2ScalePanMin"] = 0.0
            product.attrs["L2ScalePanMax"] = 1.0
        elif pan_scale == "invalid":
            product.attrs["L2ScalePanMin"] = 1.0
            product.attrs["L2ScalePanMax"] = 0.0
        data = product.require_group(validator.HCO_DATA_ROOT)
        data.create_dataset(
            "VNIR_Cube",
            data=np.array([[[1], [2]]], dtype=dtype),
            dtype=dtype,
        )
        data.create_dataset(
            "SWIR_Cube",
            data=np.array([[[3]]], dtype=dtype),
            dtype=dtype,
        )
    return path


@pytest.mark.parametrize("cube_dtype", ["<u2", ">u2"])
def test_source_loader_accepts_either_endian_uint16_and_optional_pan(
    tmp_path: Path,
    cube_dtype: str,
) -> None:
    path = _write_validator_source(
        tmp_path / f"source_{cube_dtype[-2:]}.he5",
        cube_dtype=cube_dtype,
    )

    source = validator._load_source(path)

    assert set(source.scales) == {"VNIR", "SWIR"}
    assert source.product_metadata["pan_scale_status"] == "not_applicable"
    assert "Missing source HE5 attribute L2ScalePanMin" in str(
        source.product_metadata["pan_scale_reason"]
    )


def test_source_loader_treats_invalid_pan_scale_as_not_applicable(tmp_path: Path) -> None:
    path = _write_validator_source(
        tmp_path / "invalid_pan.he5",
        cube_dtype="uint16",
        pan_scale="invalid",
    )

    source = validator._load_source(path)

    assert "PAN" not in source.scales
    assert source.product_metadata["pan_scale_status"] == "not_applicable"
    assert "Invalid PAN scale range" in str(source.product_metadata["pan_scale_reason"])


def test_source_loader_reads_provenance_from_additional_file_attributes(
    tmp_path: Path,
) -> None:
    path = _write_validator_source(
        tmp_path / "additional_provenance.he5",
        cube_dtype="uint16",
    )
    with h5py.File(path, "r+") as product:
        product.attrs["Product_Name"] = np.bytes_("ROOT PRODUCT")
        additional = product.require_group("HDFEOS/ADDITIONAL/FILE_ATTRIBUTES")
        additional.attrs["Product_Name"] = np.bytes_("ADDITIONAL PRODUCT")
        additional.attrs["Processing_Level"] = np.bytes_("L2D")
        additional.attrs["Processor_Version"] = np.bytes_("9.8.7")
        additional.attrs["Processing_Time"] = np.bytes_("2025-01-02T05:06:07Z")
        additional.attrs["Product_StartTime"] = np.bytes_("2025-01-02T03:04:05Z")

    source = validator._load_source(path)

    assert source.product_metadata == {
        "product_name": "ROOT PRODUCT",
        "processing_level": "L2D",
        "processor_version": "9.8.7",
        "processing_time": "2025-01-02T05:06:07Z",
        "product_start_time": "2025-01-02T03:04:05Z",
        "pan_scale_status": "not_applicable",
        "pan_scale_reason": "Missing source HE5 attribute L2ScalePanMin",
    }


def test_source_loader_rejects_non_uint16_spectral_samples(tmp_path: Path) -> None:
    path = _write_validator_source(
        tmp_path / "uint32.he5",
        cube_dtype=">u4",
    )

    with pytest.raises(ValueError, match="must be uint16"):
        validator._load_source(path)


def test_unresolved_noncanonical_count_skips_detector_dependent_checks() -> None:
    source = _source()
    native = _raster("native_dn", None, [ALL_BANDS[0]])
    reflectance = _raster("reflectance", None, [ALL_BANDS[0]])
    missing_metadata_band = validator.RasterBand(None, None, None, {})
    native.bands = [missing_metadata_band]
    reflectance.bands = [missing_metadata_band]
    findings = validator.Findings()

    validator._select_expected_bands(native, source, findings)
    validator._select_expected_bands(reflectance, source, findings)

    assert native.detector_mapping_available is False
    assert native.expected_bands == []
    unavailable = _findings_with_code(findings, "detector_sequence_unavailable")
    assert len(unavailable) == 2

    range_result = validator._scan_overshoots(native, source, findings)
    affine_result = validator._compare_raster_values(
        native, reflectance, source, affine=True, findings=findings
    )
    source_result = validator._compare_source_pixels(native, source, findings)
    contract_result = validator._validate_contract_tags(native, source, findings)
    assert range_result["status"] == "skipped"
    assert "detector sequence unavailable" in range_result["reason"]
    assert affine_result["status"] == "skipped"
    assert "detector sequence unavailable" in affine_result["reason"]
    assert source_result["status"] == "skipped"
    assert "detector sequence unavailable" in source_result["reason"]
    assert contract_result["detector_dependent_band_checks"]["status"] == "skipped"
    assert contract_result["ok"] is False


def test_complete_metadata_resolves_a_noncanonical_source_subsequence() -> None:
    source = _source()
    raster = _raster("reflectance", None, [ALL_BANDS[2]])
    findings = validator.Findings()

    validator._select_expected_bands(raster, source, findings)

    assert raster.detector_mapping_available is True
    assert raster.expected_bands == [ALL_BANDS[2]]
    assert raster.band_order_validation["selection_basis"] == "complete_band_metadata"
    assert not _findings_with_code(findings, "detector_sequence_unavailable")
    assert len(_findings_with_code(findings, "detector_sequence_from_band_metadata")) == 1
