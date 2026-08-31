"""Regression tests for PRISMA Level-2 radiometric decoding.

The HE5 fixtures in this module intentionally use the same dataset layout and
spectral axis order as PRISMA L2D products while remaining only a few pixels in
size.  They exercise the real HDF5 reader rather than replacing it with mocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pytest
import rasterio

from hypercoreg.config import DEFAULT_CONFIG
from hypercoreg.normalization import NormalizationParams, normalize_raster_to_path
from hypercoreg.radiometry import (
    PRISMA_DN_DENOMINATOR,
    PrismaRadiometricMetadataError,
    PrismaScalePair,
    build_prisma_radiometric_contract,
    decode_prisma_l2,
    normalize_prisma_radiometric_mode,
    prisma_active_scale_offset,
    prisma_radiometric_band_tags,
    prisma_radiometric_dataset_tags,
    validate_prisma_native_samples,
    validate_prisma_scale_pair,
)
from hypercoreg.readers.prisma import (
    extract_prisma_extended_metadata,
    read_prisma_cube_and_meta,
    read_prisma_pan_and_geo,
)


VNIR_MIN = 0.1
VNIR_MAX = 0.5
SWIR_MIN = -0.2
SWIR_MAX = 0.8
PAN_MIN = 0.05
PAN_MAX = 0.95


def _encoded_cube(dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Return detector cubes in PRISMA's (row, band, column) order."""
    vnir = np.empty((2, 3, 3), dtype=dtype)
    swir = np.empty((2, 2, 3), dtype=dtype)
    for band, value in enumerate((1000, 2000, 3000)):
        vnir[:, band, :] = value
    for band, value in enumerate((4000, 5000)):
        swir[:, band, :] = value
    return vnir, swir


def _write_synthetic_prisma(
    path: Path,
    *,
    sample_dtype: Union[str, np.dtype] = np.uint16,
    include_scales: bool = True,
    include_pan: bool = True,
) -> Path:
    """Write a minimal but structurally realistic PRISMA L2D HE5 product."""
    dtype = np.dtype(sample_dtype)
    vnir, swir = _encoded_cube(dtype)

    with h5py.File(path, "w") as product:
        product.attrs["Product_StartTime"] = np.bytes_("2025-01-02T03:04:05Z")
        product.attrs["Product_Name"] = np.bytes_("SYNTHETIC_PRISMA_L2D")
        product.attrs["Processing_Level"] = np.bytes_("L2D")
        product.attrs["Processing_Time"] = np.bytes_("2025-01-02T05:06:07Z")

        # Wavelengths are deliberately unordered across the two detectors and
        # contain one invalid VNIR band.  Expected valid order is therefore:
        # VNIR 500, SWIR 700, VNIR 900, SWIR 1200 nm.
        product.attrs["List_Cw_Vnir"] = np.array([900.0, 0.0, 500.0])
        product.attrs["List_Cw_Swir"] = np.array([1200.0, 700.0])
        product.attrs["List_Fwhm_Vnir"] = np.array([9.0, 99.0, 5.0])
        product.attrs["List_Fwhm_Swir"] = np.array([12.0, 7.0])

        if include_scales:
            product.attrs["L2ScaleVnirMin"] = VNIR_MIN
            product.attrs["L2ScaleVnirMax"] = VNIR_MAX
            product.attrs["L2ScaleSwirMin"] = SWIR_MIN
            product.attrs["L2ScaleSwirMax"] = SWIR_MAX
            product.attrs["L2ScalePanMin"] = PAN_MIN
            product.attrs["L2ScalePanMax"] = PAN_MAX

        hco_data = product.require_group("HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields")
        hco_data.create_dataset("VNIR_Cube", data=vnir, dtype=dtype)
        hco_data.create_dataset("SWIR_Cube", data=swir, dtype=dtype)

        hco_geo = product.require_group("HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields")
        hco_geo.create_dataset(
            "Latitude",
            data=np.array([[45.0, 45.0, 45.0], [44.9, 44.9, 44.9]], dtype=np.float32),
        )
        hco_geo.create_dataset(
            "Longitude",
            data=np.array([[7.0, 7.1, 7.2], [7.0, 7.1, 7.2]], dtype=np.float32),
        )

        additional = product.require_group("HDFEOS/ADDITIONAL/FILE_ATTRIBUTES")
        additional.attrs["Processor_Version"] = np.bytes_("9.8.7")

        if include_pan:
            pan_values = np.array(
                [[0, 1, PRISMA_DN_DENOMINATOR], [10, 100, 1000]],
                dtype=dtype,
            )
            pco = product.require_group("HDFEOS/SWATHS/PRS_L2D_PCO")
            pco.require_group("Data Fields").create_dataset(
                "Cube", data=pan_values[:, :, np.newaxis], dtype=dtype
            )
            pco_geo = pco.require_group("Geolocation Fields")
            pco_geo.create_dataset(
                "Latitude",
                data=np.array([[45.0, 45.0, 45.0], [44.9, 44.9, 44.9]], dtype=np.float32),
            )
            pco_geo.create_dataset(
                "Longitude",
                data=np.array([[7.0, 7.1, 7.2], [7.0, 7.1, 7.2]], dtype=np.float32),
            )

    return path


@pytest.fixture
def prisma_file(tmp_path: Path) -> Path:
    return _write_synthetic_prisma(tmp_path / "PRS_L2D_STD_20250102_TEST.he5")


def _decode_expected(value: int, minimum: float, maximum: float) -> np.float32:
    gain = np.float32((maximum - minimum) / PRISMA_DN_DENOMINATOR)
    return np.float32(value) * gain + np.float32(minimum)


def _contract_metadata() -> dict[str, object]:
    return {
        "prisma_id": "SYNTHETIC_PRISMA_L2D",
        "source_product_name": "SYNTHETIC_PRISMA_L2D",
        "source_processing_level": "L2D",
        "source_processor_version": "9.8.7",
        "source_processing_time": "2025-01-02T05:06:07Z",
        "prisma_l2_scale_vnir_min": VNIR_MIN,
        "prisma_l2_scale_vnir_max": VNIR_MAX,
        "prisma_l2_scale_swir_min": SWIR_MIN,
        "prisma_l2_scale_swir_max": SWIR_MAX,
        "prisma_l2_scale_pan_min": PAN_MIN,
        "prisma_l2_scale_pan_max": PAN_MAX,
    }


def test_reflectance_is_the_default_configuration() -> None:
    assert DEFAULT_CONFIG["prisma_radiometric_mode"] == "reflectance"
    assert normalize_prisma_radiometric_mode(None) == "reflectance"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("reflectance", "reflectance"),
        (" REFLECTANCE ", "reflectance"),
        ("native-dn", "native-dn"),
        ("native_dn", "native-dn"),
    ],
)
def test_radiometric_mode_normalization(value: str, expected: str) -> None:
    assert normalize_prisma_radiometric_mode(value) == expected


def test_invalid_radiometric_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="Invalid PRISMA radiometric mode"):
        normalize_prisma_radiometric_mode("divide-by-10000")


def test_scale_pair_validation_and_affine_coefficients() -> None:
    scale = validate_prisma_scale_pair(np.float32(0.125), np.array(0.75), "vnir")

    assert scale.minimum == pytest.approx(0.125)
    assert scale.maximum == pytest.approx(0.75)
    assert scale.offset == pytest.approx(0.125)
    assert scale.gain == pytest.approx((0.75 - 0.125) / PRISMA_DN_DENOMINATOR)


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        ([0.0], 1.0, "must be scalar"),
        (0.0, [1.0], "must be scalar"),
        ("not-a-number", 1.0, "not numeric"),
        (np.nan, 1.0, "must be finite"),
        (0.0, np.inf, "must be finite"),
        (1.0, 1.0, "must be greater"),
        (2.0, 1.0, "must be greater"),
    ],
)
def test_invalid_scale_pairs_fail_closed(minimum: object, maximum: object, message: str) -> None:
    with pytest.raises(PrismaRadiometricMetadataError, match=message):
        validate_prisma_scale_pair(minimum, maximum, "SWIR")


def test_decode_uses_product_affine_mapping_and_float32() -> None:
    encoded = np.array([0, 1, 32768, PRISMA_DN_DENOMINATOR], dtype=np.uint16)
    scale = PrismaScalePair(minimum=-0.125, maximum=0.875)

    decoded = decode_prisma_l2(encoded, scale)

    expected = encoded.astype(np.float32) * np.float32(scale.gain) + np.float32(scale.offset)
    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, expected)
    assert decoded[0] == pytest.approx(scale.minimum)
    assert decoded[-1] == pytest.approx(scale.maximum)


def test_decode_accepts_byte_swapped_uint16_without_changing_values() -> None:
    encoded = np.array([0, 1, 258, 32768, 65535], dtype=">u2")
    scale = PrismaScalePair(minimum=0.2, maximum=0.9)

    actual = decode_prisma_l2(encoded, scale)
    native_expected = np.array([0, 1, 258, 32768, 65535], dtype=np.uint16)
    expected = native_expected.astype(np.float32) * np.float32(scale.gain) + np.float32(scale.offset)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.dtype("<u2"), np.dtype(">u2")])
def test_native_sample_validation_accepts_either_uint16_byte_order(dtype: np.dtype) -> None:
    encoded = np.array([0, 258, 65535], dtype=dtype)

    validated = validate_prisma_native_samples(encoded, "VNIR")

    assert validated is encoded
    assert validated.dtype == dtype


def test_native_sample_validation_preserves_optional_none() -> None:
    assert validate_prisma_native_samples(None, "PAN") is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.uint32])
def test_native_sample_validation_rejects_non_uint16_state(dtype: np.dtype) -> None:
    with pytest.raises(TypeError, match=r"PRISMA SWIR native samples.*Refusing possible double decoding"):
        validate_prisma_native_samples(np.array([0, 1], dtype=dtype), "SWIR")


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.uint32])
def test_decode_rejects_already_decoded_or_wrong_width_samples(dtype: np.dtype) -> None:
    with pytest.raises(TypeError, match="Refusing possible double decoding"):
        decode_prisma_l2(np.array([0, 1], dtype=dtype), PrismaScalePair(0.0, 1.0))


def test_reflectance_contract_carries_source_provenance_and_identity_active_scale() -> None:
    contract = build_prisma_radiometric_contract(
        _contract_metadata(),
        "reflectance",
        normalization_mode="none",
        spatial_resampling_kernel="cubic",
        require_pan=True,
    )

    assert contract["quantity"] == "surface_reflectance"
    assert contract["units"] == "1"
    assert contract["l2_scaling_applied"] is True
    assert contract["dn_denominator"] == 65535
    assert contract["source_product_name"] == "SYNTHETIC_PRISMA_L2D"
    assert contract["source_processing_level"] == "L2D"
    assert contract["source_processor_version"] == "9.8.7"
    assert contract["spatial_resampling_kernel"] == "cubic"
    assert contract["resampling_overshoot_policy"] == "preserve_and_flag_not_clip"
    assert contract["scales"]["VNIR"]["gain"] == pytest.approx(
        (VNIR_MAX - VNIR_MIN) / PRISMA_DN_DENOMINATOR
    )
    assert prisma_active_scale_offset(contract, "VNIR") == (1.0, 0.0)

    dataset_tags = prisma_radiometric_dataset_tags(contract)
    assert dataset_tags["RADIOMETRIC_QUANTITY"] == "surface_reflectance"
    assert dataset_tags["RADIOMETRIC_UNITS"] == "1"
    assert dataset_tags["NORMALIZATION_MODE"] == "none"
    assert dataset_tags["PRISMA_RADIOMETRIC_MODE"] == "reflectance"
    assert dataset_tags["PRISMA_L2_SCALING_APPLIED"] == "true"
    assert dataset_tags["PRISMA_DN_DENOMINATOR"] == "65535"
    assert dataset_tags["SOURCE_PROCESSOR_VERSION"] == "9.8.7"
    assert dataset_tags["SPATIAL_RESAMPLING_KERNEL"] == "cubic"

    band_tags = prisma_radiometric_band_tags(contract, "VNIR")
    assert band_tags["RADIOMETRIC_QUANTITY"] == "surface_reflectance"
    assert band_tags["SOURCE_DN_GAIN"] == str(contract["scales"]["VNIR"]["gain"])
    assert band_tags["SOURCE_DN_OFFSET"] == str(VNIR_MIN)


def test_native_dn_contract_exposes_active_source_gain_and_offset() -> None:
    contract = build_prisma_radiometric_contract(_contract_metadata(), "native_dn")

    assert contract["mode"] == "native-dn"
    assert contract["quantity"] == "native_encoded_dn"
    assert contract["units"] == "DN"
    assert contract["l2_scaling_applied"] is False
    gain, offset = prisma_active_scale_offset(contract, "SWIR")
    assert gain == pytest.approx((SWIR_MAX - SWIR_MIN) / PRISMA_DN_DENOMINATOR)
    assert offset == pytest.approx(SWIR_MIN)


@pytest.mark.parametrize("mode", ["reflectance", "native-dn"])
def test_normalization_contract_is_unitless_with_identity_active_scale(mode: str) -> None:
    contract = build_prisma_radiometric_contract(
        _contract_metadata(),
        mode,
        normalization_mode="percentile",
        normalization_clip=True,
    )

    assert contract["quantity"] == "normalized_unitless"
    assert contract["units"] == "1"
    assert contract["resampling_overshoot_policy"] == (
        "transformed_by_percentile_normalization_with_clipping"
    )
    assert prisma_radiometric_dataset_tags(contract)["NORMALIZATION_MODE"] == "percentile"
    assert prisma_active_scale_offset(contract, "VNIR") == (1.0, 0.0)
    assert "INTERPOLATION_OVERSHOOTS_PRESERVED" not in prisma_radiometric_band_tags(
        contract, "VNIR"
    )


def test_unclipped_normalization_contract_does_not_claim_overshoot_preservation() -> None:
    contract = build_prisma_radiometric_contract(
        _contract_metadata(),
        "reflectance",
        normalization_mode="minmax",
        normalization_clip=False,
    )

    assert contract["resampling_overshoot_policy"] == (
        "transformed_by_minmax_normalization_without_clipping"
    )
    assert "INTERPOLATION_OVERSHOOTS_PRESERVED" not in prisma_radiometric_band_tags(
        contract, "SWIR"
    )


def test_numeric_normalization_replaces_stale_radiometric_range_claims(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "reflectance_source.tif"
    output_path = tmp_path / "normalized_output.tif"
    values = np.arange(256, dtype=np.float32).reshape(1, 16, 16) / 255.0
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype="float32",
        nodata=-9999.0,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as source:
        source.write(values)
        source.update_tags(
            PRISMA_RADIOMETRIC_MODE="reflectance",
            NORMALIZATION_MODE="none",
            NORM_CLIP="false",
            NORM_P_LOW="44.0",
            NORM_P_HIGH="55.0",
            NORM_ESTIMATOR="stale",
            RADIOMETRIC_QUANTITY="surface_reflectance",
            RADIOMETRIC_UNITS="1",
            RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY="preserve_and_flag_not_clip",
            RADIOMETRIC_RANGE_VALIDATION_STATUS="ok",
        )
        source.update_tags(
            1,
            DETECTOR="VNIR",
            SOURCE_DN_GAIN="0.00001",
            SOURCE_DN_OFFSET="0.0",
            SOURCE_VALID_RANGE_MIN="0.0",
            SOURCE_VALID_RANGE_MAX="1.0",
            INTERPOLATION_OVERSHOOTS_PRESERVED="true",
            RADIOMETRIC_QUANTITY="surface_reflectance",
            RADIOMETRIC_UNITS="1",
        )

    result = normalize_raster_to_path(
        str(source_path),
        str(output_path),
        NormalizationParams(mode="minmax", clip=True, min_valid_pixels=1),
        nodata_fallback=-9999.0,
    )

    assert result["ok"] is True, result["errors"]
    with rasterio.open(output_path) as output:
        tags = output.tags()
        assert tags["NORMALIZATION_MODE"] == "minmax"
        assert tags["NORM_CLIP"] == "true"
        assert tags["NORM_P_LOW"] == "2.0"
        assert tags["NORM_P_HIGH"] == "98.0"
        assert tags["NORM_ESTIMATOR"] == "none"
        assert tags["RADIOMETRIC_QUANTITY"] == "normalized_unitless"
        assert tags["RADIOMETRIC_UNITS"] == "1"
        assert tags["RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY"] == (
            "transformed_by_minmax_normalization_with_clipping"
        )
        assert "RADIOMETRIC_RANGE_VALIDATION_STATUS" not in tags

        band_tags = output.tags(1)
        assert band_tags["DETECTOR"] == "VNIR"
        assert band_tags["SOURCE_DN_GAIN"] == "0.00001"
        assert band_tags["SOURCE_DN_OFFSET"] == "0.0"
        assert band_tags["RADIOMETRIC_QUANTITY"] == "normalized_unitless"
        assert band_tags["RADIOMETRIC_UNITS"] == "1"
        assert "SOURCE_VALID_RANGE_MIN" not in band_tags
        assert "SOURCE_VALID_RANGE_MAX" not in band_tags
        assert "INTERPOLATION_OVERSHOOTS_PRESERVED" not in band_tags


def test_noop_normalization_preserves_selected_gdal_scale_offset_metadata(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "native_dn_source.tif"
    output_path = tmp_path / "native_dn_copy.tif"
    values = np.stack(
        (
            np.arange(256, dtype=np.float32).reshape(16, 16),
            np.arange(256, 512, dtype=np.float32).reshape(16, 16),
        )
    )
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=2,
        dtype="float32",
        nodata=-9999.0,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as source:
        source.write(values)
        source.scales = (0.00001, 0.00002)
        source.offsets = (-0.2, 0.1)

    result = normalize_raster_to_path(
        str(source_path),
        str(output_path),
        NormalizationParams(mode="none"),
        nodata_fallback=-9999.0,
        source_bands_1based=(2, 1),
    )

    assert result["ok"] is True, result["errors"]
    with rasterio.open(output_path) as output:
        assert output.scales == pytest.approx((0.00002, 0.00001))
        assert output.offsets == pytest.approx((0.1, -0.2))


def test_numeric_normalization_resets_gdal_scale_offset_metadata(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "scaled_source.tif"
    output_path = tmp_path / "unitless_output.tif"
    values = np.arange(256, dtype=np.float32).reshape(1, 16, 16)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype="float32",
        nodata=-9999.0,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as source:
        source.write(values)
        source.scales = (0.00001,)
        source.offsets = (-0.2,)

    result = normalize_raster_to_path(
        str(source_path),
        str(output_path),
        NormalizationParams(mode="minmax", min_valid_pixels=1),
        nodata_fallback=-9999.0,
    )

    assert result["ok"] is True, result["errors"]
    with rasterio.open(output_path) as output:
        assert output.scales == pytest.approx((1.0,))
        assert output.offsets == pytest.approx((0.0,))


def test_numeric_normalization_does_not_add_prisma_contract_to_other_sensors(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "enmap_like_source.tif"
    output_path = tmp_path / "enmap_like_normalized.tif"
    values = np.arange(256, dtype=np.float32).reshape(1, 16, 16)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype="float32",
        nodata=-9999.0,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as source:
        source.write(values)
        source.update_tags(SENSOR="EnMAP")
        source.update_tags(1, SENSOR_BAND="VNIR")

    result = normalize_raster_to_path(
        str(source_path),
        str(output_path),
        NormalizationParams(mode="minmax", clip=True, min_valid_pixels=1),
        nodata_fallback=-9999.0,
    )

    assert result["ok"] is True, result["errors"]
    with rasterio.open(output_path) as output:
        assert output.tags()["SENSOR"] == "EnMAP"
        assert "PRISMA_RADIOMETRIC_MODE" not in output.tags()
        assert "RADIOMETRIC_QUANTITY" not in output.tags()
        assert output.tags(1)["SENSOR_BAND"] == "VNIR"


def test_contract_requires_complete_detector_scale_metadata() -> None:
    metadata = _contract_metadata()
    del metadata["prisma_l2_scale_swir_max"]

    with pytest.raises(PrismaRadiometricMetadataError, match="SWIR scale maximum is missing"):
        build_prisma_radiometric_contract(metadata, "reflectance")


def test_contract_pan_requirement_is_explicit() -> None:
    metadata = _contract_metadata()
    del metadata["prisma_l2_scale_pan_min"]
    del metadata["prisma_l2_scale_pan_max"]

    no_pan_contract = build_prisma_radiometric_contract(metadata, "reflectance")
    assert "PAN" not in no_pan_contract["scales"]

    with pytest.raises(PrismaRadiometricMetadataError, match="PAN"):
        build_prisma_radiometric_contract(metadata, "reflectance", require_pan=True)


def test_optional_pan_scale_read_error_does_not_block_spectral_contract() -> None:
    metadata = _contract_metadata()
    metadata["prisma_l2_scale_pan_error"] = "live PAN attribute could not be decoded"

    spectral_contract = build_prisma_radiometric_contract(
        metadata,
        "reflectance",
        require_pan=False,
    )

    assert "PAN" not in spectral_contract["scales"]
    assert spectral_contract["optional_scale_errors"] == {
        "PAN": "live PAN attribute could not be decoded"
    }
    with pytest.raises(
        PrismaRadiometricMetadataError,
        match="Invalid required PRISMA PAN scale metadata.*live PAN attribute",
    ):
        build_prisma_radiometric_contract(metadata, "reflectance", require_pan=True)


def test_range_validation_samples_late_blocks_and_qualifies_counts(tmp_path: Path) -> None:
    runtime = pytest.importorskip("hypercoreg.pipeline.runtime")
    assess_range = runtime._assess_prisma_radiometric_range
    path = tmp_path / "pan_cubic_output.tif"
    values = np.full((64, 64), 0.5, dtype=np.float32)
    values[-1, -1] = 1.5
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=1,
        dtype="float32",
        transform=rasterio.transform.from_origin(0.0, 64.0, 1.0, 1.0),
        crs="EPSG:32631",
        nodata=-9999.0,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(values, 1)

    contract = build_prisma_radiometric_contract(
        _contract_metadata(),
        "reflectance",
        spatial_resampling_kernel="cubic",
        require_pan=True,
    )
    sampled = assess_range(str(path), contract, ["PAN"], max_windows=2)

    assert sampled["status"] == "ok"
    assert sampled["sampling_basis"] == "deterministic_spatially_distributed_block_sample"
    assert sampled["selected_block_indices"] == [0, 15]
    assert sampled["blocks_scanned"] == 2
    assert sampled["blocks_total"] == 16
    assert sampled["block_coverage_fraction"] == pytest.approx(2 / 16)
    assert sampled["spatial_coverage_fraction"] == pytest.approx(2 / 16)
    assert sampled["truncated"] is True
    assert sampled["sampled_above_range_band_pixels"] == 1
    assert sampled["sampled_below_range_band_pixels"] == 0
    assert "above_range_pixels" not in sampled

    contract["range_validation"] = sampled
    tags = prisma_radiometric_dataset_tags(contract)
    assert tags["RADIOMETRIC_RANGE_VALIDATION_BASIS"] == sampled["sampling_basis"]
    assert tags["RADIOMETRIC_RANGE_SAMPLED_ABOVE_SOURCE_RANGE_BAND_PIXELS"] == "1"
    assert tags["RADIOMETRIC_RANGE_BLOCKS_SCANNED"] == "2"
    assert "RADIOMETRIC_ABOVE_SOURCE_RANGE_PIXELS" not in tags

    full = assess_range(str(path), contract, ["PAN"], max_windows=0)
    assert full["sampling_basis"] == "full_raster_block_scan"
    assert full["blocks_scanned"] == 16
    assert full["blocks_total"] == 16
    assert full["spatial_coverage_fraction"] == pytest.approx(1.0)
    assert full["truncated"] is False
    assert full["sampled_above_range_band_pixels"] == 1


def test_reader_decodes_each_detector_before_filtering_and_wavelength_sorting(
    prisma_file: Path,
) -> None:
    cube, wavelengths, timestamp, bbox, lat, lon, fwhm, names, detectors = (
        read_prisma_cube_and_meta(str(prisma_file))
    )

    assert cube.shape == (2, 3, 4)
    assert cube.dtype == np.float32
    np.testing.assert_array_equal(wavelengths, [500.0, 700.0, 900.0, 1200.0])
    assert detectors == ["VNIR", "SWIR", "VNIR", "SWIR"]
    assert names == ["PRISMA_001", "PRISMA_002", "PRISMA_003", "PRISMA_004"]
    np.testing.assert_array_equal(fwhm, [5.0, 7.0, 9.0, 12.0])

    expected_band_values = np.array(
        [
            _decode_expected(3000, VNIR_MIN, VNIR_MAX),
            _decode_expected(5000, SWIR_MIN, SWIR_MAX),
            _decode_expected(1000, VNIR_MIN, VNIR_MAX),
            _decode_expected(4000, SWIR_MIN, SWIR_MAX),
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(cube, np.broadcast_to(expected_band_values, cube.shape))

    assert timestamp.isoformat() == "2025-01-02T03:04:05+00:00"
    assert bbox == pytest.approx((7.0, 44.9, 7.2, 45.0), abs=1e-5)
    assert lat.shape == (2, 3)
    assert lon.shape == (2, 3)


def test_native_dn_reader_is_exact_legacy_compatibility_path(prisma_file: Path) -> None:
    cube, wavelengths, *_rest = read_prisma_cube_and_meta(
        str(prisma_file), radiometric_mode="native-dn"
    )

    assert cube.dtype == np.uint16
    np.testing.assert_array_equal(wavelengths, [500.0, 700.0, 900.0, 1200.0])
    expected = np.array([3000, 5000, 1000, 4000], dtype=np.uint16)
    np.testing.assert_array_equal(cube, np.broadcast_to(expected, cube.shape))


def test_native_dn_reader_does_not_require_scale_metadata(tmp_path: Path) -> None:
    path = _write_synthetic_prisma(
        tmp_path / "PRS_L2D_STD_20250102_NO_SCALE.he5", include_scales=False
    )

    cube, *_ = read_prisma_cube_and_meta(str(path), radiometric_mode="native-dn")

    assert cube.dtype == np.uint16
    assert cube[0, 0, :].tolist() == [3000, 5000, 1000, 4000]


def test_overlap_removal_keeps_swir_coefficient_association(prisma_file: Path) -> None:
    with h5py.File(prisma_file, "r+") as product:
        # VNIR index 0 and SWIR index 1 now share the same spectral support.
        # The production policy keeps SWIR and drops the overlapping VNIR band.
        product.attrs["List_Cw_Swir"] = np.array([1200.0, 900.0])
        product.attrs["List_Fwhm_Vnir"] = np.array([10.0, 99.0, 5.0])
        product.attrs["List_Fwhm_Swir"] = np.array([12.0, 10.0])

    cube, wavelengths, *_rest, detectors = read_prisma_cube_and_meta(
        str(prisma_file),
        remove_detector_overlap=True,
        radiometric_mode="reflectance",
    )

    np.testing.assert_array_equal(wavelengths, [500.0, 900.0, 1200.0])
    assert detectors == ["VNIR", "SWIR", "SWIR"]
    expected = np.array(
        [
            _decode_expected(3000, VNIR_MIN, VNIR_MAX),
            _decode_expected(5000, SWIR_MIN, SWIR_MAX),
            _decode_expected(4000, SWIR_MIN, SWIR_MAX),
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(cube, np.broadcast_to(expected, cube.shape))


def test_reflectance_reader_rejects_missing_scale_metadata(tmp_path: Path) -> None:
    path = _write_synthetic_prisma(
        tmp_path / "PRS_L2D_STD_20250102_NO_SCALE.he5", include_scales=False
    )

    with pytest.raises(PrismaRadiometricMetadataError, match="Missing PRISMA VNIR"):
        read_prisma_cube_and_meta(str(path), radiometric_mode="reflectance")


def test_reflectance_reader_rejects_non_scalar_scale_metadata(prisma_file: Path) -> None:
    with h5py.File(prisma_file, "r+") as product:
        del product.attrs["L2ScaleSwirMin"]
        product.attrs["L2ScaleSwirMin"] = np.array([-0.2, -0.1])

    with pytest.raises(PrismaRadiometricMetadataError, match="SWIR.*must be scalar"):
        read_prisma_cube_and_meta(str(prisma_file), radiometric_mode="reflectance")


def test_reader_decodes_byte_swapped_uint16_he5(tmp_path: Path) -> None:
    path = _write_synthetic_prisma(
        tmp_path / "PRS_L2D_STD_20250102_BIG_ENDIAN.he5", sample_dtype=">u2"
    )

    cube, *_ = read_prisma_cube_and_meta(str(path), radiometric_mode="reflectance")

    expected = np.array(
        [
            _decode_expected(3000, VNIR_MIN, VNIR_MAX),
            _decode_expected(5000, SWIR_MIN, SWIR_MAX),
            _decode_expected(1000, VNIR_MIN, VNIR_MAX),
            _decode_expected(4000, SWIR_MIN, SWIR_MAX),
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(cube, np.broadcast_to(expected, cube.shape))


def test_reader_rejects_float_source_to_prevent_double_decode(tmp_path: Path) -> None:
    path = _write_synthetic_prisma(
        tmp_path / "PRS_L2D_STD_20250102_FLOAT.he5", sample_dtype=np.float32
    )

    with pytest.raises(TypeError, match="Refusing possible double decoding"):
        read_prisma_cube_and_meta(str(path), radiometric_mode="reflectance")

    with pytest.raises(TypeError, match=r"PRISMA VNIR native samples.*Refusing possible double decoding"):
        read_prisma_cube_and_meta(str(path), radiometric_mode="native-dn")

    with pytest.raises(TypeError, match=r"PRISMA PAN native samples.*Refusing possible double decoding"):
        read_prisma_pan_and_geo(str(path), radiometric_mode="native-dn")


def test_pan_zero_is_valid_and_decoded_with_pan_scale(prisma_file: Path) -> None:
    pan, geo = read_prisma_pan_and_geo(str(prisma_file), radiometric_mode="reflectance")

    assert pan is not None
    assert geo is not None
    assert pan.dtype == np.float32
    assert np.isfinite(pan[0, 0])
    assert pan[0, 0] == pytest.approx(PAN_MIN)
    assert pan[0, 2] == pytest.approx(PAN_MAX)
    assert pan[0, 1] == pytest.approx(_decode_expected(1, PAN_MIN, PAN_MAX))
    assert geo["rows"] == 2
    assert geo["cols"] == 3
    assert geo["ul_lon"] == pytest.approx(7.0)
    assert geo["lr_lat"] == pytest.approx(44.9)


def test_pan_native_dn_keeps_zero_finite_and_values_exact(prisma_file: Path) -> None:
    pan, _ = read_prisma_pan_and_geo(str(prisma_file), radiometric_mode="native-dn")

    assert pan is not None
    assert pan.dtype == np.float32
    assert pan[0, 0] == 0.0
    assert np.isfinite(pan[0, 0])
    np.testing.assert_array_equal(
        pan,
        np.array([[0, 1, 65535], [10, 100, 1000]], dtype=np.float32),
    )


def test_pan_reflectance_invalid_scale_failure_is_not_swallowed(prisma_file: Path) -> None:
    with h5py.File(prisma_file, "r+") as product:
        del product.attrs["L2ScalePanMax"]

    with pytest.raises(PrismaRadiometricMetadataError, match="Missing PRISMA PAN"):
        read_prisma_pan_and_geo(str(prisma_file), radiometric_mode="reflectance")

    # The compatibility path does not need scale metadata.
    pan, _ = read_prisma_pan_and_geo(str(prisma_file), radiometric_mode="native-dn")
    assert pan is not None
    assert pan[0, 0] == 0.0


def test_extended_metadata_includes_scale_and_processing_provenance(prisma_file: Path) -> None:
    metadata = extract_prisma_extended_metadata(str(prisma_file))

    assert metadata["prisma_id"] == "SYNTHETIC_PRISMA_L2D"
    assert metadata["prisma_date"] == "2025-01-02T00:00:00"
    assert metadata["source_product_name"] == "SYNTHETIC_PRISMA_L2D"
    assert metadata["source_processing_level"] == "L2D"
    assert metadata["source_processor_version"] == "9.8.7"
    assert metadata["source_processing_time"] == "2025-01-02T05:06:07Z"
    assert metadata["prisma_l2_scale_vnir_min"] == pytest.approx(VNIR_MIN)
    assert metadata["prisma_l2_scale_vnir_max"] == pytest.approx(VNIR_MAX)
    assert metadata["prisma_l2_scale_swir_min"] == pytest.approx(SWIR_MIN)
    assert metadata["prisma_l2_scale_swir_max"] == pytest.approx(SWIR_MAX)
    assert metadata["prisma_l2_scale_pan_min"] == pytest.approx(PAN_MIN)
    assert metadata["prisma_l2_scale_pan_max"] == pytest.approx(PAN_MAX)
    assert metadata["prisma_radiometric_metadata_error"] is None


def test_extended_metadata_records_invalid_radiometric_metadata(prisma_file: Path) -> None:
    with h5py.File(prisma_file, "r+") as product:
        del product.attrs["L2ScaleVnirMax"]

    metadata = extract_prisma_extended_metadata(str(prisma_file))

    assert metadata["prisma_l2_scale_vnir_min"] is None
    assert metadata["prisma_l2_scale_vnir_max"] is None
    assert "Missing PRISMA VNIR" in metadata["prisma_radiometric_metadata_error"]
