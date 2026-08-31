# PRISMA L2D Radiometry

## Release note

> HyperCoreg now decodes PRISMA Level-2D VNIR, SWIR, and PAN samples using the
> scale minimum and maximum stored in each source HE5 product. New PRISMA
> outputs default to `float32`, unitless surface reflectance and include
> radiometric provenance. Use `--prisma-radiometric-mode native-dn` to retain
> the previous encoded-value representation. Product-defined L2 decoding is
> independent of min-max or percentile image normalization.

This is a numerical default change for PRISMA processing. EnMAP radiometric
decoding and numerical defaults are unchanged.

## Processing contract

PRISMA L2D detector arrays store encoded `uint16` samples. For each source
product and detector, HyperCoreg decodes them with:

```text
reflectance = scale_min + DN * (scale_max - scale_min) / 65535
```

The coefficients come from the same HE5 file as the samples:

| Detector | Source attributes |
| --- | --- |
| VNIR | `L2ScaleVnirMin`, `L2ScaleVnirMax` |
| SWIR | `L2ScaleSwirMin`, `L2ScaleSwirMax` |
| PAN | `L2ScalePanMin`, `L2ScalePanMax` |

In `reflectance` mode, every required coefficient must exist, be a finite
scalar, and have `maximum > minimum`. Invalid metadata stops processing rather
than falling back to `/10000`, a presumed `0-1` range, or coefficients from a
different acquisition. VNIR and SWIR are decoded separately before their bands
are combined and sorted, and PAN is decoded independently when requested. PAN
coefficients are required only when PAN output is requested; an unavailable
optional PAN scale does not block spectral-only processing.

Radiometric decoding does not redefine pixel validity. NoData values and
quality masks remain separate, and native zero is not made invalid merely by
the affine conversion. Cubic interpolation can create values corresponding to
DN below `0` or above `65535`; outputs record a preserve-and-flag policy and do
not silently clamp those overshoots when normalization is `none`. Bounded range
checks sample blocks deterministically across the complete image rather than
only its first blocks. Their tags explicitly say `SAMPLED`; a full block scan
is used when the validation window limit is set to `0`.

PAN validity is read from the source
`/HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/PIXEL_L2_ERR_MATRIX` dataset.
Values `0` through `3` remain numeric samples, while flag `4` is invalid.
Source flag-4 holes are filled only in the temporary interpolation surface so
that a NoData sentinel cannot enter the cubic resampling kernel. The quality
mask itself is warped with nearest-neighbour resampling; aligned flag-4 pixels
and pixels outside the warped source footprint (`255`) are then restored as
PAN NoData. If a hole has no valid neighbour within the configured fill radius,
PAN export degrades explicitly rather than warping a source-invalid numeric
value. The exported `quality_pan` raster is `uint8`, categorical, and on the
exact output PAN grid.

## Modes and normalization

`prisma_radiometric_mode` controls the product-defined L2 conversion.
`normalization_mode` controls an optional, scene-dependent transformation after
that conversion. They answer different questions and are configured
independently.

| PRISMA mode | Normalization | Stored quantity | Quantitative interpretation |
| --- | --- | --- | --- |
| `reflectance` | `none` | `surface_reflectance`, units `1` | Recommended scientific output |
| `native-dn` | `none` | `native_encoded_dn`, units `DN` | Legacy-compatible encoded values |
| Either mode | `minmax` or `percentile` | `normalized_unitless`, units `1` | Scene-normalized values, not calibrated reflectance |

The default is `reflectance` with normalization `none`. To reproduce the
previous numerical representation explicitly:

```bash
hypercoreg single \
  -i /path/to/input.he5 \
  -o /path/to/output \
  --prisma-radiometric-mode native-dn \
  --normalization-mode none
```

For quantitative reflectance:

```bash
hypercoreg single \
  -i /path/to/input.he5 \
  -o /path/to/output \
  --prisma-radiometric-mode reflectance \
  --normalization-mode none
```

The GUI exposes the same choice as **PRISMA radiometry**. Its **Normalization
mode** control remains separate. Python callers can set:

```python
config = {
    "prisma_radiometric_mode": "reflectance",
    "normalization_mode": "none",
}
```

Pipelines that require stable behavior across HyperCoreg versions should set
both values explicitly.

## Output metadata contract

PRISMA GeoTIFFs carry a human-readable radiometric contract. PAM metadata is
given the same tags when an `.aux.xml` sidecar is emitted. The per-scene metrics
JSON and run manifest also store the resolved mode and structured radiometric
contract.

Dataset-level tags include:

| Tag | Meaning |
| --- | --- |
| `RADIOMETRIC_QUANTITY` | `surface_reflectance`, `native_encoded_dn`, or `normalized_unitless` |
| `RADIOMETRIC_UNITS` | `1` for unitless values or `DN` for encoded samples |
| `PRISMA_RADIOMETRIC_MODE` | Requested `reflectance` or `native-dn` processing mode |
| `PRISMA_L2_SCALING_APPLIED` | Whether product-defined L2 decoding was applied |
| `PRISMA_DN_DENOMINATOR` | `65535` |
| `PRISMA_L2_SCALE_<DETECTOR>_MIN/MAX` | Original per-product detector coefficients |
| `SOURCE_PRODUCT_NAME` | Source product identifier, when present |
| `SOURCE_PROCESSING_LEVEL` | Source processing level, when present |
| `SOURCE_PROCESSOR_VERSION` | Source processor version, when present |
| `SOURCE_PROCESSING_TIME` | Source processing time, when present |
| `SPATIAL_RESAMPLING_KERNEL` | Kernel used by the relevant processing stage |
| `RADIOMETRIC_RESAMPLING_OVERSHOOT_POLICY` | Recorded overshoot policy |
| `RADIOMETRIC_RANGE_VALIDATION_BASIS` | Full scan or deterministic spatial block sample |
| `RADIOMETRIC_RANGE_SAMPLED_BELOW_SOURCE_RANGE_BAND_PIXELS` | Below-range band-pixels observed in scanned blocks |
| `RADIOMETRIC_RANGE_SAMPLED_ABOVE_SOURCE_RANGE_BAND_PIXELS` | Above-range band-pixels observed in scanned blocks |
| `RADIOMETRIC_RANGE_BLOCKS_SCANNED/TOTAL` | Explicit block coverage for the range check |
| `RADIOMETRIC_RANGE_SPATIAL_COVERAGE_FRACTION` | Fraction of spatial pixels represented by scanned blocks |
| `RADIOMETRIC_RANGE_SCAN_TRUNCATED` | Whether the range check was sampled rather than complete |
| `NORMALIZATION_MODE` | Independent output normalization state |

Each spectral band also identifies its `DETECTOR`, radiometric quantity and
units, plus `SOURCE_DN_GAIN` and `SOURCE_DN_OFFSET`. These provenance values
describe how the original encoded samples map to reflectance even when output
pixels have already been decoded.

Active GDAL scale and offset are deliberately state-dependent:

| Stored quantity | Active scale/offset |
| --- | --- |
| Decoded reflectance | `1` / `0` (prevents double decoding) |
| Normalized unitless values | `1` / `0` |
| Native encoded DN | Detector-specific source gain / offset |

Some raster consumers automatically apply active scale/offset and others ignore
it. In `native-dn` mode, distinguish raw stored samples from scale-aware display
or analysis values. The explicit tags are authoritative provenance for both
types of consumer.

When min-max or percentile normalization is enabled, the output contract is
replaced with the normalized-unitless contract. It does not inherit a claim
that source-range overshoots were preserved, because normalization has changed
the numeric range and interpretation.

## Real-scene verification record

The implementation was exercised end to end on
`PRS_L2D_STD_20251004101501_20251004101506_0001.he5` with its local
Sentinel-2 stack. Registration accepted 24 VNIR and 21 SWIR GCPs. The following
checks used complete pixel scans rather than spot samples:

- All 378,167,852 valid pre-registration band-pixels matched independent HE5
  decoding exactly in both reflectance and native-DN modes.
- All 404,134,620 valid final band-pixels satisfied the detector-specific
  affine relationship between the native-DN and reflectance runs. There were
  no tolerance violations; the maximum absolute float32 difference was
  `1.1920928955078125e-7` reflectance units.
- The previous HyperCoreg output matched the new native-DN pre-registration
  raster exactly. After registration, its maximum difference from the new
  native-DN raster was `0.00390625` DN, within the documented `0.004` DN
  compatibility bound; grids and validity masks matched exactly. At the
  validator's stricter `0.001` DN setting, 1,749 of 404,134,620 compared
  values exceeded tolerance (fraction `4.327765832e-6`, RMSE
  `0.000139600744654` DN), so final compatibility is bounded rather than
  bit-identical.
- All 59,449,968 source PAN pixels satisfied `reflectance = DN / 65535`
  exactly for this product. Its 27,385 source flag-4 pixels were kept invalid,
  while 21,514,987 valid zero-DN pixels remained valid.
- A full aligned PAN/quality scan found no numeric PAN value at a flag-4 or
  outside-footprint pixel, no NoData at a quality-0 pixel, and no NoData
  sentinel contamination in valid cubic-interpolated values.
- Full categorical scans of the VNIR, SWIR, and PAN quality outputs found no
  unexpected category values. The automated suite completed with 82 passing
  and 4 skipped tests.

The reusable comparison command is
`scripts/validate_prisma_radiometry.py`; it reports grid, mask, metadata,
formula, and legacy-compatibility results as machine-readable JSON.

## Migrating legacy outputs

The safest migration is to rerun each exact source HE5 product in
`reflectance` mode with normalization `none`. A post-registration affine repair
can avoid repeating geometric processing, but only when all of these conditions
are demonstrated:

- The legacy raster is matched unambiguously to its exact source HE5 product.
- Its pixels retain the encoded values, even if stored as `float32`.
- `NORMALIZATION_MODE=none` was used.
- Values were not clipped, rounded destructively, empirically rescaled, or
  transformed nonlinearly.
- NoData and invalid pixels can be identified without treating their sentinels
  as samples.
- Every output band can be assigned reliably to VNIR, SWIR, or PAN.

Use the following decision table:

| Legacy condition | Action |
| --- | --- |
| Exact source match, encoded values preserved, normalization `none` | Product-specific post-registration decoding is possible |
| Same, with interpolation overshoots | Decode, preserve the overshoots, and record or supply a validity flag |
| Min-max or percentile normalization was applied | Rerun from the source HE5 for quantitative use |
| Source product or detector mapping is missing or ambiguous | Exclude until provenance is resolved, or rerun from a verified source |
| Values were clipped or destructively cast | Rerun from the source HE5 |

For a recoverable raster:

1. Keep the legacy product immutable and record its checksum.
2. Match it to one source HE5 by product identifier and acquisition time, then
   record the source checksum.
3. Verify its normalization metadata, numeric range, NoData convention, band
   order, and absence of destructive transformations.
4. Read the VNIR, SWIR, and PAN coefficients from that exact HE5 file.
5. Apply the detector-specific equation only to valid pixels. Do not pass a
   NoData sentinel such as `-9999` through the equation.
6. Write a new, versioned `float32` product with active scale `1`, offset `0`,
   and the radiometric metadata contract above. Do not overwrite the legacy
   raster.
7. Record coefficients, software version, source/output checksums, pixel counts,
   value ranges, and validation samples in a migration manifest.
8. Compare independently decoded HE5 samples with the migrated raster at stable
   interior pixels, allowing only the documented spatial-resampling tolerance.

An interpolated legacy value outside the native `0..65535` range can decode
outside the source scale minimum/maximum. That is evidence of geometric
resampling, not a reason to clamp the value silently.

## Downstream compatibility checklist

- Update thresholds or models that expected values in the thousands; decoded
  reflectance is normally on a unitless physical scale.
- Do not divide decoded output by `10000` or apply the source gain a second time.
- Use `RADIOMETRIC_QUANTITY`, not only the `float32` data type, to decide what
  pixels represent.
- Use `reflectance` plus normalization `none` for cross-date quantitative work.
- Use `native-dn` plus normalization `none` only for explicit legacy
  compatibility, and account for scale-aware raster readers.
