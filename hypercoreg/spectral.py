"""
Spectral band management for HyperCoreg.

This module provides data structures and utilities for handling spectral
band information from PRISMA and EnMAP hyperspectral sensors.
"""

import logging
import re
import csv
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("COREG_PROCESSING")

ENMAP_DETECTOR_MATCH_TOLERANCE_NM = 10.0
_ENMAP_REFERENCE_EXPECTED_VNIR_BANDS = 91
_ENMAP_REFERENCE_EXPECTED_SWIR_BANDS = 133
_ENMAP_REFERENCE_CSV = Path(__file__).resolve().parent / "data" / "enmap_spectral_bands.csv"


@dataclass
class SpectralBandInfo:
    """
    Information for a single spectral band.

    Attributes:
        index: 1-based band index in the data cube
        wavelength: Center wavelength in nm
        fwhm: Full width at half maximum in nm (NaN if unknown)
        name: Band name (e.g., "PRISMA_001", "EnMAP_B001")
        detector: Detector name ("VNIR", "SWIR", or "unknown")
    """
    index: int
    wavelength: float
    fwhm: float
    name: str
    detector: str


class SpectralBandTable:
    """
    Unified mapping structure holding per-band spectral information.

    Ensures band indices, wavelengths, and FWHM are properly aligned.
    Provides methods for wavelength lookup and validation.
    """

    def __init__(self, bands: Optional[List[SpectralBandInfo]] = None):
        """
        Initialize with optional list of SpectralBandInfo objects.

        Args:
            bands: List of SpectralBandInfo objects
        """
        self.bands = bands if bands is not None else []

    def __len__(self) -> int:
        return len(self.bands)

    def __getitem__(self, idx: int) -> SpectralBandInfo:
        return self.bands[idx]

    def __iter__(self):
        return iter(self.bands)

    @property
    def wavelengths(self) -> np.ndarray:
        """Return wavelength array (for backwards compatibility)."""
        return np.array([b.wavelength for b in self.bands])

    @property
    def fwhm(self) -> Optional[np.ndarray]:
        """
        Return FWHM array (for backwards compatibility).

        Returns None if all values are NaN.
        """
        arr = np.array([b.fwhm for b in self.bands])
        return arr if not np.all(np.isnan(arr)) else None

    @property
    def band_names(self) -> List[str]:
        """Return band names list (for backwards compatibility)."""
        return [b.name for b in self.bands]

    @property
    def detectors(self) -> List[str]:
        """Return detector labels list aligned with band order."""
        return [b.detector for b in self.bands]

    @property
    def n_vnir(self) -> int:
        """Return count of VNIR bands."""
        return sum(1 for b in self.bands if b.detector == "VNIR")

    @property
    def n_swir(self) -> int:
        """Return count of SWIR bands."""
        return sum(1 for b in self.bands if b.detector == "SWIR")

    @property
    def is_ascending(self) -> bool:
        """Check if wavelengths are in strictly ascending order."""
        wl = self.wavelengths
        return np.all(np.diff(wl) > 0)

    def find_closest_band(self, target_wl: float) -> Tuple[int, float, float]:
        """
        Find band closest to target wavelength.

        Args:
            target_wl: Target wavelength in nm

        Returns:
            tuple: (band_index_0based, wavelength, difference_nm)
        """
        wl = self.wavelengths
        idx = int(np.argmin(np.abs(wl - target_wl)))
        return idx, wl[idx], abs(wl[idx] - target_wl)

    def get_vnir_swir_transition(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get the wavelength where VNIR ends and SWIR begins.

        Returns:
            tuple: (last_vnir_wl, first_swir_wl, gap_nm) or (None, None, None)
        """
        vnir_bands = [b for b in self.bands if b.detector == "VNIR"]
        swir_bands = [b for b in self.bands if b.detector == "SWIR"]

        if not vnir_bands or not swir_bands:
            return None, None, None

        last_vnir = max(b.wavelength for b in vnir_bands)
        first_swir = min(b.wavelength for b in swir_bands)

        return last_vnir, first_swir, first_swir - last_vnir

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate spectral consistency.

        Returns:
            tuple: (is_valid, list_of_issues)
        """
        issues = []

        if len(self.bands) == 0:
            issues.append("No bands in table")
            return False, issues

        # Check wavelength order
        if not self.is_ascending:
            issues.append("Wavelengths not in ascending order")

        # Check for duplicate wavelengths
        wl = self.wavelengths
        if len(wl) != len(np.unique(np.round(wl, 2))):
            issues.append("Duplicate wavelengths detected")

        # Check VNIR-SWIR transition
        vnir_end, swir_start, gap = self.get_vnir_swir_transition()
        if vnir_end is not None and swir_start is not None:
            if gap < 0:
                issues.append(
                    f"VNIR-SWIR overlap: VNIR ends at {vnir_end:.1f}nm, "
                    f"SWIR starts at {swir_start:.1f}nm"
                )

        # Check FWHM validity
        fwhm_arr = np.array([b.fwhm for b in self.bands])
        valid_fwhm = fwhm_arr[~np.isnan(fwhm_arr)]
        if len(valid_fwhm) > 0:
            if np.any(valid_fwhm <= 0):
                issues.append("Invalid FWHM values (<=0) detected")
            if np.any(valid_fwhm > 100):
                issues.append("Suspiciously large FWHM values (>100nm) detected")

        return len(issues) == 0, issues

    def summary(self) -> str:
        """Return a summary string of the band table."""
        if len(self.bands) == 0:
            return "SpectralBandTable: Empty"

        wl = self.wavelengths

        return (
            f"SpectralBandTable: {len(self.bands)} bands, "
            f"{wl.min():.1f}-{wl.max():.1f}nm, "
            f"VNIR={self.n_vnir}, SWIR={self.n_swir}, "
            f"ascending={self.is_ascending}"
        )


def _band_extent(band: SpectralBandInfo) -> Tuple[float, float]:
    """Compute the spectral support interval [wl - fwhm/2, wl + fwhm/2]."""
    if np.isfinite(band.fwhm) and band.fwhm > 0:
        half = 0.5 * float(band.fwhm)
    else:
        half = 0.0
    return band.wavelength - half, band.wavelength + half


def _bands_overlap(b1: SpectralBandInfo, b2: SpectralBandInfo) -> bool:
    """Return True if two spectral support intervals overlap."""
    b1_min, b1_max = _band_extent(b1)
    b2_min, b2_max = _band_extent(b2)
    return b1_min <= b2_max and b2_min <= b1_max


def _pick_band_to_drop(vnir_band: SpectralBandInfo, swir_band: SpectralBandInfo) -> str:
    """
    Decide which detector band to drop in an overlap pair.

    Policy: always keep SWIR and drop VNIR in VNIR/SWIR overlap regions.
    """
    _ = vnir_band
    _ = swir_band
    return "VNIR"


def _remove_detector_overlap(
    bands: List[SpectralBandInfo],
    original_indices: np.ndarray
) -> Tuple[List[SpectralBandInfo], np.ndarray, int]:
    """Drop overlapping VNIR/SWIR bands with SWIR-priority, preserving index mapping."""
    keep_mask = np.ones(len(bands), dtype=bool)
    vnir_indices = [i for i, b in enumerate(bands) if b.detector == "VNIR"]
    swir_indices = [i for i, b in enumerate(bands) if b.detector == "SWIR"]

    for i in vnir_indices:
        if not keep_mask[i]:
            continue
        for j in swir_indices:
            if not keep_mask[j]:
                continue
            if not _bands_overlap(bands[i], bands[j]):
                continue

            detector_to_drop = _pick_band_to_drop(bands[i], bands[j])
            if detector_to_drop == "VNIR":
                keep_mask[i] = False
                break
            keep_mask[j] = False

    filtered_bands = [b for idx, b in enumerate(bands) if keep_mask[idx]]
    filtered_indices = original_indices[keep_mask]
    removed_count = int((~keep_mask).sum())
    return filtered_bands, filtered_indices, removed_count


@lru_cache(maxsize=1)
def _load_enmap_detector_reference() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load bundled EnMAP VNIR/SWIR wavelength references.

    Returns:
        tuple: (vnir_wavelengths_nm, swir_wavelengths_nm)
    """
    vnir_wl: List[float] = []
    swir_wl: List[float] = []

    try:
        with _ENMAP_REFERENCE_CSV.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                detector = str(row.get("detector", "")).strip().upper()
                wl_txt = row.get("wavelength_nm")
                if detector not in {"VNIR", "SWIR"} or wl_txt is None:
                    continue
                try:
                    wl_val = float(str(wl_txt).strip())
                except Exception:
                    continue
                if detector == "VNIR":
                    vnir_wl.append(wl_val)
                else:
                    swir_wl.append(wl_val)
    except FileNotFoundError:
        logger.warning(
            "Bundled EnMAP detector reference CSV missing: %s",
            _ENMAP_REFERENCE_CSV,
        )
        return np.array([], dtype=float), np.array([], dtype=float)
    except Exception as exc:
        logger.warning("Failed to load bundled EnMAP detector reference CSV: %s", exc)
        return np.array([], dtype=float), np.array([], dtype=float)

    if (
        len(vnir_wl) != _ENMAP_REFERENCE_EXPECTED_VNIR_BANDS
        or len(swir_wl) != _ENMAP_REFERENCE_EXPECTED_SWIR_BANDS
    ):
        logger.warning(
            "Bundled EnMAP detector reference has unexpected band counts "
            "(VNIR=%d, SWIR=%d; expected VNIR=%d, SWIR=%d).",
            len(vnir_wl),
            len(swir_wl),
            _ENMAP_REFERENCE_EXPECTED_VNIR_BANDS,
            _ENMAP_REFERENCE_EXPECTED_SWIR_BANDS,
        )

    return np.array(vnir_wl, dtype=float), np.array(swir_wl, dtype=float)


def _assign_enmap_detector_from_reference(
    wavelength: float,
    vnir_ref_wl: np.ndarray,
    swir_ref_wl: np.ndarray,
    tolerance_nm: float = ENMAP_DETECTOR_MATCH_TOLERANCE_NM,
) -> Tuple[str, float]:
    """
    Assign EnMAP detector by nearest bundled reference wavelength.

    Returns:
        tuple: (detector_label, nearest_delta_nm)
    """
    if vnir_ref_wl.size == 0 or swir_ref_wl.size == 0:
        return "UNKNOWN", float("inf")

    delta_vnir = float(np.min(np.abs(vnir_ref_wl - wavelength)))
    delta_swir = float(np.min(np.abs(swir_ref_wl - wavelength)))
    nearest_delta = min(delta_vnir, delta_swir)
    if nearest_delta > float(tolerance_nm):
        return "UNKNOWN", nearest_delta
    if delta_vnir <= delta_swir:
        return "VNIR", nearest_delta
    return "SWIR", nearest_delta


def _map_enmap_detectors_from_reference(
    wavelengths_nm: np.ndarray,
    tolerance_nm: float = ENMAP_DETECTOR_MATCH_TOLERANCE_NM,
) -> Tuple[List[str], Dict[str, float]]:
    """Map EnMAP detectors for all wavelengths against bundled VNIR/SWIR reference bands."""
    vnir_ref, swir_ref = _load_enmap_detector_reference()
    detectors: List[str] = []
    deltas: List[float] = []

    for wl in wavelengths_nm:
        detector, delta = _assign_enmap_detector_from_reference(
            float(wl),
            vnir_ref,
            swir_ref,
            tolerance_nm=tolerance_nm,
        )
        detectors.append(detector)
        deltas.append(float(delta))

    delta_arr = np.array(deltas, dtype=float) if deltas else np.array([], dtype=float)
    finite_delta = delta_arr[np.isfinite(delta_arr)] if delta_arr.size else np.array([], dtype=float)
    stats: Dict[str, float] = {
        "vnir_count": float(sum(1 for d in detectors if d == "VNIR")),
        "swir_count": float(sum(1 for d in detectors if d == "SWIR")),
        "unknown_count": float(sum(1 for d in detectors if d == "UNKNOWN")),
        "max_delta_nm": float(np.max(finite_delta)) if finite_delta.size else float("nan"),
    }
    return detectors, stats


def _infer_enmap_detector(wavelength: float, band_name: Optional[str]) -> str:
    """
    Infer EnMAP detector from band identifier, falling back to wavelength.

    Typical EnMAP slot IDs:
    - VNIR: low slot numbers (e.g. 1..100)
    - SWIR: high slot numbers (e.g. >=100)
    """
    if band_name:
        name_lower = str(band_name).lower()
        if "swir" in name_lower:
            return "SWIR"
        if "vnir" in name_lower:
            return "VNIR"

        match = re.search(r"\d+", str(band_name))
        if match:
            slot_id = int(match.group(0))
            if slot_id >= 100:
                return "SWIR"
            if slot_id > 0:
                return "VNIR"

    return "VNIR" if wavelength < 1000 else "SWIR"


def build_prisma_band_table(
    vnir_wl: np.ndarray,
    swir_wl: np.ndarray,
    vnir_fwhm: Optional[np.ndarray] = None,
    swir_fwhm: Optional[np.ndarray] = None,
    remove_detector_overlap: bool = False
) -> Tuple[SpectralBandTable, Optional[np.ndarray]]:
    """
    Build SpectralBandTable from PRISMA VNIR/SWIR arrays.

    Ensures wavelengths are in ascending order.
    Filters out invalid bands (wavelength <= 0).

    Args:
        vnir_wl: VNIR wavelength array
        swir_wl: SWIR wavelength array
        vnir_fwhm: VNIR FWHM array (optional)
        swir_fwhm: SWIR FWHM array (optional)
        remove_detector_overlap: If True, remove VNIR/SWIR overlap keeping SWIR bands.

    Returns:
        tuple: (SpectralBandTable, sort_indices for cube reordering)
            - sort_indices: Array to reorder valid bands to ascending wavelength
            - If both filtering and sorting needed, returns combined index array
    """
    bands = []
    original_indices = []  # Track original band indices in merged cube

    # Build VNIR bands (filter invalid wavelengths)
    vnir_valid_count = 0
    for i, wl in enumerate(vnir_wl):
        if wl <= 0:  # Skip invalid wavelengths
            continue
        fwhm_val = float(vnir_fwhm[i]) if vnir_fwhm is not None else np.nan
        bands.append(SpectralBandInfo(
            index=len(bands) + 1,
            wavelength=float(wl),
            fwhm=fwhm_val,
            name=f"PRISMA_{len(bands) + 1:03d}",
            detector="VNIR"
        ))
        original_indices.append(i)  # VNIR index in merged cube
        vnir_valid_count += 1

    # Build SWIR bands (filter invalid wavelengths)
    vnir_total = len(vnir_wl)  # Total VNIR bands in original cube
    swir_valid_count = 0
    for i, wl in enumerate(swir_wl):
        if wl <= 0:  # Skip invalid wavelengths
            continue
        fwhm_val = float(swir_fwhm[i]) if swir_fwhm is not None else np.nan
        bands.append(SpectralBandInfo(
            index=len(bands) + 1,
            wavelength=float(wl),
            fwhm=fwhm_val,
            name=f"PRISMA_{len(bands) + 1:03d}",
            detector="SWIR"
        ))
        original_indices.append(vnir_total + i)  # SWIR index in merged cube
        swir_valid_count += 1

    # Log filtering stats
    vnir_filtered = len(vnir_wl) - vnir_valid_count
    swir_filtered = len(swir_wl) - swir_valid_count
    if vnir_filtered > 0 or swir_filtered > 0:
        logger.info(
            f"Filtered invalid bands: VNIR={vnir_filtered}, "
            f"SWIR={swir_filtered} (wavelength <= 0)"
        )

    original_indices = np.array(original_indices, dtype=int)

    if remove_detector_overlap and len(bands) > 0:
        bands, original_indices, removed_overlap = _remove_detector_overlap(bands, original_indices)
        if removed_overlap > 0:
            logger.info(f"Removed {removed_overlap} overlapping VNIR/SWIR bands")
        for idx, band in enumerate(bands):
            band.index = idx + 1
            band.name = f"PRISMA_{idx + 1:03d}"

    # Check if reordering is needed
    wavelengths = np.array([b.wavelength for b in bands])
    if not np.all(np.diff(wavelengths) > 0):
        # Sort by wavelength (ascending)
        sort_order = np.argsort(wavelengths)
        bands_sorted = [bands[i] for i in sort_order]

        # Update indices and names to reflect new order
        for new_idx, band in enumerate(bands_sorted):
            band.index = new_idx + 1
            band.name = f"PRISMA_{new_idx + 1:03d}"

        # Combine filtering + sorting: original_indices[sort_order] gives final cube indices
        final_indices = original_indices[sort_order]
        return SpectralBandTable(bands_sorted), final_indices

    # Only filtering needed (already ascending)
    if len(original_indices) < len(vnir_wl) + len(swir_wl):
        return SpectralBandTable(bands), original_indices

    return SpectralBandTable(bands), None


def build_enmap_band_table(
    wl_list: np.ndarray,
    fwhm_list: Optional[np.ndarray] = None,
    band_names_list: Optional[List[str]] = None,
    remove_detector_overlap: bool = False,
    enmap_processing_version: Optional[str] = None,
) -> Tuple[SpectralBandTable, Optional[np.ndarray]]:
    """
    Build SpectralBandTable from EnMAP arrays.

    Ensures wavelengths are in ascending order.

    Args:
        wl_list: Wavelength array/list
        fwhm_list: FWHM array/list (optional)
        band_names_list: Band name list (optional)
        remove_detector_overlap: If True, remove VNIR/SWIR overlap keeping SWIR bands.
        enmap_processing_version: Optional EnMAP processing version from metadata.

    Returns:
        tuple: (SpectralBandTable, sort_indices or None if no reordering needed)
    """
    bands = []
    wl_array = np.asarray(wl_list, dtype=float)
    original_indices = np.arange(len(wl_array), dtype=int)
    detectors, detector_stats = _map_enmap_detectors_from_reference(
        wl_array,
        tolerance_nm=ENMAP_DETECTOR_MATCH_TOLERANCE_NM,
    )

    # Determine detector based on bundled EnMAP reference table.
    for i, wl in enumerate(wl_array):
        fwhm_val = float(fwhm_list[i]) if fwhm_list is not None and i < len(fwhm_list) else np.nan
        name = band_names_list[i] if band_names_list is not None and i < len(band_names_list) else f"EnMAP_{i + 1:03d}"
        detector = detectors[i] if i < len(detectors) else "UNKNOWN"

        bands.append(SpectralBandInfo(
            index=i + 1,
            wavelength=float(wl),
            fwhm=fwhm_val,
            name=name,
            detector=detector
        ))

    unknown_count = int(detector_stats.get("unknown_count", 0.0))
    max_delta_nm = float(detector_stats.get("max_delta_nm", float("nan")))
    if np.isfinite(max_delta_nm):
        logger.info(
            "EnMAP detector mapping (bundled table): VNIR=%d, SWIR=%d, UNKNOWN=%d "
            "(tol=%.1f nm, max_delta=%.3f nm, version=%s)",
            int(detector_stats.get("vnir_count", 0.0)),
            int(detector_stats.get("swir_count", 0.0)),
            unknown_count,
            ENMAP_DETECTOR_MATCH_TOLERANCE_NM,
            max_delta_nm,
            str(enmap_processing_version) if enmap_processing_version else "unknown",
        )
    else:
        logger.warning(
            "EnMAP detector mapping (bundled table) unavailable; assigning UNKNOWN detector labels."
        )
    if unknown_count > 0:
        logger.warning(
            "EnMAP detector mapping left %d bands as UNKNOWN "
            "(nearest reference delta > %.1f nm, version=%s).",
            unknown_count,
            ENMAP_DETECTOR_MATCH_TOLERANCE_NM,
            str(enmap_processing_version) if enmap_processing_version else "unknown",
        )

    if remove_detector_overlap and len(bands) > 0:
        bands, original_indices, removed_overlap = _remove_detector_overlap(bands, original_indices)
        if removed_overlap > 0:
            logger.info(f"Removed {removed_overlap} overlapping VNIR/SWIR bands")
        for new_idx, band in enumerate(bands):
            band.index = new_idx + 1

    # Check if reordering is needed
    wavelengths = np.array([b.wavelength for b in bands])
    if not np.all(np.diff(wavelengths) > 0):
        # Sort by wavelength (ascending)
        sort_idx = np.argsort(wavelengths)
        bands_sorted = [bands[i] for i in sort_idx]

        # Update indices to reflect new order
        for new_idx, band in enumerate(bands_sorted):
            band.index = new_idx + 1

        return SpectralBandTable(bands_sorted), original_indices[sort_idx]

    if len(original_indices) < len(wl_list):
        return SpectralBandTable(bands), original_indices

    return SpectralBandTable(bands), None
