"""
EnMAP hyperspectral data reader.

This module provides functions for reading EnMAP L2A data products,
including spectral images, metadata, and auxiliary bands.
"""

import os
import shutil
import logging
import subprocess
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from pyproj import CRS

from hypercoreg.spectral import build_enmap_band_table, SpectralBandTable
from hypercoreg.logging_config import log_section_header
from hypercoreg.utils import resolve_gdalwarp_exe

logger = logging.getLogger("COREG_PROCESSING")
_ENMAP_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}


def _local_tag_name(tag: str) -> str:
    """Return XML local tag name (namespace-agnostic)."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _iter_local(root: ET.Element, name: str):
    """Iterate elements matching local-name, regardless of XML namespace."""
    target = str(name)
    for elem in root.iter():
        if _local_tag_name(elem.tag) == target:
            yield elem


def _find_first_text(node: ET.Element, local_name: str) -> Optional[str]:
    """Find first text value for a local-name under node."""
    for elem in _iter_local(node, local_name):
        if elem.text:
            text = elem.text.strip()
            if text:
                return text
    return None


def _find_first_float(node: ET.Element, local_name: str) -> Optional[float]:
    """Find first float value for a local-name under node."""
    txt = _find_first_text(node, local_name)
    if txt is None:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def _collect_text_values(node: ET.Element, local_name: str) -> List[str]:
    """Collect all non-empty text values for a local-name under node."""
    values: List[str] = []
    for elem in _iter_local(node, local_name):
        if not elem.text:
            continue
        text = elem.text.strip()
        if text:
            values.append(text)
    return values


def _collect_float_values(node: ET.Element, local_name: str) -> List[float]:
    """Collect all parseable float values for a local-name under node."""
    values: List[float] = []
    for txt in _collect_text_values(node, local_name):
        try:
            values.append(float(txt))
        except Exception:
            continue
    return values


def _parse_float_list_text(text: Optional[str]) -> List[float]:
    """Parse a scalar or ENVI-style numeric list into finite floats."""
    if text is None:
        return []
    cleaned = str(text).replace("{", " ").replace("}", " ").replace(";", ",")
    values: List[float] = []
    for part in re.split(r"[,\s]+", cleaned):
        if not part:
            continue
        try:
            val = float(part)
        except Exception:
            continue
        if np.isfinite(val):
            values.append(val)
    return values


def _align_band_float_values(
    values: List[float],
    band_count: int,
    label: str,
) -> Optional[np.ndarray]:
    """Return per-band values, accepting exact arrays or one scalar repeated."""
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return None
    if band_count > 0:
        if len(vals) == band_count:
            return np.asarray(vals, dtype=float)
        if len(vals) == 1:
            return np.full(int(band_count), float(vals[0]), dtype=float)
        logger.warning(
            "EnMAP metadata %s length mismatch (%d vs bands %d); ignoring %s.",
            label,
            len(vals),
            int(band_count),
            label,
        )
        return None
    return np.asarray(vals, dtype=float)


def _iter_children_local(node: ET.Element, name: str):
    """Iterate direct children with matching local-name."""
    target = str(name)
    for child in list(node):
        if _local_tag_name(child.tag) == target:
            yield child


def _find_first_child_local(node: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    """Return first direct child matching local-name."""
    if node is None:
        return None
    for child in _iter_children_local(node, name):
        return child
    return None


def _parse_float_text(text: Optional[str]) -> Optional[float]:
    """Safely parse float text."""
    if text is None:
        return None
    try:
        return float(str(text).strip())
    except Exception:
        return None


def _extract_numeric_value(node: Optional[ET.Element]) -> Optional[float]:
    """Extract numeric value from direct text, nested value/center tags, or attributes."""
    if node is None:
        return None

    direct = _parse_float_text(node.text)
    if direct is not None:
        return direct

    for key in ("value", "center"):
        nested = _find_first_float(node, key)
        if nested is not None:
            return nested

    for attr_key in ("value", "center"):
        attr_val = _parse_float_text(node.attrib.get(attr_key))
        if attr_val is not None:
            return attr_val

    return None


def _normalize_enmap_processing_version(value: Optional[str]) -> Optional[str]:
    """Normalize EnMAP processing version text to dotted MAJOR.MINOR.PATCH."""
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None

    dotted_match = re.search(r"(\d{2}\.\d{2}\.\d{2})", txt)
    if dotted_match:
        return dotted_match.group(1)

    compact_match = re.search(r"\b(\d{6})\b", txt)
    if compact_match:
        raw = compact_match.group(1)
        return f"{raw[:2]}.{raw[2:4]}.{raw[4:6]}"

    return None


def _extract_enmap_processing_version_from_root(
    root: ET.Element,
    metadata_xml: Optional[str] = None,
) -> Optional[str]:
    """Extract EnMAP processing version with schema-aware priority."""
    base_node = _find_first_child_local(root, "base")
    if base_node is None:
        base_node = next(_iter_local(root, "base"), None)

    for key in ("revision", "archivedVersion"):
        if base_node is None:
            break
        candidate = _normalize_enmap_processing_version(_find_first_text(base_node, key))
        if candidate:
            return candidate

    metadata_node = _find_first_child_local(root, "metadata")
    if metadata_node is None:
        metadata_node = next(_iter_local(root, "metadata"), None)

    citation_text = _find_first_text(metadata_node, "citation") if metadata_node is not None else None
    if not citation_text:
        citation_text = _find_first_text(root, "citation")
    if citation_text:
        citation_match = re.search(
            r"version\s*[:=]\s*(\d{2}\.\d{2}\.\d{2}|\d{6})",
            str(citation_text),
            flags=re.IGNORECASE,
        )
        if citation_match:
            candidate = _normalize_enmap_processing_version(citation_match.group(1))
            if candidate:
                return candidate

    metadata_name = _find_first_text(metadata_node, "name") if metadata_node is not None else None
    if metadata_name:
        name_match = re.search(r"_V(\d{6})_", str(metadata_name), flags=re.IGNORECASE)
        if name_match:
            candidate = _normalize_enmap_processing_version(name_match.group(1))
            if candidate:
                return candidate

    if metadata_xml:
        filename = os.path.basename(str(metadata_xml))
        file_match = re.search(r"_V(\d{6})_", filename, flags=re.IGNORECASE)
        if file_match:
            candidate = _normalize_enmap_processing_version(file_match.group(1))
            if candidate:
                return candidate

    return None


def _bbox_from_bounding_polygon(polygon_node: Optional[ET.Element]) -> Optional[Tuple[float, float, float, float]]:
    """Compute bbox from a boundingPolygon/point list."""
    if polygon_node is None:
        return None

    lats: List[float] = []
    lons: List[float] = []
    for point in _iter_children_local(polygon_node, "point"):
        frame = _find_first_text(point, "frame")
        lat = _find_first_float(point, "latitude")
        lon = _find_first_float(point, "longitude")
        if frame is not None and frame.strip().lower() == "center":
            continue
        if lat is None or lon is None:
            continue
        lats.append(float(lat))
        lons.append(float(lon))

    if not lats or not lons:
        return None

    west = float(min(lons))
    east = float(max(lons))
    south = float(min(lats))
    north = float(max(lats))
    if east <= west or north <= south:
        return None
    return (west, south, east, north)


def _extract_scene_bbox_from_base_spatial_coverage(root: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    """Extract scene bbox from spatialCoverage/boundingPolygon."""
    spatial_coverage = _find_first_child_local(root, "spatialCoverage")
    if spatial_coverage is None:
        spatial_coverage = next(_iter_local(root, "spatialCoverage"), None)
    bounding_polygon = _find_first_child_local(spatial_coverage, "boundingPolygon")
    if bounding_polygon is None and spatial_coverage is not None:
        bounding_polygon = next(_iter_local(spatial_coverage, "boundingPolygon"), None)
    return _bbox_from_bounding_polygon(bounding_polygon)


def _extract_bbox_from_datatake_spatial_coverage(root: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    """Extract fallback bbox from spatialCoverageOfDatatake/boundingPolygon."""
    datatake_cov = _find_first_child_local(root, "spatialCoverageOfDatatake")
    if datatake_cov is None:
        datatake_cov = next(_iter_local(root, "spatialCoverageOfDatatake"), None)
    bounding_polygon = _find_first_child_local(datatake_cov, "boundingPolygon")
    if bounding_polygon is None and datatake_cov is not None:
        bounding_polygon = next(_iter_local(datatake_cov, "boundingPolygon"), None)
    return _bbox_from_bounding_polygon(bounding_polygon)


def find_enmap_metadata_for_spectral_image(spectral_image_path: str) -> Optional[str]:
    """
    Locate the EnMAP metadata XML file for a given spectral image.

    Args:
        spectral_image_path: Path to EnMAP SPECTRAL_IMAGE raster (.tif/.tiff/.bsq)

    Returns:
        str: Path to metadata XML file

    Raises:
        ValueError: If the metadata pair is missing or ambiguous.
    """
    parent_dir = os.path.dirname(spectral_image_path)
    basename = os.path.basename(spectral_image_path)
    base_stem = os.path.splitext(basename)[0]
    lower_stem = base_stem.lower()

    if "-spectral_image" in lower_stem:
        expected_stem = base_stem[:lower_stem.rfind("-spectral_image")]
    else:
        expected_stem = base_stem
    expected_meta_stem = f"{expected_stem}-METADATA"
    expected_meta_stem_l = expected_meta_stem.lower()

    metadata_files = []
    try:
        for name in os.listdir(parent_dir):
            name_l = name.lower()
            if name_l.endswith("-metadata.xml"):
                metadata_files.append(name)
    except Exception as e:
        raise ValueError(f"Could not list EnMAP directory for metadata discovery: {e}") from e

    if not metadata_files:
        raise ValueError(f"Could not find EnMAP metadata XML for {spectral_image_path}")

    exact_matches = []
    for name in metadata_files:
        stem = os.path.splitext(name)[0]
        if stem.lower() == expected_meta_stem_l:
            exact_matches.append(name)
    if exact_matches:
        exact_matches.sort()
        if len(exact_matches) > 1:
            raise ValueError(
                f"Ambiguous EnMAP metadata match for {spectral_image_path}: "
                f"{', '.join(exact_matches)}"
            )
        return os.path.join(parent_dir, exact_matches[0])

    prefix_matches = []
    expected_prefix_l = f"{expected_stem}-".lower()
    for name in metadata_files:
        if name.lower().startswith(expected_prefix_l):
            prefix_matches.append(name)
    if prefix_matches:
        prefix_matches.sort()
        if len(prefix_matches) > 1:
            raise ValueError(
                f"Ambiguous EnMAP metadata prefix match for {spectral_image_path}: "
                f"{', '.join(prefix_matches)}"
            )
        chosen = prefix_matches[0]
        logger.warning(
            f"Using prefix metadata match for {basename}: {chosen}. "
            "Please verify scene pairing."
        )
        return os.path.join(parent_dir, chosen)

    raise ValueError(
        f"Could not find a matching EnMAP metadata XML for {spectral_image_path}. "
        "Expected an exact or unambiguous prefix match."
    )


def _extract_center_value(parent_node: Optional[ET.Element]) -> Optional[float]:
    """Extract center value from EnMAP angle node variants."""
    if parent_node is None:
        return None
    center = _find_first_float(parent_node, "center")
    if center is not None:
        return center
    if parent_node.text:
        try:
            return float(parent_node.text.strip())
        except Exception:
            return None
    return None


def _extract_enmap_extended_metadata_from_root(
    root: ET.Element,
    metadata_xml: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract EnMAP extended metadata fields from an already-parsed XML root."""
    result: Dict[str, Any] = {
        'enmap_id': None,
        'enmap_date': None,
        'enmap_processing_version': None,
        'prisma_cloud_pct': None,  # Mapped for compatibility
        'enmap_cloud_pct': None,
        'enmap_haze_pct': None,
        'enmap_cirrus_pct': None,
        'enmap_snow_pct': None,
        'enmap_water_pct': None,
        'sun_azimuth_angle': None,
        'sun_elevation_angle': None,
        'sun_zenith_angle': None,
        'across_offnadir_angle': None,
        'along_offnadir_angle': None,
        'scene_azimuth_angle': None,
        'observation_angle': None,
    }

    result["enmap_processing_version"] = _extract_enmap_processing_version_from_root(
        root,
        metadata_xml=metadata_xml,
    )

    # Product ID
    for elem in _iter_local(root, 'productId'):
        if elem.text:
            result['enmap_id'] = elem.text.strip()
            break
    if result['enmap_id'] is None:
        metadata_node = _find_first_child_local(root, "metadata")
        metadata_name_node = _find_first_child_local(metadata_node, "name")
        metadata_name = None if metadata_name_node is None else metadata_name_node.text
        if metadata_name is not None and str(metadata_name).strip():
            meta_name = str(metadata_name).strip()
            if meta_name.lower().endswith("-metadata.xml"):
                meta_name = meta_name[:-len("-metadata.xml")]
            result['enmap_id'] = meta_name
    if result['enmap_id'] is None and metadata_xml:
        stem = os.path.splitext(os.path.basename(str(metadata_xml)))[0]
        if stem and stem.lower().endswith("-metadata"):
            stem = stem[:-len("-metadata")]
        if stem:
            result['enmap_id'] = stem

    # Date
    for elem in _iter_local(root, 'startTime'):
        if elem.text:
            try:
                dt = datetime.fromisoformat(elem.text.strip().replace('Z', '+00:00'))
                result['enmap_date'] = dt.isoformat()
            except Exception:
                pass
            break

    # Cloud / atmospheric quality
    quality_tag_map = {
        "enmap_cloud_pct": ("cloudCover", "cloudPercentage", "cloudFraction", "cloudyPixelPercentage"),
        "enmap_haze_pct": ("hazeCover", "hazePercentage"),
        "enmap_cirrus_pct": ("cirrusCover", "cirrusPercentage"),
        "enmap_snow_pct": ("snowCover", "snowPercentage"),
        "enmap_water_pct": ("waterCover", "waterPercentage"),
    }
    quality_tag_names: set[str] = set()

    for container_name in ("qualityInformation", "qualityFlag"):
        for quality in _iter_local(root, container_name):
            for tag in quality.iter():
                quality_tag_names.add(_local_tag_name(tag.tag))

            for metric_key, aliases in quality_tag_map.items():
                if result.get(metric_key) is not None:
                    continue
                found: Optional[float] = None
                for alias in aliases:
                    for node in _iter_local(quality, alias):
                        val = _extract_numeric_value(node)
                        if val is not None:
                            found = float(val)
                            break
                    if found is not None:
                        break
                if found is not None:
                    result[metric_key] = found

    if result['enmap_cloud_pct'] is not None:
        result['prisma_cloud_pct'] = result['enmap_cloud_pct']
    else:
        seen = sorted(quality_tag_names)
        if seen:
            logger.debug(
                "EnMAP cloud metadata not parsed; quality tags discovered: %s",
                ", ".join(seen[:80]),
            )

    # Angles are direct descendants under specific (or elsewhere); search from root.
    angle_map = (
        ('sun_azimuth_angle', 'sunAzimuthAngle'),
        ('sun_elevation_angle', 'sunElevationAngle'),
        ('sun_zenith_angle', 'sunZenithAngle'),
        ('across_offnadir_angle', 'acrossOffNadirAngle'),
        ('along_offnadir_angle', 'alongOffNadirAngle'),
        ('scene_azimuth_angle', 'sceneAzimuthAngle'),
    )
    for out_key, xml_tag in angle_map:
        for node in _iter_local(root, xml_tag):
            val = _extract_center_value(node)
            if val is not None:
                result[out_key] = float(val)
                break

    if result['across_offnadir_angle'] is not None and result['along_offnadir_angle'] is not None:
        result['observation_angle'] = float(np.sqrt(
            result['across_offnadir_angle'] ** 2 + result['along_offnadir_angle'] ** 2
        ))

    return result


def _normalize_epsg_text(value: Optional[str]) -> Optional[str]:
    """Normalize EPSG value from raw metadata text (e.g. 'EPSG:32632', '32632')."""
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None

    epsg_match = re.search(r"epsg\s*[:_ ]\s*(\d{4,6})", txt, flags=re.IGNORECASE)
    if epsg_match:
        return epsg_match.group(1)

    numeric_match = re.fullmatch(r"\d{4,6}", txt)
    if numeric_match:
        return numeric_match.group(0)

    return None


def _infer_epsg_from_projection_text(value: Optional[str]) -> Optional[str]:
    """
    Infer EPSG code from UTM-style projection labels.

    Examples supported:
    - UTM_Zone32_North -> EPSG:32632
    - UTM Zone 32 South -> EPSG:32732
    """
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None

    match = re.search(r"zone[\s_]*(\d{1,2})", txt, flags=re.IGNORECASE)
    if not match:
        return None

    zone = int(match.group(1))
    if zone < 1 or zone > 60:
        return None

    text_lower = txt.lower()
    hemisphere: Optional[str] = None
    if "north" in text_lower:
        hemisphere = "north"
    elif "south" in text_lower:
        hemisphere = "south"
    else:
        letter_match = re.search(r"zone[\s_]*\d{1,2}\s*([ns])\b", text_lower)
        if letter_match:
            hemisphere = "north" if letter_match.group(1) == "n" else "south"

    if hemisphere is None:
        return None

    if hemisphere == "north":
        return str(32600 + zone)
    return str(32700 + zone)


def _extract_crs_from_metadata_tree(root: ET.Element) -> Optional[str]:
    """Extract CRS from EnMAP metadata tree using explicit EPSG and projection tags."""
    for epsg_elem in _iter_local(root, "epsgCode"):
        epsg = _normalize_epsg_text(epsg_elem.text)
        if epsg is not None:
            return f"EPSG:{epsg}"

    projection_tags = (
        "mapProjection",
        "projection",
        "crs",
        "CRS",
        "srsName",
        "coordinateReferenceSystem",
    )
    for tag in projection_tags:
        for elem in _iter_local(root, tag):
            epsg = _normalize_epsg_text(elem.text)
            if epsg is None:
                epsg = _infer_epsg_from_projection_text(elem.text)
            if epsg is not None:
                return f"EPSG:{epsg}"

    return None


def _extract_crs_from_raster_path(spectral_image_path: Optional[str]) -> Optional[str]:
    """Extract CRS from EnMAP spectral image georeferencing."""
    if spectral_image_path is None:
        return None
    if not os.path.exists(spectral_image_path):
        return None

    try:
        with rasterio.open(spectral_image_path) as src:
            if src.crs is None:
                return None
            epsg = src.crs.to_epsg()
            if epsg is not None:
                return f"EPSG:{int(epsg)}"
            return str(src.crs)
    except Exception as exc:
        logger.debug("Could not derive CRS from EnMAP raster fallback: %s", exc)
        return None


def read_enmap_metadata(metadata_xml: str, spectral_image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse EnMAP metadata XML file to extract wavelengths, FWHM, and other info.

    Args:
        metadata_xml: Path to EnMAP -METADATA.XML file
        spectral_image_path: Optional path to spectral raster for CRS fallback

    Returns:
        dict: Metadata including wavelengths, fwhm, band_names, acquisition_time, bbox, crs
    """
    result = {
        'wavelengths': None,
        'fwhm': None,
        'band_names': None,
        'acquisition_time': None,
        'bbox': None,
        'crs': None,
        'n_rows': None,
        'n_cols': None,
        'n_bands': None,
        'data_gain_values': None,
        'data_offset_values': None,
        'background_value': None,
        'enmap_id': None,
        'enmap_date': None,
        'enmap_processing_version': None,
        'prisma_cloud_pct': None,
        'enmap_cloud_pct': None,
        'enmap_haze_pct': None,
        'enmap_cirrus_pct': None,
        'enmap_snow_pct': None,
        'enmap_water_pct': None,
        'sun_azimuth_angle': None,
        'sun_elevation_angle': None,
        'sun_zenith_angle': None,
        'across_offnadir_angle': None,
        'along_offnadir_angle': None,
        'scene_azimuth_angle': None,
        'observation_angle': None,
    }

    try:
        tree = ET.parse(metadata_xml)
        root = tree.getroot()

        # Find spectral characteristics.
        wavelengths: List[float] = []
        fwhm_list: List[float] = []
        band_names: List[str] = []
        data_gain_values: List[float] = []
        data_offset_values: List[float] = []

        band_blocks: List[List[Tuple[float, Optional[float], Optional[str], Optional[float], Optional[float]]]] = []
        for band_char in _iter_local(root, "bandCharacterisation"):
            block_entries: List[Tuple[float, Optional[float], Optional[str], Optional[float], Optional[float]]] = []
            band_nodes = list(_iter_children_local(band_char, "bandID"))

            if band_nodes:
                for band_node in band_nodes:
                    wl_val = _find_first_float(band_node, "wavelengthCenterOfBand")
                    if wl_val is None:
                        wl_val = _find_first_float(band_node, "centralWavelength")
                    if wl_val is None:
                        continue

                    fwhm_val = _find_first_float(band_node, "FWHMOfBand")
                    if fwhm_val is None:
                        fwhm_val = _find_first_float(band_node, "FWHM")

                    name_val: Optional[str] = None
                    num_attr = band_node.attrib.get("number")
                    if num_attr is not None and str(num_attr).strip():
                        name_val = str(num_attr).strip()
                    elif band_node.text and band_node.text.strip():
                        name_val = band_node.text.strip()

                    gain_val = _find_first_float(band_node, "GainOfBand")
                    offset_val = _find_first_float(band_node, "OffsetOfBand")

                    block_entries.append(
                        (
                            float(wl_val),
                            None if fwhm_val is None else float(fwhm_val),
                            name_val,
                            None if gain_val is None else float(gain_val),
                            None if offset_val is None else float(offset_val),
                        )
                    )
            else:
                wl_val = _find_first_float(band_char, "wavelengthCenterOfBand")
                if wl_val is None:
                    wl_val = _find_first_float(band_char, "centralWavelength")
                if wl_val is not None:
                    fwhm_val = _find_first_float(band_char, "FWHMOfBand")
                    if fwhm_val is None:
                        fwhm_val = _find_first_float(band_char, "FWHM")
                    name_val = _find_first_text(band_char, "bandID")
                    gain_val = _find_first_float(band_char, "GainOfBand")
                    offset_val = _find_first_float(band_char, "OffsetOfBand")
                    block_entries.append(
                        (
                            float(wl_val),
                            None if fwhm_val is None else float(fwhm_val),
                            name_val,
                            None if gain_val is None else float(gain_val),
                            None if offset_val is None else float(offset_val),
                        )
                    )

            if block_entries:
                band_blocks.append(block_entries)

        if band_blocks:
            selected_entries = [entry for block in band_blocks for entry in block]
            if len(band_blocks) > 1:
                logger.info(
                    "EnMAP spectral parsing merged %d bandCharacterisation blocks (%d total bands).",
                    len(band_blocks),
                    len(selected_entries),
                )

            for wl_val, fwhm_val, name_val, gain_val, offset_val in selected_entries:
                wavelengths.append(float(wl_val))
                if fwhm_val is not None:
                    fwhm_list.append(float(fwhm_val))
                if name_val is not None and str(name_val).strip():
                    band_names.append(str(name_val).strip())
                if gain_val is not None:
                    data_gain_values.append(float(gain_val))
                if offset_val is not None:
                    data_offset_values.append(float(offset_val))

        # Alternative structure
        if not wavelengths:
            for spec in _iter_local(root, 'spectralCharacterisation'):
                band_names.extend(_collect_text_values(spec, "bandID"))
                wavelengths.extend(_collect_float_values(spec, "centralWavelength"))
                fwhm_list.extend(_collect_float_values(spec, "FWHM"))
                data_gain_values.extend(_collect_float_values(spec, "GainOfBand"))
                data_offset_values.extend(_collect_float_values(spec, "OffsetOfBand"))

        if not data_gain_values:
            data_gain_values = _collect_float_values(root, "GainOfBand")
        if not data_offset_values:
            data_offset_values = _collect_float_values(root, "OffsetOfBand")

        n_wl = len(wavelengths)
        n_fwhm = len(fwhm_list)
        n_names = len(band_names)

        if wavelengths:
            result['wavelengths'] = np.array(wavelengths)
        if fwhm_list:
            if n_fwhm == n_wl:
                result['fwhm'] = np.array(fwhm_list)
            else:
                logger.warning(
                    f"EnMAP metadata FWHM length mismatch ({n_fwhm} vs wavelengths {n_wl}); ignoring FWHM."
                )
        if band_names:
            if n_names == n_wl:
                result['band_names'] = band_names
            else:
                logger.warning(
                    f"EnMAP metadata band-name length mismatch ({n_names} vs wavelengths {n_wl}); ignoring band names."
                )
        gains_arr = _align_band_float_values(data_gain_values, n_wl, "data gain values")
        offsets_arr = _align_band_float_values(data_offset_values, n_wl, "data offset values")
        if gains_arr is not None:
            result['data_gain_values'] = gains_arr
        if offsets_arr is not None:
            result['data_offset_values'] = offsets_arr

        background_value = _find_first_float(root, "backgroundValue")
        if background_value is None:
            background_value = _find_first_float(root, "dataIgnoreValue")
        if background_value is not None and np.isfinite(float(background_value)):
            result['background_value'] = float(background_value)

        # Extract acquisition time
        for time_elem in _iter_local(root, 'startTime'):
            if not time_elem.text:
                continue
            try:
                result['acquisition_time'] = datetime.fromisoformat(
                    time_elem.text.strip().replace('Z', '+00:00')
                )
            except Exception:
                pass
            break

        # Extract bounding box
        for bbox_elem in _iter_local(root, 'boundingBox'):
            try:
                west = _find_first_float(bbox_elem, "westBoundLongitude")
                east = _find_first_float(bbox_elem, "eastBoundLongitude")
                south = _find_first_float(bbox_elem, "southBoundLatitude")
                north = _find_first_float(bbox_elem, "northBoundLatitude")
                if None not in (west, east, south, north):
                    result['bbox'] = (west, south, east, north)
            except Exception:
                pass
            break

        if result['bbox'] is None:
            scene_bbox = _extract_scene_bbox_from_base_spatial_coverage(root)
            if scene_bbox is not None:
                result['bbox'] = scene_bbox
                logger.info("EnMAP metadata bbox parsed from spatialCoverage/boundingPolygon.")

        if result['bbox'] is None:
            datatake_bbox = _extract_bbox_from_datatake_spatial_coverage(root)
            if datatake_bbox is not None:
                result['bbox'] = datatake_bbox
                logger.warning(
                    "EnMAP metadata scene bbox unavailable; using datatake coverage bounding polygon fallback."
                )

        # Extract CRS
        result['crs'] = _extract_crs_from_metadata_tree(root)
        if result['crs'] is None:
            raster_crs = _extract_crs_from_raster_path(spectral_image_path)
            if raster_crs is not None:
                result['crs'] = raster_crs
                logger.info("EnMAP metadata CRS inferred from spectral raster georeferencing.")

        # Extract dimensions from current EnMAP schema.
        rows_txt = _find_first_text(root, "heightOfOrthoScene")
        cols_txt = _find_first_text(root, "widthOfOrthoScene")

        merge_node = next(_iter_local(root, "merge"), None)
        bands_txt = _find_first_text(merge_node, "channels") if merge_node is not None else None
        if bands_txt is None:
            bands_txt = _find_first_text(root, "channels")

        if rows_txt is not None:
            try:
                result['n_rows'] = int(float(rows_txt))
            except Exception:
                logger.debug("Could not parse EnMAP heightOfOrthoScene as integer: %s", rows_txt)
        if cols_txt is not None:
            try:
                result['n_cols'] = int(float(cols_txt))
            except Exception:
                logger.debug("Could not parse EnMAP widthOfOrthoScene as integer: %s", cols_txt)
        if bands_txt is not None:
            try:
                result['n_bands'] = int(float(bands_txt))
            except Exception:
                logger.debug("Could not parse EnMAP channels as integer: %s", bands_txt)

        # Extract extended metadata in the same single XML parse.
        result.update(_extract_enmap_extended_metadata_from_root(root, metadata_xml=metadata_xml))

        if result['acquisition_time'] is None:
            logger.warning("EnMAP metadata missing acquisition time.")
        if result['bbox'] is None:
            logger.warning("EnMAP metadata missing geographic bounding box.")
        if result['wavelengths'] is None:
            logger.warning("EnMAP metadata missing wavelength list.")

        logger.info(f"EnMAP metadata: {len(wavelengths)} bands, CRS={result['crs']}")

    except Exception as e:
        logger.warning(f"Error parsing EnMAP metadata XML: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return result


def stage_enmap_metadata_for_session(raster_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage parsed EnMAP metadata without mutating the source raster.

    The refactored pipeline keeps the source raster immutable. Metadata is staged
    in an in-memory cache so immediate readback can still recover the parsed
    values without mutating the user input file.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "error": None,
        "bands_updated": 0,
        "dataset_tags_updated": 0,
        "metadata_staged": False,
        "source_mutated": False,
        "staging_mode": "memory_cache",
        "cache_scope": "process",
    }
    try:
        if not raster_path or not os.path.exists(raster_path):
            out["error"] = f"Raster not found: {raster_path}"
            return out
        if not isinstance(metadata, dict):
            out["error"] = "Metadata must be a dictionary."
            return out

        wavelengths = np.asarray(metadata.get("wavelengths"), dtype=float).reshape(-1) \
            if metadata.get("wavelengths") is not None else np.array([], dtype=float)
        fwhm_vals = np.asarray(metadata.get("fwhm"), dtype=float).reshape(-1) \
            if metadata.get("fwhm") is not None else np.array([], dtype=float)
        gain_vals = np.asarray(metadata.get("data_gain_values"), dtype=float).reshape(-1) \
            if metadata.get("data_gain_values") is not None else np.array([], dtype=float)
        offset_vals = np.asarray(metadata.get("data_offset_values"), dtype=float).reshape(-1) \
            if metadata.get("data_offset_values") is not None else np.array([], dtype=float)
        band_names_raw = metadata.get("band_names")
        band_names: List[str] = []
        if isinstance(band_names_raw, (list, tuple, np.ndarray)):
            band_names = [str(v).strip() for v in list(band_names_raw) if str(v).strip()]

        try:
            with rasterio.open(raster_path) as src:
                count = int(src.count)
                for bidx in range(1, count + 1):
                    has_metadata = False
                    if bidx <= len(band_names):
                        has_metadata = True
                    if bidx <= wavelengths.size and np.isfinite(float(wavelengths[bidx - 1])):
                        has_metadata = True
                    if bidx <= fwhm_vals.size and np.isfinite(float(fwhm_vals[bidx - 1])):
                        has_metadata = True
                    if bidx <= gain_vals.size and np.isfinite(float(gain_vals[bidx - 1])):
                        has_metadata = True
                    if bidx <= offset_vals.size and np.isfinite(float(offset_vals[bidx - 1])):
                        has_metadata = True
                    if has_metadata:
                        out["bands_updated"] += 1
        except Exception as exc:
            out["error"] = str(exc)
            return out

        global_tags: Dict[str, str] = {}
        acq_time = metadata.get("acquisition_time")
        if isinstance(acq_time, datetime):
            global_tags["acquisition_time"] = acq_time.isoformat()
        elif acq_time is not None and str(acq_time).strip():
            global_tags["acquisition_time"] = str(acq_time).strip()

        bbox_val = metadata.get("bbox")
        if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
            try:
                west, south, east, north = [float(v) for v in bbox_val]
                global_tags["bbox_wgs84"] = f"{west:.10f},{south:.10f},{east:.10f},{north:.10f}"
            except Exception:
                pass

        scalar_keys = [
            "crs",
            "n_rows",
            "n_cols",
            "n_bands",
            "background_value",
            "enmap_id",
            "enmap_date",
            "enmap_processing_version",
            "prisma_cloud_pct",
            "enmap_cloud_pct",
            "enmap_haze_pct",
            "enmap_cirrus_pct",
            "enmap_snow_pct",
            "enmap_water_pct",
            "sun_azimuth_angle",
            "sun_elevation_angle",
            "sun_zenith_angle",
            "across_offnadir_angle",
            "along_offnadir_angle",
            "scene_azimuth_angle",
            "observation_angle",
        ]
        for key in scalar_keys:
            val = metadata.get(key)
            if val is None:
                continue
            if isinstance(val, float):
                if not np.isfinite(float(val)):
                    continue
                global_tags[key] = f"{float(val):.10f}"
            else:
                txt = str(val).strip()
                if txt:
                    global_tags[key] = txt

        out["dataset_tags_updated"] = len(global_tags)
        _ENMAP_METADATA_CACHE[os.path.abspath(raster_path)] = dict(metadata)
        out["metadata_staged"] = True
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def read_enmap_metadata_from_raster(raster_path: str) -> Dict[str, Any]:
    """Read EnMAP metadata from staged cache first, then from raster tags."""
    out: Dict[str, Any] = {
        'wavelengths': None,
        'fwhm': None,
        'band_names': None,
        'acquisition_time': None,
        'bbox': None,
        'crs': None,
        'n_rows': None,
        'n_cols': None,
        'n_bands': None,
        'data_gain_values': None,
        'data_offset_values': None,
        'background_value': None,
        'enmap_id': None,
        'enmap_date': None,
        'enmap_processing_version': None,
        'prisma_cloud_pct': None,
        'enmap_cloud_pct': None,
        'enmap_haze_pct': None,
        'enmap_cirrus_pct': None,
        'enmap_snow_pct': None,
        'enmap_water_pct': None,
        'sun_azimuth_angle': None,
        'sun_elevation_angle': None,
        'sun_zenith_angle': None,
        'across_offnadir_angle': None,
        'along_offnadir_angle': None,
        'scene_azimuth_angle': None,
        'observation_angle': None,
    }
    try:
        abs_raster_path = os.path.abspath(raster_path)
        cached = _ENMAP_METADATA_CACHE.get(abs_raster_path)

        with rasterio.open(raster_path) as src:
            out['n_rows'] = int(src.height)
            out['n_cols'] = int(src.width)
            out['n_bands'] = int(src.count)
            if src.crs is not None:
                epsg = src.crs.to_epsg()
                out['crs'] = f"EPSG:{int(epsg)}" if epsg is not None else str(src.crs)

            if cached is not None:
                for key, value in cached.items():
                    if key in out and value is not None:
                        out[key] = value
                if out.get("bbox") is not None and isinstance(out["bbox"], (list, tuple)) and len(out["bbox"]) == 4:
                    out["bbox"] = tuple(float(v) for v in out["bbox"])
                return out

            ds_tags = src.tags()
            try:
                envi_tags = src.tags(ns="ENVI")
            except Exception:
                envi_tags = {}
            all_ds_tags = dict(ds_tags)
            all_ds_tags.update({str(k): v for k, v in envi_tags.items()})

            def _tag_key(value: Any) -> str:
                return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")

            def _get_tag(*names: str) -> Optional[str]:
                lookup = {_tag_key(k): v for k, v in all_ds_tags.items()}
                for name in names:
                    val = lookup.get(_tag_key(name))
                    if val is not None:
                        return str(val)
                return None

            acq_txt = ds_tags.get("acquisition_time")
            if acq_txt:
                try:
                    out['acquisition_time'] = datetime.fromisoformat(str(acq_txt).replace('Z', '+00:00'))
                except Exception:
                    pass

            bbox_txt = ds_tags.get("bbox_wgs84")
            if bbox_txt:
                try:
                    parts = [float(v.strip()) for v in str(bbox_txt).split(",")]
                    if len(parts) == 4:
                        out['bbox'] = (parts[0], parts[1], parts[2], parts[3])
                except Exception:
                    pass

            def _parse_float_tag(key: str) -> Optional[float]:
                raw = _get_tag(key)
                if raw is None:
                    return None
                text = str(raw).strip().strip("{}[]()").strip().strip("'\"")
                try:
                    val = float(text)
                    if np.isfinite(val):
                        return val
                except Exception:
                    return None
                return None

            out["background_value"] = _parse_float_tag("background_value")
            if out["background_value"] is None:
                out["background_value"] = _parse_float_tag("backgroundValue")
            if out["background_value"] is None:
                out["background_value"] = _parse_float_tag("data_ignore_value")
            if out["background_value"] is None:
                out["background_value"] = _parse_float_tag("data ignore value")
            if out["background_value"] is None and src.nodata is not None:
                try:
                    nodata_val = float(src.nodata)
                    if np.isfinite(nodata_val):
                        out["background_value"] = nodata_val
                except Exception:
                    pass

            for key in (
                "prisma_cloud_pct",
                "enmap_cloud_pct",
                "enmap_haze_pct",
                "enmap_cirrus_pct",
                "enmap_snow_pct",
                "enmap_water_pct",
                "sun_azimuth_angle",
                "sun_elevation_angle",
                "sun_zenith_angle",
                "across_offnadir_angle",
                "along_offnadir_angle",
                "scene_azimuth_angle",
                "observation_angle",
            ):
                out[key] = _parse_float_tag(key)

            for key in ("enmap_id", "enmap_date", "enmap_processing_version"):
                raw = ds_tags.get(key)
                if raw is not None and str(raw).strip():
                    if key == "enmap_processing_version":
                        normalized = _normalize_enmap_processing_version(str(raw).strip())
                        out[key] = normalized if normalized is not None else str(raw).strip()
                    else:
                        out[key] = str(raw).strip()

            band_count = int(src.count)
            wl_by_band = np.full(band_count, np.nan, dtype=float)
            fwhm_by_band = np.full(band_count, np.nan, dtype=float)
            gain_vals: List[float] = []
            offset_vals: List[float] = []
            band_names_by_band: List[Optional[str]] = [None] * band_count
            for bidx in range(1, band_count + 1):
                btags = src.tags(bidx)
                wl_txt = btags.get("wavelength")
                if wl_txt is not None:
                    try:
                        wl = float(str(wl_txt).strip())
                        if np.isfinite(wl):
                            wl_by_band[bidx - 1] = wl
                    except Exception:
                        pass
                fw_txt = btags.get("fwhm")
                if fw_txt is not None:
                    try:
                        fw = float(str(fw_txt).strip())
                        if np.isfinite(fw):
                            fwhm_by_band[bidx - 1] = fw
                    except Exception:
                        pass
                gain_txt = btags.get("data_gain") or btags.get("gain")
                if gain_txt is not None:
                    try:
                        gain = float(str(gain_txt).strip())
                        if np.isfinite(gain):
                            gain_vals.append(gain)
                    except Exception:
                        pass
                offset_txt = btags.get("data_offset") or btags.get("offset")
                if offset_txt is not None:
                    try:
                        offset = float(str(offset_txt).strip())
                        if np.isfinite(offset):
                            offset_vals.append(offset)
                    except Exception:
                        pass
                desc = src.descriptions[bidx - 1]
                if desc is not None and str(desc).strip():
                    band_names_by_band[bidx - 1] = str(desc).strip()

            if not gain_vals:
                gain_vals = _parse_float_list_text(
                    _get_tag("data_gain_values", "data gain values", "GainOfBand")
                )
            if not offset_vals:
                offset_vals = _parse_float_list_text(
                    _get_tag("data_offset_values", "data offset values", "OffsetOfBand")
                )

            wl_valid_mask = np.isfinite(wl_by_band)
            if bool(np.all(wl_valid_mask)) and band_count > 0:
                out['wavelengths'] = np.array(wl_by_band, dtype=float)
            elif bool(np.any(wl_valid_mask)):
                logger.warning(
                    "Sparse EnMAP raster wavelength tags (%d/%d bands); ignoring raster tags and trying XML fallback.",
                    int(np.count_nonzero(wl_valid_mask)),
                    band_count,
                )
            if out['wavelengths'] is not None and bool(np.all(np.isfinite(fwhm_by_band))):
                out['fwhm'] = np.array(fwhm_by_band, dtype=float)
            if out['wavelengths'] is not None and all(name for name in band_names_by_band):
                out['band_names'] = [str(name) for name in band_names_by_band]
            gains_arr = _align_band_float_values(gain_vals, band_count, "data gain values")
            offsets_arr = _align_band_float_values(offset_vals, band_count, "data offset values")
            if gains_arr is not None:
                out['data_gain_values'] = gains_arr
            if offsets_arr is not None:
                out['data_offset_values'] = offsets_arr

        if out["wavelengths"] is None:
            try:
                metadata_xml = find_enmap_metadata_for_spectral_image(raster_path)
                xml_meta = read_enmap_metadata(metadata_xml, spectral_image_path=raster_path)
                for key, value in xml_meta.items():
                    if key in out and out[key] is None and value is not None:
                        out[key] = value
            except Exception as exc:
                logger.debug(
                    "Could not fall back to EnMAP XML metadata for %s: %s",
                    raster_path,
                    exc,
                )

        return out
    except Exception as e:
        logger.debug("Could not read staged EnMAP raster metadata from %s: %s", raster_path, e)
        return out


def derive_enmap_bbox_from_raster(enmap_file: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Derive geographic WGS84 bbox from EnMAP spectral image georeferencing.

    Args:
        enmap_file: Path to EnMAP spectral image GeoTIFF

    Returns:
        tuple: (west, south, east, north) in EPSG:4326, or None on failure
    """
    try:
        with rasterio.open(enmap_file) as src:
            if src.crs is None:
                logger.warning("EnMAP raster bbox fallback failed: source CRS is missing.")
                return None
            source_crs = src.crs
            bounds = src.bounds

        if bounds is None:
            logger.warning("EnMAP raster bbox fallback failed: raster bounds unavailable.")
            return None

        left = float(bounds.left)
        bottom = float(bounds.bottom)
        right = float(bounds.right)
        top = float(bounds.top)
        if not np.all(np.isfinite([left, bottom, right, top])):
            logger.warning("EnMAP raster bbox fallback failed: non-finite raster bounds.")
            return None

        west, south, east, north = transform_bounds(
            source_crs,
            "EPSG:4326",
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
        if not np.all(np.isfinite([west, south, east, north])):
            logger.warning("EnMAP raster bbox fallback failed: non-finite transformed bounds.")
            return None

        # Keep bbox monotonic even if source orientation is unusual.
        west_f = float(min(west, east))
        east_f = float(max(west, east))
        south_f = float(min(south, north))
        north_f = float(max(south, north))
        if east_f <= west_f or north_f <= south_f:
            logger.warning("EnMAP raster bbox fallback failed: invalid transformed bounds ordering.")
            return None

        return (west_f, south_f, east_f, north_f)
    except Exception as e:
        logger.warning(f"EnMAP raster bbox fallback failed: {e}")
        return None


def extract_enmap_extended_metadata(
    enmap_file: str,
    metadata_xml: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract extended EnMAP metadata for metrics reporting.

    Args:
        enmap_file: Path to EnMAP spectral image
        metadata_xml: Path to metadata XML (optional, will be auto-discovered)

    Returns:
        dict: Extended metadata including angles, cloud cover, etc.
    """
    result = {
        'enmap_id': None,
        'enmap_date': None,
        'enmap_processing_version': None,
        'prisma_cloud_pct': None,  # Mapped for compatibility
        'enmap_cloud_pct': None,
        'enmap_haze_pct': None,
        'enmap_cirrus_pct': None,
        'enmap_snow_pct': None,
        'enmap_water_pct': None,
        'sun_azimuth_angle': None,
        'sun_elevation_angle': None,
        'sun_zenith_angle': None,
        'across_offnadir_angle': None,
        'along_offnadir_angle': None,
        'scene_azimuth_angle': None,
        'observation_angle': None
    }

    # Auto-discover metadata XML if not provided
    if metadata_xml is None:
        metadata_xml = find_enmap_metadata_for_spectral_image(enmap_file)

    if not os.path.exists(metadata_xml):
        raise FileNotFoundError(f"EnMAP metadata XML not found for {enmap_file}: {metadata_xml}")

    try:
        tree = ET.parse(metadata_xml)
        root = tree.getroot()

        result.update(_extract_enmap_extended_metadata_from_root(root, metadata_xml=metadata_xml))

    except Exception as e:
        logger.warning(f"Error extracting EnMAP extended metadata: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return result


def copy_enmap_auxiliary_tifs(
    enmap_spectral_image: str,
    output_dir: str
) -> Dict[str, str]:
    """
    Copy EnMAP auxiliary band TIFFs to output directory.

    Args:
        enmap_spectral_image: Path to EnMAP spectral image
        output_dir: Output directory

    Returns:
        dict: Mapping of auxiliary type to copied file path
    """
    parent_dir = os.path.dirname(enmap_spectral_image)
    basename = os.path.basename(enmap_spectral_image)

    # Extract base pattern extension-agnostically so .tif/.tiff/.bsq behave the same.
    base_stem = os.path.splitext(basename)[0]
    lower_stem = base_stem.lower()
    if "-spectral_image" in lower_stem:
        base_pattern = base_stem[:lower_stem.rfind("-spectral_image")]
    else:
        base_pattern = base_stem

    # Auxiliary band types
    aux_types = [
        'QUALITY_CLASSES',
        'QUALITY_CLOUD',
        'QUALITY_CLOUDSHADOW',
        'QUALITY_HAZE',
        'QUALITY_CIRRUS',
        'QUALITY_SNOW',
        'QUALITY_TESTFLAGS',
    ]

    copied = {}
    os.makedirs(output_dir, exist_ok=True)

    for aux_type in aux_types:
        aux_name = f"{base_pattern}-{aux_type}.TIF"
        aux_path = os.path.join(parent_dir, aux_name)

        if os.path.exists(aux_path):
            dest_path = os.path.join(output_dir, aux_name)
            try:
                shutil.copy2(aux_path, dest_path)
                copied[aux_type] = dest_path
                logger.debug(f"Copied EnMAP auxiliary: {aux_type}")
            except Exception as e:
                logger.warning(f"Failed to copy {aux_type}: {e}")
        else:
            # Try lowercase
            aux_name_lower = f"{base_pattern}-{aux_type}.tif"
            aux_path_lower = os.path.join(parent_dir, aux_name_lower)
            if os.path.exists(aux_path_lower):
                dest_path = os.path.join(output_dir, aux_name_lower)
                try:
                    shutil.copy2(aux_path_lower, dest_path)
                    copied[aux_type] = dest_path
                    logger.debug(f"Copied EnMAP auxiliary: {aux_type}")
                except Exception as e:
                    logger.warning(f"Failed to copy {aux_type}: {e}")

    logger.info(f"Copied {len(copied)} EnMAP auxiliary bands")
    return copied


def check_enmap_crs_compatibility(enmap_crs, s2_crs) -> bool:
    """
    Check if EnMAP CRS is compatible with Sentinel-2 CRS.

    Args:
        enmap_crs: EnMAP CRS (rasterio CRS or string)
        s2_crs: Sentinel-2 CRS

    Returns:
        bool: True if CRS are compatible (same or can be transformed)
    """
    try:
        c1 = CRS.from_user_input(enmap_crs)
        c2 = CRS.from_user_input(s2_crs)

        if c1.equals(c2):
            return True

        # Both should be UTM zones for best compatibility
        if c1.is_projected and c2.is_projected:
            return True

        return False

    except Exception as e:
        logger.warning(f"CRS compatibility check failed: {e}")
        return False


def _format_crs_for_gdal(target_crs: Any) -> str:
    """Normalize CRS input to a gdalwarp-compatible string."""
    if target_crs is None:
        raise ValueError("target_crs cannot be None for EnMAP reprojection.")

    try:
        crs_obj = CRS.from_user_input(target_crs)
        epsg = crs_obj.to_epsg()
        if epsg is not None:
            return f"EPSG:{epsg}"
        return crs_obj.to_wkt()
    except Exception:
        # Preserve backwards compatibility for already-string CRS expressions.
        crs_text = str(target_crs).strip()
        if not crs_text:
            raise ValueError("target_crs cannot be empty for EnMAP reprojection.")
        return crs_text


def reproject_enmap_to_s2_crs(
    enmap_path: str,
    target_crs: Any,
    output_path: str,
    resolution: float = 30.0
) -> str:
    """
    Reproject EnMAP image to Sentinel-2 CRS.

    Args:
        enmap_path: Path to EnMAP GeoTIFF
        target_crs: Target CRS (e.g., "EPSG:32632")
        output_path: Output file path
        resolution: Target resolution in meters

    Returns:
        str: Path to reprojected file
    """
    try:
        gdalwarp = resolve_gdalwarp_exe()
        target_crs_arg = _format_crs_for_gdal(target_crs)
        source_nodata: Optional[float] = None
        try:
            with rasterio.open(str(enmap_path)) as src:
                nodata_val = src.nodata
                if nodata_val is not None and np.isfinite(float(nodata_val)):
                    source_nodata = float(nodata_val)
        except Exception as nodata_exc:
            logger.warning(
                "Could not read EnMAP source nodata for reprojection (%s); "
                "proceeding without explicit src/dst nodata.",
                nodata_exc,
            )

        cmd = [
            gdalwarp,
            '-t_srs', target_crs_arg,
            '-tr', str(resolution), str(resolution),
            '-r', 'bilinear',
        ]
        if source_nodata is not None:
            nodata_arg = str(source_nodata)
            cmd.extend(['-srcnodata', nodata_arg, '-dstnodata', nodata_arg])
            logger.info(
                "EnMAP reprojection nodata policy: src=%s, dst=%s",
                nodata_arg,
                nodata_arg,
            )
        else:
            logger.info("EnMAP reprojection nodata policy: no explicit src/dst nodata")
        cmd.extend([
            '-co', 'COMPRESS=LZW',
            '-co', 'TILED=YES',
            '-co', 'BIGTIFF=YES',
            '-overwrite',
            str(enmap_path),
            str(output_path)
        ])

        logger.info(f"Reprojecting EnMAP to {target_crs_arg}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info(f"Reprojected EnMAP saved to: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        msg = stderr if stderr else stdout
        if not msg:
            msg = f"exit code {e.returncode}"
        logger.error(f"gdalwarp failed for EnMAP reprojection: {msg}")
        raise RuntimeError(f"gdalwarp failed: {msg}") from e
    except Exception as e:
        logger.error(f"Failed to reproject EnMAP: {e}")
        raise
