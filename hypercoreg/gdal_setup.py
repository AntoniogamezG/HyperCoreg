"""
GDAL Environment Setup for HyperCoreg.

This module configures GDAL paths and environment variables.
It must be imported early, before rasterio or other GDAL-dependent imports.
"""

import os
import sys
import platform

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"
EXE_EXT = ".exe" if IS_WINDOWS else ""

# Get conda/Python environment base path
ENV_PREFIX = os.environ.get("CONDA_PREFIX", sys.prefix)
# Only a conda environment ships GDAL/PROJ data under its prefix. For a plain
# system Python, sys.prefix is e.g. /usr, whose PROJ data belongs to the system
# PROJ, not the one bundled with rasterio/pyproj wheels.
IS_CONDA_ENV = bool(os.environ.get("CONDA_PREFIX")) or os.path.isdir(
    os.path.join(sys.prefix, "conda-meta")
)


def _get_gdal_candidates() -> list:
    """
    Get list of candidate GDAL_DATA paths to search.

    Returns:
        List of directory paths to check for GDAL data files.
    """
    candidates = []

    # User-specified GDAL_DATA takes priority
    if os.environ.get("GDAL_DATA"):
        candidates.append(os.environ.get("GDAL_DATA"))

    if IS_WINDOWS:
        # Windows conda paths
        candidates.extend([
            os.path.join(ENV_PREFIX, "Library", "share", "gdal"),
            os.path.join(ENV_PREFIX, "Library", "data"),
            os.path.join(sys.prefix, "Library", "share", "gdal"),
        ])
    else:
        # Linux/macOS paths
        candidates.extend([
            os.path.join(ENV_PREFIX, "share", "gdal"),
            os.path.join(ENV_PREFIX, "share", "data"),
            os.path.join(sys.prefix, "share", "gdal"),
            "/usr/share/gdal",
            "/usr/local/share/gdal",
        ])

    # Additional conda-forge paths
    candidates.extend([
        os.path.join(ENV_PREFIX, "share", "gdal"),
        os.path.join(ENV_PREFIX, "share", "data"),
    ])

    return candidates


def _resolve_gdal_data() -> str:
    """
    Resolve a valid GDAL_DATA path that contains required files.

    Searches for gdalvrt.xsd as a marker for valid GDAL data directory.

    Returns:
        str: Valid GDAL_DATA path, or None if not found.
    """
    candidates = _get_gdal_candidates()

    for path in candidates:
        if not path:
            continue

        # Check for marker file
        xsd = os.path.join(path, "gdalvrt.xsd")
        if os.path.isfile(xsd):
            os.environ["GDAL_DATA"] = path
            return path

    return None


def _setdefault_env_dir(name: str, path: str) -> bool:
    """Set ``name`` to ``path`` only when the user has not set it and the directory exists."""
    if os.environ.get(name):
        return True
    if path and os.path.isdir(path):
        os.environ[name] = path
        return True
    return False


def _setup_gdal_windows():
    """Configure GDAL paths for Windows."""
    if not IS_CONDA_ENV:
        return
    _setdefault_env_dir(
        "GDAL_DRIVER_PATH", os.path.join(ENV_PREFIX, "Library", "lib", "gdalplugins")
    )
    _setdefault_env_dir("PROJ_LIB", os.path.join(ENV_PREFIX, "Library", "share", "proj"))
    warp_exe = os.path.join(ENV_PREFIX, "Library", "bin", f"gdalwarp{EXE_EXT}")
    if not os.environ.get("GDAL_WARP_EXE") and os.path.isfile(warp_exe):
        os.environ["GDAL_WARP_EXE"] = warp_exe


def _setup_gdal_unix():
    """Configure GDAL paths for Linux/macOS."""
    # Only point at an active conda environment. System-wide fallbacks such as
    # /usr/share/proj belong to a different PROJ build than the one bundled with
    # rasterio/pyproj wheels and break every CRS lookup; a system GDAL already knows
    # its own default paths.
    if not IS_CONDA_ENV:
        return
    _setdefault_env_dir("GDAL_DRIVER_PATH", os.path.join(ENV_PREFIX, "lib", "gdalplugins"))
    _setdefault_env_dir("PROJ_LIB", os.path.join(ENV_PREFIX, "share", "proj"))

    # gdalwarp is resolved from GDAL_WARP_EXE (if the user set it) or PATH;
    # see hypercoreg.utils.resolve_gdalwarp_exe.


def setup_gdal_environment():
    """
    Configure GDAL environment variables for the current platform.

    This function should be called early, before importing rasterio
    or other GDAL-dependent libraries.
    """
    # Platform-specific setup
    if IS_WINDOWS:
        _setup_gdal_windows()
    else:
        _setup_gdal_unix()

    # Multithreaded JPEG2000 decoding (Sentinel-2 bands) and GeoTIFF compression.
    # Batch runs with several worker processes scale this down per worker.
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

    # Resolve GDAL_DATA
    gdal_data = _resolve_gdal_data()

    if gdal_data:
        # Suppress some GDAL warnings
        os.environ.setdefault("GDAL_PAM_ENABLED", "YES")
        os.environ.setdefault("CPL_LOG", "/dev/null" if not IS_WINDOWS else "NUL")


# Auto-run setup on import
setup_gdal_environment()
