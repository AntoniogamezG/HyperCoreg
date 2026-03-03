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


def _setup_gdal_windows():
    """Configure GDAL paths for Windows."""
    os.environ["GDAL_DRIVER_PATH"] = os.path.join(
        ENV_PREFIX, "Library", "lib", "gdalplugins"
    )
    os.environ["PROJ_LIB"] = os.path.join(
        ENV_PREFIX, "Library", "share", "proj"
    )
    os.environ["GDAL_WARP_EXE"] = os.path.join(
        ENV_PREFIX, "Library", "bin", f"gdalwarp{EXE_EXT}"
    )


def _setup_gdal_unix():
    """Configure GDAL paths for Linux/macOS."""
    # GDAL plugins
    plugin_candidates = [
        os.path.join(ENV_PREFIX, "lib", "gdalplugins"),
        "/usr/lib/gdalplugins",
        "/usr/local/lib/gdalplugins",
    ]
    for path in plugin_candidates:
        if os.path.isdir(path):
            os.environ["GDAL_DRIVER_PATH"] = path
            break

    # PROJ library data
    proj_candidates = [
        os.path.join(ENV_PREFIX, "share", "proj"),
        "/usr/share/proj",
        "/usr/local/share/proj",
    ]
    for path in proj_candidates:
        if os.path.isdir(path):
            os.environ["PROJ_LIB"] = path
            break

    # gdalwarp executable
    os.environ["GDAL_WARP_EXE"] = "gdalwarp"


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

    # Resolve GDAL_DATA
    gdal_data = _resolve_gdal_data()

    if gdal_data:
        # Suppress some GDAL warnings
        os.environ.setdefault("GDAL_PAM_ENABLED", "YES")
        os.environ.setdefault("CPL_LOG", "/dev/null" if not IS_WINDOWS else "NUL")


# Auto-run setup on import
setup_gdal_environment()
