"""
HyperCoreg - Automated hyperspectral PRISMA/EnMAP to Sentinel-2 coregistration
===============================================================================

HyperCoreg is a Python tool that automatically aligns hyperspectral satellite
imagery (PRISMA, EnMAP) to geometrically accurate Sentinel-2 reference data.

Basic usage:
    >>> from hypercoreg import run_coregistration
    >>> from hypercoreg.utils import detect_hyp_type
    >>> hyp_type = detect_hyp_type("image.he5")
    >>> metrics = run_coregistration("image.he5", hyp_type, "./output", config)

For GUI mode:
    $ python runGUI.py

For CLI mode:
    $ python runCLI.py single -i image.he5 -o ./output
"""

from hypercoreg._version import __version__, __version_info__

# Import GDAL setup first (must run before other geospatial imports)
from hypercoreg import gdal_setup  # noqa: F401

# Main API
from hypercoreg.coregistration import run_coregistration, run_batch_coregistration
from hypercoreg.utils import detect_hyp_type, SentinelNotFoundError
from hypercoreg.spectral import SpectralBandInfo, SpectralBandTable

__all__ = [
    "__version__",
    "__version_info__",
    "run_coregistration",
    "run_batch_coregistration",
    "detect_hyp_type",
    "SentinelNotFoundError",
    "SpectralBandInfo",
    "SpectralBandTable",
]
