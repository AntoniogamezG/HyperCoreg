#!/usr/bin/env python
"""
HyperCoreg CLI Launcher
=======================

Command-line interface for HyperCoreg coregistration.

Usage:
    python runCLI.py --help
    python runCLI.py single --input file.he5 --output ./results
    python runCLI.py batch --input ./prisma_folder --output ./results

Examples:
    # Process a single PRISMA file
    python runCLI.py single -i PRS_L2D_STD_20230615.he5 -o ./output

    # Process a single EnMAP file (.tif or .bsq)
    python runCLI.py single -i ENMAP01-SPECTRAL_IMAGE.TIF -o ./output
    python runCLI.py single -i ENMAP01-SPECTRAL_IMAGE.BSQ -o ./output

    # Batch process a folder
    python runCLI.py batch -i ./prisma_images -o ./output

    # With custom parameters
    python runCLI.py single -i image.he5 -o ./output \\
        --days-window 60 \\
        --max-cloud 30 \\
        --min-tie-points 15 \\
        --verbose

For GUI mode, see runGUI.py
"""

import sys

# Ensure the package is importable
if __name__ == "__main__":
    # Add parent directory to path if running directly
    import os
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from hypercoreg.cli import main

if __name__ == "__main__":
    sys.exit(main())
