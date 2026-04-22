#!/usr/bin/env python
"""
HyperCoreg GUI Launcher
=======================

Launch the graphical user interface for HyperCoreg coregistration.

Usage:
    python runGUI.py

The GUI will guide you through:
1. Selecting single file or batch mode
2. Choosing input file/folder and output directory
3. Configuring processing parameters    
4. Running the coregistration

For command-line usage, see runCLI.py
"""

import sys

# Ensure the package is importable
if __name__ == "__main__":
    # Add parent directory to path if running directly
    import os
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from hypercoreg.gui import main

if __name__ == "__main__":
    sys.exit(main())