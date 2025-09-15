#!/usr/bin/env python3
"""
Main entry point for cuBLASter application.
This provides the same interface as BLASter but with CUDA acceleration.
"""

import sys
from pathlib import Path

# Add the cublaster module to the path
sys.path.insert(0, str(Path(__file__).parent))

from cublaster.app import main

if __name__ == "__main__":
    sys.exit(main())