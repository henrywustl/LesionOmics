#!/usr/bin/env python3
"""Backward-compatible shim for ``lesionomics compare-splitting``.

Prefer the installed CLI:

    lesionomics compare-splitting --seg data/truenet.nii.gz --output-dir splitting_comparison/

This wrapper forwards to the same code so existing ``python compare_splitting.py ...``
calls keep working.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lesionomics.cli import main

if __name__ == "__main__":
    sys.exit(main(["compare-splitting", *sys.argv[1:]]))
