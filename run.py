#!/usr/bin/env python3
"""Backward-compatible shim for ``lesionomics quantify``.

Prefer the installed CLI:

    lesionomics quantify --seg data/truenet.nii.gz --modalities T1=data/T1.nii.gz

This wrapper forwards to the same code so existing ``python run.py ...`` calls
keep working. Note: the legacy ``--log`` flag is now ``--log-filter``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lesionomics.cli import main

if __name__ == "__main__":
    sys.exit(main(["quantify", *sys.argv[1:]]))
