"""Shared I/O helpers for the LesionOmics command-line tools.

Keeps NIfTI loading and ``NAME=PATH`` modality parsing in one place so every
sub-command behaves identically.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import nibabel as nib

logger = logging.getLogger(__name__)


def load_nifti(path: str | Path, *, required: bool = True):
    """Load a NIfTI image.

    Args:
        path:     Path to the .nii / .nii.gz file.
        required: If True, exit the program when the file is missing.
                  If False, return None instead.

    Returns:
        ``nib.Nifti1Image`` or ``None`` (when not required and absent).
    """
    p = Path(path)
    if not p.exists():
        if required:
            logger.error("File not found: %s", p)
            sys.exit(1)
        logger.warning("File not found, skipping: %s", p)
        return None
    logger.info("Loading: %s", p)
    return nib.load(p)


def parse_modalities(items: list[str]) -> dict:
    """Parse a list of ``NAME=PATH`` strings into ``{name: Nifti1Image}``.

    Malformed entries and missing files are skipped with a warning rather than
    aborting, so a single bad path doesn't sink the whole run.
    """
    modalities: dict = {}
    for item in items or []:
        if "=" not in item:
            logger.warning("Modality must be NAME=PATH, skipping: %s", item)
            continue
        name, path = item.split("=", 1)
        img = load_nifti(path, required=False)
        if img is not None:
            modalities[name] = img
    return modalities
