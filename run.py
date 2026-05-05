#!/usr/bin/env python3
"""
LesionOmics end-to-end pipeline.

Labels lesions from a probability map, quantifies per-lesion features across
any set of co-registered 3D modality images, and writes a CSV of measurements.

Default usage (sample data in data/):
    python run.py

Custom usage:
    python run.py \\
        --prob-map  path/to/prob.nii.gz \\
        --modalities T1=path/to/T1.nii.gz T2=path/to/T2.nii.gz \\
        --output    lesion_measurements.csv
"""

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from src.label_lesions import label_lesions
from src.quantifying_lesions import quantify_lesions

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def _parse_modality(s: str) -> tuple:
    """Parse 'NAME=path/to/file.nii.gz' into (name, path)."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"Modality must be in NAME=PATH format, got: {s!r}"
        )
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser(
        description="LesionOmics: end-to-end lesion feature extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--prob-map", default=None,
        help="Lesion probability map NIfTI (default: data/truenet.nii.gz)",
    )
    parser.add_argument(
        "--modalities", nargs="+", type=_parse_modality, metavar="NAME=PATH",
        help="Co-registered modality images. "
             "Default: T1=data/T1.nii.gz T2=data/T2.nii.gz",
    )
    parser.add_argument(
        "--output", default="lesion_measurements.csv",
        help="Output CSV path (default: lesion_measurements.csv)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Binarisation threshold for probability map (default: 0.3)",
    )
    parser.add_argument(
        "--split-confluent", action="store_true",
        help="Apply erosion-based confluent lesion splitting before quantification",
    )
    parser.add_argument(
        "--nawm-mask", default=None,
        help="Binary NAWM mask NIfTI for NAWM-normalised intensity columns",
    )
    parser.add_argument(
        "--no-radiomics", action="store_true",
        help="Skip PyRadiomics texture features (faster; use when pyradiomics "
             "is not installed)",
    )
    args = parser.parse_args()

    # ── Resolve inputs ──────────────────────────────────────────────────
    prob_path = (
        Path(args.prob_map) if args.prob_map
        else DEFAULT_DATA_DIR / "truenet.nii.gz"
    )
    if not prob_path.exists():
        sys.exit(f"Error: probability map not found: {prob_path}")

    if args.modalities:
        modality_paths = dict(args.modalities)
    else:
        modality_paths = {
            "T1": str(DEFAULT_DATA_DIR / "T1.nii.gz"),
            "T2": str(DEFAULT_DATA_DIR / "T2.nii.gz"),
        }

    print("=" * 60)
    print("LesionOmics Pipeline")
    print("=" * 60)
    print(f"  Probability map : {prob_path}")
    for name, path in modality_paths.items():
        print(f"  Modality {name:<8}: {path}")
    print(f"  Threshold       : {args.threshold}")
    print(f"  Split confluent : {args.split_confluent}")
    print(f"  Radiomics       : {not args.no_radiomics}")
    print(f"  Output          : {args.output}")
    print()

    # ── Load images ────────────────────────────────────────────────────
    print("[1/3] Loading images...")
    prob_img = nib.load(str(prob_path))

    modalities = {}
    for name, path in modality_paths.items():
        p = Path(path)
        if not p.exists():
            sys.exit(f"Error: modality image not found: {p}")
        modalities[name] = nib.load(str(p))
        print(f"      Loaded {name} from {p}")

    nawm_mask = None
    if args.nawm_mask:
        nawm_path = Path(args.nawm_mask)
        if not nawm_path.exists():
            sys.exit(f"Error: NAWM mask not found: {nawm_path}")
        nawm_mask = nib.load(str(nawm_path))
        print(f"      Loaded NAWM mask from {nawm_path}")

    # ── Step 1: Label lesions ────────────────────────────────────────────
    print(f"\n[2/3] Labelling lesions "
          f"(threshold={args.threshold}, "
          f"split_confluent={args.split_confluent})...")
    t0 = time.perf_counter()
    labeled_img = label_lesions(
        prob_img,
        threshold=args.threshold,
        split_confluent=args.split_confluent,
    )
    n_lesions = int(labeled_img.get_fdata().max())
    print(f"      Found {n_lesions} lesion(s)  [{time.perf_counter() - t0:.1f}s]")

    if n_lesions == 0:
        sys.exit(
            "No lesions found above threshold. "
            "Try lowering --threshold and re-run."
        )

    # ── Step 2: Quantify ────────────────────────────────────────────────
    print(f"\n[3/3] Quantifying {n_lesions} lesion(s)...")
    t0 = time.perf_counter()
    df = quantify_lesions(
        labeled_img=labeled_img,
        modalities=modalities,
        prob_map=prob_img,
        nawm_mask=nawm_mask,
        include_radiomics=not args.no_radiomics,
    )
    elapsed = time.perf_counter() - t0
    print(f"      Done  [{elapsed:.1f}s]")

    # ── Save ───────────────────────────────────────────────────────────
    df.to_csv(args.output, index=False)

    print()
    print("=" * 60)
    print(f"Results: {len(df)} lesions x {len(df.columns)} features")
    print(f"Saved  : {args.output}")
    print("=" * 60)
    print()

    # Volume summary
    print("Volume summary (mm³):")
    vol = df["volume_mm3"]
    print(f"  count : {len(vol)}")
    print(f"  min   : {vol.min():.1f}")
    print(f"  median: {vol.median():.1f}")
    print(f"  mean  : {vol.mean():.1f}")
    print(f"  max   : {vol.max():.1f}")
    print()

    # Top 5 by volume
    print("Top 5 lesions by volume:")
    top = df.nlargest(min(5, len(df)), "volume_mm3")[
        ["label", "volume_mm3", "sphericity", "hemisphere"]
        + (["mean_confidence"] if "mean_confidence" in df.columns else [])
    ]
    print(top.to_string(index=False))
    print()

    return df


if __name__ == "__main__":
    main()
