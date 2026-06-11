#!/usr/bin/env python3
"""Example 4 — Full pipeline: split + quantify + track.

Puts everything together in the order you would use it on real data:

  1. Label and split confluent lesions at each timepoint
  2. Quantify per-lesion features (shape, intensity, texture)
  3. Match lesions longitudinally and attach quantitative changes

Run from the repo root:
    python examples/04_full_pipeline.py

Real-data usage (co-register TP2 to TP1 space first):
    python examples/04_full_pipeline.py \
        --seg-tp1 data/seg_tp1.nii.gz  --t1-tp1 data/T1_tp1.nii.gz \
        --seg-tp2 data/seg_tp2.nii.gz  --t1-tp2 data/T1_tp2.nii.gz \
        --split-method watershed
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lesionomics.label_lesions import label_lesions
from lesionomics.quantifying_lesions import quantify_lesions
from lesionomics.tracking import match_lesions_longitudinal


# ---------------------------------------------------------------------------
# CLI (all arguments optional — falls back to synthetic data)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Full LesionOmics pipeline example.")
    p.add_argument("--seg-tp1", default=None)
    p.add_argument("--t1-tp1", default=None)
    p.add_argument("--seg-tp2", default=None)
    p.add_argument("--t1-tp2", default=None)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument(
        "--split-method",
        choices=["erosion", "watershed", "random_walker"],
        default="watershed",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic scene helpers
# ---------------------------------------------------------------------------

def _synthetic_timepoint(seed: int):
    """Return (seg_img, t1_img) for a small synthetic scene."""
    rng = np.random.default_rng(seed)
    shape = (50, 30, 30)
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    hdr = nib.Nifti1Header()
    hdr.set_data_shape(shape)
    hdr.set_zooms((1.5, 1.5, 1.5))

    seg = np.zeros(shape, dtype=np.float32)
    # Two spheres close together — they form a single contiguous lesion
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]]
    for cz in (10, 18):
        seg[(z - cz) ** 2 + (y - 15) ** 2 + (x - 15) ** 2 <= 16] = 0.85 + rng.uniform(-0.05, 0.05)
    # Isolated lesion
    seg[35:39, 12:16, 12:16] = 0.7 + rng.uniform(-0.05, 0.05)

    t1 = np.ones(shape, dtype=np.float32) * 850
    t1[seg > 0] = 380 + rng.uniform(-20, 20, size=shape)[seg > 0]

    return nib.Nifti1Image(seg, affine, hdr), nib.Nifti1Image(t1, affine, hdr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Load or synthesise data -------------------------------------------
    if args.seg_tp1 and Path(args.seg_tp1).exists():
        print("Loading real data …")
        seg_tp1 = nib.load(args.seg_tp1)
        t1_tp1 = nib.load(args.t1_tp1)
        seg_tp2 = nib.load(args.seg_tp2)
        t1_tp2 = nib.load(args.t1_tp2)
    else:
        print("No data files supplied — using synthetic scene.")
        seg_tp1, t1_tp1 = _synthetic_timepoint(seed=0)
        seg_tp2, t1_tp2 = _synthetic_timepoint(seed=1)

    # --- Step 1: label + split confluent lesions at each timepoint ----------
    print(f"\n[Step 1] Labeling lesions (split_method='{args.split_method}') …")
    result_tp1 = label_lesions(
        seg_tp1,
        threshold=args.threshold,
        split_confluent=True,
        split_method=args.split_method,
    )
    result_tp2 = label_lesions(
        seg_tp2,
        threshold=args.threshold,
        split_confluent=True,
        split_method=args.split_method,
    )

    print(f"  TP1: {result_tp1.n_confluent_lesions} contiguous region(s), "
          f"{result_tp1.n_lesions} lesion(s) after splitting")
    print(f"  TP2: {result_tp2.n_confluent_lesions} contiguous region(s), "
          f"{result_tp2.n_lesions} lesion(s) after splitting")

    print("\n  TP1 hierarchy:")
    print(result_tp1.hierarchy.to_string(index=False))
    print("\n  TP2 hierarchy:")
    print(result_tp2.hierarchy.to_string(index=False))

    # --- Step 2: quantify features at each timepoint -----------------------
    print("\n[Step 2] Quantifying features …")
    df_tp1 = quantify_lesions(
        labeled_img=result_tp1,
        modalities={"T1": t1_tp1},
        include_radiomics=False,
    )
    df_tp2 = quantify_lesions(
        labeled_img=result_tp2,
        modalities={"T1": t1_tp2},
        include_radiomics=False,
    )
    print(f"  TP1: {len(df_tp1)} rows × {len(df_tp1.columns)} columns")
    print(f"  TP2: {len(df_tp2)} rows × {len(df_tp2.columns)} columns")

    df_tp1.to_csv("features_tp1.csv", index=False)
    df_tp2.to_csv("features_tp2.csv", index=False)

    # --- Step 3: longitudinal matching -------------------------------------
    print("\n[Step 3] Matching lesions across timepoints …")
    matches = match_lesions_longitudinal(result_tp1, result_tp2, min_iou=0.1)

    print("\nMatch results:")
    print(matches[["tp1_display_id", "tp2_display_id", "iou", "match_type"]].to_string(index=False))

    print("\nSummary:")
    print(matches["match_type"].value_counts().to_string())

    # --- Attach volume change to matched pairs -----------------------------
    vol_tp1 = df_tp1.set_index("display_id")["volume_mm3"].rename("vol_tp1_mm3")
    vol_tp2 = df_tp2.set_index("display_id")["volume_mm3"].rename("vol_tp2_mm3")

    matches = matches.join(vol_tp1, on="tp1_display_id").join(vol_tp2, on="tp2_display_id")
    matches["delta_volume_mm3"] = matches["vol_tp2_mm3"] - matches["vol_tp1_mm3"]

    stable = matches[matches["match_type"] == "stable"]
    if not stable.empty:
        print("\nVolume change for stable lesions:")
        print(stable[["tp1_display_id", "vol_tp1_mm3", "vol_tp2_mm3", "delta_volume_mm3"]].to_string(index=False))

    matches.to_csv("longitudinal_matches_with_volumes.csv", index=False)
    print("\nSaved: features_tp1.csv, features_tp2.csv, longitudinal_matches_with_volumes.csv")


if __name__ == "__main__":
    main()
