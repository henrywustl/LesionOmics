#!/usr/bin/env python3
"""Example 1 — Basic lesion labeling and quantification.

Demonstrates the minimal workflow:
  1. Load a lesion probability map (or binary mask)
  2. Label connected components with label_lesions()
  3. Quantify shape and intensity features with quantify_lesions()
  4. Save results to CSV

Run from the repo root:
    python examples/01_basic_quantification.py

If real data files are absent a small synthetic scene is created
so the script always runs end-to-end.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.label_lesions import label_lesions
from src.quantifying_lesions import quantify_lesions


# ---------------------------------------------------------------------------
# Data — use real files when available, otherwise synthesise a scene
# ---------------------------------------------------------------------------

def _make_synthetic():
    shape = (40, 30, 30)
    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    hdr = nib.Nifti1Header()
    hdr.set_data_shape(shape)
    hdr.set_zooms((1.5, 1.5, 1.5))

    seg = np.zeros(shape, dtype=np.float32)
    seg[4:10, 8:14, 8:14] = 0.9   # large lesion
    seg[20:24, 8:12, 8:12] = 0.7  # medium lesion
    seg[30:33, 5:8, 5:8] = 0.5    # small lesion

    t1 = np.ones(shape, dtype=np.float32) * 800
    t1[seg > 0] = 400  # lesions are hypointense on T1

    t2 = np.ones(shape, dtype=np.float32) * 500
    t2[seg > 0] = 900  # lesions are hyperintense on T2

    return (
        nib.Nifti1Image(seg, affine, hdr),
        nib.Nifti1Image(t1, affine, hdr),
        nib.Nifti1Image(t2, affine, hdr),
    )


seg_path = Path("data/truenet.nii.gz")
t1_path = Path("data/T1.nii.gz")
t2_path = Path("data/T2.nii.gz")

if seg_path.exists() and t1_path.exists():
    print("Loading real data …")
    seg_img = nib.load(seg_path)
    modalities = {"T1": nib.load(t1_path)}
    if t2_path.exists():
        modalities["T2"] = nib.load(t2_path)
else:
    print("Data files not found — using synthetic scene.")
    seg_img, t1_img, t2_img = _make_synthetic()
    modalities = {"T1": t1_img, "T2": t2_img}


# ---------------------------------------------------------------------------
# Step 1 — label lesions
# ---------------------------------------------------------------------------

result = label_lesions(seg_img, threshold=0.3)

print(f"\nFound {result.n_lesions} lesion(s) across "
      f"{result.n_confluent_lesions} contiguous region(s).")

# LabeledResult exposes:
#   result.label_img          — NIfTI: flat integer labels 1..N
#   result.confluent_lesion_img — NIfTI: pre-split contiguous lesion labels
#   result.hierarchy          — DataFrame: label / confluent_lesion_id /
#                               sub_label / display_id
#   result.n_lesions          — total labeled lesions (post-split)
#   result.n_confluent_lesions — contiguous connected components (pre-split)

print("\nHierarchy table:")
print(result.hierarchy.to_string(index=False))


# ---------------------------------------------------------------------------
# Step 2 — quantify
# ---------------------------------------------------------------------------

df = quantify_lesions(
    labeled_img=result,
    modalities=modalities,
    include_radiomics=False,   # set True to add PyRadiomics texture features
)

print(f"\nFeature matrix: {len(df)} lesions × {len(df.columns)} columns")
print(df[["display_id", "volume_mm3", "sphericity"]].to_string(index=False))


# ---------------------------------------------------------------------------
# Step 3 — save
# ---------------------------------------------------------------------------

out = Path("lesion_measurements.csv")
df.to_csv(out, index=False)
print(f"\nSaved → {out}")
