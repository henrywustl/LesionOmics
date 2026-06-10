#!/usr/bin/env python3
"""Example 2 — Confluent lesion splitting.

Shows how to use each of the three splitting algorithms and how the
lesion hierarchy reflects the split:

  erosion     — iterative morphological erosion → Voronoi expansion
  watershed   — distance-transform watershed
  random_walker — erosion-derived seeds + random-walk boundary placement

The sub-lesion naming convention is:
  * Isolated (unsplit) contiguous lesion  → display_id = "L1", "L2", …
  * Split sub-lesions from contiguous lesion 2 → "L2A", "L2B", "L2C", …

Run from the repo root:
    python examples/02_confluent_lesion_splitting.py
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from lesionomics.label_lesions import label_lesions


# ---------------------------------------------------------------------------
# Build a synthetic scene with two merged spheres (one confluent lesion)
# plus one isolated lesion
# ---------------------------------------------------------------------------

def _two_merged_spheres(shape=(30, 20, 20), r=4, sep=8):
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    cy, cx = shape[1] // 2, shape[2] // 2
    c1 = (z - (shape[0] // 2 - sep // 2)) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2
    c2 = (z - (shape[0] // 2 + sep // 2)) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2
    return (c1 | c2).astype(np.float32)


shape = (60, 20, 20)
affine = np.eye(4)

data = np.zeros(shape, dtype=np.float32)
data[:30] = _two_merged_spheres((30, 20, 20))   # confluent lesion
data[40:44, 8:12, 8:12] = 1.0                   # isolated lesion

seg_img = nib.Nifti1Image(data, affine)


# ---------------------------------------------------------------------------
# Label WITHOUT splitting (default)
# ---------------------------------------------------------------------------

result_no_split = label_lesions(seg_img, threshold=0.5, split_confluent=False)

print("=== No splitting ===")
print(f"  Contiguous regions : {result_no_split.n_confluent_lesions}")
print(f"  Labeled lesions    : {result_no_split.n_lesions}")
print(result_no_split.hierarchy.to_string(index=False))


# ---------------------------------------------------------------------------
# Label WITH each splitting method
# ---------------------------------------------------------------------------

for method in ("erosion", "watershed", "random_walker"):
    result = label_lesions(
        seg_img,
        threshold=0.5,
        split_confluent=True,
        split_method=method,          # "erosion" | "watershed" | "random_walker"
    )
    print(f"\n=== split_method='{method}' ===")
    print(f"  Contiguous regions : {result.n_confluent_lesions}")
    print(f"  Labeled lesions    : {result.n_lesions}")
    print(result.hierarchy.to_string(index=False))
    # Expected output for both watershed and erosion (scene is clean):
    #   label  confluent_lesion_id sub_label display_id
    #       1                   1         A        L1A   ← sub-lesion A of confluent region 1
    #       2                   1         B        L1B   ← sub-lesion B of confluent region 1
    #       3                   2      None         L2   ← isolated lesion


# ---------------------------------------------------------------------------
# Accessing the contiguous lesion image
# ---------------------------------------------------------------------------

result = label_lesions(seg_img, threshold=0.5, split_confluent=True)

# confluent_lesion_img: each contiguous connected component has a single label
# regardless of whether it was later split. Use this for spatial grouping.
cl_arr = result.confluent_lesion_img.get_fdata().astype(int)
print(f"\ncontiguous_lesion_img unique labels: {np.unique(cl_arr[cl_arr > 0]).tolist()}")

# label_img: post-split flat labels used as input to quantify_lesions()
lbl_arr = result.label_img.get_fdata().astype(int)
print(f"label_img unique labels            : {np.unique(lbl_arr[lbl_arr > 0]).tolist()}")
