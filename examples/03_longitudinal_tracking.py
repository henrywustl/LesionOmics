#!/usr/bin/env python3
"""Example 3 — Longitudinal lesion tracking.

Demonstrates how to match lesions between two co-registered timepoints
and classify each match as: stable / split / merge / new / resolved.

Key concepts
------------
* Both segmentations must already be in the same voxel space.
  (Register the follow-up image to the baseline before calling label_lesions.)
* match_lesions_longitudinal() uses voxel-level IoU. Pairs below min_iou
  (default 0.1) are classified as independent new/resolved events.
* Classification is contiguous-lesion-aware: sub-lesions from the same
  contiguous region are grouped together when detecting split/merge events.

Output columns
--------------
  tp1_label, tp1_confluent_lesion_id, tp1_sub_label, tp1_display_id
  tp2_label, tp2_confluent_lesion_id, tp2_sub_label, tp2_display_id
  overlap_vox, iou, match_type

Run from the repo root:
    python examples/03_longitudinal_tracking.py
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from lesionomics.label_lesions import label_lesions
from lesionomics.tracking import match_lesions_longitudinal


# ---------------------------------------------------------------------------
# Synthetic scene — two timepoints in the same voxel space
# ---------------------------------------------------------------------------

shape = (60, 20, 20)
affine = np.eye(4)


def _box(data, slices, val=1.0):
    data[slices] = val
    return data


# TP1: three lesions
data_tp1 = np.zeros(shape, dtype=np.float32)
_box(data_tp1, np.s_[2:8, 5:12, 5:12])    # lesion A — persists (stable)
_box(data_tp1, np.s_[20:26, 5:12, 5:12])  # lesion B — resolves at TP2
_box(data_tp1, np.s_[40:46, 5:12, 5:12])  # lesion C — grows and merges at TP2

# TP2: lesion B resolved, new lesion D, lesions A+C coalesced (merged)
data_tp2 = np.zeros(shape, dtype=np.float32)
_box(data_tp2, np.s_[2:8, 5:12, 5:12])    # lesion A — stable
_box(data_tp2, np.s_[36:50, 4:14, 4:14])  # lesions C + new territory (merged)
_box(data_tp2, np.s_[50:55, 5:10, 5:10])  # new lesion D


result_tp1 = label_lesions(nib.Nifti1Image(data_tp1, affine), threshold=0.5)
result_tp2 = label_lesions(nib.Nifti1Image(data_tp2, affine), threshold=0.5)

print(f"TP1: {result_tp1.n_lesions} lesion(s)")
print(result_tp1.hierarchy.to_string(index=False))
print(f"\nTP2: {result_tp2.n_lesions} lesion(s)")
print(result_tp2.hierarchy.to_string(index=False))


# ---------------------------------------------------------------------------
# Match across timepoints
# ---------------------------------------------------------------------------

matches = match_lesions_longitudinal(
    result_tp1,
    result_tp2,
    min_iou=0.1,   # lower → more matches captured; higher → stricter criterion
)

print("\n=== Longitudinal matches ===")
print(matches.to_string(index=False))

print("\nSummary by match_type:")
print(matches["match_type"].value_counts().to_string())


# ---------------------------------------------------------------------------
# Filtering examples
# ---------------------------------------------------------------------------

resolved = matches[matches["match_type"] == "resolved"]
new_lesions = matches[matches["match_type"] == "new"]
stable = matches[matches["match_type"] == "stable"]

print(f"\nResolved lesions : {len(resolved)}")
for _, r in resolved.iterrows():
    print(f"  {r['tp1_display_id']} (label {r['tp1_label']}) resolved")

print(f"\nNew lesions      : {len(new_lesions)}")
for _, r in new_lesions.iterrows():
    print(f"  {r['tp2_display_id']} (label {r['tp2_label']}) appeared")

print(f"\nStable lesions   : {len(stable)}")
for _, r in stable.iterrows():
    print(f"  {r['tp1_display_id']} → {r['tp2_display_id']}  IoU={r['iou']:.2f}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out = Path("longitudinal_matches.csv")
matches.to_csv(out, index=False)
print(f"\nSaved → {out}")
