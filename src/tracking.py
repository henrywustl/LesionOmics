"""Longitudinal lesion tracking via spatial overlap matching.

Matches lesions between two co-registered LabeledResult objects from
different timepoints using voxel-level IoU, then classifies each match:

  stable    — 1-to-1 spatial correspondence above min_iou
  split     — one TP1 blob maps to ≥2 TP2 blobs (blob grew apart)
  merge     — ≥2 TP1 blobs map to one TP2 blob (lesions coalesced)
  new       — present at TP2 only (new lesion)
  resolved  — present at TP1 only (lesion resolved)

Both label images must already be in the same voxel space (i.e. the
follow-up segmentation has been co-registered to the baseline before
calling label_lesions, or the label arrays have been resampled to a
common grid).

Typical usage
-------------
>>> from src.label_lesions import label_lesions
>>> from src.tracking import match_lesions_longitudinal
>>>
>>> r_tp1 = label_lesions(seg_tp1)
>>> r_tp2 = label_lesions(seg_tp2_registered_to_tp1_space)
>>> matches = match_lesions_longitudinal(r_tp1, r_tp2)
>>> matches.to_csv("longitudinal_matches.csv", index=False)
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .label_lesions import LabeledResult

logger = logging.getLogger(__name__)

_MATCH_COLS = [
    "tp1_label", "tp1_confluent_lesion_id", "tp1_sub_label", "tp1_display_id",
    "tp2_label", "tp2_confluent_lesion_id", "tp2_sub_label", "tp2_display_id",
    "overlap_vox", "iou", "match_type",
]


def _label_volumes(arr: np.ndarray) -> dict:
    labels, counts = np.unique(arr[arr > 0], return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, counts)}


def _overlap_matrix(arr1: np.ndarray, arr2: np.ndarray) -> pd.DataFrame:
    fg = (arr1 > 0) | (arr2 > 0)
    v1 = arr1[fg].astype(np.int64)
    v2 = arr2[fg].astype(np.int64)
    max2 = int(arr2.max()) + 1
    pairs = v1 * max2 + v2
    n_buckets = (int(arr1.max()) + 1) * max2
    counts = np.bincount(pairs, minlength=n_buckets).reshape(int(arr1.max()) + 1, max2)
    rows = [
        {"label_tp1": l1, "label_tp2": l2, "overlap_vox": int(counts[l1, l2])}
        for l1 in range(1, int(arr1.max()) + 1)
        for l2 in range(1, int(arr2.max()) + 1)
        if counts[l1, l2] > 0
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["label_tp1", "label_tp2", "overlap_vox"])


def _h(hierarchy: pd.DataFrame, label: int, col: str, default=None):
    if hierarchy.empty or label not in hierarchy.index:
        return default
    return hierarchy.at[label, col]


def match_lesions_longitudinal(
    result_tp1: "LabeledResult",
    result_tp2: "LabeledResult",
    min_iou: float = 0.1,
) -> pd.DataFrame:
    """Match lesions between two co-registered timepoints.

    Args:
        result_tp1: LabeledResult from the baseline timepoint.
        result_tp2: LabeledResult from the follow-up timepoint, already in
                    the same voxel space as result_tp1.
        min_iou:    Minimum IoU to consider two lesions as overlapping
                    (default 0.1). Pairs below this are treated as
                    independent new/resolved lesions.

    Returns:
        DataFrame with columns:
          tp1_label, tp1_confluent_lesion_id, tp1_sub_label, tp1_display_id,
          tp2_label, tp2_confluent_lesion_id, tp2_sub_label, tp2_display_id,
          overlap_vox, iou, match_type

        match_type is one of: "stable", "split", "merge", "new", "resolved".

        Classification is contiguous-lesion-aware: split sub-lesions from the
        same contiguous lesion are treated as a unit when deciding whether a
        contiguous lesion split or merged.
    """
    arr1 = result_tp1.label_img.get_fdata().astype(np.int32)
    arr2 = result_tp2.label_img.get_fdata().astype(np.int32)
    vol1 = _label_volumes(arr1)
    vol2 = _label_volumes(arr2)
    h1 = result_tp1.hierarchy.set_index("label") if not result_tp1.hierarchy.empty else pd.DataFrame()
    h2 = result_tp2.hierarchy.set_index("label") if not result_tp2.hierarchy.empty else pd.DataFrame()

    if not vol1 and not vol2:
        return pd.DataFrame(columns=_MATCH_COLS)

    ovlp = _overlap_matrix(arr1, arr2)
    if not ovlp.empty:
        ovlp["iou"] = ovlp.apply(
            lambda r: r["overlap_vox"] / (
                vol1.get(int(r["label_tp1"]), 0)
                + vol2.get(int(r["label_tp2"]), 0)
                - r["overlap_vox"]
            ), axis=1,
        )
        ovlp = ovlp[ovlp["iou"] >= min_iou].copy()

    rows = []
    matched_tp1: set = set()
    matched_tp2: set = set()

    if not ovlp.empty:
        ovlp["cl_tp1"] = ovlp["label_tp1"].map(lambda l: _h(h1, int(l), "confluent_lesion_id", int(l)))
        ovlp["cl_tp2"] = ovlp["label_tp2"].map(lambda l: _h(h2, int(l), "confluent_lesion_id", int(l)))
        cl1_to_cls2 = ovlp.groupby("cl_tp1")["cl_tp2"].apply(set).to_dict()
        cl2_to_cls1 = ovlp.groupby("cl_tp2")["cl_tp1"].apply(set).to_dict()

        for _, row in ovlp.iterrows():
            l1, l2 = int(row["label_tp1"]), int(row["label_tp2"])
            b1, b2 = int(row["cl_tp1"]), int(row["cl_tp2"])
            matched_tp1.add(l1)
            matched_tp2.add(l2)
            n2 = len(cl1_to_cls2.get(b1, set()))
            n1 = len(cl2_to_cls1.get(b2, set()))
            if n2 > 1 and n1 == 1:
                match_type = "split"
            elif n1 > 1:
                match_type = "merge"
            else:
                match_type = "stable"
            rows.append({
                "tp1_label": l1,
                "tp1_confluent_lesion_id": _h(h1, l1, "confluent_lesion_id"),
                "tp1_sub_label": _h(h1, l1, "sub_label"),
                "tp1_display_id": _h(h1, l1, "display_id", f"L{l1}"),
                "tp2_label": l2,
                "tp2_confluent_lesion_id": _h(h2, l2, "confluent_lesion_id"),
                "tp2_sub_label": _h(h2, l2, "sub_label"),
                "tp2_display_id": _h(h2, l2, "display_id", f"L{l2}"),
                "overlap_vox": int(row["overlap_vox"]),
                "iou": float(row["iou"]), "match_type": match_type,
            })

    for l1 in vol1:
        if l1 not in matched_tp1:
            rows.append({
                "tp1_label": l1,
                "tp1_confluent_lesion_id": _h(h1, l1, "confluent_lesion_id"),
                "tp1_sub_label": _h(h1, l1, "sub_label"),
                "tp1_display_id": _h(h1, l1, "display_id", f"L{l1}"),
                "tp2_label": None, "tp2_confluent_lesion_id": None,
                "tp2_sub_label": None, "tp2_display_id": None,
                "overlap_vox": 0, "iou": 0.0, "match_type": "resolved",
            })

    for l2 in vol2:
        if l2 not in matched_tp2:
            rows.append({
                "tp1_label": None, "tp1_confluent_lesion_id": None,
                "tp1_sub_label": None, "tp1_display_id": None,
                "tp2_label": l2,
                "tp2_confluent_lesion_id": _h(h2, l2, "confluent_lesion_id"),
                "tp2_sub_label": _h(h2, l2, "sub_label"),
                "tp2_display_id": _h(h2, l2, "display_id", f"L{l2}"),
                "overlap_vox": 0, "iou": 0.0, "match_type": "new",
            })

    return pd.DataFrame(rows, columns=_MATCH_COLS) if rows else pd.DataFrame(columns=_MATCH_COLS)
