import numpy as np
import nibabel as nib
import pytest
import pandas as pd

from src.label_lesions import label_lesions, LabeledResult
from src.tracking import match_lesions_longitudinal


def _make_img(data, affine=None):
    if affine is None:
        affine = np.eye(4)
    hdr = nib.Nifti1Header()
    hdr.set_data_shape(data.shape)
    hdr.set_zooms((1.0, 1.0, 1.0))
    return nib.Nifti1Image(data.astype(np.float32), affine, hdr)


def _lesion_img(shape, lesion_slices):
    data = np.zeros(shape, dtype=np.float32)
    for s in lesion_slices:
        data[s] = 1.0
    return _make_img(data)


class TestMatchLesionsLongitudinal:
    def test_returns_dataframe(self):
        img = _lesion_img((30, 20, 20), [
            np.s_[2:6, 5:10, 5:10],
            np.s_[15:19, 5:10, 5:10],
        ])
        r1 = label_lesions(img, threshold=0.5)
        r2 = label_lesions(img, threshold=0.5)
        df = match_lesions_longitudinal(r1, r2)
        assert isinstance(df, pd.DataFrame)

    def test_identical_images_all_stable(self):
        img = _lesion_img((30, 20, 20), [
            np.s_[2:8, 5:12, 5:12],
            np.s_[15:20, 5:12, 5:12],
        ])
        r1 = label_lesions(img, threshold=0.5)
        r2 = label_lesions(img, threshold=0.5)
        df = match_lesions_longitudinal(r1, r2)
        assert set(df["match_type"].unique()) == {"stable"}

    def test_resolved_lesion_detected(self):
        # TP1 has two lesions; TP2 has only one (second lesion resolved)
        shape = (40, 20, 20)
        data_tp1 = np.zeros(shape, dtype=np.float32)
        data_tp1[2:7, 5:12, 5:12] = 1.0
        data_tp1[25:30, 5:12, 5:12] = 1.0
        data_tp2 = np.zeros(shape, dtype=np.float32)
        data_tp2[2:7, 5:12, 5:12] = 1.0  # only first lesion remains

        r1 = label_lesions(_make_img(data_tp1), threshold=0.5)
        r2 = label_lesions(_make_img(data_tp2), threshold=0.5)
        df = match_lesions_longitudinal(r1, r2)
        assert "resolved" in df["match_type"].values

    def test_new_lesion_detected(self):
        shape = (40, 20, 20)
        data_tp1 = np.zeros(shape, dtype=np.float32)
        data_tp1[2:7, 5:12, 5:12] = 1.0
        data_tp2 = np.zeros(shape, dtype=np.float32)
        data_tp2[2:7, 5:12, 5:12] = 1.0
        data_tp2[25:30, 5:12, 5:12] = 1.0  # new lesion at TP2

        r1 = label_lesions(_make_img(data_tp1), threshold=0.5)
        r2 = label_lesions(_make_img(data_tp2), threshold=0.5)
        df = match_lesions_longitudinal(r1, r2)
        assert "new" in df["match_type"].values

    def test_display_ids_present(self):
        img = _lesion_img((30, 20, 20), [np.s_[2:8, 5:12, 5:12]])
        r = label_lesions(img, threshold=0.5)
        df = match_lesions_longitudinal(r, r)
        assert "tp1_display_id" in df.columns
        assert "tp2_display_id" in df.columns
        assert df["tp1_display_id"].iloc[0] == "L1"

    def test_empty_tp2_all_resolved(self):
        shape = (30, 20, 20)
        data = np.zeros(shape, dtype=np.float32)
        data[2:8, 5:12, 5:12] = 1.0
        r1 = label_lesions(_make_img(data), threshold=0.5)
        r2 = label_lesions(_make_img(np.zeros(shape, dtype=np.float32)), threshold=0.5)
        df = match_lesions_longitudinal(r1, r2)
        assert all(df["match_type"] == "resolved")

    def test_match_columns_present(self):
        img = _lesion_img((30, 20, 20), [np.s_[2:8, 5:12, 5:12]])
        r = label_lesions(img, threshold=0.5)
        df = match_lesions_longitudinal(r, r)
        expected = [
            "tp1_label", "tp1_confluent_lesion_id", "tp1_sub_label", "tp1_display_id",
            "tp2_label", "tp2_confluent_lesion_id", "tp2_sub_label", "tp2_display_id",
            "overlap_vox", "iou", "match_type",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"
