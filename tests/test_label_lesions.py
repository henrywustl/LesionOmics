"""Tests for src/label_lesions.py"""

import numpy as np
import nibabel as nib
import pytest
from skimage.measure import label as sk_label

from src.label_lesions import (
    erosion_derived_confluent_lesion_splitting,
    label_lesions,
    relabel_from_largest_to_smallest,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _nifti(data, voxel_size=1.0):
    affine = np.eye(4) * voxel_size
    affine[3, 3] = 1.0
    return nib.Nifti1Image(data.astype(np.float32), affine)


# ── relabel_from_largest_to_smallest ────────────────────────────────────────

class TestRelabelFromLargestToSmallest:
    def test_larger_blob_gets_label_1(self):
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[1:8, 1:8, 1:8] = 2    # 343 voxels — larger; initially label 2
        arr[10:13, 10:13, 10:13] = 1  # 27 voxels  — smaller; initially label 1
        result = relabel_from_largest_to_smallest(arr)
        assert result[1, 1, 1] == 1    # formerly-label-2 block is now 1
        assert result[10, 10, 10] == 2  # formerly-label-1 block is now 2

    def test_already_ordered_unchanged(self):
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[1:8, 1:8, 1:8] = 1    # 343 voxels
        arr[10:13, 10:13, 10:13] = 2   # 27 voxels
        result = relabel_from_largest_to_smallest(arr)
        assert result[1, 1, 1] == 1
        assert result[10, 10, 10] == 2

    def test_single_blob_is_label_1(self):
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        arr[2:8, 2:8, 2:8] = 1
        result = relabel_from_largest_to_smallest(arr)
        assert result[5, 5, 5] == 1

    def test_background_stays_zero(self):
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        arr[2:5, 2:5, 2:5] = 1
        result = relabel_from_largest_to_smallest(arr)
        assert result[0, 0, 0] == 0

    def test_voxel_count_preserved(self):
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[1:8, 1:8, 1:8] = 1
        arr[10:13, 10:13, 10:13] = 2
        result = relabel_from_largest_to_smallest(arr)
        assert np.sum(result > 0) == np.sum(arr > 0)


# ── label_lesions ──────────────────────────────────────────────────────────────

class TestLabelLesions:
    def test_two_separate_lesions_give_two_labels(self):
        data = np.zeros((40, 40, 40), dtype=np.float32)
        data[3:13, 3:13, 3:13] = 0.8
        data[25:32, 25:32, 25:32] = 0.8
        result = label_lesions(_nifti(data), threshold=0.5)
        assert int(result.get_fdata().max()) == 2

    def test_threshold_below_values_finds_lesion(self):
        data = np.zeros((20, 20, 20), dtype=np.float32)
        data[5:10, 5:10, 5:10] = 0.4
        result = label_lesions(_nifti(data), threshold=0.3)
        assert result.get_fdata().max() == 1

    def test_threshold_above_values_finds_nothing(self):
        data = np.zeros((20, 20, 20), dtype=np.float32)
        data[5:10, 5:10, 5:10] = 0.4
        result = label_lesions(_nifti(data), threshold=0.5)
        assert result.get_fdata().max() == 0

    def test_larger_lesion_is_label_1(self):
        data = np.zeros((50, 50, 50), dtype=np.float32)
        data[2:12, 2:12, 2:12] = 0.8   # 1000 voxels
        data[30:35, 30:35, 30:35] = 0.8  # 125 voxels
        result = label_lesions(_nifti(data), threshold=0.5)
        rd = result.get_fdata()
        assert rd[2, 2, 2] == 1     # larger blob
        assert rd[30, 30, 30] == 2  # smaller blob

    def test_output_affine_preserved(self):
        data = np.zeros((20, 20, 20), dtype=np.float32)
        data[5:10, 5:10, 5:10] = 0.8
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img = nib.Nifti1Image(data, affine)
        result = label_lesions(img, threshold=0.5)
        np.testing.assert_array_equal(result.affine, affine)

    def test_output_dtype_is_int32(self):
        data = np.zeros((20, 20, 20), dtype=np.float32)
        data[5:10, 5:10, 5:10] = 0.8
        result = label_lesions(_nifti(data), threshold=0.5)
        assert result.get_fdata().dtype in (np.float32, np.float64)  # get_fdata casts
        assert result.header.get_data_dtype() == np.int32

    def test_split_confluent_separates_touching_cubes(self):
        # Two cubes sharing a face => 1 connected component without splitting
        data = np.zeros((50, 50, 50), dtype=np.float32)
        data[5:20, 10:20, 10:20] = 0.8
        data[20:35, 10:20, 10:20] = 0.8
        img = _nifti(data)

        no_split = label_lesions(img, threshold=0.5, split_confluent=False)
        assert int(no_split.get_fdata().max()) == 1  # confirms they merged

        with_split = label_lesions(img, threshold=0.5, split_confluent=True)
        assert int(with_split.get_fdata().max()) == 2  # correctly split

    def test_split_confluent_preserves_voxel_count(self):
        data = np.zeros((50, 50, 50), dtype=np.float32)
        data[5:20, 10:20, 10:20] = 0.8
        data[20:35, 10:20, 10:20] = 0.8
        img = _nifti(data)
        result = label_lesions(img, threshold=0.5, split_confluent=True)
        original_vox = int((data > 0.5).sum())
        result_vox = int((result.get_fdata() > 0).sum())
        assert result_vox == original_vox


# ── erosion_derived_confluent_lesion_splitting ───────────────────────────────

class TestErosionDerivedSplitting:
    def test_touching_cubes_split_into_two(self):
        arr = np.zeros((40, 40, 40), dtype=np.int32)
        arr[5:20, 5:15, 5:15] = 1
        arr[20:35, 5:15, 5:15] = 1
        labeled = sk_label(arr.astype(bool), connectivity=3)  # -> 1 component
        assert labeled.max() == 1
        result = erosion_derived_confluent_lesion_splitting(labeled)
        assert int(result.max()) == 2

    def test_all_voxels_assigned(self):
        arr = np.zeros((40, 40, 40), dtype=np.int32)
        arr[5:20, 5:15, 5:15] = 1
        arr[20:35, 5:15, 5:15] = 1
        labeled = sk_label(arr.astype(bool), connectivity=3)
        result = erosion_derived_confluent_lesion_splitting(labeled)
        assert np.sum(result > 0) == np.sum(arr > 0)

    def test_already_separate_blobs_preserved(self):
        arr = np.zeros((30, 30, 30), dtype=np.int32)
        arr[2:6, 2:6, 2:6] = 1
        arr[20:24, 20:24, 20:24] = 2
        result = erosion_derived_confluent_lesion_splitting(arr)
        assert int(result.max()) == 2
        assert np.sum(result > 0) == np.sum(arr > 0)

    def test_background_stays_zero(self):
        arr = np.zeros((30, 30, 30), dtype=np.int32)
        arr[5:15, 5:15, 5:15] = 1
        result = erosion_derived_confluent_lesion_splitting(arr)
        assert result[0, 0, 0] == 0
        assert result[29, 29, 29] == 0
