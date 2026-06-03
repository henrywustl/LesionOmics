import numpy as np
import nibabel as nib
import pytest

from src.label_lesions import (
    relabel_from_largest_to_smallest,
    erosion_derived_confluent_lesion_splitting,
    label_lesions,
)


def _make_img(data, affine=None):
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(data.astype(np.float32), affine)


# ---------------------------------------------------------------------------
# relabel_from_largest_to_smallest
# ---------------------------------------------------------------------------

class TestRelabelFromLargestToSmallest:
    def test_empty_array_returns_zeros(self):
        arr = np.zeros((5, 5, 5), dtype=np.int32)
        result = relabel_from_largest_to_smallest(arr)
        assert result.sum() == 0

    def test_single_region_becomes_label_one(self):
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        arr[2:5, 2:5, 2:5] = 7  # arbitrary old label
        result = relabel_from_largest_to_smallest(arr)
        assert set(np.unique(result)) == {0, 1}
        assert result[3, 3, 3] == 1

    def test_largest_region_gets_label_one(self):
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[0:3, 0:3, 0:3] = 1   # 27 vox — smaller
        arr[5:12, 5:12, 5:12] = 2  # 343 vox — larger
        result = relabel_from_largest_to_smallest(arr)
        assert result[7, 7, 7] == 1   # large block → label 1
        assert result[1, 1, 1] == 2   # small block → label 2

    def test_background_stays_zero(self):
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        arr[3:7, 3:7, 3:7] = 5
        result = relabel_from_largest_to_smallest(arr)
        assert result[0, 0, 0] == 0

    def test_preserves_output_shape(self):
        arr = np.zeros((15, 15, 15), dtype=np.int32)
        arr[0:4, 0:4, 0:4] = 1
        arr[7:10, 7:10, 7:10] = 2
        result = relabel_from_largest_to_smallest(arr)
        assert result.shape == arr.shape


# ---------------------------------------------------------------------------
# label_lesions
# ---------------------------------------------------------------------------

class TestLabelLesions:
    def test_returns_nifti_image(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 1.0
        result = label_lesions(_make_img(data))
        assert isinstance(result, nib.Nifti1Image)

    def test_above_threshold_labeled(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 0.5  # above default threshold 0.3
        result = label_lesions(_make_img(data))
        assert result.get_fdata().max() == 1

    def test_below_threshold_gives_background(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 0.2  # below 0.3
        result = label_lesions(_make_img(data))
        assert result.get_fdata().max() == 0

    def test_two_components_labeled(self):
        data = np.zeros((20, 10, 10), dtype=np.float32)
        data[0:4, 3:7, 3:7] = 1.0
        data[8:12, 3:7, 3:7] = 1.0
        result = label_lesions(_make_img(data))
        assert result.get_fdata().max() == 2

    def test_custom_threshold(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 0.5  # below custom threshold of 0.8
        result = label_lesions(_make_img(data), threshold=0.8)
        assert result.get_fdata().max() == 0

    def test_preserves_affine(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 1.0
        result = label_lesions(nib.Nifti1Image(data, affine))
        np.testing.assert_array_equal(result.affine, affine)

    def test_largest_lesion_labeled_first(self):
        data = np.zeros((30, 10, 10), dtype=np.float32)
        data[0:3, 3:7, 3:7] = 1.0   # smaller
        data[10:18, 3:7, 3:7] = 1.0  # larger
        result = label_lesions(_make_img(data))
        labeled = result.get_fdata()
        assert labeled[13, 4, 4] == 1  # large block → label 1
        assert labeled[1, 4, 4] == 2   # small block → label 2

    def test_empty_image_gives_all_zeros(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        result = label_lesions(_make_img(data))
        assert result.get_fdata().max() == 0


# ---------------------------------------------------------------------------
# erosion_derived_confluent_lesion_splitting
# ---------------------------------------------------------------------------

class TestErosionDerivedSplitting:
    def test_empty_array_unchanged(self):
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        result = erosion_derived_confluent_lesion_splitting(arr)
        assert result.sum() == 0

    def test_two_separate_components_preserved(self):
        arr = np.zeros((30, 10, 10), dtype=np.int32)
        arr[0:5, 3:7, 3:7] = 1
        arr[10:15, 3:7, 3:7] = 2
        result = erosion_derived_confluent_lesion_splitting(arr)
        assert result.max() == 2

    def test_preserves_total_foreground_voxels(self):
        arr = np.zeros((30, 20, 20), dtype=np.int32)
        z, y, x = np.mgrid[0:30, 0:20, 0:20]
        # Two overlapping spheres merged into a single label
        mask1 = (z - 8) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 16
        mask2 = (z - 14) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 16
        arr[mask1 | mask2] = 1
        original_fg = int(np.sum(arr > 0))
        result = erosion_derived_confluent_lesion_splitting(arr)
        assert int(np.sum(result > 0)) == original_fg

    def test_result_is_non_negative(self):
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[5:15, 5:15, 5:15] = 1
        result = erosion_derived_confluent_lesion_splitting(arr)
        assert result.min() >= 0
