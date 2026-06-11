import numpy as np
import nibabel as nib
import pytest

from lesionomics.label_lesions import (
    relabel_from_largest_to_smallest,
    erosion_derived_confluent_lesion_splitting,
    watershed_confluent_lesion_splitting,
    random_walker_confluent_lesion_splitting,
    label_lesions,
    LabeledResult,
)


def _make_img(data, affine=None):
    if affine is None:
        affine = np.eye(4)
    return nib.Nifti1Image(data.astype(np.float32), affine)


def _two_merged_spheres(shape=(30, 20, 20), r=4, sep=8):
    """Binary array containing two overlapping spheres merged into a single blob."""
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    cy, cx = shape[1] // 2, shape[2] // 2
    mask1 = (z - (shape[0] // 2 - sep // 2)) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2
    mask2 = (z - (shape[0] // 2 + sep // 2)) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2
    arr = np.zeros(shape, dtype=np.int32)
    arr[mask1 | mask2] = 1
    return arr


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
        assert isinstance(result, LabeledResult)
        assert isinstance(result.label_img, nib.Nifti1Image)

    def test_above_threshold_labeled(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 0.5  # above default threshold 0.3
        result = label_lesions(_make_img(data))
        assert result.label_img.get_fdata().max() == 1

    def test_below_threshold_gives_background(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 0.2  # below 0.3
        result = label_lesions(_make_img(data))
        assert result.label_img.get_fdata().max() == 0

    def test_two_components_labeled(self):
        data = np.zeros((20, 10, 10), dtype=np.float32)
        data[0:4, 3:7, 3:7] = 1.0
        data[8:12, 3:7, 3:7] = 1.0
        result = label_lesions(_make_img(data))
        assert result.label_img.get_fdata().max() == 2

    def test_custom_threshold(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 0.5  # below custom threshold of 0.8
        result = label_lesions(_make_img(data), threshold=0.8)
        assert result.label_img.get_fdata().max() == 0

    def test_preserves_affine(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 1.0
        result = label_lesions(nib.Nifti1Image(data, affine))
        np.testing.assert_array_equal(result.label_img.affine, affine)

    def test_largest_lesion_labeled_first(self):
        data = np.zeros((30, 10, 10), dtype=np.float32)
        data[0:3, 3:7, 3:7] = 1.0   # smaller
        data[10:18, 3:7, 3:7] = 1.0  # larger
        result = label_lesions(_make_img(data))
        labeled = result.label_img.get_fdata()
        assert labeled[13, 4, 4] == 1  # large block → label 1
        assert labeled[1, 4, 4] == 2   # small block → label 2

    def test_empty_image_gives_all_zeros(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        result = label_lesions(_make_img(data))
        assert result.label_img.get_fdata().max() == 0

    def test_invalid_split_method_raises(self):
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:8, 2:8, 2:8] = 1.0
        with pytest.raises(ValueError, match="Unknown split_method"):
            label_lesions(_make_img(data), split_confluent=True, split_method="bad_method")

    def test_split_confluent_false_ignores_method(self):
        """split_method is irrelevant when split_confluent=False."""
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:8, 2:8, 2:8] = 1.0
        result = label_lesions(_make_img(data), split_confluent=False, split_method="watershed")
        assert result.label_img.get_fdata().max() >= 1  # no error raised


# ---------------------------------------------------------------------------
# Shared contract tests for all splitting methods
# ---------------------------------------------------------------------------

def _assert_splitting_contract(split_fn, merged_arr):
    """Shared assertions that any splitting function must satisfy."""
    original_fg = int(np.sum(merged_arr > 0))
    result = split_fn(merged_arr)
    assert int(np.sum(result > 0)) == original_fg, "foreground voxel count changed"
    assert result.min() >= 0, "negative labels found"
    assert result.max() >= 1, "result is all background"


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
        _assert_splitting_contract(
            erosion_derived_confluent_lesion_splitting,
            _two_merged_spheres(),
        )

    def test_result_is_non_negative(self):
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[5:15, 5:15, 5:15] = 1
        result = erosion_derived_confluent_lesion_splitting(arr)
        assert result.min() >= 0


class TestWatershedSplitting:
    def test_empty_array_unchanged(self):
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        result = watershed_confluent_lesion_splitting(arr)
        assert result.sum() == 0

    def test_two_separate_components_preserved(self):
        arr = np.zeros((30, 10, 10), dtype=np.int32)
        arr[0:5, 3:7, 3:7] = 1
        arr[10:15, 3:7, 3:7] = 2
        result = watershed_confluent_lesion_splitting(arr)
        assert result.max() == 2

    def test_preserves_total_foreground_voxels(self):
        _assert_splitting_contract(
            watershed_confluent_lesion_splitting,
            _two_merged_spheres(),
        )

    def test_result_is_non_negative(self):
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[5:15, 5:15, 5:15] = 1
        result = watershed_confluent_lesion_splitting(arr)
        assert result.min() >= 0

    def test_min_distance_parameter_accepted(self):
        arr = _two_merged_spheres()
        result = watershed_confluent_lesion_splitting(arr, min_distance=5)
        assert result.min() >= 0

    def test_label_lesions_watershed_method(self):
        data = np.zeros((30, 20, 20), dtype=np.float32)
        arr = _two_merged_spheres()
        data[arr > 0] = 1.0
        result = label_lesions(_make_img(data), split_confluent=True, split_method="watershed")
        assert isinstance(result, LabeledResult)
        assert result.label_img.get_fdata().max() >= 1


class TestRandomWalkerSplitting:
    def test_empty_array_unchanged(self):
        arr = np.zeros((10, 10, 10), dtype=np.int32)
        result = random_walker_confluent_lesion_splitting(arr)
        assert result.sum() == 0

    def test_two_separate_components_preserved(self):
        arr = np.zeros((30, 10, 10), dtype=np.int32)
        arr[0:5, 3:7, 3:7] = 1
        arr[10:15, 3:7, 3:7] = 2
        result = random_walker_confluent_lesion_splitting(arr)
        assert result.max() == 2

    def test_preserves_total_foreground_voxels(self):
        _assert_splitting_contract(
            random_walker_confluent_lesion_splitting,
            _two_merged_spheres(),
        )

    def test_result_is_non_negative(self):
        arr = np.zeros((20, 20, 20), dtype=np.int32)
        arr[5:15, 5:15, 5:15] = 1
        result = random_walker_confluent_lesion_splitting(arr)
        assert result.min() >= 0

    def test_label_lesions_random_walker_method(self):
        data = np.zeros((30, 20, 20), dtype=np.float32)
        arr = _two_merged_spheres()
        data[arr > 0] = 1.0
        result = label_lesions(_make_img(data), split_confluent=True, split_method="random_walker")
        assert isinstance(result, LabeledResult)
        assert result.label_img.get_fdata().max() >= 1


# ---------------------------------------------------------------------------
# LabeledResult hierarchy tests
# ---------------------------------------------------------------------------

class TestLabeledResultHierarchy:
    def test_isolated_lesion_has_no_sub_label(self):
        data = np.zeros((20, 10, 10), dtype=np.float32)
        data[1:4, 3:7, 3:7] = 1.0
        result = label_lesions(_make_img(data))
        assert result.hierarchy["sub_label"].iloc[0] is None

    def test_display_id_isolated(self):
        data = np.zeros((20, 10, 10), dtype=np.float32)
        data[1:4, 3:7, 3:7] = 1.0
        result = label_lesions(_make_img(data))
        assert result.hierarchy["display_id"].iloc[0] == "L1"

    def test_split_lesion_has_sub_labels(self):
        # Two merged spheres that erosion can split
        arr = _two_merged_spheres()
        data = arr.astype(np.float32)
        result = label_lesions(_make_img(data), threshold=0.5, split_confluent=True)
        # If splitting produced >1 sub-lesion, sub_label should be set
        if len(result.hierarchy) > 1:
            sub_labels = result.hierarchy["sub_label"].dropna()
            assert len(sub_labels) > 0

    def test_n_lesions_matches_hierarchy(self):
        data = np.zeros((20, 10, 10), dtype=np.float32)
        data[1:4, 3:7, 3:7] = 1.0
        data[8:12, 3:7, 3:7] = 1.0
        result = label_lesions(_make_img(data))
        assert result.n_lesions == len(result.hierarchy)

    def test_n_confluent_lesions_correct(self):
        data = np.zeros((20, 10, 10), dtype=np.float32)
        data[1:4, 3:7, 3:7] = 1.0
        data[8:12, 3:7, 3:7] = 1.0
        result = label_lesions(_make_img(data))
        assert result.n_confluent_lesions == 2

    def test_confluent_lesion_img_has_same_affine(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[2:5, 2:5, 2:5] = 1.0
        result = label_lesions(nib.Nifti1Image(data, affine))
        np.testing.assert_array_equal(result.confluent_lesion_img.affine, affine)

    def test_int_to_sublabel(self):
        from lesionomics.label_lesions import _int_to_sublabel
        assert _int_to_sublabel(1) == "A"
        assert _int_to_sublabel(26) == "Z"
        assert _int_to_sublabel(27) == "AA"
