"""
Lesion segmentation and connected-component labeling utilities.
"""

import numpy as np
import nibabel as nib
import skimage.measure
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, distance_transform_edt


def relabel_from_largest_to_smallest(labeled_array: np.ndarray) -> np.ndarray:
    """Re-index connected-component labels in descending order of size (largest = 1)."""
    props = regionprops(labeled_array)
    sorted_indices = np.argsort([p.area for p in props])[::-1]
    new_labeled = np.zeros_like(labeled_array)
    for new_label, idx in enumerate(sorted_indices, start=1):
        new_labeled[labeled_array == props[idx].label] = new_label
    return new_labeled


def erosion_derived_confluent_lesion_splitting(labeled_array: np.ndarray) -> np.ndarray:
    """
    Split confluent (touching) lesions using erosion-derived centre points
    and distance-transform Voronoi partitioning.

    Each connected region is eroded until a further erosion step would change
    the number of connected components, yielding stable centre cores.  The
    original volume is then re-partitioned by assigning each voxel to the
    nearest core via argmin over stacked EDT distance maps.
    """
    props = regionprops(labeled_array)
    original_volume = np.zeros_like(labeled_array, dtype=bool)
    for prop in props:
        original_volume |= (labeled_array == prop.label)

    current_state = original_volume.copy()
    n_current = skimage.measure.label(current_state, connectivity=3).max()

    while True:
        eroded = binary_erosion(current_state)
        if eroded.sum() == 0:
            break
        n_eroded = skimage.measure.label(eroded, connectivity=3).max()
        if n_eroded != n_current:
            break
        current_state = eroded
        n_current = n_eroded

    center_labels = skimage.measure.label(current_state, connectivity=3)
    center_labels = relabel_from_largest_to_smallest(center_labels)
    n_centers = int(center_labels.max())

    if n_centers == 0:
        return labeled_array.copy()

    # Stack EDT maps then argmin for fast Voronoi assignment
    dist_stack = np.stack(
        [distance_transform_edt(center_labels != i) for i in range(1, n_centers + 1)],
        axis=0,
    )
    assignment = np.argmin(dist_stack, axis=0) + 1  # 1-indexed
    return np.where(original_volume, assignment, 0).astype(labeled_array.dtype)


def label_lesions(
    input_img: nib.Nifti1Image,
    threshold: float = 0.3,
    split_confluent: bool = False,
) -> nib.Nifti1Image:
    """
    Threshold, 3D-label, and optionally split confluent lesions.

    Args:
        input_img:       NIfTI image — probability map or binary mask.
        threshold:       Binarisation threshold (values > threshold are lesion).
        split_confluent: If True, apply erosion-based confluent lesion splitting.

    Returns:
        NIfTI image with integer lesion labels; largest lesion = 1.
    """
    data = input_img.get_fdata()
    binary = data > threshold
    labeled = label(binary, connectivity=3)
    labeled = relabel_from_largest_to_smallest(labeled)
    if split_confluent:
        labeled = erosion_derived_confluent_lesion_splitting(labeled)
        labeled = relabel_from_largest_to_smallest(labeled)
    return nib.Nifti1Image(
        labeled.astype(np.int32), input_img.affine, input_img.header
    )
