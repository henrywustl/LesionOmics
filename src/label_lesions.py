import logging

import numpy as np
import nibabel as nib
import skimage.measure
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, distance_transform_edt

logger = logging.getLogger(__name__)


def relabel_from_largest_to_smallest(labeled_array: np.ndarray) -> np.ndarray:
    """Reassign labels 1..N in decreasing order of blob size (largest → label 1)."""
    props = regionprops(labeled_array)
    if not props:
        return labeled_array.copy()

    sorted_props = sorted(props, key=lambda p: p.area, reverse=True)
    new_labeled = np.zeros_like(labeled_array)
    for new_val, prop in enumerate(sorted_props, start=1):
        new_labeled[labeled_array == prop.label] = new_val

    return new_labeled


def erosion_derived_confluent_lesion_splitting(labeled_array: np.ndarray) -> np.ndarray:
    """Split confluent lesions via iterative erosion then Voronoi expansion.

    Erodes the binary union of all labeled voxels until connected components
    either split (n_eroded > n_current) or a component would be eliminated
    (n_eroded < n_current). The eroded state at that point serves as Voronoi
    seeds; each original voxel is assigned to its nearest seed.

    Args:
        labeled_array: Integer-labeled array (0 = background).

    Returns:
        New labeled array with the same foreground voxels, relabeled
        largest-to-smallest after splitting.
    """
    if not np.any(labeled_array):
        return labeled_array.copy()

    original_volume = labeled_array > 0
    current_state = original_volume.copy()
    n_current = skimage.measure.label(current_state, connectivity=3).max()

    while True:
        eroded = binary_erosion(current_state)
        if not np.any(eroded):
            break

        n_eroded = int(skimage.measure.label(eroded, connectivity=3).max())

        if n_eroded > n_current:
            # Erosion revealed separate sub-regions — use them as centre seeds
            current_state = eroded
            break
        if n_eroded < n_current:
            # A component would vanish — keep current_state as seeds
            break

        current_state = eroded
        n_current = n_eroded

    center_labels = skimage.measure.label(current_state, connectivity=3)
    n_centers = int(center_labels.max())

    if n_centers == 0:
        return labeled_array.copy()

    # Voronoi expansion: EDT from each center, assign each foreground voxel
    # to whichever center it is nearest to
    dist_maps = np.zeros((n_centers, *original_volume.shape), dtype=np.float32)
    for i in range(1, n_centers + 1):
        dist_maps[i - 1] = distance_transform_edt(center_labels != i)

    nearest_center = np.argmin(dist_maps, axis=0) + 1  # 1-indexed
    result = np.where(original_volume, nearest_center, 0).astype(np.int32)

    return relabel_from_largest_to_smallest(result)


def label_lesions(
    input_img: nib.Nifti1Image,
    threshold: float = 0.3,
    split_confluent: bool = False,
) -> nib.Nifti1Image:
    """Label connected lesion components in a NIfTI image.

    Args:
        input_img: Probability or binary lesion map.
        threshold: Voxels above this value are considered lesion.
        split_confluent: If True, attempt to split confluent lesions via erosion.

    Returns:
        NIfTI image with integer labels (0 = background, 1..N = lesions,
        sorted largest-first).
    """
    data = input_img.get_fdata()
    binary_mask = data > threshold
    labeled = label(binary_mask, connectivity=3)
    labeled = relabel_from_largest_to_smallest(labeled)

    if split_confluent and np.any(labeled):
        labeled = erosion_derived_confluent_lesion_splitting(labeled)

    return nib.Nifti1Image(labeled.astype(np.int32), input_img.affine, input_img.header)
