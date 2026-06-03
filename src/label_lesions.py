import logging
from typing import Literal

import numpy as np
import nibabel as nib
import skimage.measure
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import binary_erosion, distance_transform_edt

logger = logging.getLogger(__name__)

SplitMethod = Literal["erosion", "watershed", "random_walker"]


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


def _find_erosion_seeds(binary_volume: np.ndarray) -> np.ndarray:
    """Iteratively erode until components split or would vanish; return labeled seed array.

    Shared by both the erosion-Voronoi and random-walker splitting methods.
    """
    current_state = binary_volume.copy()
    n_current = int(skimage.measure.label(current_state, connectivity=3).max())

    while True:
        eroded = binary_erosion(current_state)
        if not np.any(eroded):
            break
        n_eroded = int(skimage.measure.label(eroded, connectivity=3).max())
        if n_eroded > n_current:
            current_state = eroded
            break
        if n_eroded < n_current:
            break
        current_state = eroded
        n_current = n_eroded

    return skimage.measure.label(current_state, connectivity=3)


def erosion_derived_confluent_lesion_splitting(labeled_array: np.ndarray) -> np.ndarray:
    """Split confluent lesions via iterative erosion then Voronoi expansion.

    Erodes the binary union of all labeled voxels until connected components
    either split (n_eroded > n_current) or a component would be eliminated
    (n_eroded < n_current). The eroded state at that point serves as Voronoi
    seeds; each original voxel is assigned to its nearest seed via EDT.

    Args:
        labeled_array: Integer-labeled array (0 = background).

    Returns:
        New labeled array with the same foreground voxels, relabeled
        largest-to-smallest after splitting.
    """
    if not np.any(labeled_array):
        return labeled_array.copy()

    original_volume = labeled_array > 0
    center_labels = _find_erosion_seeds(original_volume)
    n_centers = int(center_labels.max())

    if n_centers == 0:
        return labeled_array.copy()

    dist_maps = np.zeros((n_centers, *original_volume.shape), dtype=np.float32)
    for i in range(1, n_centers + 1):
        dist_maps[i - 1] = distance_transform_edt(center_labels != i)

    nearest_center = np.argmin(dist_maps, axis=0) + 1
    result = np.where(original_volume, nearest_center, 0).astype(np.int32)
    return relabel_from_largest_to_smallest(result)


def watershed_confluent_lesion_splitting(
    labeled_array: np.ndarray,
    min_distance: int = 3,
) -> np.ndarray:
    """Split confluent lesions via distance-transform watershed.

    Computes the Euclidean distance transform of the binary mask, locates
    local maxima (the "peaks" most central to each sub-region) as watershed
    seeds, then floods downhill to assign every foreground voxel to a region.
    Compared to erosion-Voronoi the watershed respects the topology of the DT
    landscape rather than just nearest-seed distance, producing boundaries
    that tend to sit on the narrowest neck between merged blobs.

    Args:
        labeled_array: Integer-labeled array (0 = background).
        min_distance: Minimum voxel separation between DT peaks used as
                      seeds (default 3). Increase to reduce over-segmentation
                      when lesions are large or densely packed.

    Returns:
        New labeled array with the same foreground voxels, relabeled
        largest-to-smallest after splitting.
    """
    if not np.any(labeled_array):
        return labeled_array.copy()

    binary = labeled_array > 0
    dist = distance_transform_edt(binary)

    coords = peak_local_max(dist, min_distance=min_distance, labels=binary)
    if len(coords) == 0:
        return labeled_array.copy()

    seed_mask = np.zeros(dist.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers = skimage.measure.label(seed_mask, connectivity=3)

    result = watershed(-dist, markers, mask=binary)
    return relabel_from_largest_to_smallest(result.astype(np.int32))


def random_walker_confluent_lesion_splitting(labeled_array: np.ndarray) -> np.ndarray:
    """Split confluent lesions via random-walker segmentation.

    Uses the same erosion-derived seeds as the erosion method, but expands
    regions through a diffusion (random-walk) process on the Euclidean
    distance transform rather than nearest-neighbour assignment. The
    probabilistic boundary placement tends to produce smoother, more
    natural-looking splits than Voronoi expansion, and is less sensitive to
    seed placement than the watershed approach.

    Falls back to erosion-Voronoi if scikit-image's random_walker is
    unavailable or raises an exception.

    Args:
        labeled_array: Integer-labeled array (0 = background).

    Returns:
        New labeled array with the same foreground voxels, relabeled
        largest-to-smallest after splitting.
    """
    try:
        from skimage.segmentation import random_walker as _rw
    except ImportError:
        logger.warning(
            "skimage.segmentation.random_walker unavailable; "
            "falling back to erosion method."
        )
        return erosion_derived_confluent_lesion_splitting(labeled_array)

    if not np.any(labeled_array):
        return labeled_array.copy()

    binary = labeled_array > 0
    seed_labels = _find_erosion_seeds(binary)
    n_seeds = int(seed_labels.max())

    if n_seeds == 0:
        return labeled_array.copy()

    # If every foreground voxel is already covered by a seed, skip the walker
    if np.all(seed_labels[binary] > 0):
        result = np.where(binary, seed_labels, 0).astype(np.int32)
        return relabel_from_largest_to_smallest(result)

    # Use the DT as image data so the walker is guided toward region centres
    dist = distance_transform_edt(binary).astype(np.float64)

    try:
        walked = _rw(dist, seed_labels, beta=10, mode="bf")
        result = np.where(binary, walked, 0).astype(np.int32)
        return relabel_from_largest_to_smallest(result)
    except Exception as exc:
        logger.warning(
            "random_walker failed (%s); falling back to erosion method.", exc
        )
        return erosion_derived_confluent_lesion_splitting(labeled_array)


# Registry — extend here to add future methods
_SPLIT_METHODS: dict[str, callable] = {
    "erosion": erosion_derived_confluent_lesion_splitting,
    "watershed": watershed_confluent_lesion_splitting,
    "random_walker": random_walker_confluent_lesion_splitting,
}


def label_lesions(
    input_img: nib.Nifti1Image,
    threshold: float = 0.3,
    split_confluent: bool = False,
    split_method: SplitMethod = "erosion",
) -> nib.Nifti1Image:
    """Label connected lesion components in a NIfTI image.

    Args:
        input_img: Probability or binary lesion map.
        threshold: Voxels above this value are considered lesion.
        split_confluent: If True, attempt to split confluent lesions.
        split_method: Algorithm used when ``split_confluent=True``.

            * ``"erosion"`` *(default)* — iterative morphological erosion to
              find seeds, then Voronoi (EDT nearest-seed) expansion.
            * ``"watershed"`` — distance-transform watershed; seeds are
              local maxima of the EDT. Respects narrow necks between blobs.
            * ``"random_walker"`` — erosion-derived seeds expanded via
              random-walk diffusion on the distance transform. Produces
              smooth, probabilistically placed boundaries.

    Returns:
        NIfTI image with integer labels (0 = background, 1..N = lesions,
        sorted largest-first).

    Raises:
        ValueError: If ``split_method`` is not one of the recognised options.
    """
    data = input_img.get_fdata()
    binary_mask = data > threshold
    labeled = label(binary_mask, connectivity=3)
    labeled = relabel_from_largest_to_smallest(labeled)

    if split_confluent and np.any(labeled):
        if split_method not in _SPLIT_METHODS:
            raise ValueError(
                f"Unknown split_method {split_method!r}. "
                f"Choose from: {list(_SPLIT_METHODS)}"
            )
        labeled = _SPLIT_METHODS[split_method](labeled)

    return nib.Nifti1Image(labeled.astype(np.int32), input_img.affine, input_img.header)
