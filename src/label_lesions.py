import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import nibabel as nib
import pandas as pd
import skimage.measure
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import binary_erosion, distance_transform_edt

logger = logging.getLogger(__name__)

SplitMethod = Literal["erosion", "watershed", "random_walker"]


def _int_to_sublabel(n: int) -> str:
    """1→'A', 2→'B', 26→'Z', 27→'AA', etc. (Excel-style column naming)."""
    result = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        result = chr(65 + r) + result
    return result


@dataclass
class LabeledResult:
    """Hierarchical lesion labeling result from label_lesions().

    Attributes:
        label_img:  NIfTI with flat integer labels 1..N — one per distinct
                    lesion or sub-lesion.  Safe to pass to any NIfTI tool.
        blob_img:   NIfTI with blob labels 1..M — one per original
                    pre-split connected component.
        hierarchy:  DataFrame with columns:
                      label      – integer label in label_img
                      blob_id    – integer label in blob_img this came from
                      sub_label  – None (isolated) or 'A'/'B'/… (sub-lesion)
                      display_id – human-readable id e.g. 'L1' or 'L3B'
    """
    label_img: nib.Nifti1Image
    blob_img: nib.Nifti1Image
    hierarchy: "pd.DataFrame"

    @property
    def n_lesions(self) -> int:
        return int(self.label_img.get_fdata().max())

    @property
    def n_blobs(self) -> int:
        return int(self.blob_img.get_fdata().max())


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

    Uses the same erosion-derived seeds as the erosion method, but assigns
    unlabeled foreground voxels through a diffusion (random-walk) process on
    the Euclidean distance transform rather than nearest-neighbour assignment.
    The probabilistic boundary placement tends to produce smoother, more
    natural-looking splits than Voronoi expansion, and is less sensitive to
    seed placement than the watershed approach.

    To keep run-time tractable on clinical-size volumes, each connected
    component is processed independently within its own tight bounding box.

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
    component_labels = skimage.measure.label(binary, connectivity=3)
    n_components = int(component_labels.max())
    result = np.zeros_like(labeled_array, dtype=np.int32)
    global_label = 0

    for comp_id in range(1, n_components + 1):
        comp_mask = component_labels == comp_id

        # Crop to bounding box (+ 1 voxel padding) for speed
        coords = np.column_stack(np.where(comp_mask))
        lo = np.maximum(coords.min(axis=0) - 1, 0)
        hi = np.minimum(coords.max(axis=0) + 2, np.array(binary.shape))
        slices = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))

        sub_mask = comp_mask[slices]
        sub_seeds = _find_erosion_seeds(sub_mask)

        if int(sub_seeds.max()) == 0:
            global_label += 1
            result[comp_mask] = global_label
            continue

        if np.all(sub_seeds[sub_mask] > 0):
            sub_result = np.where(sub_mask, sub_seeds, 0).astype(np.int32)
        else:
            sub_dist = distance_transform_edt(sub_mask).astype(np.float64)
            sub_result = None
            # Conjugate-gradient solvers are O(N log N); brute-force is O(N³)
            for mode in ("cg_j", "cg", "bf"):
                try:
                    walked = _rw(sub_dist, sub_seeds, beta=10, mode=mode)
                    sub_result = np.where(sub_mask, walked, 0).astype(np.int32)
                    break
                except Exception as exc:
                    logger.debug(
                        "random_walker mode=%s failed for component %d (%s).",
                        mode, comp_id, exc,
                    )
            if sub_result is None:
                logger.warning(
                    "random_walker failed for component %d; "
                    "using Voronoi fallback for this component.", comp_id,
                )
                sub_result = np.where(sub_mask, sub_seeds, 0).astype(np.int32)

        # Map local labels (1..k) to globally unique labels
        for local_lbl in range(1, int(sub_result.max()) + 1):
            global_label += 1
            local_vox = np.where(sub_result == local_lbl)
            full_vox = tuple(lo[d] + local_vox[d] for d in range(3))
            result[full_vox] = global_label

    return relabel_from_largest_to_smallest(result)


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
) -> "LabeledResult":
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
        :class:`LabeledResult` containing:
        - ``label_img``: NIfTI with flat labels 1..N (one per lesion/sub-lesion)
        - ``blob_img``: NIfTI with pre-split blob labels 1..M
        - ``hierarchy``: DataFrame mapping labels to blobs and sub-labels

    Raises:
        ValueError: If ``split_method`` is not one of the recognised options.
    """
    if split_confluent and split_method not in _SPLIT_METHODS:
        raise ValueError(
            f"Unknown split_method {split_method!r}. "
            f"Choose from: {list(_SPLIT_METHODS)}"
        )

    data = input_img.get_fdata()
    binary_mask = data > threshold

    # Find original connected components (blobs), sorted largest-first
    blob_array = label(binary_mask, connectivity=3)
    blob_array = relabel_from_largest_to_smallest(blob_array).astype(np.int32)
    n_blobs = int(blob_array.max())

    blob_img = nib.Nifti1Image(blob_array, input_img.affine, input_img.header)

    # Build flat label array and hierarchy
    label_array = np.zeros_like(blob_array, dtype=np.int32)
    hierarchy_rows = []
    next_label = 1

    for blob_id in range(1, n_blobs + 1):
        blob_mask = blob_array == blob_id
        blob_only = np.where(blob_mask, 1, 0).astype(np.int32)

        if split_confluent:
            split_result = _SPLIT_METHODS[split_method](blob_only)
        else:
            split_result = blob_only

        n_sub = int(split_result.max())

        if n_sub <= 1:
            # No split — isolated lesion
            label_array[blob_mask] = next_label
            hierarchy_rows.append({
                "label": next_label,
                "blob_id": blob_id,
                "sub_label": None,
                "display_id": f"L{blob_id}",
            })
            next_label += 1
        else:
            # Blob split into n_sub sub-lesions
            # Relabel split_result by size (largest sub-lesion first → sub_label A)
            split_result = relabel_from_largest_to_smallest(split_result)
            for sub_idx in range(1, n_sub + 1):
                sub_mask = split_result == sub_idx
                label_array[sub_mask] = next_label
                sl = _int_to_sublabel(sub_idx)
                hierarchy_rows.append({
                    "label": next_label,
                    "blob_id": blob_id,
                    "sub_label": sl,
                    "display_id": f"L{blob_id}{sl}",
                })
                next_label += 1

    label_img = nib.Nifti1Image(label_array, input_img.affine, input_img.header)
    hierarchy = pd.DataFrame(hierarchy_rows) if hierarchy_rows else pd.DataFrame(
        columns=["label", "blob_id", "sub_label", "display_id"]
    )

    return LabeledResult(label_img=label_img, blob_img=blob_img, hierarchy=hierarchy)
