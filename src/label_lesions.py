import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt


def relabel_from_largest_to_smallest(labeled_array):
    """
    Given a labeled_array (integer labels), this function reassigns 
    labels from 1..N in decreasing order of blob size (largest->smallest).
    """
    # Extract region properties
    props = regionprops(labeled_array)
    
    # Sort regions by area (descending order)
    areas = [prop.area for prop in props]
    sorted_indices = np.argsort(areas)[::-1]  # largest first
    
    # Reassign labels
    new_labeled = np.zeros_like(labeled_array)
    for new_label, idx in enumerate(sorted_indices, start=1):
        old_label = props[idx].label
        new_labeled[labeled_array == old_label] = new_label
        
    return new_labeled

def erosion_derived_confluent_lesion_splitting(labeled_array):
    """
    Split confluent lesions by eroding to minimal center points and expanding back.

    Args:
        labeled_array (ndarray): Input array with integer labels for different regions

    Returns:
        ndarray: New labeled array with split regions based on center points
    """
    # Extract region properties
    props = regionprops(labeled_array)

    # Initialize arrays
    original_volume = np.zeros_like(labeled_array, dtype=bool)
    center_points = np.zeros_like(labeled_array, dtype=bool)

    # Store original volume
    for prop in props:
        original_volume |= (labeled_array == prop.label)

    # Erode to center points
    current_state = original_volume.copy()
    while True:
        eroded = binary_erosion(current_state)
        labeled_eroded = label(eroded, connectivity=3)

        # Check if further erosion would eliminate any region completely
        if np.sum(eroded) == 0 or label(eroded, connectivity=3).max() != label(current_state, connectivity=3).max():
            center_points = current_state
            break

        current_state = eroded

    # Label center points
    center_labels = label(center_points, connectivity=3)
    center_labels = relabel_from_largest_to_smallest(center_labels)

    # Create distance maps for expansion
    distance_maps = {}
    for i in range(1, center_labels.max() + 1):
        distance_maps[i] = distance_transform_edt(~(center_labels == i))

    # Assign original volume to nearest center point
    result = np.zeros_like(labeled_array)
    unassigned = original_volume.copy()

    while np.any(unassigned):
        min_dist = np.inf
        best_label = None

        for label, dist_map in distance_maps.items():
            dist = dist_map[unassigned]
            if dist.min() < min_dist:
                min_dist = dist.min()
                best_label = label

        assign_mask = unassigned & (distance_maps[best_label] == min_dist)
        result[assign_mask] = best_label
        unassigned[assign_mask] = False

    return result


# 1) Load the NIfTI file
input_nii = nib.load("input.nii.gz")
data = input_nii.get_fdata()

# 2) Threshold at 0.4 to create a binary mask
binary_data = data > 0.3

# 3) Label connected components in 3D
#    connectivity=3 ensures 6-connectivity in 3D
labeled_data = label(binary_data, connectivity=3)

# 4) Re-label blobs from largest to smallest
re_labeled_data = relabel_from_largest_to_smallest(labeled_data)

# 5) Save as a new NIfTI image
out_nii = nib.Nifti1Image(re_labeled_data.astype(np.int32), input_nii.affine, input_nii.header)
nib.save(out_nii, "output_labeled.nii.gz")

