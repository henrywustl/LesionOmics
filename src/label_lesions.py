import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops

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

