import numpy as np
import nibabel as nib
import pandas as pd
from skimage.measure import label, regionprops, shannon_entropy
from scipy.spatial import ConvexHull
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import cv2


def voxel_volume_from_affine(header):
    """
    Returns the volume (in mm^3) of a single voxel,
    computed from the 3x3 submatrix of the affine.
    """
    voxel_sizes = header.get_zooms()[:3]
    voxel_volume = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
    return voxel_volume


def compute_3d_convex_hull_volume(coords_vox):
    """
    Given an (N x 3) array of real-space coordinates in mm,
    compute the volume of the 3D convex hull around these points.
    Returns 0.0 (or np.nan) if the region is degenerate (2D or 1D).
    """
    # If fewer than 4 points, the hull is degenerate
    if coords_vox.shape[0] < 4:
        return 0.0

    # Check if points span at least some 3D extent
    extents = coords_vox.max(axis=0) - coords_vox.min(axis=0)
    if np.count_nonzero(extents) < 3:
        # The shape is effectively 2D or 1D in 3D space
        return 0.0

    hull = ConvexHull(coords_vox)
    return hull.volume


def compute_4d_inertia(coords_vox, intensities):
    """
    Given:
      coords_mm: (N, 3) array of spatial coordinates in mm
      intensities: (N,) array of T1 or T2 intensities (the 4th dimension)

    Returns:
      inertia_4d: (4x4) second-moment matrix about the centroid in 4D
      eigvals_4d, eigvecs_4d: eigenvalues and eigenvectors of that matrix
      centroid_4d: the mean of points in (x, y, z, intensity) space
    """
    if len(coords_vox) == 0:
        return None, None, None, None

    # Stack to create an (N, 4) array
    points_4d = np.column_stack([coords_vox, intensities])

    # Compute centroid in 4D
    centroid_4d = points_4d.mean(axis=0)  # shape: (4,)

    # Center the points on the 4D centroid
    centered_4d = points_4d - centroid_4d

    # Inertia (second moment) = sum of outer products
    # Typically for an "inertia" about the centroid, we sum (or average) (p - mean) (p - mean)^T
    # If you want a pure 'sum', use dot directly:
    inertia_4d = centered_4d.T @ centered_4d  # shape: (4,4)

    # For the eigen-decomposition
    eigvals_4d, eigvecs_4d = np.linalg.eigh(inertia_4d)

    return inertia_4d, eigvals_4d, eigvecs_4d, centroid_4d


# -------------------------------------------------------------------------
# 1. Load images
# -------------------------------------------------------------------------
labeled_img = nib.load("/ceph/chpc/mapped/brier/SLOWLY_EVOLVING_LESIONS/output_labeled.nii.gz")
labeled_data = labeled_img.get_fdata()  # shape: X x Y x Z
affine = labeled_img.affine
header = labeled_img.header

t1_img = nib.load("/ceph/chpc/mapped/brier/SLOWLY_EVOLVING_LESIONS/T1.nii.gz")
t1_data = t1_img.get_fdata()

t2_img = nib.load("/ceph/chpc/mapped/brier/SLOWLY_EVOLVING_LESIONS/T2.nii.gz")
t2_data = t2_img.get_fdata()

voxel_vol = voxel_volume_from_affine(header)

# List of all labels (excluding 0 = background)
all_labels = np.unique(labeled_data)
all_labels = all_labels[all_labels != 0]

# Prepare storage for results
results = []

# -------------------------------------------------------------------------
# 2. For each label (blob), compute shape measurements + T1/T2 metrics
# -------------------------------------------------------------------------
for label_val in all_labels:
    mask = (labeled_data == label_val)

    # ----------------------------------------------------------
    # Basic shape/volume
    # ----------------------------------------------------------
    region_volume_vox = np.count_nonzero(mask)  # * voxel_vol

    # ----------------------------------------------------------
    # Center of mass (in voxel and mm space)
    # ----------------------------------------------------------
    # Using ndimage's center_of_mass
    com_vox = ndimage.center_of_mass(mask)

    # ----------------------------------------------------------
    # Principal axes (inertia tensor) in 3D
    # ----------------------------------------------------------
    # regionprops can compute 3D inertia tensors, eigenvalues, centroids, etc.
    # But regionprops works on labeled images. We'll isolate just this region as label=1:
    temp_label = np.zeros_like(mask, dtype=np.int32)
    temp_label[mask] = 1

    # First compute spatial moments
    moments = cv2.moments(temp_label)

    # Then compute Hu Moments
    hu_moments = cv2.HuMoments(moments).flatten()  # Flatten to 1D array

    props = regionprops(temp_label)
    # regionprops returns a list (one entry for each label). Here we have exactly one label=1.
    rprop = props[0]

    # 3D inertia
    # rprop.inertia_tensor is a 3x3 matrix
    inertia_tensor = rprop.inertia_tensor
    inertia_eigs = rprop.inertia_tensor_eigvals  # 3 eigenvalues
    # You could sort them descending or ascending as needed

    # For the 3D centroid, regionprops also gives:
    # rprop.centroid, which should match com_vox in many cases.

    # ----------------------------------------------------------
    # Compactness / "convexness" in 3D
    # ----------------------------------------------------------
    # As regionprops3D does not yet provide a built-in "convex_volume" for 3D,
    # we can do it manually via the convex hull of the voxel coordinates in mm space.

    # Get all voxel coordinates for this blob
    coords_vox = np.column_stack(np.where(mask))

    # Compute 3D convex hull
    convex_vol = compute_3d_convex_hull_volume(coords_vox)
    # "Solidity" or "convexness" = volume of region / volume of hull
    # If convex_vol is zero (e.g., degenerate region), set ratio to 1.0 or np.nan
    if convex_vol > 0.0:
        convexness = region_volume_vox / convex_vol
    else:
        convexness = np.nan

    # ----------------------------------------------------------
    # T1/T2 intensity metrics within the blob
    # ----------------------------------------------------------
    t1_vals = t1_data[mask]
    t2_vals = t2_data[mask]

    t1_mean = np.mean(t1_vals) if t1_vals.size > 0 else np.nan
    t1_median = np.median(t1_vals) if t1_vals.size > 0 else np.nan
    t1_mode = stats.mode(t1_vals, keepdims=False)[0] if t1_vals.size > 0 else np.nan
    t1_min, t1_max = (np.min(t1_vals), np.max(t1_vals)) if t1_vals.size > 0 else (np.nan, np.nan)
    np.var(t1_vals)
    np.std(t1_vals)
    scipy.stats.skew(t1_vals)
    scipy.stats.kurtosis(t1_vals)
    n = 5  # 5th moment
    moment5 = np.mean((t1_vals - np.mean(t1_vals)) ** n)
    entropy = shannon_entropy(t1_vals)
    # First compute spatial moments
    moments = cv2.moments(temp_label)
    # Then compute Hu Moments
    hu_moments = cv2.HuMoments(moments).flatten()  # Flatten to 1D array

    t2_mean = np.mean(t2_vals) if t2_vals.size > 0 else np.nan
    t2_median = np.median(t2_vals) if t2_vals.size > 0 else np.nan
    t2_mode = stats.mode(t2_vals, keepdims=False)[0] if t2_vals.size > 0 else np.nan
    t2_min, t2_max = (np.min(t2_vals), np.max(t2_vals)) if t2_vals.size > 0 else (np.nan, np.nan)

    # ------------------------------------------------------
    # 3D inertia with regionprops
    # ------------------------------------------------------
    # regionprops expects a labeled image. We'll isolate just this region as label=1.
    temp_label = np.zeros_like(mask, dtype=np.int32)
    temp_label[mask] = 1

    rprops = regionprops(temp_label)
    # There should be exactly one region
    rp = rprops[0]

    inertia_tensor_3d = rp.inertia_tensor  # 3x3
    inertia_eigs_3d = rp.inertia_tensor_eigvals  # length 3
    centroid_vox_3d = rp.centroid  # (z, y, x) in voxel coords
    # regionprops centroid is (row, col, slice) => (y, x, z) in 2D, or (z, y, x) in 3D
    # For nib, we might want to reorder or carefully interpret if needed
    # Note: The above reversing depends on your dimension ordering. 
    # Check carefully. Or just compare with e.g. ndimage.center_of_mass.

    # ------------------------------------------------------
    # "4D" inertia with T1 intensities
    # ------------------------------------------------------
    t1_vals = t1_data[mask]
    inertia_4d_T1, eigvals_4d_T1, eigvecs_4d_T1, centroid_4d_T1 = \
        compute_4d_inertia(coords_vox, t1_vals)

    # ------------------------------------------------------
    # "4D" inertia with T2 intensities
    # ------------------------------------------------------
    t2_vals = t2_data[mask]
    inertia_4d_T2, eigvals_4d_T2, eigvecs_4d_T2, centroid_4d_T2 = \
        compute_4d_inertia(coords_vox, t2_vals)

    # ----------------------------------------------------------
    # Assemble results for this label
    # ----------------------------------------------------------
    result_dict = {
        "label": label_val,
        "volume_vox": region_volume_vox,
        "center_of_mass_vox": com_vox,
        "convex_hull_volume_vox": convex_vol,
        "convexness": convexness,

        # 3D inertia:
        "inertia_tensor_3d": inertia_tensor_3d,  # 3x3
        "inertia_eigs_3d": inertia_eigs_3d,  # length 3
        "centroid_vox_3d": centroid_vox_3d,

        # 4D inertia with T1:
        "inertia_4d_T1": inertia_4d_T1 if inertia_4d_T1 is not None else np.nan,
        "eigvals_4d_T1": eigvals_4d_T1 if eigvals_4d_T1 is not None else np.nan,
        "centroid_4d_T1": centroid_4d_T1 if centroid_4d_T1 is not None else np.nan,

        # 4D inertia with T2:
        "inertia_4d_T2": inertia_4d_T2 if inertia_4d_T2 is not None else np.nan,
        "eigvals_4d_T2": eigvals_4d_T2 if eigvals_4d_T2 is not None else np.nan,
        "centroid_4d_T2": centroid_4d_T2 if centroid_4d_T2 is not None else np.nan,

        "t1_mean": t1_mean,
        "t1_median": t1_median,
        "t1_mode": t1_mode,
        "t1_min": t1_min,
        "t1_max": t1_max,

        "t2_mean": t2_mean,
        "t2_median": t2_median,
        "t2_mode": t2_mode,
        "t2_min": t2_min,
        "t2_max": t2_max,
    }

    results.append(result_dict)

# -------------------------------------------------------------------------
# 3. Create a DataFrame of all results and save
# -------------------------------------------------------------------------
df = pd.DataFrame(results)
print(df)

df.to_csv("lesion_measurements.csv", index=False)
