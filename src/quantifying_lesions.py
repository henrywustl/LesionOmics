"""Modality-agnostic lesion quantification.

Extracts morphological, intensity, texture, neighbourhood, probabilistic, and
Jacobian features for every labeled lesion in a NIfTI image.
"""

import logging

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import ConvexHull
from skimage.measure import regionprops, marching_cubes, mesh_surface_area, shannon_entropy

logger = logging.getLogger(__name__)

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False

try:
    from radiomics import featureextractor as _rad_extractor
    import SimpleITK as _sitk
    _RADIOMICS_AVAILABLE = True
except Exception:
    _RADIOMICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public helper: voxel volume
# ---------------------------------------------------------------------------

def voxel_volume_from_header(header) -> float:
    """Return the volume of one voxel in mm³ from a NIfTI header."""
    zooms = header.get_zooms()[:3]
    return float(zooms[0] * zooms[1] * zooms[2])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _convex_hull_volume(coords: np.ndarray) -> float:
    """Convex-hull volume of an (N×3) mm-space coordinate array."""
    if coords.shape[0] < 4:
        return 0.0
    extents = coords.max(axis=0) - coords.min(axis=0)
    if np.count_nonzero(extents) < 3:
        return 0.0
    try:
        return float(ConvexHull(coords).volume)
    except Exception:
        return 0.0


def _surface_area_mm2(mask: np.ndarray, voxel_sizes: tuple) -> float:
    """Surface area in mm² via marching cubes."""
    try:
        verts, faces, _, _ = marching_cubes(mask.astype(float), level=0.5, spacing=voxel_sizes)
        return float(mesh_surface_area(verts, faces))
    except Exception:
        return np.nan


def _surface_curvature_stats(mask: np.ndarray, voxel_sizes: tuple) -> dict:
    """Discrete mean-curvature statistics at the lesion surface (requires trimesh)."""
    nan_result = {
        "curvature_mean": np.nan,
        "curvature_std": np.nan,
        "curvature_abs_max": np.nan,
    }
    if not _TRIMESH_AVAILABLE:
        return nan_result
    try:
        verts, faces, _, _ = marching_cubes(mask.astype(float), level=0.5, spacing=voxel_sizes)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        radius = max(voxel_sizes)
        curv = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, radius=radius)
        return {
            "curvature_mean": float(np.mean(curv)),
            "curvature_std": float(np.std(curv)),
            "curvature_abs_max": float(np.max(np.abs(curv))),
        }
    except Exception:
        return nan_result


def _compute_4d_inertia(coords: np.ndarray, intensities: np.ndarray) -> tuple:
    """Eigenvalues of the (spatial + intensity) 4-D inertia tensor."""
    if len(coords) == 0:
        return np.full(4, np.nan), None
    points_4d = np.column_stack([coords, intensities])
    centroid = points_4d.mean(axis=0)
    centered = points_4d - centroid
    inertia = centered.T @ centered
    eigvals = np.linalg.eigh(inertia)[0]
    return eigvals, centroid


def _intensity_stats(vals: np.ndarray, prefix: str) -> dict:
    """11 intensity statistics for a set of lesion voxel values.

    Dropped from the original 19: var (= std²), cv (= std/mean), rms
    (correlated with mean/energy), moment5 (noise-dominated beyond kurtosis),
    and the four granular percentiles p10/p25/p75/p90 (iqr already covers
    the interquartile spread; min/max bound the full range).
    """
    keys = [
        "mean", "median", "mode", "min", "max", "std",
        "skewness", "kurtosis", "entropy", "iqr", "energy",
    ]
    if vals.size == 0:
        return {f"{prefix}_{k}": np.nan for k in keys}

    mean = float(np.mean(vals))
    p25 = float(np.percentile(vals, 25))
    p75 = float(np.percentile(vals, 75))

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": float(np.median(vals)),
        f"{prefix}_mode": float(stats.mode(vals, keepdims=False)[0]),
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_max": float(np.max(vals)),
        f"{prefix}_std": float(np.std(vals)),
        f"{prefix}_skewness": float(stats.skew(vals)),
        f"{prefix}_kurtosis": float(stats.kurtosis(vals)),
        f"{prefix}_entropy": float(shannon_entropy(vals)),
        f"{prefix}_iqr": float(p75 - p25),
        f"{prefix}_energy": float(np.sum(vals ** 2)),
    }


def _boundary_gradient_stats(data: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    """Gradient-magnitude statistics sampled at the lesion surface."""
    surface = mask & ~ndimage.binary_erosion(mask)
    if not np.any(surface):
        return {
            f"{prefix}_boundary_grad_mean": np.nan,
            f"{prefix}_boundary_grad_std": np.nan,
        }
    gradients = np.gradient(data)
    grad_magnitude = np.sqrt(sum(g ** 2 for g in gradients))
    grad_vals = grad_magnitude[surface]
    return {
        f"{prefix}_boundary_grad_mean": float(np.mean(grad_vals)),
        f"{prefix}_boundary_grad_std": float(np.std(grad_vals)),
    }


def _perilesional_ring_stats(
    data: np.ndarray,
    mask: np.ndarray,
    prefix: str,
    n_dilations: int = 3,
) -> dict:
    """Intensity stats in the perilesional ring (dilation band around the lesion)."""
    ring = ndimage.binary_dilation(mask, iterations=n_dilations) & ~mask
    ring_vals = data[ring] if np.any(ring) else np.array([])
    lesion_vals = data[mask]

    ring_mean = float(np.mean(ring_vals)) if ring_vals.size > 0 else np.nan
    ring_std = float(np.std(ring_vals)) if ring_vals.size > 0 else np.nan
    lesion_mean = float(np.mean(lesion_vals)) if lesion_vals.size > 0 else np.nan

    if ring_mean and not np.isnan(ring_mean) and ring_mean != 0:
        ratio = lesion_mean / ring_mean
    else:
        ratio = np.nan

    return {
        f"{prefix}_perilesional_mean": ring_mean,
        f"{prefix}_perilesional_std": ring_std,
        f"{prefix}_lesion_perilesional_ratio": ratio,
    }


def _concentric_shell_stats(
    data: np.ndarray,
    mask: np.ndarray,
    prefix: str,
    n_shells: int = 5,
) -> dict:
    """Radial intensity profile via concentric erosion shells.

    Shell 0 = outermost surface layer; shell n-1 = innermost core.
    Also computes rim-to-core ratio and linear gradient slope across shells.
    """
    shells = []
    current = mask.copy()

    for i in range(n_shells):
        eroded = ndimage.binary_erosion(current)
        if not np.any(eroded):
            shells.append(current.copy())  # remaining voxels form the last shell
            # Pad with empty shells for consistent column count
            for _ in range(i + 1, n_shells):
                shells.append(np.zeros_like(mask, dtype=bool))
            break
        shells.append(current & ~eroded)
        current = eroded
    else:
        # All n_shells completed; remaining core is NOT added as an extra shell
        pass

    result = {}
    shell_means = []

    for i, shell in enumerate(shells):
        vals = data[shell] if np.any(shell) else np.array([])
        n_vox = int(np.sum(shell))
        mean_val = float(np.mean(vals)) if vals.size > 0 else np.nan
        std_val = float(np.std(vals)) if vals.size > 0 else np.nan
        result[f"{prefix}_shell_{i}_mean"] = mean_val
        result[f"{prefix}_shell_{i}_std"] = std_val
        shell_means.append(mean_val)

    rim_mean = shell_means[0] if shell_means else np.nan
    core_mean = shell_means[-1] if shell_means else np.nan

    if not np.isnan(rim_mean) and not np.isnan(core_mean) and core_mean != 0:
        result[f"{prefix}_rim_core_ratio"] = float(rim_mean / core_mean)
    else:
        result[f"{prefix}_rim_core_ratio"] = np.nan

    valid = [(i, m) for i, m in enumerate(shell_means) if not np.isnan(m)]
    if len(valid) >= 2:
        xs = np.array([v[0] for v in valid], dtype=float)
        ys = np.array([v[1] for v in valid], dtype=float)
        slope, _, r, _, _ = stats.linregress(xs, ys)
        result[f"{prefix}_shell_gradient_slope"] = float(slope)
        result[f"{prefix}_shell_gradient_r2"] = float(r ** 2)
    else:
        result[f"{prefix}_shell_gradient_slope"] = np.nan
        result[f"{prefix}_shell_gradient_r2"] = np.nan

    return result


def _mutual_information(v1: np.ndarray, v2: np.ndarray, n_bins: int = 32) -> float:
    """Histogram-based mutual information (pure numpy, no sklearn required)."""
    if v1.size != v2.size or v1.size == 0:
        return np.nan
    hist_2d, _, _ = np.histogram2d(v1.ravel(), v2.ravel(), bins=n_bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    px_py = px * py
    nz = (pxy > 0) & (px_py > 0)
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz])))


def _build_radiomics_extractor(include_wavelet: bool = False, include_log: bool = False):
    """Build a PyRadiomics extractor (GLCM, GLRLM, GLSZM, NGTDM)."""
    if not _RADIOMICS_AVAILABLE:
        return None
    try:
        settings = {"binWidth": 25, "enableCExtensions": True}
        extractor = _rad_extractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        for cls in ("glcm", "glrlm", "glszm", "ngtdm"):
            extractor.enableFeatureClassByName(cls)
        if include_wavelet:
            extractor.enableImageTypeByName("Wavelet")
        if include_log:
            extractor.enableImageTypeByName("LoG")
        return extractor
    except Exception as exc:
        logger.warning("Could not build PyRadiomics extractor: %s", exc)
        return None


def _radiomics_features(
    mask: np.ndarray,
    data: np.ndarray,
    header,
    prefix: str,
    extractor,
) -> dict:
    """Run PyRadiomics on a single lesion mask and return scalar features."""
    if extractor is None or not _RADIOMICS_AVAILABLE:
        return {}
    try:
        vox_sizes = [float(z) for z in header.get_zooms()[:3]]
        sitk_img = _sitk.GetImageFromArray(data.astype(np.float32))
        sitk_img.SetSpacing(vox_sizes)
        sitk_mask = _sitk.GetImageFromArray(mask.astype(np.uint32))
        sitk_mask.SetSpacing(vox_sizes)
        feat = extractor.execute(sitk_img, sitk_mask)
        return {
            f"{prefix}_radiomics_{k}": float(v)
            for k, v in feat.items()
            if any(k.startswith(p) for p in ("original_", "wavelet", "log"))
        }
    except Exception as exc:
        logger.warning("PyRadiomics failed for %s: %s", prefix, exc)
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quantify_lesions(
    labeled_img: nib.Nifti1Image,
    modalities: dict,
    prob_map: nib.Nifti1Image = None,
    nawm_mask: nib.Nifti1Image = None,
    jacobian_map: nib.Nifti1Image = None,
    include_radiomics: bool = True,
    include_wavelet: bool = False,
    include_log: bool = False,
    n_shells: int = 5,
) -> pd.DataFrame:
    """Quantify features for every labeled lesion.

    Args:
        labeled_img: NIfTI with integer lesion labels (0 = background).
        modalities: Dict mapping modality name to NIfTI image, e.g.
                    ``{"T1": t1_img, "T2": t2_img}``.  Any number of
                    modalities accepted.
        prob_map: Optional probabilistic segmentation (same space).
        nawm_mask: Optional NAWM binary mask for intensity normalisation.
        jacobian_map: Optional Jacobian-determinant image.
        include_radiomics: Compute PyRadiomics texture features if available.
        include_wavelet: Add wavelet filter bank to PyRadiomics (slow).
        include_log: Add LoG filter bank to PyRadiomics (slow).
        n_shells: Number of concentric erosion shells for radial profiles.

    Returns:
        DataFrame with one row per lesion and one column per feature.
    """
    labeled_data = labeled_img.get_fdata()
    header = labeled_img.header
    affine = labeled_img.affine
    voxel_sizes = tuple(float(z) for z in header.get_zooms()[:3])
    voxel_vol = float(voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2])

    mod_arrays = {name: img.get_fdata() for name, img in modalities.items()}
    mod_names = list(mod_arrays.keys())

    prob_data = prob_map.get_fdata() if prob_map is not None else None
    nawm_data = nawm_mask.get_fdata().astype(bool) if nawm_mask is not None else None
    jac_data = jacobian_map.get_fdata() if jacobian_map is not None else None

    all_labels = np.unique(labeled_data)
    all_labels = all_labels[all_labels != 0].astype(int)

    # Precompute mm-space centroids for neighbourhood density
    centroids_mm: dict[int, np.ndarray] = {}
    for lv in all_labels:
        coords = np.column_stack(np.where(labeled_data == lv))
        com_vox = coords.mean(axis=0)
        centroids_mm[int(lv)] = nib.affines.apply_affine(affine, com_vox)

    radiomics_extractor = (
        _build_radiomics_extractor(include_wavelet, include_log)
        if include_radiomics
        else None
    )

    results = []

    for label_val in all_labels:
        mask = labeled_data == label_val
        coords_vox = np.column_stack(np.where(mask))
        r: dict = {"label": int(label_val)}

        # ---- Volume -------------------------------------------------------
        n_vox = int(np.count_nonzero(mask))
        vol_mm3 = n_vox * voxel_vol
        r["volume_mm3"] = vol_mm3

        # ---- Bounding box (computed for spatial tracking; not emitted to CSV) -
        # bbox_{z,y,x}_{min,max} are retained here for longitudinal pipelines.

        # ---- Centre of mass -----------------------------------------------
        com_vox = tuple(float(c) for c in ndimage.center_of_mass(mask))
        com_mm = tuple(float(c) for c in nib.affines.apply_affine(affine, com_vox))
        for i, ax in enumerate(["z", "y", "x"]):
            r[f"com_mm_{ax}"] = com_mm[i]

        # ---- Convex hull --------------------------------------------------
        coords_mm_arr = nib.affines.apply_affine(affine, coords_vox)
        hull_vol = _convex_hull_volume(coords_mm_arr)
        r["convex_hull_volume_mm3"] = hull_vol
        r["solidity"] = (vol_mm3 / hull_vol) if hull_vol > 0 else np.nan

        # ---- Surface area + sphericity ------------------------------------
        sa_mm2 = _surface_area_mm2(mask, voxel_sizes)
        r["surface_area_mm2"] = sa_mm2
        if not np.isnan(sa_mm2) and sa_mm2 > 0:
            r["sphericity"] = float((np.pi ** (1 / 3)) * (6 * vol_mm3) ** (2 / 3) / sa_mm2)
        else:
            r["sphericity"] = np.nan

        # ---- Surface curvature --------------------------------------------
        r.update(_surface_curvature_stats(mask, voxel_sizes))

        # ---- 3-D inertia tensor -------------------------------------------
        temp_label = np.zeros_like(mask, dtype=np.int32)
        temp_label[mask] = 1
        rp = regionprops(temp_label)[0]
        # Raw eigenvalues omitted from CSV — elongation/flatness/isotropy capture
        # the meaningful relationships between them.
        eigs = np.sort(rp.inertia_tensor_eigvals)[::-1]  # descending
        e1, e2, e3 = eigs[0], eigs[1], eigs[2]
        r["elongation"] = float(np.sqrt(e2 / e1)) if e1 > 0 else np.nan
        r["flatness"] = float(np.sqrt(e3 / e1)) if e1 > 0 else np.nan
        r["isotropy"] = float(np.sqrt(e3 / e2)) if e2 > 0 else np.nan

        # ---- Topology (Euler number) --------------------------------------
        r["euler_number"] = int(rp.euler_number)

        # ---- Neighbourhood density ----------------------------------------
        own_mm = centroids_mm[int(label_val)]
        dists_mm = [
            float(np.linalg.norm(own_mm - other_mm))
            for other_lv, other_mm in centroids_mm.items()
            if other_lv != int(label_val)
        ]
        if dists_mm:
            r["nearest_lesion_dist_mm"] = float(min(dists_mm))
            r["lesions_within_10mm"] = int(sum(d < 10 for d in dists_mm))
            r["lesions_within_20mm"] = int(sum(d < 20 for d in dists_mm))
            r["lesions_within_30mm"] = int(sum(d < 30 for d in dists_mm))
            r["mean_dist_to_all_lesions_mm"] = float(np.mean(dists_mm))
        else:
            r["nearest_lesion_dist_mm"] = np.nan
            r["lesions_within_10mm"] = 0
            r["lesions_within_20mm"] = 0
            r["lesions_within_30mm"] = 0
            r["mean_dist_to_all_lesions_mm"] = np.nan

        # ---- Per-modality features ----------------------------------------
        for mod_name, data_arr in mod_arrays.items():
            mod_vals = data_arr[mask]

            # 4-D inertia
            eigvals_4d, _ = _compute_4d_inertia(coords_vox.astype(float), mod_vals)
            for i, ev in enumerate(eigvals_4d):
                r[f"{mod_name}_4d_eigval_{i}"] = float(ev)

            # 19 intensity stats
            r.update(_intensity_stats(mod_vals, mod_name))

            # Boundary gradient
            r.update(_boundary_gradient_stats(data_arr, mask, mod_name))

            # Perilesional ring
            r.update(_perilesional_ring_stats(data_arr, mask, mod_name))

            # Concentric shells
            r.update(_concentric_shell_stats(data_arr, mask, mod_name, n_shells=n_shells))

            # NAWM normalisation
            if nawm_data is not None and np.any(nawm_data):
                nawm_vals = data_arr[nawm_data]
                nawm_mean = float(np.mean(nawm_vals)) if nawm_vals.size > 0 else np.nan
                r[f"{mod_name}_nawm_normalized_mean"] = (
                    float(np.mean(mod_vals) / nawm_mean)
                    if nawm_mean and nawm_mean != 0
                    else np.nan
                )

            # PyRadiomics
            if radiomics_extractor is not None:
                r.update(_radiomics_features(mask, data_arr, header, mod_name, radiomics_extractor))

        # ---- Pairwise modality features -----------------------------------
        for i, n1 in enumerate(mod_names):
            for n2 in mod_names[i + 1:]:
                v1 = mod_arrays[n1][mask]
                v2 = mod_arrays[n2][mask]
                mean1 = float(np.mean(v1))
                mean2 = float(np.mean(v2))
                r[f"{n1}_over_{n2}_mean_ratio"] = mean1 / mean2 if mean2 != 0 else np.nan
                if v1.size > 1:
                    corr, _ = stats.pearsonr(v1, v2)
                    r[f"{n1}_{n2}_correlation"] = float(corr)
                else:
                    r[f"{n1}_{n2}_correlation"] = np.nan
                r[f"{n1}_{n2}_mutual_info"] = _mutual_information(v1, v2)

        # ---- Jacobian features --------------------------------------------
        if jac_data is not None:
            jac_vals = jac_data[mask]
            r["jacobian_mean"] = float(np.mean(jac_vals))
            r["jacobian_std"] = float(np.std(jac_vals))
            r["jacobian_median"] = float(np.median(jac_vals))
            r["jacobian_expanding_fraction"] = float(np.mean(jac_vals > 1))
            r["jacobian_contracting_fraction"] = float(np.mean(jac_vals < 1))

        # ---- Probabilistic features ---------------------------------------
        if prob_data is not None:
            prob_vals = prob_data[mask]
            total_prob = float(np.sum(prob_vals))
            r["fuzzy_volume_mm3"] = total_prob * voxel_vol
            r["mean_confidence"] = float(np.mean(prob_vals))
            r["std_confidence"] = float(np.std(prob_vals))
            if total_prob > 0:
                for dim, ax in enumerate(["z", "y", "x"]):
                    r[f"prob_weighted_centroid_{ax}"] = float(
                        np.sum(coords_vox[:, dim] * prob_vals) / total_prob
                    )
                pn = prob_vals / total_prob
                nz = pn > 0
                r["prob_entropy"] = float(-np.sum(pn[nz] * np.log(pn[nz])))
            else:
                for ax in ["z", "y", "x"]:
                    r[f"prob_weighted_centroid_{ax}"] = np.nan
                r["prob_entropy"] = np.nan
            r["core_volume_mm3"] = float(np.sum(prob_vals > 0.8) * voxel_vol)
            r["rim_volume_mm3"] = float(np.sum((prob_vals > 0.3) & (prob_vals <= 0.8)) * voxel_vol)

        results.append(r)

    return pd.DataFrame(results)
