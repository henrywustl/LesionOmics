"""
Per-lesion feature extraction for 3D neuroimaging data.

Computes shape, intensity, texture (via PyRadiomics), and probabilistic
features for every labeled region in a NIfTI segmentation image.  Any
number of imaging modalities can be supplied — the same feature set is
computed for each channel automatically.
"""

import warnings
from itertools import combinations

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.ndimage import sobel
from scipy.spatial import ConvexHull
from skimage.measure import euler_number, marching_cubes, mesh_surface_area, regionprops

try:
    import radiomics  # noqa: F401
    _RADIOMICS_AVAILABLE = True
except ImportError:
    _RADIOMICS_AVAILABLE = False
    warnings.warn(
        "pyradiomics is not installed — texture features will be skipped. "
        "Install with: pip install pyradiomics",
        ImportWarning,
        stacklevel=2,
    )

try:
    import trimesh as _trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    _TRIMESH_AVAILABLE = False


# ── Helpers ────────────────────────────────────────────────────────────────────


def voxel_volume_from_header(header) -> float:
    """Voxel volume in mm³ from a NIfTI header."""
    zooms = header.get_zooms()[:3]
    return float(zooms[0] * zooms[1] * zooms[2])


def _convex_hull_volume(coords: np.ndarray) -> float:
    """Convex hull volume (voxel units) of an (N, 3) integer coordinate array."""
    if coords.shape[0] < 4:
        return 0.0
    if np.count_nonzero(coords.max(axis=0) - coords.min(axis=0)) < 3:
        return 0.0
    return float(ConvexHull(coords).volume)


def _surface_area_mm2(mask: np.ndarray, voxel_sizes: tuple) -> float:
    """Surface area in mm² via marching cubes.  Returns NaN for degenerate regions."""
    try:
        verts, faces, _, _ = marching_cubes(
            mask.astype(np.float32), level=0.5, spacing=voxel_sizes
        )
        return float(mesh_surface_area(verts, faces))
    except Exception:
        return np.nan


def _surface_curvature_stats(mask: np.ndarray, voxel_sizes: tuple) -> dict:
    """
    Discrete mean curvature statistics on the lesion surface mesh.

    Requires trimesh (pip install trimesh).  Returns NaN columns when
    trimesh is unavailable or the mesh is degenerate.

    Returns keys: curvature_mean, curvature_std, curvature_abs_max
    """
    nan_row = {"curvature_mean": np.nan, "curvature_std": np.nan, "curvature_abs_max": np.nan}
    if not _TRIMESH_AVAILABLE:
        return nan_row
    try:
        verts, faces, _, _ = marching_cubes(
            mask.astype(np.float32), level=0.5, spacing=voxel_sizes
        )
        mesh = _trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        radius = float(max(voxel_sizes)) * 2.0
        H = _trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=radius
        )
        # Normalise by integration ball area to get curvature per mm²
        H_norm = H / (np.pi * radius ** 2)
        return {
            "curvature_mean":    float(np.mean(H_norm)),
            "curvature_std":     float(np.std(H_norm)),
            "curvature_abs_max": float(np.max(np.abs(H_norm))),
        }
    except Exception:
        return nan_row


def _compute_4d_inertia(coords: np.ndarray, intensities: np.ndarray):
    """
    Second-moment matrix in 4D: (x, y, z, intensity).
    Returns (eigvals_4, centroid_4) or (None, None) for empty regions.
    """
    if coords.shape[0] == 0:
        return None, None
    points = np.column_stack([coords, intensities])
    centroid = points.mean(axis=0)
    centered = points - centroid
    eigvals = np.linalg.eigvalsh(centered.T @ centered)
    return eigvals, centroid


def _intensity_stats(vals: np.ndarray, prefix: str) -> dict:
    """Full intensity statistics for a flat array of voxel values."""
    _keys = (
        "mean", "median", "mode", "min", "max", "var", "std",
        "skewness", "kurtosis", "moment5", "entropy",
        "iqr", "p10", "p25", "p75", "p90", "cov", "energy", "rms",
    )
    if vals.size == 0:
        return {f"{prefix}_{k}": np.nan for k in _keys}

    mean = float(np.mean(vals))
    std = float(np.std(vals))
    p10, p25, p75, p90 = np.percentile(vals, [10, 25, 75, 90])
    hist_counts, _ = np.histogram(vals, bins=min(64, vals.size))

    return {
        f"{prefix}_mean":     mean,
        f"{prefix}_median":   float(np.median(vals)),
        f"{prefix}_mode":     float(stats.mode(vals, keepdims=False)[0]),
        f"{prefix}_min":      float(np.min(vals)),
        f"{prefix}_max":      float(np.max(vals)),
        f"{prefix}_var":      float(np.var(vals)),
        f"{prefix}_std":      std,
        f"{prefix}_skewness": float(stats.skew(vals)),
        f"{prefix}_kurtosis": float(stats.kurtosis(vals)),
        f"{prefix}_moment5":  float(np.mean((vals - mean) ** 5)),
        f"{prefix}_entropy":  float(stats.entropy(hist_counts + 1e-9)),
        f"{prefix}_iqr":      float(p75 - p25),
        f"{prefix}_p10":      float(p10),
        f"{prefix}_p25":      float(p25),
        f"{prefix}_p75":      float(p75),
        f"{prefix}_p90":      float(p90),
        f"{prefix}_cov":      (std / mean) if mean != 0 else np.nan,
        f"{prefix}_energy":   float(np.sum(vals ** 2)),
        f"{prefix}_rms":      float(np.sqrt(np.mean(vals ** 2))),
    }


def _boundary_gradient_stats(data: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    """
    Mean and std of gradient magnitude at the 1-voxel erosion boundary.
    Captures rim-enhancing vs. diffuse signal patterns.
    """
    gx = sobel(data, axis=0)
    gy = sobel(data, axis=1)
    gz = sobel(data, axis=2)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    surface = mask & ~ndimage.binary_erosion(mask)
    surf_vals = grad_mag[surface]
    if surf_vals.size == 0:
        return {
            f"{prefix}_boundary_grad_mean": np.nan,
            f"{prefix}_boundary_grad_std":  np.nan,
        }
    return {
        f"{prefix}_boundary_grad_mean": float(np.mean(surf_vals)),
        f"{prefix}_boundary_grad_std":  float(np.std(surf_vals)),
    }


def _perilesional_ring_stats(
    data: np.ndarray, mask: np.ndarray, prefix: str, n_dilations: int = 3
) -> dict:
    """
    Intensity statistics in the perilesional tissue ring surrounding the lesion.

    Dilates the mask n_dilations times and computes stats on the band of voxels
    that are in the dilated mask but not in the original lesion mask.  Captures
    peri-lesional oedema and tissue reaction distinct from global NAWM.

    Returns: {prefix}_perilesional_mean/std and {prefix}_lesion_perilesional_ratio.
    """
    dilated = ndimage.binary_dilation(mask, iterations=n_dilations)
    ring = dilated & ~mask
    ring_vals = data[ring]
    lesion_vals = data[mask]

    if ring_vals.size == 0 or lesion_vals.size == 0:
        return {
            f"{prefix}_perilesional_mean":           np.nan,
            f"{prefix}_perilesional_std":            np.nan,
            f"{prefix}_lesion_perilesional_ratio":   np.nan,
        }

    ring_mean   = float(np.mean(ring_vals))
    lesion_mean = float(np.mean(lesion_vals))
    return {
        f"{prefix}_perilesional_mean":         ring_mean,
        f"{prefix}_perilesional_std":          float(np.std(ring_vals)),
        f"{prefix}_lesion_perilesional_ratio": lesion_mean / ring_mean if ring_mean != 0 else np.nan,
    }


def _concentric_shell_stats(
    data: np.ndarray, mask: np.ndarray, prefix: str, n_shells: int = 5
) -> dict:
    """
    Intensity statistics in concentric erosion shells from surface inward.

    Shell 0 is the outermost 1-voxel layer; shell n_shells-1 is the core.
    Captures the radial intensity profile that distinguishes rim-active
    from diffuse lesions.  Also computes a linear slope across shells
    (negative slope = brighter at the rim than core).

    Returns per-shell mean/std/n_vox, rim_core_ratio, and shell_gradient_slope/r2.
    """
    results: dict = {}
    shell_means: list = []

    current = mask.copy()
    for i in range(n_shells - 1):
        eroded = ndimage.binary_erosion(current)
        if eroded.sum() == 0:
            # Remaining mass becomes the final shell; fill rest with NaN
            shell = current
            vals = data[shell]
            results[f"{prefix}_shell_{i}_mean"]  = float(np.mean(vals)) if vals.size else np.nan
            results[f"{prefix}_shell_{i}_std"]   = float(np.std(vals))  if vals.size else np.nan
            results[f"{prefix}_shell_{i}_n_vox"] = int(vals.size)
            if vals.size:
                shell_means.append(float(np.mean(vals)))
            for j in range(i + 1, n_shells):
                results[f"{prefix}_shell_{j}_mean"]  = np.nan
                results[f"{prefix}_shell_{j}_std"]   = np.nan
                results[f"{prefix}_shell_{j}_n_vox"] = 0
            break
        shell = current & ~eroded
        vals = data[shell]
        results[f"{prefix}_shell_{i}_mean"]  = float(np.mean(vals)) if vals.size else np.nan
        results[f"{prefix}_shell_{i}_std"]   = float(np.std(vals))  if vals.size else np.nan
        results[f"{prefix}_shell_{i}_n_vox"] = int(vals.size)
        if vals.size:
            shell_means.append(float(np.mean(vals)))
        current = eroded
    else:
        # Core shell (innermost)
        vals = data[current]
        results[f"{prefix}_shell_{n_shells - 1}_mean"]  = float(np.mean(vals)) if vals.size else np.nan
        results[f"{prefix}_shell_{n_shells - 1}_std"]   = float(np.std(vals))  if vals.size else np.nan
        results[f"{prefix}_shell_{n_shells - 1}_n_vox"] = int(vals.size)
        if vals.size:
            shell_means.append(float(np.mean(vals)))

    # Rim-to-core ratio (shell_0 / deepest valid shell)
    core_mean = results.get(f"{prefix}_shell_{min(n_shells - 1, len(shell_means) - 1)}_mean", np.nan)
    rim_mean  = results.get(f"{prefix}_shell_0_mean", np.nan)
    results[f"{prefix}_rim_core_ratio"] = (
        rim_mean / core_mean
        if not (np.isnan(rim_mean) or np.isnan(core_mean) or core_mean == 0)
        else np.nan
    )

    # Linear slope from surface (i=0) inward: negative = higher at rim
    if len(shell_means) >= 2:
        xs = np.arange(len(shell_means), dtype=float)
        slope, _, r, _, _ = stats.linregress(xs, shell_means)
        results[f"{prefix}_shell_gradient_slope"] = float(slope)
        results[f"{prefix}_shell_gradient_r2"]    = float(r ** 2)
    else:
        results[f"{prefix}_shell_gradient_slope"] = np.nan
        results[f"{prefix}_shell_gradient_r2"]    = np.nan

    return results


def _mutual_information(v1: np.ndarray, v2: np.ndarray, n_bins: int = 32) -> float:
    """
    Histogram-based mutual information between two intensity arrays.

    Complements Pearson correlation by capturing non-linear co-dependence.
    Pure numpy — no sklearn required.
    """
    if v1.size < 4:
        return np.nan
    try:
        hist_2d, _, _ = np.histogram2d(v1, v2, bins=n_bins)
        pxy = hist_2d / (hist_2d.sum() + 1e-12)
        px  = pxy.sum(axis=1, keepdims=True)
        py  = pxy.sum(axis=0, keepdims=True)
        px_py = px * py
        nonzero = pxy > 0
        mi = float(np.sum(pxy[nonzero] * np.log(pxy[nonzero] / (px_py[nonzero] + 1e-12))))
        return max(mi, 0.0)  # numerical noise can produce tiny negatives
    except Exception:
        return np.nan


def _build_radiomics_extractor(
    include_wavelet: bool = False,
    include_log: bool = False,
):
    """Instantiate a PyRadiomics extractor with texture (and optionally filter-bank) features."""
    import logging
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    from radiomics import featureextractor

    settings = {"binWidth": 25, "enableCExtensions": True, "geometryTolerance": 1e-6}
    ex = featureextractor.RadiomicsFeatureExtractor(**settings)
    ex.disableAllFeatures()
    for cls in ("glcm", "glrlm", "glszm", "ngtdm"):
        ex.enableFeatureClassByName(cls)
    if include_wavelet:
        ex.enableImageTypeByName("Wavelet")
    if include_log:
        ex.enableImageTypeByName("LoG", customArgs={"sigma": [1.0, 2.0, 3.0]})
    return ex


def _radiomics_features(
    mask: np.ndarray,
    data: np.ndarray,
    header,
    prefix: str,
    extractor,
) -> dict:
    """Run PyRadiomics on a single lesion mask crop and return a flat feature dict."""
    if extractor is None:
        return {}
    try:
        import SimpleITK as sitk

        coords = np.argwhere(mask)
        lo = np.maximum(coords.min(axis=0) - 2, 0)
        hi = np.minimum(coords.max(axis=0) + 3, np.array(mask.shape))
        slc = tuple(slice(lo[i], hi[i]) for i in range(3))

        img_crop = data[slc].astype(np.float32)
        msk_crop = mask[slc].astype(np.uint8)
        spacing = [float(z) for z in header.get_zooms()[:3]]

        img_sitk = sitk.GetImageFromArray(img_crop)
        img_sitk.SetSpacing(spacing)
        msk_sitk = sitk.GetImageFromArray(msk_crop)
        msk_sitk.SetSpacing(spacing)

        result = extractor.execute(img_sitk, msk_sitk, label=1)
        return {
            f"{prefix}_radiomics_{k.split('_', 1)[-1]}": float(v)
            for k, v in result.items()
            if not k.startswith("diagnostics")
            and isinstance(v, (int, float, np.floating))
        }
    except Exception as exc:
        warnings.warn(
            f"PyRadiomics skipped for '{prefix}' ({mask.sum()} vox): {exc}"
        )
        return {}


# ── Public API ─────────────────────────────────────────────────────────────────


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
    """
    Compute per-lesion features from a labeled NIfTI segmentation.

    Args:
        labeled_img:       NIfTI with integer lesion labels (0 = background).
        modalities:        Dict mapping modality name to NIfTI image, e.g.
                           {"T1": t1_img, "T2": t2_img, "FLAIR": flair_img}.
                           Any 3D imaging modality is accepted.
        prob_map:          Optional probabilistic segmentation NIfTI (0-1 floats).
                           Enables fuzzy/confidence feature columns.
        nawm_mask:         Optional binary NAWM mask NIfTI.  Enables NAWM-
                           normalised intensity columns per modality.
        jacobian_map:      Optional Jacobian determinant NIfTI (from
                           jacobian_determinant.compute_lesion_jacobian_pipeline).
                           Enables per-lesion expansion/contraction statistics.
        include_radiomics: Whether to run PyRadiomics texture extraction.
        include_wavelet:   Enable PyRadiomics wavelet filter bank (slow).
        include_log:       Enable PyRadiomics LoG filter bank at sigma 1/2/3 mm (slow).
        n_shells:          Number of concentric erosion shells (default 5).

    Returns:
        pd.DataFrame with one row per lesion.
    """
    labeled_data = labeled_img.get_fdata()
    affine       = labeled_img.affine
    header       = labeled_img.header
    voxel_vol    = voxel_volume_from_header(header)
    voxel_sizes  = tuple(float(z) for z in header.get_zooms()[:3])

    mod_arrays = {name: img.get_fdata() for name, img in modalities.items()}
    mod_names  = list(mod_arrays.keys())

    # NAWM reference statistics per modality
    nawm_stats: dict = {}
    if nawm_mask is not None:
        nawm_bin = nawm_mask.get_fdata().astype(bool)
        for name, arr in mod_arrays.items():
            w = arr[nawm_bin]
            nawm_stats[name] = (float(np.mean(w)), float(np.std(w)))

    prob_data     = prob_map.get_fdata()    if prob_map     is not None else None
    jacobian_data = jacobian_map.get_fdata() if jacobian_map is not None else None

    all_labels = np.unique(labeled_data)
    all_labels = all_labels[all_labels != 0]

    # Precompute centroids in both voxel and mm space
    centroids: dict = {
        lv: ndimage.center_of_mass(labeled_data == lv) for lv in all_labels
    }
    centroids_mm: dict = {
        lv: nib.affines.apply_affine(affine, np.array(c))
        for lv, c in centroids.items()
    }

    extractor = None
    if include_radiomics and _RADIOMICS_AVAILABLE:
        extractor = _build_radiomics_extractor(
            include_wavelet=include_wavelet, include_log=include_log
        )

    results = []
    for label_val in all_labels:
        mask   = labeled_data == label_val
        coords = np.argwhere(mask)       # (N, 3) voxel indices
        row: dict = {"label": int(label_val)}

        # ── Volume ─────────────────────────────────────────────────────────
        n_vox   = int(coords.shape[0])
        vol_mm3 = n_vox * voxel_vol
        row["volume_vox"] = n_vox
        row["volume_mm3"] = vol_mm3

        # ── Centre of mass ──────────────────────────────────────────────────
        com_vox = centroids[label_val]
        com_mm  = centroids_mm[label_val]
        row.update({
            "com_x_vox": com_vox[0], "com_y_vox": com_vox[1], "com_z_vox": com_vox[2],
            "com_x_mm":  float(com_mm[0]),
            "com_y_mm":  float(com_mm[1]),
            "com_z_mm":  float(com_mm[2]),
            "hemisphere": "R" if com_mm[0] >= 0 else "L",
        })

        # ── Bounding box ────────────────────────────────────────────────────
        rp = regionprops(mask.astype(np.int32))[0]
        bb = rp.bbox  # (min0, min1, min2, max0, max1, max2)
        bb_dims = (bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
        bb_vol  = int(bb_dims[0] * bb_dims[1] * bb_dims[2])
        row.update({
            "bbox_dim0_vox":   bb_dims[0],
            "bbox_dim1_vox":   bb_dims[1],
            "bbox_dim2_vox":   bb_dims[2],
            "bbox_volume_vox": bb_vol,
            "bbox_fill_ratio": n_vox / bb_vol if bb_vol > 0 else np.nan,
        })

        # ── Convex hull ────────────────────────────────────────────────────
        ch_vol = _convex_hull_volume(coords)
        row["convex_hull_volume_vox"] = ch_vol
        row["convexness"] = (n_vox / ch_vol) if ch_vol > 0 else np.nan

        # ── Surface area & sphericity ───────────────────────────────────────
        sa = _surface_area_mm2(mask, voxel_sizes)
        row["surface_area_mm2"]     = sa
        row["surface_volume_ratio"] = (
            sa / vol_mm3 if (not np.isnan(sa) and vol_mm3 > 0) else np.nan
        )
        if not np.isnan(sa) and sa > 0:
            row["sphericity"] = (np.pi ** (1 / 3) * (6 * vol_mm3) ** (2 / 3)) / sa
        else:
            row["sphericity"] = np.nan

        # ── Surface curvature (requires trimesh) ────────────────────────────
        row.update(_surface_curvature_stats(mask, voxel_sizes))

        # ── Inertia tensor & derived shape ratios ───────────────────────────
        eigvals_3d = np.sort(rp.inertia_tensor_eigvals)[::-1]  # λ1 ≥ λ2 ≥ λ3
        lam1, lam2, lam3 = eigvals_3d
        row.update({
            "inertia_eigval_1": float(lam1),
            "inertia_eigval_2": float(lam2),
            "inertia_eigval_3": float(lam3),
            "elongation": float(np.sqrt(lam2 / lam1)) if lam1 > 0 else np.nan,
            "flatness":   float(np.sqrt(lam3 / lam1)) if lam1 > 0 else np.nan,
            "isotropy":   float(lam3 / lam1)           if lam1 > 0 else np.nan,
        })

        # ── Topology ────────────────────────────────────────────────────────
        row["euler_number"] = int(euler_number(mask, connectivity=3))

        # ── Inter-lesion distances & neighbourhood density ─────────────────────
        other_vox = [c for lv, c in centroids.items()    if lv != label_val]
        other_mm  = [c for lv, c in centroids_mm.items() if lv != label_val]
        if other_vox:
            dists_vox = [np.linalg.norm(np.array(com_vox) - np.array(c)) for c in other_vox]
            dists_mm  = [np.linalg.norm(com_mm - np.array(c))             for c in other_mm]
            row["nearest_lesion_dist_vox"]      = float(min(dists_vox))
            row["nearest_lesion_dist_mm"]       = float(min(dists_mm))
            row["lesions_within_10mm"]          = int(sum(1 for d in dists_mm if d <= 10))
            row["lesions_within_20mm"]          = int(sum(1 for d in dists_mm if d <= 20))
            row["lesions_within_30mm"]          = int(sum(1 for d in dists_mm if d <= 30))
            row["mean_dist_to_all_lesions_mm"]  = float(np.mean(dists_mm))
        else:
            row["nearest_lesion_dist_vox"]     = np.nan
            row["nearest_lesion_dist_mm"]      = np.nan
            row["lesions_within_10mm"]         = 0
            row["lesions_within_20mm"]         = 0
            row["lesions_within_30mm"]         = 0
            row["mean_dist_to_all_lesions_mm"] = np.nan

        # ── 4D inertia (spatial + intensity) per modality ────────────────────
        for name, arr in mod_arrays.items():
            eigvals_4d, _ = _compute_4d_inertia(coords, arr[mask])
            if eigvals_4d is not None:
                for i, ev in enumerate(eigvals_4d, 1):
                    row[f"inertia_4d_{name}_eigval_{i}"] = float(ev)
            else:
                for i in range(1, 5):
                    row[f"inertia_4d_{name}_eigval_{i}"] = np.nan

        # ── Intensity statistics, gradient, perilesional ring & shells ────────
        for name, arr in mod_arrays.items():
            vals = arr[mask]
            row.update(_intensity_stats(vals, prefix=name))
            row.update(_boundary_gradient_stats(arr, mask, prefix=name))
            row.update(_perilesional_ring_stats(arr, mask, prefix=name))
            row.update(_concentric_shell_stats(arr, mask, prefix=name, n_shells=n_shells))
            if name in nawm_stats:
                nm, ns = nawm_stats[name]
                row[f"{name}_nawm_norm_mean"] = (
                    (float(np.mean(vals)) - nm) / ns if ns > 0 else np.nan
                )

        # ── Pairwise: ratio, Pearson correlation, mutual information ──────────
        for n1, n2 in combinations(mod_names, 2):
            v1 = mod_arrays[n1][mask]
            v2 = mod_arrays[n2][mask]
            m2 = float(np.mean(v2))
            row[f"{n1}_over_{n2}_mean_ratio"] = (
                float(np.mean(v1)) / m2 if m2 != 0 else np.nan
            )
            row[f"{n1}_{n2}_correlation"] = (
                float(np.corrcoef(v1, v2)[0, 1]) if v1.size > 1 else np.nan
            )
            row[f"{n1}_{n2}_mutual_info"] = _mutual_information(v1, v2)

        # ── PyRadiomics texture features ─────────────────────────────────────
        for name, arr in mod_arrays.items():
            row.update(
                _radiomics_features(mask, arr, header, prefix=name, extractor=extractor)
            )

        # ── Jacobian determinant statistics ─────────────────────────────────
        if jacobian_data is not None:
            jac = jacobian_data[mask]
            row.update({
                "jacobian_mean":                 float(np.mean(jac)),
                "jacobian_std":                  float(np.std(jac)),
                "jacobian_median":               float(np.median(jac)),
                # Jacobian > 1 → local expansion; < 1 → contraction
                "jacobian_expanding_fraction":   float(np.mean(jac > 1.0)),
                "jacobian_contracting_fraction": float(np.mean(jac < 1.0)),
            })

        # ── Probabilistic segmentation features ─────────────────────────────
        if prob_data is not None:
            probs = prob_data[mask]
            row.update({
                "fuzzy_volume_vox":    float(np.sum(probs)),
                "mean_confidence":     float(np.mean(probs)),
                "std_confidence":      float(np.std(probs)),
                "prob_weighted_com_x": float(np.average(coords[:, 0], weights=probs)),
                "prob_weighted_com_y": float(np.average(coords[:, 1], weights=probs)),
                "prob_weighted_com_z": float(np.average(coords[:, 2], weights=probs)),
                "prob_entropy":        float(stats.entropy(probs + 1e-9)),
                "core_volume_vox":     int(np.sum(probs > 0.9)),
                "rim_volume_vox":      int(np.sum((probs > 0.3) & (probs < 0.7))),
            })

        results.append(row)

    return pd.DataFrame(results)
