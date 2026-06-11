import numpy as np
import nibabel as nib
import pytest

from lesionomics.quantifying_lesions import (
    voxel_volume_from_header,
    _convex_hull_volume,
    _surface_area_mm2,
    _surface_curvature_stats,
    _intensity_stats,
    _compute_4d_inertia,
    _boundary_gradient_stats,
    _perilesional_ring_stats,
    _concentric_shell_stats,
    _mutual_information,
    quantify_lesions,
)

# Expected column sets for structural assertions
SHAPE_COLS = {
    "volume_mm3",
    "com_mm_z", "com_mm_y", "com_mm_x",
    "convex_hull_volume_mm3", "solidity",
    "surface_area_mm2", "sphericity",
    "curvature_mean", "curvature_std", "curvature_abs_max",
    "elongation", "flatness", "isotropy",
    "euler_number",
    "nearest_lesion_dist_mm",
    "lesions_within_10mm", "lesions_within_20mm", "lesions_within_30mm",
    "mean_dist_to_all_lesions_mm",
}

# Removed from INTENSITY_STATS: var (=std²), cv (=std/mean), rms,
# moment5, p10, p25, p75, p90 (iqr + min/max sufficient).
INTENSITY_STATS = {
    "mean", "median", "mode", "min", "max", "std",
    "skewness", "kurtosis", "entropy", "iqr", "energy",
}


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------

class TestVoxelVolumeFromHeader:
    def test_isotropic_1mm(self):
        hdr = nib.Nifti1Header()
        hdr.set_data_shape((10, 10, 10))
        hdr.set_zooms((1.0, 1.0, 1.0))
        assert voxel_volume_from_header(hdr) == pytest.approx(1.0)

    def test_anisotropic(self):
        hdr = nib.Nifti1Header()
        hdr.set_data_shape((10, 10, 10))
        hdr.set_zooms((2.0, 3.0, 4.0))
        assert voxel_volume_from_header(hdr) == pytest.approx(24.0)


class TestConvexHullVolume:
    def test_fewer_than_4_points_returns_zero(self):
        assert _convex_hull_volume(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])) == 0.0

    def test_planar_points_returns_zero(self):
        pts = np.array([[i, j, 0] for i in range(5) for j in range(5)], dtype=float)
        assert _convex_hull_volume(pts) == 0.0

    def test_unit_cube_approx_1(self):
        corners = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=float)
        assert _convex_hull_volume(corners) == pytest.approx(1.0, abs=0.01)


class TestSurfaceAreaMm2:
    def test_sphere_approximate(self):
        # r=5 sphere in 30×30×30 volume
        shape = (30, 30, 30)
        z, y, x = np.mgrid[0:30, 0:30, 0:30]
        mask = (z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2 <= 25
        sa = _surface_area_mm2(mask, (1.0, 1.0, 1.0))
        expected = 4 * np.pi * 25  # 4πr²
        assert sa == pytest.approx(expected, rel=0.15)

    def test_empty_mask_returns_nan(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        result = _surface_area_mm2(mask, (1.0, 1.0, 1.0))
        assert np.isnan(result)


class TestSurfaceCurvatureStats:
    def test_returns_three_keys(self):
        shape = (20, 20, 20)
        z, y, x = np.mgrid[0:20, 0:20, 0:20]
        mask = (z - 10) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 16
        result = _surface_curvature_stats(mask, (1.0, 1.0, 1.0))
        assert set(result.keys()) == {"curvature_mean", "curvature_std", "curvature_abs_max"}

    def test_graceful_without_trimesh(self, monkeypatch):
        import lesionomics.quantifying_lesions as qm
        monkeypatch.setattr(qm, "_TRIMESH_AVAILABLE", False)
        shape = (20, 20, 20)
        z, y, x = np.mgrid[0:20, 0:20, 0:20]
        mask = (z - 10) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 16
        result = qm._surface_curvature_stats(mask, (1.0, 1.0, 1.0))
        assert all(np.isnan(v) for v in result.values())


class TestIntensityStats:
    def test_returns_11_stats(self):
        vals = np.random.default_rng(0).normal(100, 10, 200).astype(np.float32)
        result = _intensity_stats(vals, "T1")
        assert len(result) == 11

    def test_empty_vals_all_nan(self):
        result = _intensity_stats(np.array([]), "T1")
        assert all(np.isnan(v) for v in result.values())

    def test_mean_correct(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _intensity_stats(vals, "T1")
        assert result["T1_mean"] == pytest.approx(3.0)

    def test_column_names_prefixed(self):
        vals = np.ones(10)
        result = _intensity_stats(vals, "FLAIR")
        assert all(k.startswith("FLAIR_") for k in result)


class TestCompute4dInertia:
    def test_empty_returns_nans(self):
        eigvals, centroid = _compute_4d_inertia(np.empty((0, 3)), np.array([]))
        assert all(np.isnan(eigvals))
        assert centroid is None

    def test_returns_four_eigenvalues(self):
        coords = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        intensities = np.array([1.0, 2.0, 3.0, 4.0])
        eigvals, centroid = _compute_4d_inertia(coords, intensities)
        assert eigvals.shape == (4,)
        assert centroid.shape == (4,)


class TestBoundaryGradientStats:
    def test_returns_two_keys(self):
        shape = (20, 20, 20)
        z, y, x = np.mgrid[0:20, 0:20, 0:20]
        mask = (z - 10) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 9
        data = np.random.default_rng(1).normal(100, 10, shape)
        result = _boundary_gradient_stats(data, mask, "T1")
        assert set(result.keys()) == {"T1_boundary_grad_mean", "T1_boundary_grad_std"}

    def test_gradient_finite_for_spherical_mask(self):
        shape = (20, 20, 20)
        z, y, x = np.mgrid[0:20, 0:20, 0:20]
        mask = (z - 10) ** 2 + (y - 10) ** 2 + (x - 10) ** 2 <= 9
        data = np.where(mask, 800.0, 1000.0).astype(float)
        result = _boundary_gradient_stats(data, mask, "T1")
        assert np.isfinite(result["T1_boundary_grad_mean"])
        assert result["T1_boundary_grad_mean"] > 0


class TestPerilesionalRingStats:
    def test_returns_three_keys(self):
        shape = (20, 20, 20)
        mask = np.zeros(shape, dtype=bool)
        mask[8:12, 8:12, 8:12] = True
        data = np.ones(shape) * 100.0
        result = _perilesional_ring_stats(data, mask, "T1")
        assert set(result.keys()) == {
            "T1_perilesional_mean", "T1_perilesional_std", "T1_lesion_perilesional_ratio"
        }

    def test_bright_rim_ratio_above_one(self):
        # Lesion brighter than surroundings → ratio > 1
        shape = (30, 30, 30)
        mask = np.zeros(shape, dtype=bool)
        mask[12:18, 12:18, 12:18] = True
        data = np.where(mask, 1000.0, 500.0).astype(float)
        result = _perilesional_ring_stats(data, mask, "T1")
        assert result["T1_lesion_perilesional_ratio"] > 1.0


class TestConcentricShellStats:
    def test_returns_expected_keys(self):
        shape = (30, 30, 30)
        z, y, x = np.mgrid[0:30, 0:30, 0:30]
        mask = (z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2 <= 25
        data = np.ones(shape) * 100.0
        result = _concentric_shell_stats(data, mask, "T1", n_shells=3)
        for i in range(3):
            assert f"T1_shell_{i}_mean" in result
            assert f"T1_shell_{i}_std" in result
        assert "T1_rim_core_ratio" in result
        assert "T1_shell_gradient_slope" in result
        assert "T1_shell_gradient_r2" in result

    def test_rim_core_ratio_above_one_for_bright_rim(self):
        # Bright rim: intensity INCREASES with distance from centre
        # → surface (shell 0, dist≈7) is bright; core (shell n-1, dist≈0) is dark
        shape = (30, 30, 30)
        z, y, x = np.mgrid[0:30, 0:30, 0:30]
        dist_from_center = np.sqrt((z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2)
        mask = dist_from_center <= 7
        data = np.where(mask, 50.0 + dist_from_center * 20.0, 0.0)
        result = _concentric_shell_stats(data, mask, "T1", n_shells=4)
        assert result["T1_rim_core_ratio"] > 1.0

    def test_gradient_slope_negative_for_bright_rim(self):
        # Bright at surface (shell 0) → dark at core (shell n-1) → negative slope
        # (intensity decreases as shell index grows from 0=rim toward n-1=core)
        shape = (30, 30, 30)
        z, y, x = np.mgrid[0:30, 0:30, 0:30]
        dist_from_center = np.sqrt((z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2)
        mask = dist_from_center <= 7
        data = np.where(mask, 50.0 + dist_from_center * 20.0, 0.0)
        result = _concentric_shell_stats(data, mask, "T1", n_shells=4)
        assert result["T1_shell_gradient_slope"] < 0


class TestMutualInformation:
    def test_identical_arrays_high_mi(self):
        rng = np.random.default_rng(7)
        v = rng.normal(0, 1, 500)
        mi_same = _mutual_information(v, v)
        mi_indep = _mutual_information(v, rng.normal(0, 1, 500))
        assert mi_same > mi_indep

    def test_non_negative(self):
        rng = np.random.default_rng(8)
        v1 = rng.normal(0, 1, 200)
        v2 = rng.normal(0, 1, 200)
        assert _mutual_information(v1, v2) >= 0

    def test_size_mismatch_returns_nan(self):
        v1 = np.ones(10)
        v2 = np.ones(5)
        assert np.isnan(_mutual_information(v1, v2))


# ---------------------------------------------------------------------------
# Integration tests — quantify_lesions
# ---------------------------------------------------------------------------

class TestQuantifyLesions:
    def test_returns_dataframe(self, synthetic_dataset):
        import pandas as pd
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_lesion(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        assert len(df) == 2  # two lesions

    def test_shape_columns_present(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        missing = SHAPE_COLS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_intensity_stat_columns_present(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        for mod in ("T1", "T2"):
            for stat in INTENSITY_STATS:
                col = f"{mod}_{stat}"
                assert col in df.columns, f"Missing: {col}"

    def test_volume_matches_sphere_approximately(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        # Lesion 1 (r=7): expected ~1437 mm³; Lesion 2 (r=4): ~268 mm³ (1 mm isotropic)
        vol1 = df.loc[df["label"] == 1, "volume_mm3"].iloc[0]
        vol2 = df.loc[df["label"] == 2, "volume_mm3"].iloc[0]
        expected1 = (4 / 3) * np.pi * 7 ** 3
        expected2 = (4 / 3) * np.pi * 4 ** 3
        assert vol1 == pytest.approx(expected1, rel=0.15)
        assert vol2 == pytest.approx(expected2, rel=0.15)

    def test_lesion1_larger_than_lesion2(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        v1 = df.loc[df["label"] == 1, "volume_mm3"].iloc[0]
        v2 = df.loc[df["label"] == 2, "volume_mm3"].iloc[0]
        assert v1 > v2

    def test_intensity_mean_close_to_800(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        t1_mean = df.loc[df["label"] == 1, "T1_mean"].iloc[0]
        assert t1_mean == pytest.approx(800, abs=30)

    def test_shell_columns_present(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False, n_shells=3)
        for i in range(3):
            assert f"T1_shell_{i}_mean" in df.columns

    def test_mutual_info_non_negative(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        assert "T1_T2_mutual_info" in df.columns
        assert (df["T1_T2_mutual_info"] >= 0).all()

    def test_neighbourhood_density_columns(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        for col in ("nearest_lesion_dist_mm", "lesions_within_10mm", "lesions_within_20mm",
                    "lesions_within_30mm", "mean_dist_to_all_lesions_mm"):
            assert col in df.columns

    def test_jacobian_columns_when_map_given(self, synthetic_dataset):
        ds = synthetic_dataset
        jac_data = np.ones(ds["shape"], dtype=np.float32) * 1.1
        header = nib.Nifti1Header()
        header.set_data_shape(ds["shape"])
        header.set_zooms((1.0, 1.0, 1.0))
        jac_img = nib.Nifti1Image(jac_data, ds["affine"], header)
        df = quantify_lesions(
            ds["labeled_img"], ds["modalities"],
            jacobian_map=jac_img,
            include_radiomics=False,
        )
        for col in ("jacobian_mean", "jacobian_std", "jacobian_median",
                    "jacobian_expanding_fraction", "jacobian_contracting_fraction"):
            assert col in df.columns
        assert df["jacobian_expanding_fraction"].iloc[0] == pytest.approx(1.0)

    def test_jacobian_columns_absent_without_map(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        assert "jacobian_mean" not in df.columns

    def test_probabilistic_columns_present(self, synthetic_dataset):
        ds = synthetic_dataset
        prob_data = np.where(ds["labeled"] > 0, 0.9, 0.0).astype(np.float32)
        header = nib.Nifti1Header()
        header.set_data_shape(ds["shape"])
        header.set_zooms((1.0, 1.0, 1.0))
        prob_img = nib.Nifti1Image(prob_data, ds["affine"], header)
        df = quantify_lesions(
            ds["labeled_img"], ds["modalities"],
            prob_map=prob_img,
            include_radiomics=False,
        )
        for col in ("fuzzy_volume_mm3", "mean_confidence", "std_confidence",
                    "prob_weighted_centroid_z", "prob_entropy",
                    "core_volume_mm3", "rim_volume_mm3"):
            assert col in df.columns

    def test_nawm_normalization_columns_present(self, synthetic_dataset):
        ds = synthetic_dataset
        nawm_data = np.zeros(ds["shape"], dtype=np.uint8)
        nawm_data[30:60, 10:60, 10:60] = 1  # simple NAWM region
        header = nib.Nifti1Header()
        header.set_data_shape(ds["shape"])
        header.set_zooms((1.0, 1.0, 1.0))
        nawm_img = nib.Nifti1Image(nawm_data, ds["affine"], header)
        df = quantify_lesions(
            ds["labeled_img"], ds["modalities"],
            nawm_mask=nawm_img,
            include_radiomics=False,
        )
        assert "T1_nawm_normalized_mean" in df.columns
        assert "T2_nawm_normalized_mean" in df.columns

    def test_pairwise_columns_present(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        assert "T1_over_T2_mean_ratio" in df.columns
        assert "T1_T2_correlation" in df.columns
        assert "T1_T2_mutual_info" in df.columns

    def test_single_modality_works(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(
            ds["labeled_img"],
            {"T1": ds["modalities"]["T1"]},
            include_radiomics=False,
        )
        assert len(df) == 2
        assert "T1_mean" in df.columns
        assert "T1_over_T2_mean_ratio" not in df.columns

    def test_shape_metric_ranges(self, synthetic_dataset):
        """Shape metrics have known mathematical bounds that should always hold."""
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        # Sphericity: (0, 1] — sphere achieves 1, all other convex shapes < 1
        sph = df["sphericity"].dropna()
        assert (sph > 0).all() and (sph <= 1.0 + 1e-6).all(), f"sphericity out of (0,1]: {sph.values}"
        # Solidity: (0, 1] — convex hull volume >= lesion volume
        sol = df["solidity"].dropna()
        assert (sol > 0).all() and (sol <= 1.0 + 1e-6).all(), f"solidity out of (0,1]: {sol.values}"
        # Elongation, flatness, isotropy: sqrt of eigenvalue ratios ≤ 1 when eigenvalues sorted descending
        for col in ("elongation", "flatness", "isotropy"):
            vals = df[col].dropna()
            assert (vals >= 0).all() and (vals <= 1.0 + 1e-6).all(), f"{col} out of [0,1]: {vals.values}"

    def test_no_nan_in_core_columns(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        core_cols = ["volume_mm3", "T1_mean", "T2_mean",
                     "sphericity", "surface_area_mm2"]
        for col in core_cols:
            assert not df[col].isna().any(), f"NaN found in {col}"

    def test_4d_inertia_columns_present(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        for mod in ("T1", "T2"):
            for i in range(4):
                assert f"{mod}_4d_eigval_{i}" in df.columns

    def test_perilesional_columns_present(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        for mod in ("T1", "T2"):
            assert f"{mod}_perilesional_mean" in df.columns
            assert f"{mod}_lesion_perilesional_ratio" in df.columns

    def test_boundary_gradient_columns_present(self, synthetic_dataset):
        ds = synthetic_dataset
        df = quantify_lesions(ds["labeled_img"], ds["modalities"], include_radiomics=False)
        for mod in ("T1", "T2"):
            assert f"{mod}_boundary_grad_mean" in df.columns

    def test_hierarchy_columns_present_when_labeled_result(self):
        """When a LabeledResult is passed, confluent_lesion_id and display_id appear in output."""
        pass  # covered by test_hierarchy_columns_present_from_label_lesions

    def test_hierarchy_columns_present_from_label_lesions(self):
        """quantify_lesions adds confluent_lesion_id/sub_label/display_id when given a LabeledResult."""
        shape = (40, 20, 20)
        affine = np.eye(4)
        header = nib.Nifti1Header()
        header.set_data_shape(shape)
        header.set_zooms((1.0, 1.0, 1.0))

        data = np.zeros(shape, dtype=np.float32)
        data[2:8, 5:12, 5:12] = 1.0
        data[20:26, 5:12, 5:12] = 1.0
        seg_img = nib.Nifti1Image(data, affine, header)

        t1 = nib.Nifti1Image(np.ones(shape, dtype=np.float32) * 800, affine, header)
        from lesionomics.label_lesions import label_lesions
        from lesionomics.quantifying_lesions import quantify_lesions
        result = label_lesions(seg_img, threshold=0.5)
        df = quantify_lesions(result, {"T1": t1}, include_radiomics=False)
        for col in ("confluent_lesion_id", "sub_label", "display_id"):
            assert col in df.columns, f"Missing column: {col}"
        assert set(df["display_id"]) == {"L1", "L2"}
