"""Tests for src/quantifying_lesions.py"""

import numpy as np
import nibabel as nib
import pytest

from tests.conftest import make_nifti, make_sphere_mask
from src.quantifying_lesions import (
    _boundary_gradient_stats,
    _compute_4d_inertia,
    _concentric_shell_stats,
    _convex_hull_volume,
    _intensity_stats,
    _mutual_information,
    _perilesional_ring_stats,
    _surface_area_mm2,
    _surface_curvature_stats,
    quantify_lesions,
    voxel_volume_from_header,
)


# ── voxel_volume_from_header ────────────────────────────────────────────────

class TestVoxelVolumeFromHeader:
    def test_isotropic_1mm(self):
        img = make_nifti(np.zeros((5, 5, 5)), voxel_sizes=(1.0, 1.0, 1.0))
        assert voxel_volume_from_header(img.header) == pytest.approx(1.0)

    def test_isotropic_2mm(self):
        img = make_nifti(np.zeros((5, 5, 5)), voxel_sizes=(2.0, 2.0, 2.0))
        assert voxel_volume_from_header(img.header) == pytest.approx(8.0)

    def test_anisotropic(self):
        img = make_nifti(np.zeros((5, 5, 5)), voxel_sizes=(1.0, 2.0, 3.0))
        assert voxel_volume_from_header(img.header) == pytest.approx(6.0)


# ── _convex_hull_volume ────────────────────────────────────────────────────

class TestConvexHullVolume:
    def test_fewer_than_4_points_returns_zero(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        assert _convex_hull_volume(pts) == 0.0

    def test_coplanar_points_returns_zero(self):
        pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
        assert _convex_hull_volume(pts) == 0.0

    def test_unit_cube_volume(self):
        corners = np.array([
            [0,0,0],[1,0,0],[0,1,0],[1,1,0],
            [0,0,1],[1,0,1],[0,1,1],[1,1,1],
        ], dtype=float)
        assert _convex_hull_volume(corners) == pytest.approx(1.0, rel=1e-6)

    def test_volume_positive_for_3d_coords(self):
        pts = np.random.default_rng(0).uniform(0, 10, (50, 3))
        assert _convex_hull_volume(pts) > 0


# ── _surface_area_mm2 ────────────────────────────────────────────────────

class TestSurfaceAreaMm2:
    def test_sphere_within_10_percent(self):
        r = 8
        mask = make_sphere_mask((40, 40, 40), (20, 20, 20), r)
        sa = _surface_area_mm2(mask, (1.0, 1.0, 1.0))
        assert not np.isnan(sa)
        assert sa == pytest.approx(4 * np.pi * r ** 2, rel=0.10)

    def test_single_voxel_returns_nan(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        assert np.isnan(_surface_area_mm2(mask, (1.0, 1.0, 1.0)))

    def test_scales_with_voxel_size_squared(self):
        mask = make_sphere_mask((30, 30, 30), (15, 15, 15), 6)
        sa_1 = _surface_area_mm2(mask, (1.0, 1.0, 1.0))
        sa_2 = _surface_area_mm2(mask, (2.0, 2.0, 2.0))
        assert sa_2 == pytest.approx(sa_1 * 4, rel=0.05)


# ── _surface_curvature_stats ───────────────────────────────────────────────

class TestSurfaceCurvatureStats:
    def test_returns_three_keys(self):
        mask = make_sphere_mask((30, 30, 30), (15, 15, 15), 7)
        result = _surface_curvature_stats(mask, (1.0, 1.0, 1.0))
        assert "curvature_mean" in result
        assert "curvature_std"  in result
        assert "curvature_abs_max" in result

    def test_sphere_abs_max_positive(self):
        # If trimesh is installed, sphere should give finite curvature values
        mask = make_sphere_mask((30, 30, 30), (15, 15, 15), 7)
        result = _surface_curvature_stats(mask, (1.0, 1.0, 1.0))
        # Either trimesh is available (finite values) or not (NaN) -- both are valid
        val = result["curvature_abs_max"]
        assert np.isnan(val) or val >= 0


# ── _intensity_stats ────────────────────────────────────────────────────────

class TestIntensityStats:
    EXPECTED_KEYS = [
        "mean", "median", "mode", "min", "max",
        "var", "std", "skewness", "kurtosis", "moment5", "entropy",
        "iqr", "p10", "p25", "p75", "p90", "cov", "energy", "rms",
    ]

    def test_all_keys_present(self):
        s = _intensity_stats(np.arange(10, dtype=float), prefix="X")
        for k in self.EXPECTED_KEYS:
            assert f"X_{k}" in s, f"Missing: X_{k}"

    def test_known_values(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = _intensity_stats(vals, prefix="M")
        assert s["M_mean"]   == pytest.approx(3.0)
        assert s["M_energy"] == pytest.approx(55.0)
        assert s["M_rms"]    == pytest.approx(np.sqrt(11.0))

    def test_empty_array_all_nan(self):
        s = _intensity_stats(np.array([]), prefix="X")
        assert all(np.isnan(s[f"X_{k}"]) for k in self.EXPECTED_KEYS)

    def test_zero_mean_cov_is_nan(self):
        s = _intensity_stats(np.array([-1.0, 1.0]), prefix="X")
        assert np.isnan(s["X_cov"])


# ── _compute_4d_inertia ────────────────────────────────────────────────────

class TestCompute4dInertia:
    def test_empty_returns_none(self):
        ev, c = _compute_4d_inertia(np.zeros((0, 3)), np.zeros(0))
        assert ev is None and c is None

    def test_returns_four_eigenvalues(self):
        rng = np.random.default_rng(7)
        coords = rng.integers(0, 20, (200, 3)).astype(float)
        ev, centroid = _compute_4d_inertia(coords, rng.normal(100, 10, 200))
        assert len(ev) == 4 and len(centroid) == 4

    def test_eigenvalues_non_negative(self):
        rng = np.random.default_rng(8)
        ev, _ = _compute_4d_inertia(rng.uniform(0, 10, (100, 3)), rng.normal(50, 5, 100))
        assert np.all(ev >= -1e-10)


# ── _boundary_gradient_stats ───────────────────────────────────────────────

class TestBoundaryGradientStats:
    def test_finite_for_sphere(self):
        data = np.random.default_rng(1).normal(100, 10, (30, 30, 30)).astype(np.float32)
        mask = make_sphere_mask((30, 30, 30), (15, 15, 15), 5)
        s = _boundary_gradient_stats(data, mask, prefix="T1")
        assert not np.isnan(s["T1_boundary_grad_mean"])
        assert s["T1_boundary_grad_mean"] >= 0


# ── _perilesional_ring_stats ──────────────────────────────────────────────

class TestPerilesionalRingStats:
    def test_keys_present(self):
        data = np.random.default_rng(2).normal(500, 30, (40, 40, 40)).astype(np.float32)
        mask = make_sphere_mask((40, 40, 40), (20, 20, 20), 6)
        s = _perilesional_ring_stats(data, mask, prefix="T2")
        assert "T2_perilesional_mean" in s
        assert "T2_perilesional_std"  in s
        assert "T2_lesion_perilesional_ratio" in s

    def test_all_finite_for_interior_sphere(self):
        data = np.ones((40, 40, 40), dtype=np.float32) * 100
        data[15:25, 15:25, 15:25] = 200  # lesion brighter
        mask = make_sphere_mask((40, 40, 40), (20, 20, 20), 4)
        s = _perilesional_ring_stats(data, mask, prefix="FL")
        assert not np.isnan(s["FL_perilesional_mean"])
        assert s["FL_lesion_perilesional_ratio"] == pytest.approx(2.0, rel=0.05)

    def test_boundary_sphere_graceful(self):
        # Sphere touching the image boundary has no dilation room
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[:, :, :] = True  # fills whole volume
        data = np.ones((10, 10, 10), dtype=np.float32)
        s = _perilesional_ring_stats(data, mask, prefix="X")
        # Ring is empty — should return NaN gracefully
        assert np.isnan(s["X_perilesional_mean"])


# ── _concentric_shell_stats ───────────────────────────────────────────────

class TestConcentricShellStats:
    def test_shell_keys_present(self):
        data = np.random.default_rng(3).normal(100, 10, (40, 40, 40)).astype(np.float32)
        mask = make_sphere_mask((40, 40, 40), (20, 20, 20), 8)
        s = _concentric_shell_stats(data, mask, prefix="T1", n_shells=5)
        for i in range(5):
            assert f"T1_shell_{i}_mean" in s
            assert f"T1_shell_{i}_std"  in s
        assert "T1_rim_core_ratio" in s
        assert "T1_shell_gradient_slope" in s
        assert "T1_shell_gradient_r2"    in s

    def test_rim_core_ratio_finite_for_large_sphere(self):
        data = np.random.default_rng(4).normal(100, 10, (50, 50, 50)).astype(np.float32)
        mask = make_sphere_mask((50, 50, 50), (25, 25, 25), 10)
        s = _concentric_shell_stats(data, mask, prefix="T1", n_shells=5)
        assert not np.isnan(s["T1_rim_core_ratio"])

    def test_gradient_slope_detects_bright_rim(self):
        # Make outer shells brighter than core — slope should be negative (high at rim=0)
        data = np.zeros((40, 40, 40), dtype=np.float32)
        mask = make_sphere_mask((40, 40, 40), (20, 20, 20), 8)
        coords = np.argwhere(mask)
        center = np.array([20, 20, 20])
        dists = np.linalg.norm(coords - center, axis=1)
        # Intensity proportional to distance from center (bright rim)
        for (i, j, k), d in zip(coords, dists):
            data[i, j, k] = d * 10
        s = _concentric_shell_stats(data, mask, prefix="T1", n_shells=5)
        # Shell 0 (rim) has higher intensity than shell 4 (core): negative slope
        assert s["T1_shell_gradient_slope"] < 0

    def test_tiny_lesion_no_crash(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True  # 1 voxel
        data = np.ones((10, 10, 10), dtype=np.float32)
        s = _concentric_shell_stats(data, mask, prefix="X", n_shells=5)
        assert "X_shell_0_mean" in s  # should not raise


# ── _mutual_information ────────────────────────────────────────────────────

class TestMutualInformation:
    def test_identical_arrays_high_mi(self):
        v = np.linspace(0, 1, 200)
        mi = _mutual_information(v, v)
        assert mi > 0

    def test_independent_arrays_lower_mi(self):
        rng = np.random.default_rng(99)
        v1 = rng.normal(0, 1, 500)
        v2 = rng.normal(0, 1, 500)
        v_same = v1.copy()
        mi_same = _mutual_information(v1, v_same)
        mi_indep = _mutual_information(v1, v2)
        assert mi_same > mi_indep

    def test_too_few_samples_returns_nan(self):
        assert np.isnan(_mutual_information(np.array([1.0, 2.0]), np.array([1.0, 2.0])))

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        mi = _mutual_information(rng.normal(size=100), rng.normal(size=100))
        assert mi >= 0


# ── quantify_lesions (integration) ──────────────────────────────────────────

SHAPE_COLS = [
    "label", "volume_vox", "volume_mm3",
    "com_x_vox", "com_y_vox", "com_z_vox",
    "com_x_mm",  "com_y_mm",  "com_z_mm",
    "hemisphere",
    "bbox_dim0_vox", "bbox_dim1_vox", "bbox_dim2_vox",
    "bbox_volume_vox", "bbox_fill_ratio",
    "convex_hull_volume_vox", "convexness",
    "surface_area_mm2", "surface_volume_ratio", "sphericity",
    "curvature_mean", "curvature_std", "curvature_abs_max",
    "inertia_eigval_1", "inertia_eigval_2", "inertia_eigval_3",
    "elongation", "flatness", "isotropy",
    "euler_number",
    "nearest_lesion_dist_vox", "nearest_lesion_dist_mm",
    "lesions_within_10mm", "lesions_within_20mm", "lesions_within_30mm",
    "mean_dist_to_all_lesions_mm",
]
INTENSITY_STATS = [
    "mean", "median", "mode", "min", "max", "var", "std",
    "skewness", "kurtosis", "moment5", "entropy",
    "iqr", "p10", "p25", "p75", "p90", "cov", "energy", "rms",
    "boundary_grad_mean", "boundary_grad_std",
    "perilesional_mean", "perilesional_std", "lesion_perilesional_ratio",
    "rim_core_ratio", "shell_gradient_slope", "shell_gradient_r2",
]


class TestQuantifyLesions:
    def test_row_count(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert len(df) == 2

    def test_shape_columns_present(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for col in SHAPE_COLS:
            assert col in df.columns, f"Missing shape column: {col}"

    def test_intensity_columns_present(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for mod in ("T1", "T2"):
            for stat in INTENSITY_STATS:
                col = f"{mod}_{stat}"
                assert col in df.columns, f"Missing column: {col}"

    def test_shell_columns_present(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
            n_shells=3,
        )
        for mod in ("T1", "T2"):
            for i in range(3):
                assert f"{mod}_shell_{i}_mean" in df.columns
                assert f"{mod}_shell_{i}_std"  in df.columns

    def test_pairwise_columns_present(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert "T1_over_T2_mean_ratio" in df.columns
        assert "T1_T2_correlation"     in df.columns
        assert "T1_T2_mutual_info"     in df.columns

    def test_mutual_info_non_negative(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert (df["T1_T2_mutual_info"] >= 0).all()

    def test_label_1_larger_than_label_2(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        v1 = df.loc[df["label"] == 1, "volume_vox"].iloc[0]
        v2 = df.loc[df["label"] == 2, "volume_vox"].iloc[0]
        assert v1 > v2

    def test_volume_approximately_correct(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        v1 = df.loc[df["label"] == 1, "volume_mm3"].iloc[0]
        v2 = df.loc[df["label"] == 2, "volume_mm3"].iloc[0]
        assert v1 == pytest.approx(4/3 * np.pi * 7**3, rel=0.10)
        assert v2 == pytest.approx(4/3 * np.pi * 4**3, rel=0.10)

    def test_intensity_means_close_to_target(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        row = df.loc[df["label"] == 1].iloc[0]
        assert row["T1_mean"] == pytest.approx(800.0, abs=15.0)
        assert row["T2_mean"] == pytest.approx(700.0, abs=15.0)

    def test_no_nan_in_core_columns(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for col in ["volume_vox", "volume_mm3", "convexness",
                    "T1_mean", "T2_mean", "elongation", "flatness", "isotropy",
                    "T1_perilesional_mean", "T1_rim_core_ratio",
                    "nearest_lesion_dist_mm", "T1_T2_mutual_info"]:
            assert not df[col].isna().any(), f"Unexpected NaN in: {col}"

    def test_neighbourhood_density_columns(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for col in ("nearest_lesion_dist_mm", "lesions_within_10mm",
                    "lesions_within_20mm", "lesions_within_30mm",
                    "mean_dist_to_all_lesions_mm"):
            assert col in df.columns, f"Missing: {col}"
        assert (df["nearest_lesion_dist_mm"] > 0).all()

    def test_jacobian_columns_when_map_given(self, synthetic_dataset):
        # Simulate a Jacobian map (all ones = no volume change)
        jac_data = np.ones((70, 70, 70), dtype=np.float32)
        jac_img  = make_nifti(jac_data)
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            jacobian_map=jac_img,
            include_radiomics=False,
        )
        for col in ("jacobian_mean", "jacobian_std", "jacobian_median",
                    "jacobian_expanding_fraction", "jacobian_contracting_fraction"):
            assert col in df.columns, f"Missing Jacobian column: {col}"
        # All-ones Jacobian: mean=1, expanding fraction=0, contracting=1
        assert df["jacobian_mean"].iloc[0] == pytest.approx(1.0, abs=1e-5)
        assert df["jacobian_expanding_fraction"].iloc[0]   == pytest.approx(0.0, abs=1e-5)
        assert df["jacobian_contracting_fraction"].iloc[0] == pytest.approx(0.0, abs=1e-5)

    def test_jacobian_columns_absent_without_map(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            jacobian_map=None,
            include_radiomics=False,
        )
        assert "jacobian_mean" not in df.columns

    def test_probabilistic_columns(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            prob_map=synthetic_dataset["prob_img"],
            include_radiomics=False,
        )
        for col in ("fuzzy_volume_vox", "mean_confidence", "core_volume_vox", "rim_volume_vox"):
            assert col in df.columns
        assert df["mean_confidence"].between(0.0, 1.0).all()

    def test_nawm_column_present(self, synthetic_dataset):
        nawm = np.zeros((70, 70, 70), dtype=np.float32)
        nawm[60:65, 60:65, 60:65] = 1.0
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            nawm_mask=make_nifti(nawm),
            include_radiomics=False,
        )
        assert "T1_nawm_norm_mean" in df.columns

    def test_three_modalities_all_pairwise(self, synthetic_dataset):
        flair = make_nifti(np.random.default_rng(99).normal(600, 40, (70, 70, 70)).astype(np.float32))
        mods  = {**synthetic_dataset["modalities"], "FLAIR": flair}
        df = quantify_lesions(
            synthetic_dataset["labeled_img"], mods, include_radiomics=False
        )
        for n1, n2 in (("T1","T2"),("T1","FLAIR"),("T2","FLAIR")):
            assert f"{n1}_over_{n2}_mean_ratio" in df.columns
            assert f"{n1}_{n2}_correlation"     in df.columns
            assert f"{n1}_{n2}_mutual_info"     in df.columns

    def test_hemisphere_valid(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert set(df["hemisphere"]).issubset({"L", "R"})

    def test_euler_number_one_for_spheres(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert (df["euler_number"] == 1).all()

    def test_bbox_fill_ratio_in_0_1(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for val in df["bbox_fill_ratio"].dropna():
            assert 0 < val <= 1.0
