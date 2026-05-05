"""Tests for src/quantifying_lesions.py"""

import numpy as np
import nibabel as nib
import pytest

from tests.conftest import make_nifti, make_sphere_mask
from src.quantifying_lesions import (
    _boundary_gradient_stats,
    _compute_4d_inertia,
    _convex_hull_volume,
    _intensity_stats,
    _surface_area_mm2,
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
        pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])  # z=0 plane
        assert _convex_hull_volume(pts) == 0.0

    def test_unit_cube_volume(self):
        corners = np.array([
            [0,0,0],[1,0,0],[0,1,0],[1,1,0],
            [0,0,1],[1,0,1],[0,1,1],[1,1,1],
        ], dtype=float)
        assert _convex_hull_volume(corners) == pytest.approx(1.0, rel=1e-6)

    def test_volume_positive_for_3d_coords(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 10, (50, 3))
        vol = _convex_hull_volume(pts)
        assert vol > 0


# ── _surface_area_mm2 ─────────────────────────────────────────────────────

class TestSurfaceAreaMm2:
    def test_sphere_surface_area_within_10_percent(self):
        r = 8
        shape = (40, 40, 40)
        mask = make_sphere_mask(shape, (20, 20, 20), r)
        sa = _surface_area_mm2(mask, (1.0, 1.0, 1.0))
        expected = 4 * np.pi * r ** 2  # ~804 mm2
        assert not np.isnan(sa)
        assert sa == pytest.approx(expected, rel=0.10)

    def test_single_voxel_returns_nan(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        assert np.isnan(_surface_area_mm2(mask, (1.0, 1.0, 1.0)))

    def test_larger_voxel_size_scales_area(self):
        r = 6
        shape = (30, 30, 30)
        mask = make_sphere_mask(shape, (15, 15, 15), r)
        sa_1mm = _surface_area_mm2(mask, (1.0, 1.0, 1.0))
        sa_2mm = _surface_area_mm2(mask, (2.0, 2.0, 2.0))
        # Surface area scales with voxel_size^2
        assert sa_2mm == pytest.approx(sa_1mm * 4, rel=0.05)


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
            assert f"X_{k}" in s, f"Missing key: X_{k}"

    def test_known_values(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = _intensity_stats(vals, prefix="M")
        assert s["M_mean"]   == pytest.approx(3.0)
        assert s["M_min"]    == pytest.approx(1.0)
        assert s["M_max"]    == pytest.approx(5.0)
        assert s["M_std"]    == pytest.approx(np.std(vals))
        assert s["M_energy"] == pytest.approx(55.0)   # 1+4+9+16+25
        assert s["M_rms"]    == pytest.approx(np.sqrt(55.0 / 5))

    def test_empty_array_all_nan(self):
        s = _intensity_stats(np.array([]), prefix="X")
        for k in self.EXPECTED_KEYS:
            assert np.isnan(s[f"X_{k}"]), f"Expected NaN for X_{k}"

    def test_zero_mean_cov_is_nan(self):
        s = _intensity_stats(np.array([-1.0, 1.0]), prefix="X")
        assert np.isnan(s["X_cov"])

    def test_constant_array_std_is_zero(self):
        s = _intensity_stats(np.ones(20) * 5.0, prefix="X")
        assert s["X_std"] == pytest.approx(0.0, abs=1e-10)
        assert s["X_var"] == pytest.approx(0.0, abs=1e-10)


# ── _compute_4d_inertia ─────────────────────────────────────────────────────

class TestCompute4dInertia:
    def test_empty_returns_none(self):
        ev, c = _compute_4d_inertia(np.zeros((0, 3)), np.zeros(0))
        assert ev is None and c is None

    def test_returns_four_eigenvalues(self):
        rng = np.random.default_rng(7)
        coords = rng.integers(0, 20, (200, 3)).astype(float)
        intensities = rng.normal(100, 10, 200)
        ev, centroid = _compute_4d_inertia(coords, intensities)
        assert ev is not None and len(ev) == 4
        assert centroid is not None and len(centroid) == 4

    def test_eigenvalues_non_negative(self):
        rng = np.random.default_rng(8)
        coords = rng.uniform(0, 10, (100, 3))
        intensities = rng.normal(50, 5, 100)
        ev, _ = _compute_4d_inertia(coords, intensities)
        assert np.all(ev >= -1e-10)  # non-negative (up to float precision)


# ── _boundary_gradient_stats ───────────────────────────────────────────────

class TestBoundaryGradientStats:
    def test_finite_values_for_sphere_mask(self):
        data = np.random.default_rng(1).normal(100, 10, (30, 30, 30)).astype(np.float32)
        mask = make_sphere_mask((30, 30, 30), (15, 15, 15), 5)
        s = _boundary_gradient_stats(data, mask, prefix="T1")
        assert not np.isnan(s["T1_boundary_grad_mean"])
        assert not np.isnan(s["T1_boundary_grad_std"])
        assert s["T1_boundary_grad_mean"] >= 0

    def test_keys_present(self):
        data = np.ones((20, 20, 20), dtype=np.float32)
        mask = make_sphere_mask((20, 20, 20), (10, 10, 10), 4)
        s = _boundary_gradient_stats(data, mask, prefix="FL")
        assert "FL_boundary_grad_mean" in s
        assert "FL_boundary_grad_std" in s


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
    "inertia_eigval_1", "inertia_eigval_2", "inertia_eigval_3",
    "elongation", "flatness", "isotropy",
    "euler_number", "nearest_lesion_dist_vox",
]
INTENSITY_STATS = [
    "mean", "median", "mode", "min", "max", "var", "std",
    "skewness", "kurtosis", "moment5", "entropy",
    "iqr", "p10", "p25", "p75", "p90", "cov", "energy", "rms",
    "boundary_grad_mean", "boundary_grad_std",
]


class TestQuantifyLesions:
    def test_row_count_matches_lesion_count(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert len(df) == 2

    def test_all_shape_columns_present(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for col in SHAPE_COLS:
            assert col in df.columns, f"Missing shape column: {col}"

    def test_all_intensity_columns_present(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for mod in ("T1", "T2"):
            for stat in INTENSITY_STATS:
                col = f"{mod}_{stat}"
                assert col in df.columns, f"Missing intensity column: {col}"

    def test_4d_inertia_columns_present(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for mod in ("T1", "T2"):
            for i in range(1, 5):
                assert f"inertia_4d_{mod}_eigval_{i}" in df.columns

    def test_pairwise_columns_present(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert "T1_over_T2_mean_ratio" in df.columns
        assert "T1_T2_correlation" in df.columns

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
        r1, r2 = 7, 4
        v1 = df.loc[df["label"] == 1, "volume_mm3"].iloc[0]
        v2 = df.loc[df["label"] == 2, "volume_mm3"].iloc[0]
        assert v1 == pytest.approx(4/3 * np.pi * r1**3, rel=0.10)
        assert v2 == pytest.approx(4/3 * np.pi * r2**3, rel=0.10)

    def test_intensity_means_close_to_target(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        row = df.loc[df["label"] == 1].iloc[0]
        assert row["T1_mean"] == pytest.approx(800.0, abs=15.0)
        assert row["T2_mean"] == pytest.approx(700.0, abs=15.0)

    def test_convexness_at_most_one(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for val in df["convexness"].dropna():
            assert 0 < val <= 1.05  # small tolerance for voxelisation

    def test_sphericity_positive(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for val in df["sphericity"].dropna():
            assert val > 0

    def test_hemisphere_valid_values(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert set(df["hemisphere"]).issubset({"L", "R"})

    def test_euler_number_is_one_for_solid_spheres(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert (df["euler_number"] == 1).all()

    def test_no_nan_in_core_columns(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for col in ["volume_vox", "volume_mm3", "convexness",
                    "T1_mean", "T2_mean", "elongation", "flatness", "isotropy"]:
            assert not df[col].isna().any(), f"Unexpected NaN in: {col}"

    def test_probabilistic_columns_present_when_prob_map_given(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            prob_map=synthetic_dataset["prob_img"],
            include_radiomics=False,
        )
        for col in ("fuzzy_volume_vox", "mean_confidence", "std_confidence",
                    "prob_weighted_com_x", "prob_weighted_com_y", "prob_weighted_com_z",
                    "prob_entropy", "core_volume_vox", "rim_volume_vox"):
            assert col in df.columns, f"Missing probabilistic column: {col}"
        assert df["mean_confidence"].between(0.0, 1.0).all()
        assert (df["fuzzy_volume_vox"] > 0).all()

    def test_probabilistic_columns_absent_without_prob_map(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            prob_map=None,
            include_radiomics=False,
        )
        assert "fuzzy_volume_vox" not in df.columns
        assert "mean_confidence" not in df.columns

    def test_nawm_norm_column_when_nawm_mask_given(self, synthetic_dataset):
        nawm = np.zeros((70, 70, 70), dtype=np.float32)
        nawm[60:65, 60:65, 60:65] = 1.0
        nawm_img = make_nifti(nawm)
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            nawm_mask=nawm_img,
            include_radiomics=False,
        )
        assert "T1_nawm_norm_mean" in df.columns
        assert "T2_nawm_norm_mean" in df.columns

    def test_three_modalities_all_pairs_present(self, synthetic_dataset):
        flair = np.random.default_rng(99).normal(600, 40, (70, 70, 70)).astype(np.float32)
        mods = {**synthetic_dataset["modalities"], "FLAIR": make_nifti(flair)}
        df = quantify_lesions(
            synthetic_dataset["labeled_img"], mods, include_radiomics=False
        )
        for pair in (("T1", "T2"), ("T1", "FLAIR"), ("T2", "FLAIR")):
            assert f"{pair[0]}_over_{pair[1]}_mean_ratio" in df.columns
            assert f"{pair[0]}_{pair[1]}_correlation" in df.columns

    def test_single_modality_no_pairwise_columns(self):
        shape = (40, 40, 40)
        mask = make_sphere_mask(shape, (20, 20, 20), 6).astype(np.int32)
        labeled_img = nib.Nifti1Image(mask, np.eye(4))
        t1 = make_nifti(np.random.default_rng(5).normal(1000, 50, shape).astype(np.float32))
        df = quantify_lesions(labeled_img, {"T1": t1}, include_radiomics=False)
        assert len(df) == 1
        assert "T1_mean" in df.columns
        # No pairwise columns with only one modality
        pairwise_cols = [c for c in df.columns if "_over_" in c or "_correlation" in c]
        assert pairwise_cols == []

    def test_bbox_fill_ratio_between_0_and_1(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        for val in df["bbox_fill_ratio"].dropna():
            assert 0 < val <= 1.0

    def test_nearest_lesion_dist_positive(self, synthetic_dataset):
        df = quantify_lesions(
            synthetic_dataset["labeled_img"],
            synthetic_dataset["modalities"],
            include_radiomics=False,
        )
        assert (df["nearest_lesion_dist_vox"] > 0).all()
