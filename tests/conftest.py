"""Shared pytest fixtures for the LesionOmics test suite."""

import numpy as np
import nibabel as nib
import pytest


def make_sphere_mask(shape, center, radius):
    """Boolean mask with a filled sphere."""
    idx = np.indices(shape).transpose(1, 2, 3, 0).astype(float)
    return np.linalg.norm(idx - np.array(center, dtype=float), axis=-1) <= radius


def make_nifti(data, voxel_sizes=(1.0, 1.0, 1.0)):
    """Wrap a numpy array in a NIfTI image with isotropic or anisotropic voxels."""
    affine = np.diag([*voxel_sizes, 1.0])
    return nib.Nifti1Image(data.astype(np.float32), affine)


@pytest.fixture(scope="module")
def synthetic_dataset():
    """
    Two spherical lesions embedded in noisy background.

    Lesion 1: radius 7, label 1  (larger)
    Lesion 2: radius 4, label 2  (smaller)

    T1 inside lesion 1 ~ N(800, 20), T2 ~ N(700, 20)
    T1 inside lesion 2 ~ N(850, 15), T2 ~ N(650, 15)
    Background T1 ~ N(1000, 50),     T2 ~ N(500,  30)
    """
    shape = (70, 70, 70)
    rng = np.random.default_rng(42)

    t1 = rng.normal(1000, 50, shape).astype(np.float32)
    t2 = rng.normal(500,  30, shape).astype(np.float32)
    prob = np.zeros(shape, dtype=np.float32)
    labeled_data = np.zeros(shape, dtype=np.int32)

    m1 = make_sphere_mask(shape, (20, 35, 35), 7)
    t1[m1]   = rng.normal(800, 20, m1.sum()).astype(np.float32)
    t2[m1]   = rng.normal(700, 20, m1.sum()).astype(np.float32)
    prob[m1] = rng.uniform(0.6, 1.0, m1.sum()).astype(np.float32)
    labeled_data[m1] = 1

    m2 = make_sphere_mask(shape, (50, 35, 35), 4)
    t1[m2]   = rng.normal(850, 15, m2.sum()).astype(np.float32)
    t2[m2]   = rng.normal(650, 15, m2.sum()).astype(np.float32)
    prob[m2] = rng.uniform(0.5, 0.9, m2.sum()).astype(np.float32)
    labeled_data[m2] = 2

    affine = np.eye(4)
    return {
        "labeled_img": nib.Nifti1Image(labeled_data, affine),
        "modalities":  {"T1": make_nifti(t1), "T2": make_nifti(t2)},
        "prob_img":    make_nifti(prob),
        "m1": m1, "m2": m2,
        "t1": t1, "t2": t2,
    }
