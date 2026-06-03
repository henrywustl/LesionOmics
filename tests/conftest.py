import numpy as np
import nibabel as nib
import pytest


@pytest.fixture(scope="module")
def synthetic_dataset():
    """70×70×70 volume with two spherical lesions at known positions.

    Lesion 1: radius=7, centre=(20,35,35), label=1  (~1437 vox)
    Lesion 2: radius=4, centre=(50,35,35), label=2  (~268 vox)

    T1: background~N(1000,50), lesion~N(800,20)
    T2: background~N(500,30),  lesion~N(700,20)
    """
    shape = (70, 70, 70)
    affine = np.eye(4)  # 1 mm isotropic

    rng = np.random.default_rng(42)
    t1_data = rng.normal(1000, 50, shape).astype(np.float32)
    t2_data = rng.normal(500, 30, shape).astype(np.float32)

    z, y, x = np.mgrid[0:70, 0:70, 0:70]
    mask1 = (z - 20) ** 2 + (y - 35) ** 2 + (x - 35) ** 2 <= 49  # r=7
    mask2 = (z - 50) ** 2 + (y - 35) ** 2 + (x - 35) ** 2 <= 16  # r=4

    labeled = np.zeros(shape, dtype=np.int32)
    labeled[mask1] = 1
    labeled[mask2] = 2

    t1_data[mask1] = rng.normal(800, 20, int(mask1.sum())).astype(np.float32)
    t2_data[mask1] = rng.normal(700, 20, int(mask1.sum())).astype(np.float32)
    t1_data[mask2] = rng.normal(800, 20, int(mask2.sum())).astype(np.float32)
    t2_data[mask2] = rng.normal(700, 20, int(mask2.sum())).astype(np.float32)

    header = nib.Nifti1Header()
    header.set_data_shape(shape)
    header.set_zooms((1.0, 1.0, 1.0))

    labeled_img = nib.Nifti1Image(labeled, affine, header)
    t1_img = nib.Nifti1Image(t1_data, affine, header)
    t2_img = nib.Nifti1Image(t2_data, affine, header)

    return {
        "labeled_img": labeled_img,
        "modalities": {"T1": t1_img, "T2": t2_img},
        "labeled": labeled,
        "t1_data": t1_data,
        "t2_data": t2_data,
        "mask1": mask1,
        "mask2": mask2,
        "affine": affine,
        "header": header,
        "shape": shape,
    }
