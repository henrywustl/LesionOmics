"""Jacobian determinant pipeline for lesion volumetric-change analysis.

Implements the approach from:
- Slowly expanding/evolving lesions as an MRI marker of chronic active MS
  (doi:10.1177/1352458518814117)
- Jacobian integration method increases statistical power to measure GM atrophy in MS
  (doi:10.1016/j.nicl.2013.10.015)
"""

import logging

logger = logging.getLogger(__name__)

try:
    import ants
    _ANTS_AVAILABLE = True
except ImportError:
    _ANTS_AVAILABLE = False
    logger.warning("antspyx not installed; Jacobian pipeline unavailable.")


def _require_ants():
    if not _ANTS_AVAILABLE:
        raise ImportError(
            "antspyx is required for the Jacobian pipeline. "
            "Install it with: pip install antspyx"
        )


def resample_to_isotropic(image: "ants.ANTsImage", voxel_size: float = 1.0) -> "ants.ANTsImage":
    """Resample image to isotropic voxel dimensions.

    Args:
        image: The image to resample.
        voxel_size: Desired isotropic voxel size in mm.

    Returns:
        Resampled ANTsImage.
    """
    _require_ants()
    return ants.resample_image(
        image,
        resample_params=(voxel_size, voxel_size, voxel_size),
        use_voxels=False,
    )


def register_to_halfway_space(
    fixed_image: "ants.ANTsImage",
    moving_image: "ants.ANTsImage",
) -> tuple:
    """Affine registration to a halfway space between two images.

    Args:
        fixed_image: Reference image (e.g. baseline T1).
        moving_image: Image to align (e.g. follow-up T1).

    Returns:
        Tuple of (halfway_fixed, halfway_moving) in the halfway space.
    """
    _require_ants()
    reg_f2m = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform="Affine")
    reg_m2f = ants.registration(fixed=moving_image, moving=fixed_image, type_of_transform="Affine")

    halfway_fixed = ants.apply_transforms(
        fixed=fixed_image,
        moving=fixed_image,
        transformlist=reg_m2f["fwdtransforms"],
        whichtoinvert=[True],
        interpolator="linear",
    )
    halfway_moving = ants.apply_transforms(
        fixed=moving_image,
        moving=moving_image,
        transformlist=reg_f2m["fwdtransforms"],
        whichtoinvert=[True],
        interpolator="linear",
    )
    return halfway_fixed, halfway_moving


def nonlinear_registration(
    fixed_image: "ants.ANTsImage",
    moving_image: "ants.ANTsImage",
    fixed_auxiliary: "ants.ANTsImage" = None,
    moving_auxiliary: "ants.ANTsImage" = None,
    step_size: float = 0.7,
    gaussian_sigma: float = 2,
) -> dict:
    """SyN non-linear registration, optionally incorporating an auxiliary modality.

    Args:
        fixed_image: Fixed reference image (T1).
        moving_image: Moving image (T1).
        fixed_auxiliary: Optional auxiliary modality for fixed (e.g. T2).
        moving_auxiliary: Optional auxiliary modality for moving (e.g. T2).
        step_size: SyN gradient step size.
        gaussian_sigma: Gaussian smoothing sigma.

    Returns:
        ANTs registration result dict (contains 'fwdtransforms', etc.).
    """
    _require_ants()
    if fixed_auxiliary is not None and moving_auxiliary is not None:
        combined_fixed = fixed_image + fixed_auxiliary
        combined_moving = moving_image + moving_auxiliary
    else:
        combined_fixed = fixed_image
        combined_moving = moving_image

    return ants.registration(
        combined_fixed,
        combined_moving,
        type_of_transform="SyN",
        reg_iterations=(40, 20, 0),
        transformation_parameters=(step_size, gaussian_sigma),
    )


def compute_jacobian_determinant(
    domain_image: "ants.ANTsImage",
    transform_path: str,
) -> "ants.ANTsImage":
    """Compute the Jacobian determinant of a displacement field.

    Args:
        domain_image: Image defining the domain (resolution/space).
        transform_path: Path to the displacement field transform file.

    Returns:
        Jacobian determinant image (values > 1 = local expansion, < 1 = contraction).
    """
    _require_ants()
    return ants.create_jacobian_determinant_image(domain_image, transform_path, do_log=False)


def resample_jacobian_determinant(
    jacobian_image: "ants.ANTsImage",
    original_image: "ants.ANTsImage",
) -> "ants.ANTsImage":
    """Resample a Jacobian image back to the original image resolution.

    Args:
        jacobian_image: Jacobian determinant image to resample.
        original_image: Target resolution image.

    Returns:
        Resampled Jacobian determinant image.
    """
    _require_ants()
    return ants.resample_image_to_target(jacobian_image, original_image, interp_type="linear")


def compute_lesion_jacobian_pipeline(
    t1_fixed: "ants.ANTsImage",
    t1_moving: "ants.ANTsImage",
    t2_fixed: "ants.ANTsImage" = None,
    t2_moving: "ants.ANTsImage" = None,
    voxel_size: float = 1.0,
) -> "ants.ANTsImage":
    """End-to-end Jacobian pipeline for lesion volumetric change.

    Resamples to isotropic, registers to halfway space, performs SyN, and
    returns the Jacobian determinant resampled to the original T1 space.

    Args:
        t1_fixed: Baseline T1 image.
        t1_moving: Follow-up T1 image.
        t2_fixed: Optional baseline T2 (for multi-modal registration).
        t2_moving: Optional follow-up T2.
        voxel_size: Isotropic voxel size for resampling before registration.

    Returns:
        Jacobian determinant image in the original T1 space.
    """
    _require_ants()
    t1_f = resample_to_isotropic(t1_fixed, voxel_size)
    t1_m = resample_to_isotropic(t1_moving, voxel_size)
    t2_f = resample_to_isotropic(t2_fixed, voxel_size) if t2_fixed is not None else None
    t2_m = resample_to_isotropic(t2_moving, voxel_size) if t2_moving is not None else None

    halfway_fixed, halfway_moving = register_to_halfway_space(t1_f, t1_m)

    reg = nonlinear_registration(
        halfway_fixed,
        halfway_moving,
        fixed_auxiliary=t2_f,
        moving_auxiliary=t2_m,
    )

    syn_transform = reg["fwdtransforms"][0]
    jacobian = compute_jacobian_determinant(halfway_fixed, syn_transform)
    return resample_jacobian_determinant(jacobian, t1_fixed)
