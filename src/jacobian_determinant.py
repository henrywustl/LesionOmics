"""
Jacobian determinant pipeline for detecting slowly expanding/evolving MS lesions.

References:
    - https://doi.org/10.1177/1352458518814117
    - https://doi.org/10.1016/j.nicl.2013.10.015
"""

import ants


def resample_to_isotropic(image: ants.ANTsImage, voxel_size: float = 1.0) -> ants.ANTsImage:
    """Resample an image to isotropic voxels of the given size (mm)."""
    return ants.resample_image(image, resample_params=(voxel_size,) * 3, use_voxels=False)


def register_to_halfway_space(
    fixed_image: ants.ANTsImage,
    moving_image: ants.ANTsImage,
) -> tuple:
    """
    Affine-register both images to their halfway space.

    Each image is moved halfway toward the other, reducing interpolation
    bias before subsequent non-linear registration.

    Returns:
        (halfway_fixed, halfway_moving) as ANTsImage objects.
    """
    reg_fwd = ants.registration(
        fixed=fixed_image, moving=moving_image, type_of_transform="Affine"
    )
    reg_rev = ants.registration(
        fixed=moving_image, moving=fixed_image, type_of_transform="Affine"
    )
    halfway_fixed = ants.apply_transforms(
        fixed=fixed_image, moving=fixed_image,
        transformlist=reg_rev["fwdtransforms"], whichtoinvert=[True],
        interpolator="linear",
    )
    halfway_moving = ants.apply_transforms(
        fixed=moving_image, moving=moving_image,
        transformlist=reg_fwd["fwdtransforms"], whichtoinvert=[True],
        interpolator="linear",
    )
    return halfway_fixed, halfway_moving


def nonlinear_registration(
    fixed_image: ants.ANTsImage,
    moving_image: ants.ANTsImage,
    fixed_auxiliary: ants.ANTsImage = None,
    moving_auxiliary: ants.ANTsImage = None,
    step_size: float = 0.7,
    gaussian_sigma: float = 2.0,
) -> dict:
    """
    SyN non-linear registration, optionally combining a second modality
    for multi-channel registration.

    Args:
        fixed_image:      Fixed reference image (primary modality, e.g. T1).
        moving_image:     Moving image (primary modality).
        fixed_auxiliary:  Optional auxiliary fixed image (e.g. T2).
        moving_auxiliary: Optional auxiliary moving image (e.g. T2).
        step_size:        SyN gradient step size.
        gaussian_sigma:   Smoothing sigma.

    Returns:
        ANTs registration result dict ('fwdtransforms', 'invtransforms', etc.).
    """
    if fixed_auxiliary is not None and moving_auxiliary is not None:
        combined_fixed  = fixed_image  + fixed_auxiliary
        combined_moving = moving_image + moving_auxiliary
    else:
        combined_fixed  = fixed_image
        combined_moving = moving_image

    return ants.registration(
        fixed=combined_fixed,
        moving=combined_moving,
        type_of_transform="SyN",
        reg_iterations=(40, 20, 0),
        transformation_parameters=(step_size, gaussian_sigma),
    )


def compute_jacobian_determinant(
    domain_image: ants.ANTsImage,
    transform_path: str,
) -> ants.ANTsImage:
    """
    Compute the Jacobian determinant of a displacement field transform.

    Args:
        domain_image:   Image defining the space of the transform.
        transform_path: Path to the displacement field file
                        (first entry in registration['fwdtransforms']).

    Returns:
        Jacobian determinant image in the domain space.
    """
    return ants.create_jacobian_determinant_image(
        domain_image=domain_image, tx=transform_path, do_log=False
    )


def resample_jacobian_determinant(
    jacobian_image: ants.ANTsImage,
    target_image: ants.ANTsImage,
) -> ants.ANTsImage:
    """Resample Jacobian determinant image to match the resolution of target_image."""
    return ants.resample_image_to_target(
        jacobian_image, target_image, interp_type="linear"
    )


def compute_lesion_jacobian_pipeline(
    t1_fixed: ants.ANTsImage,
    t1_moving: ants.ANTsImage,
    t2_fixed: ants.ANTsImage = None,
    t2_moving: ants.ANTsImage = None,
    voxel_size: float = 1.0,
) -> ants.ANTsImage:
    """
    End-to-end Jacobian determinant pipeline for longitudinal lesion analysis.

    Steps: resample to isotropic -> halfway-space affine registration ->
    SyN non-linear registration -> Jacobian determinant -> resample back.

    Args:
        t1_fixed:   T1-weighted fixed (reference) image.
        t1_moving:  T1-weighted moving image.
        t2_fixed:   Optional T2-weighted fixed image (multi-modal registration).
        t2_moving:  Optional T2-weighted moving image (multi-modal registration).
        voxel_size: Isotropic voxel size (mm) for intermediate processing.

    Returns:
        Jacobian determinant image resampled to the original t1_fixed resolution.
    """
    t1_f = resample_to_isotropic(t1_fixed,  voxel_size)
    t1_m = resample_to_isotropic(t1_moving, voxel_size)
    t2_f = resample_to_isotropic(t2_fixed,  voxel_size) if t2_fixed  is not None else None
    t2_m = resample_to_isotropic(t2_moving, voxel_size) if t2_moving is not None else None

    t1_f_half, t1_m_half = register_to_halfway_space(t1_f, t1_m)
    if t2_f is not None:
        t2_f_half, t2_m_half = register_to_halfway_space(t2_f, t2_m)
    else:
        t2_f_half = t2_m_half = None

    reg = nonlinear_registration(
        t1_f_half, t1_m_half,
        fixed_auxiliary=t2_f_half,
        moving_auxiliary=t2_m_half,
    )

    # fwdtransforms[0] is the displacement field (warp); [1] is the affine component
    jacobian = compute_jacobian_determinant(t1_f_half, reg["fwdtransforms"][0])
    return resample_jacobian_determinant(jacobian, t1_fixed)
