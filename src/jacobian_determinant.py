"""This module implements the Jacobian Determinant image processing pipeline outlined in `Slowly expanding/evolving
lesions as a magnetic resonance imaging marker of chronic active multiple sclerosis lesions
<https://doi.org/10.1177/1352458518814117>`_ and `Jacobian integration method increases the statistical power to
measure gray matter atrophy in multiple sclerosis <https://doi.org/10.1016/j.nicl.2013.10.015>`_."""

import ants


def resample_to_isotropic(image: ants.ANTsImage, voxel_size: float = 1.0) -> ants.ANTsImage:
    """Resample image to isotropic voxel dimensions.

    Args:
        image (ants.ANTsImage): The image to be resampled.
        voxel_size (float): The desired voxel size in millimeters for all dimensions.

    Returns:
        ants.ANTsImage: The resampled image.
    """
    return ants.resample_image(image, resample_params=(voxel_size, voxel_size, voxel_size), use_voxels=False)


def register_to_halfway_space(fixed_image: ants.ANTsImage, moving_image: ants.ANTsImage) -> (ants.ANTsImage, ants.ANTsImage):
    """
    Perform affine registration to a halfway space between two images.

    Args:
        fixed_image (ants.ANTsImage): The reference image.
        moving_image (ants.ANTsImage): The image to be aligned to the reference.

    Returns:
        tuple: A tuple containing the two images transformed into the halfway space.
    """
    registration_fixed_to_moving = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Affine')

    registration_moving_to_fixed = ants.registration(fixed=moving_image, moving=fixed_image, type_of_transform='Affine')

    halfway_fixed = ants.apply_transforms(fixed=fixed_image, moving=fixed_image,
                                          transformlist=registration_moving_to_fixed['fwdtransforms'],
                                          whichtoinvert=[True], interpolator='linear')
    halfway_moving = ants.apply_transforms(fixed=moving_image, moving=moving_image,
                                           transformlist=registration_fixed_to_moving['fwdtransforms'],
                                           whichtoinvert=[True], interpolator='linear')

    return halfway_fixed, halfway_moving

def nonlinear_registration(fixed_image: ants.ANTsImage, moving_image: ants.ANTsImage,
                           additional_image: ants.ANTsImage, step_size: float = 0.7,
                           gaussian_sigma: float = 2) -> dict:
    """Perform non-linear registration with specific parameters using multiple modalities.

    Args:
        fixed_image (ants.ANTsImage): The fixed reference image (T1-weighted).
        moving_image (ants.ANTsImage): The moving image (T1-weighted) to be registered.
        additional_image (ants.ANTsImage): An additional modality (T2-weighted) that needs to be
                                           incorporated into the registration process.
        step_size (float): Step size for the registration.
        gaussian_sigma (float): Gaussian sigma for smoothing in registration.

    Returns:
        dict: The registration result including transformations.
    """
    # Combining both T1 and T2 images for registration; this assumes both images are aligned.
    combined_fixed = ants.image_math(fixed_image, "add", additional_image)
    combined_moving = ants.image_math(moving_image, "add", additional_image)
    return ants.registration(combined_fixed, combined_moving, type_of_transform='SyN',
                             reg_iterations=(40, 20, 0), transformation_parameters=(step_size, gaussian_sigma))


def compute_jacobian_determinant(tx: ants.ANTsTransform) -> ants.ANTsImage:
    """Compute the Jacobian determinant of the deformation field.

    Args:
        tx (ants.ANTsTransform): The transformation for which to compute the Jacobian determinant.

    Returns:
        ants.ANTsImage: The Jacobian determinant image.
    """
    return ants.create_jacobian_determinant_image(tx, do_log=False)


def resample_jacobian_determinant(jacobian_image: ants.ANTsImage, original_image: ants.ANTsImage) -> ants.ANTsImage:
    """Resample the Jacobian determinant image to the original resolution.

    Args:
        jacobian_image (ants.ANTsImage): The Jacobian determinant image to resample.
        original_image (ants.ANTsImage): The image whose resolution will be matched.

    Returns:
        ants.ANTsImage: The resampled Jacobian determinant image.
    """
    return ants.resample_image_to_target(jacobian_image, original_image, interp_type='linear')
