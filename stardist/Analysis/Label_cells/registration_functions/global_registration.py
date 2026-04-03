" Functions for global image registration. "

# imports
from typing import Tuple, Union
import math
import numpy as np
import scipy
import skimage
import cv2
from .calculate_dislocation import calculate_dislocation

# pylint: disable=no-member


def register_global_im(
    image: np.ndarray,  # "im" in MATLAB code
    transform: np.ndarray,  # "tform" in MATLAB code
    flipped: bool,  # "f" in MATLAB code
    fillval: Union[np.ndarray, float, int],
) -> np.ndarray:
    """Applies previously calculated global image registration to an image

    Args:
        image: Image to register.
        transform: Transformation matrix.
        flipped: True if the input image should be flipped, False otherwise.
        fillval: Value to fill in empty space (may be an array with different values
            for each color channel).

    Returns:
        Registered image.

    Raises:
        ValueError: If the input image is not a 2D or 3D array, or the "fillval"
            argument doesn't match the shape of the input image.
    """
    # flip if necessary
    if flipped == 1:
        image = cv2.flip(image, 0)
    # register
    if len(image.shape) == 2 and isinstance(fillval, (float, int, np.integer)):
        return cv2.warpAffine(  # type: ignore[call-overload]
            image,
            transform,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=float(fillval),
        )
    if (
        len(image.shape) == 3
        and isinstance(fillval, np.ndarray)
        and fillval.shape == (3,)
    ):
        warped_channels = [
            cv2.warpAffine(  # type: ignore[call-overload]
                channel,
                transform,
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=float(fillval[i]),
            )
            for i, channel in enumerate(cv2.split(image))
        ]
        return cv2.merge(warped_channels)
    raise ValueError(
        (
            "Failed to determine global registration method for image with shape "
            f"{image.shape} and fill value {fillval} of type {type(fillval)}."
        )
    )


def _bandpass_img(arr: np.ndarray, lnoise: float, lobject: int) -> np.ndarray:
    """Apply a bandpass filter to a given image.

    Originally called "bpassW" in MATLAB code.

    Args:
        arr: Image to filter.
        lnoise: Noise level (standard deviation for Gaussian blur).
        lobject: Object level (size of the averaging filter).

    Returns:
        Filtered image.
    """
    filter_size = 2 * lobject + 1
    arr_gaussian_blurred = cv2.GaussianBlur(
        arr,
        (filter_size, filter_size),
        lnoise * math.sqrt(2),
        borderType=cv2.BORDER_REFLECT_101,
    )
    arr_avg = cv2.blur(
        arr, (filter_size, filter_size), borderType=cv2.BORDER_REFLECT_101
    )
    # remove extremely small values
    result = np.maximum(arr_gaussian_blurred - arr_avg, 1e-10)
    result[result <= 1e-10] = 0
    return result


def _get_opencv_transformation_matrix(
    rotation_angle_degrees: Union[float, np.float64] = 0,
    rotation_center: Tuple[float, float] = (0.0, 0.0),
    translation_xy: np.ndarray = np.array([0.0, 0.0]),
) -> np.ndarray:
    """Get an OpenCV transformation matrix for a given rotation angle and translation.

    The rotation is applied about the center point of the image first, followed by the
    translation.

    Args:
        rotation_angle_degrees: Angle of rotation in degrees.
        rotation_center_point: Center point (x, y) for rotation.
        translation_xy: Translation vector (x, y).

    Returns:
        OpenCV-formatted combined transformation matrix.
    """
    matrix = cv2.getRotationMatrix2D(rotation_center, float(rotation_angle_degrees), 1)
    matrix[0, 2] += translation_xy[0]
    matrix[1, 2] += translation_xy[1]
    return matrix


def _apply_transformation(
    image: np.ndarray,
    rotation_angle: Union[float, np.float64] = 0.0,
    translation_xy: np.ndarray = np.array([0.0, 0.0]),
) -> np.ndarray:
    """
    Apply rotation and translation to an image.

    Helper function for _reg_ims_com.

    Args:
        image: The input image.
        rotation_angle: The angle to rotate the image.
        translation_xy: The translation vector (x, y).

    Returns:
        The transformed image.
    """
    height, width = image.shape[:2]
    transformation_matrix = _get_opencv_transformation_matrix(
        rotation_angle, (width / 2, height / 2), translation_xy
    )
    interpolation = cv2.INTER_LINEAR  # if translation_xy.any() else cv2.INTER_NEAREST
    return cv2.warpAffine(
        image, transformation_matrix, (width, height), flags=interpolation
    )


def _get_com(image: np.ndarray) -> np.ndarray:
    """Get the center of mass of an image.

    Args:
        image: Image to calculate the center of mass for.

    Returns:
        Center of mass (x, y) of the image.
    """
    mask = (_bandpass_img(image, 2, 50) > 0).astype(np.uint8)
    mom = cv2.moments(mask)
    return np.array([mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]])


def _calculate_rr_metric(
    image1: np.ndarray, image2: np.ndarray, valid_indices: np.ndarray
) -> float:
    """Calculate the registration quality metric between two images.

    Helper function for _reg_ims_com.

    Args:
        image1: First image to compare.
        image2: Second image to compare.
        valid_indices: Indices of valid pixels in the images.
    """
    return np.corrcoef(
        image1[valid_indices].flatten(),
        image2[valid_indices].flatten(),
    )[0, 1]


def _reg_ims_com(
    im_ref_arr: np.ndarray,
    im_moving_arr: np.ndarray,
    n_iters: int,
    init_rot_deg: float = 0.0,
    init_trans_xy: np.ndarray = np.array([0.0, 0.0]),
    do_center_of_mass_translation: bool = False,
    theta_range: float = 90.0,
    theta_step: float = 0.5,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Determines the global rotation angle and overall translation to maximize cross
    correlation of two images

    Args:
        im_ref_arr: Reference image array to register to.
        im_moving_arr: Image array to register with a rotation and a translation.
        n_iters: Number of iterations to run registration.
        init_rot_deg: Angle of initial rotation to apply about image center (degrees).
            This rotation is applied before any other transformations.
        init_trans_xy: Initial translation vector (x, y). If
            do_center_of_mass_translation is True, this translation will be applied in
            addition to the center of mass translation.
        do_center_of_mass_translation: If True, calculate an initial translation based
            on the center of mass of the images and add it to init_trans_xy.
        theta_range: The limit of the range of angles (in degrees) to use for the radon
            transform in determining the rotation transformation. Angles between +/-
            theta_range will be tested at increments of theta_step.
        theta_step: The step size (in degrees) to use for the radon transform in
            determining the rotation transformation.

    Returns:
        Registered image, rotation angle, translation vector, and registration quality
            metric.
    """
    # start by applying the initial rotation
    (height, width) = im_moving_arr.shape[:2]
    im_moving_init_rot = (
        _apply_transformation(im_moving_arr, init_rot_deg)
        if init_rot_deg != 0
        else im_moving_arr
    )
    if np.sum(im_moving_init_rot) == 0:
        return im_moving_init_rot, init_rot_deg, np.array([0.0, 0.0]), 0.0
    # apply initial translation, optionally shifted by center of mass
    trans_xy = init_trans_xy
    if do_center_of_mass_translation:
        trans_xy = trans_xy + (_get_com(im_ref_arr) - _get_com(im_moving_init_rot))
    im_moving_first_pass = (
        _apply_transformation(im_moving_init_rot, translation_xy=trans_xy)
        if trans_xy.any()
        else im_moving_init_rot
    )
    if np.sum(im_moving_first_pass) == 0:
        return im_moving_first_pass, init_rot_deg, trans_xy, 0.0
    # init some other variables
    im_moving_mode = scipy.stats.mode(im_moving_arr).mode[0]
    inc_rot_angles = np.zeros(n_iters + 1)
    inc_rot_angles[0] = init_rot_deg
    valid_ref_indices = im_ref_arr > 0
    rr_metric_init = _calculate_rr_metric(
        im_moving_first_pass, im_ref_arr, valid_ref_indices
    )
    # iterate "n_iters" times to achieve sufficient global registration
    theta = np.arange(-1.0 * theta_range, theta_range + theta_step, theta_step)
    radon_ref = skimage.transform.radon(im_ref_arr, theta, circle=True)
    radon_ref = _bandpass_img(radon_ref, 1, 3)
    im_moving_iter = im_moving_first_pass
    for kk in range(n_iters):
        # use radon for rotational registration
        radon_moving = skimage.transform.radon(im_moving_iter, theta, circle=True)
        radon_moving = _bandpass_img(radon_moving, 1, 3)
        radon_disloc = calculate_dislocation(radon_ref, radon_moving)
        rot_angle = radon_disloc[0] * theta_step
        # rotate image then calculate translational registration
        im_moving_iter_rot = _apply_transformation(im_moving_iter, rot_angle)
        im_moving_iter_rot[im_moving_iter_rot == 0] = im_moving_mode
        trans_xy_iter = calculate_dislocation(im_ref_arr, im_moving_iter_rot)
        # keep old transform in case update is bad
        rot_angles_prev = inc_rot_angles.copy()
        trans_xy_prev = trans_xy.copy()
        # update total rotation angle
        inc_rot_angles[kk + 1] = rot_angle
        # update rotation of total translation and add new iteration
        trans_xy = (
            cv2.getRotationMatrix2D((width / 2, height / 2), rot_angle, 1.0)[:2, :2]
            @ trans_xy
        ) + trans_xy_iter
        # update registration image
        im_moving_iter = _apply_transformation(
            im_moving_arr, np.sum(inc_rot_angles), trans_xy
        )
        im_moving_iter[im_moving_iter == 0] = im_moving_mode
        rr_metric = _calculate_rr_metric(im_moving_iter, im_ref_arr, valid_ref_indices)
        # if iteration hasn't improved correlation of images, then stop
        if rr_metric + 0.02 < rr_metric_init and n_iters > 2:
            inc_rot_angles = rot_angles_prev
            trans_xy = trans_xy_prev
            im_moving_iter = _apply_transformation(
                im_moving_arr, np.sum(inc_rot_angles), trans_xy
            )
            im_moving_iter[im_moving_iter == 0] = im_moving_mode
            rr_metric = _calculate_rr_metric(
                im_moving_iter, im_ref_arr, valid_ref_indices
            )
            break
        # maximum distance a point in the image moves
        center_x = width / 2
        center_y = height / 2
        corners = np.array(
            [
                [-center_x, -center_y],
                [center_x, -center_y],
                [center_x, center_y],
                [-center_x, center_y],
            ]
        )
        angle_rad = np.radians(rot_angle)
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )
        rotated_corners = np.dot(corners, rotation_matrix.T)
        translated_corners = rotated_corners + trans_xy_iter
        distances = np.sqrt(np.sum(translated_corners**2, axis=1))
        rff = np.max(distances)
        if rff < 0.75 or rr_metric > 0.9:
            break
    return im_moving_iter, float(np.sum(inc_rot_angles)), trans_xy, float(rr_metric)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image.

    Normalize by subtracting the mean and dividing by the standard deviation,
    flooring to the resulting minimum and maintaining locations of empty pixels.

    Args:
        image: Image to normalize.

    Returns:
        Normalized image.
    """
    empty_pixels = image == 0
    image = (image - np.mean(image)) / np.std(image)
    image -= np.min(image)
    image[empty_pixels] = 0
    return image


def _group_of_reg(
    im_ref_arr: np.ndarray,
    im_moving_arr: np.ndarray,
    n_iters: int,
    metric_threshold: float,
) -> Tuple[float, float, np.ndarray]:
    """Calculates sets of global registrations considering different initialization
    angles for a pair of greyscale images.

    Iteratively refines initial guesses instead of using a hardcoded set of angles.

    Args:
        im_ref_arr: Reference image to register to.
        im_moving_arr: Image to register.
        n_iters: Number of iterations to run registration.
        metric_threshold: Registration quality metric threshold.

    Returns:
        Registration quality metric, rotation angle, and translation vector for best
            registration.
    """
    # Initial angles and step size
    angles = [0.0, 180.0, 90.0, 270.0]
    step = 45.0
    # Normalize images
    im_ref_arr = _normalize_image(im_ref_arr)
    im_moving_arr = _normalize_image(im_moving_arr)
    # Set up variables for the best transformation and result
    best_metric_value = 0.2
    best_rotation_angle = 0.0
    best_trans_xy = np.array([0.0, 0.0])
    # Iterate until the step size is less than 2 degrees
    while step >= 2:
        # Test angles between +/- 2x the step size, at 1/10th of the step size
        # Except in the final iteration, which is very granular
        theta_range = step * 2 if step / 2 >= 2 else 90
        theta_step = step / 10.0 if step / 2 >= 2 else 0.5
        for angle in angles:
            registered_im, rot_angle, trans_xy, rr_metric = _reg_ims_com(
                im_ref_arr,
                im_moving_arr,
                n_iters,
                angle,
                do_center_of_mass_translation=True,
                theta_range=theta_range,
                theta_step=theta_step,
            )
            if rr_metric != 0:
                aa = (im_ref_arr > 0).astype(np.uint8) + (registered_im > 0).astype(
                    np.uint8
                )
                rr_metric = np.sum(aa == 2) / np.sum(aa > 0)
            if rr_metric > best_metric_value:
                best_metric_value = rr_metric
                best_rotation_angle = rot_angle
                best_trans_xy = trans_xy
            if best_metric_value > metric_threshold:
                break
        # Refine angles around the best angle found
        angles = [best_rotation_angle + step * i for i in (0, -1, 1, -2, 2)]
        step /= 2  # Reduce step size for finer search
    return best_metric_value, best_rotation_angle, best_trans_xy


def _rescale_and_blur_image(
    image: np.ndarray, rescale: int, sigma: int = 2
) -> np.ndarray:
    """Rescale and blur an image.

    Args:
        image: Image to rescale and blur.
        rescale: Factor to downsample images by.
        sigma: Standard deviation for Gaussian blur.

    Returns:
        Rescaled and blurred image.
    """
    return cv2.GaussianBlur(
        cv2.resize(
            image,
            (round(image.shape[1] / rescale), round(image.shape[0] / rescale)),
            interpolation=cv2.INTER_AREA,
        ),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )


def calculate_global_reg(
    im_ref_arr: np.ndarray,
    im_moving_arr: np.ndarray,
    rescale: int,
    max_iters: int,
    includes_ihc: bool,
) -> Tuple[np.ndarray, bool, float]:
    """Calculates global registration of a pair of greyscale, downsampled images.

    Args:
        ref_im: Reference image to register to.
        moving_im: Image to register.
        rescale: Factor to downsample images by.
        max_iters: Maximum number of iterations to run registration.
        includes_ihc: True if images are IHC, false if images are H&E

    Returns:
        Transformation matrix, flip status (True if flipped, False otherwise), Rout
            (registration quality metric).
    """
    rr_metric_threshold = 0.8 if includes_ihc else 0.9
    # pre-registration image processing
    a_ref = _rescale_and_blur_image(im_ref_arr, rescale)
    a_moving = _rescale_and_blur_image(im_moving_arr, rescale)
    if includes_ihc:
        a_ref = skimage.exposure.equalize_adapthist(a_ref)
        a_moving = skimage.exposure.equalize_adapthist(a_moving)
    # calculate registration, flipping image if necessary
    n_init_iters = 2
    reg_metric, rot_angle, trans_xy = _group_of_reg(
        a_ref, a_moving, n_init_iters, rr_metric_threshold
    )
    flipped = False  # "f" in MATLAB code
    if reg_metric < 0.8:
        print("try flipping image")
        a_moving_flipped = cv2.flip(a_moving, 0)
        reg_metric_2, rot_angle_flipped, trans_xy_flipped = _group_of_reg(
            a_ref, a_moving_flipped, n_init_iters, rr_metric_threshold
        )
        if reg_metric_2 > reg_metric:
            flipped = True
            rot_angle = rot_angle_flipped
            trans_xy = trans_xy_flipped
            a_moving = a_moving_flipped
    amvout, rot_angle, translation_xy, reg_metric_out = _reg_ims_com(
        a_ref, a_moving, max_iters - n_init_iters, rot_angle, trans_xy
    )
    translation_xy = translation_xy * rescale
    aa = (a_ref > 0).astype(np.uint8) + (amvout > 0).astype(np.uint8)
    reg_metric_out = np.sum(aa == 2) / np.sum(aa > 0)
    # create output image
    tform = _get_opencv_transformation_matrix(
        rot_angle,
        (im_moving_arr.shape[1] / 2, im_moving_arr.shape[0] / 2),
        translation_xy,
    )
    return tform, flipped, reg_metric_out
