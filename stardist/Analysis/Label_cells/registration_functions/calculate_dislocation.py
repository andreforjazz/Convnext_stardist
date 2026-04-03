" Common functions for calculating translation transformations."

# imports
from typing import Optional, Tuple
import numpy as np
import scipy
import cv2

# pylint: disable=no-member


def _get_bright_spot_centroid(
    image_array: np.ndarray,
    max_locs: Tuple[np.ndarray, ...],
    window_size: int = 25,
    threshold_percentile: float = 95.0,
    brightness_weight: float = 1.5,
) -> np.ndarray:
    """Calculates the centroid of the brightest spot in an image to sub-pixel accuracy.

    Args:
        image_array: Image to process. Particle should be bright spots on dark
            background with little noise. Often a bandpass filtered brightfield image
            or a nice fluorescent image.
        max_locs: Locations of local maxima to pixel-level accuracy. (Tuple of
            (y_locs, x_locs))
        window_size: Diameter of the window over which to average to calculate the
            centroid. Should be big enough to capture the whole particle but not so big
            that it captures others. If initial guess of center is far from the
            centroid, the window will need to be larger than the particle size.
        threshold_percentile: Pixels within the window around each candidate location
            will be excluded if they are more dim than this percentile within the
            window.
        brightness_weight: Exponent to weight the brightness values. Higher values give
            more weight to brighter pixels. 1.0 corresponds to a normal center of mass
            calculation.

    Returns:
        Centroid of the bright spots to sub-pixel accuracy.
    """
    mx = np.array(list(max_locs)).reshape(1, -1)
    kk = round(window_size / 2)
    max_intensity = 0
    brightest_region_mask = None
    brightest_cand_i = None
    for cand_i in range(mx.shape[0]):
        # Slice the window around the candidate
        window_im = image_array[
            mx[cand_i, 0] - kk : mx[cand_i, 0] + kk + 1,
            mx[cand_i, 1] - kk : mx[cand_i, 1] + kk + 1,
        ]
        # Calculate the percentile-based threshold
        try:
            threshold = np.percentile(window_im[window_im > 0], threshold_percentile)
        except IndexError:
            # If the window is empty, skip this candidate
            continue
        # Create a mask using the threshold
        _, mask = cv2.threshold(window_im, threshold, 1, cv2.THRESH_BINARY)
        # Label the connected regions in the mask
        num_features, labeled_array, _, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        # Find the brightest region for this candidate
        for label_i in range(1, num_features):  # Skip the background label 0
            region_intensity = np.sum(window_im[labeled_array == label_i])
            if region_intensity > max_intensity:
                max_intensity = region_intensity
                brightest_region_mask = labeled_array == label_i
                brightest_cand_i = cand_i
    if brightest_region_mask is None:
        # If no bright region was found, return the first max location
        return np.array([mx[0, 0], mx[0, 1]])
    # Calculate the weighted center of mass based on brightness for the brightest region
    offset = np.array([mx[brightest_cand_i, 0] - kk, mx[brightest_cand_i, 1] - kk])
    region_intensity = image_array[
        offset[0] : offset[0] + window_size,
        offset[1] : offset[1] + window_size,
    ][brightest_region_mask]
    weighted_intensity = region_intensity**brightness_weight
    sum_weights = np.sum(weighted_intensity)
    y_coords, x_coords = np.nonzero(brightest_region_mask)
    center_of_mass = np.array(
        [
            np.sum(y_coords * weighted_intensity) / sum_weights,
            np.sum(x_coords * weighted_intensity) / sum_weights,
        ]
    )
    # if the center of mass is outside the mask, return the max location within the mask
    if brightest_region_mask[int(center_of_mass[0]), int(center_of_mass[1])] == 0:
        return (
            np.unravel_index(np.argmax(region_intensity), region_intensity.shape)
            + offset
        )
    # otherwise return the center of mass
    brightest_centroid = center_of_mass + offset
    return brightest_centroid


def calculate_dislocation(
    im_ref_arr: np.ndarray,
    im_moving_arr: np.ndarray,
    ref_yx: Optional[np.ndarray] = None,
    moving_yx: Optional[np.ndarray] = None,
    rm: Optional[np.ndarray] = None,
    rs: Optional[np.ndarray] = None,
    rg: Optional[int] = None,
) -> np.ndarray:
    """Estimate the dislocation between two images using cross correlation.

    Args:
        im_ref_arr: Static image array.
        im_moving_arr: Moving image array.
        ref_yx: Central point of the image pattern in the reference image (default is
            the center of the reference image).
        moving_yx: Central point of the image pattern in the moving image (default is
            the center of the moving image).
        rm: Size (y, x) of the image pattern (default is 95% of the image size to
            exclude edge effects).
        rs: Size (y, x) of the search range (default is 95% of the image size to exclude
            edge effects).
        rg: Search range (default is the max dimension of the images).

    Returns:
        np.ndarray: Estimated dislocation (x, y) of imnxt with respect to im.
    """
    imly = min(im_ref_arr.shape[0], im_moving_arr.shape[0])
    imlx = min(im_ref_arr.shape[1], im_moving_arr.shape[1])
    center_yx = np.array([imly, imlx]) // 2
    # rounding in line below is for excluding edge effects
    def_rm_rs = np.round(0.95 * center_yx).astype(int)
    # set default argument values
    if ref_yx is None:
        ref_yx = np.array([im_ref_arr.shape[0], im_ref_arr.shape[1]]) // 2
    if moving_yx is None:
        moving_yx = np.array([im_moving_arr.shape[0], im_moving_arr.shape[1]]) // 2
    if rm is None:
        rm = def_rm_rs
    if rs is None:
        rs = def_rm_rs
    if rg is None:
        rg = max(imly, imlx)
    # Slice out the ranges of the reference and moving images
    imptn = im_ref_arr[
        ref_yx[0] - rm[0] : ref_yx[0] + rm[0] + 1,
        ref_yx[1] - rm[1] : ref_yx[1] + rm[1] + 1,
    ]
    imgrid = im_moving_arr[
        moving_yx[0] - rs[0] : moving_yx[0] + rs[0] + 1,
        moving_yx[1] - rs[1] : moving_yx[1] + rs[1] + 1,
    ]
    # intensity normalization (may help to take off scale effects,
    # expecially for fft-based transformation)
    imptn = (imptn - np.mean(imptn)) / np.std(imptn)
    imgrid = (imgrid - np.mean(imgrid)) / np.std(imgrid)
    cross_corr = scipy.signal.correlate(imptn, imgrid, method="fft")
    msk = cv2.circle(  # type: ignore[call-overload]
        np.zeros(cross_corr.shape, dtype=np.uint8),
        center=(cross_corr.shape[1] // 2, cross_corr.shape[0] // 2),
        radius=rg // 2,
        color=1,  # values in the circle set to 1
        thickness=-1,  # fully filled in
    )
    cross_corr_m = cross_corr * msk
    # Get the location of the centroid of the bright spot to subpixel accuracy
    centroid_yx = _get_bright_spot_centroid(
        cross_corr_m, np.where(cross_corr_m == np.max(cross_corr_m))
    )
    # Calculate the translation from the centroid to the reference point
    translation_yx = centroid_yx - rs - rm
    return translation_yx[[1, 0]]  # Swap to xy
