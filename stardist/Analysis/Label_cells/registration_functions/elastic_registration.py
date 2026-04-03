" Functions for calculating elastic registration of images. "

# imports
from typing import Tuple, Union
import random
import pathlib
import numpy as np
import scipy
import cv2
import scipy.interpolate
from .calculate_dislocation import calculate_dislocation

# pylint: disable=no-member


def _reg_ims_elastic(
    im_ref_arr: np.ndarray,
    im_moving_arr: np.ndarray,
    rescale: int,
) -> np.ndarray:
    """Calculates registration translation only for a pair of tissue images.

    Original MATLAB function called "reg_ims_ELS". Original function also included a
    flag to return the registered moving image and the correlation coefficient.

    Args:
        im_ref_arr: Reference image.
        im_moving_arr: Moving image.
        rescale: Resize factor for downsampling.

    Returns:
        Displacement in x and y directions.
    """
    a_ref = cv2.resize(
        im_ref_arr,
        (round(im_ref_arr.shape[1] / rescale), round(im_ref_arr.shape[0] / rescale)),
        interpolation=cv2.INTER_LINEAR,
    )
    amv = cv2.resize(
        im_moving_arr,
        (
            round(im_moving_arr.shape[1] / rescale),
            round(im_moving_arr.shape[0] / rescale),
        ),
        interpolation=cv2.INTER_LINEAR,
    )
    xyt = calculate_dislocation(a_ref, amv)
    return -(xyt * rescale)


NN_GRIDS_FILTERS = [
    np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
]


def _get_nn_grids(grid: np.ndarray) -> np.ndarray:
    """Get nearest neighbor grids for a displacement map.

    Args:
        grid: Displacement map.

    Returns:
        Nearest neighbor grids.
    """
    if grid.dtype == np.bool:
        grid = grid.astype(np.uint8)
    nn_grids = [
        cv2.filter2D(grid, -1, f, borderType=cv2.BORDER_CONSTANT)
        for f in NN_GRIDS_FILTERS
    ]
    return np.stack(nn_grids, axis=-1)


FILL_VALS_SURROUNDING_PIXELS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def _fill_vals(
    xgg: np.ndarray, ygg: np.ndarray, cc: np.ndarray, xystd: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill in values in the displacement map that are outside the tissue region.

    Args:
        xgg: X displacement map to smooth.
        ygg: Y displacement map to smooth.
        cc: Boolean map of which locations in xgg/ygg should be smoothed.
        xystd: True if standard deviation of x and y should be calculated and returned.

    Returns:
        Smoothed x displacement map, smoothed y displacement map, denominator map for
        smoothing, standard deviation of x displacement map, standard deviation of y
        displacement map.
    """
    denom = cv2.filter2D(
        (~cc).astype(np.uint8),
        -1,
        FILL_VALS_SURROUNDING_PIXELS,
        borderType=cv2.BORDER_CONSTANT,
    )
    if xystd:
        grid_x = _get_nn_grids(xgg)
        grid_y = _get_nn_grids(ygg)
        grid_d = _get_nn_grids(~cc)
        grid_x = (grid_x - xgg[:, :, np.newaxis]) ** 2 * grid_d
        grid_y = (grid_y - ygg[:, :, np.newaxis]) ** 2 * grid_d
        sxgg = np.sqrt(
            np.divide(
                np.sum(grid_x, axis=-1),
                denom,
                out=np.zeros_like(grid_x[:, :, 0]),
                where=denom != 0,
            )
        )
        sygg = np.sqrt(
            np.divide(
                np.sum(grid_y, axis=-1),
                denom,
                out=np.zeros_like(grid_y[:, :, 0]),
                where=denom != 0,
            )
        )
        sxgg[cc] = 0
        sygg[cc] = 0
    else:
        sxgg = np.array([])
        sygg = np.array([])
    denom[denom == 0] = 1
    dxgg = (
        cv2.filter2D(
            xgg, -1, FILL_VALS_SURROUNDING_PIXELS, borderType=cv2.BORDER_CONSTANT
        )
        / denom
    )
    dygg = (
        cv2.filter2D(
            ygg, -1, FILL_VALS_SURROUNDING_PIXELS, borderType=cv2.BORDER_CONSTANT
        )
        / denom
    )
    return dxgg, dygg, denom, sxgg, sygg


def _make_final_grids(
    xgg0: np.ndarray,
    ygg0: np.ndarray,
    bf: int,
    x: np.ndarray,
    y: np.ndarray,
    szim: Tuple[int, int],
) -> np.ndarray:
    """Creates final nonlinear image registration matrices for a pair of registered
    images.

    In the original MATLAB code, this function had an additional case in it that would
    never run.

    Args:
        xgg0: Initial x displacement map.
        ygg0: Initial y displacement map.
        bf: size of buffer to add to displacement maps.
        x: tile x coordinates.
        y: tile y coordinates.
        szim: Image size.

    Returns:
        Displacement map.
    """
    xgg = np.copy(xgg0)
    ygg = np.copy(ygg0)
    mxy = 75  # 50 # allow no translation larger than this cutoff
    xgg[(np.abs(xgg) > mxy) | (np.abs(ygg) > mxy)] = -5000  # non-continuous values
    # find points where registration was calculated
    cempty = xgg == -5000
    xgg[cempty] = 0
    ygg[cempty] = 0
    # replace non-continuous values with mean of neighbors
    dxgg, dygg, _, sxgg, sygg = _fill_vals(xgg, ygg, cempty, True)
    m1 = np.divide(
        np.abs(xgg - dxgg), np.abs(dxgg), out=np.zeros_like(xgg), where=dxgg != 0
    )  # percent difference between x and mean of surrounding
    m2 = np.divide(
        np.abs(ygg - dygg), np.abs(dygg), out=np.zeros_like(ygg), where=dygg != 0
    )
    dd = (
        ((sxgg > 50) | (sygg > 50))  # large standard deviation
        | ((m1 > 5) | (m2 > 5))  # large percent difference
    ) & ~cempty
    xgg[dd] = dxgg[dd]
    ygg[dd] = dygg[dd]
    # fill in values outside tissue region with mean of neighbors
    count = 1
    while np.sum(cempty) > 0 and count < 500:
        dxgg, dygg, denom, _, _ = _fill_vals(xgg, ygg, cempty)
        cfill = (denom > 2) & cempty  # touching 3+ numbers and needs to be filled
        xgg[cfill] = dxgg[cfill]
        ygg[cfill] = dygg[cfill]
        cempty = cempty & ~cfill  # needs to be filled and has not been filled
        count += 1
    print(f"count = {count}/500")
    xgg = cv2.GaussianBlur(xgg, (0, 0), 1, borderType=cv2.BORDER_REPLICATE)
    ygg = cv2.GaussianBlur(ygg, (0, 0), 1, borderType=cv2.BORDER_REPLICATE)
    # add buffer to outline of displacement map to avoid discontinuity
    xgg = np.pad(xgg, ((1, 1), (1, 1)), mode="edge")
    ygg = np.pad(ygg, ((1, 1), (1, 1)), mode="edge")
    x = np.concatenate(([1], np.unique(x) - bf, [szim[1]]))
    y = np.concatenate(([1], np.unique(y) - bf, [szim[0]]))
    # get interpolated displacement map
    xq, yq = np.meshgrid(np.arange(szim[1]) + 1, np.arange(szim[0]) + 1)
    xmesh, ymesh = np.meshgrid(x, y)
    points = np.column_stack((xmesh.flatten(), ymesh.flatten()))
    xgq = scipy.interpolate.griddata(points, xgg.flatten(), (xq, yq), method="cubic")
    ygq = scipy.interpolate.griddata(points, ygg.flatten(), (xq, yq), method="cubic")
    return np.stack((xgq, ygq), axis=-1)


def calculate_elastic_registration(
    im_ref: np.ndarray,
    im_moving: np.ndarray,
    mask_ref: np.ndarray,
    mask_moving: np.ndarray,
    tile_size: int,
    n_buffer_pix: int,
    intertile_distance: int,
    cutoff: float = 0.15,
    skipstep: int = 1,
) -> np.ndarray:
    """Iterative calculation of registration translation on small tiles for
    determination of nonlinear alignment of globally aligned images.

    Args:
        im_ref: Reference image.
        im_moving: Moving image.
        mask_ref: Reference mask.
        mask_moving: Moving mask.
        tile_size: Size of tiles for elastic registration.
        n_buffer_pix: number of buffer pixels (border for padding images).
        intertile_distance: Distance between registration points/tiles.
        cutoff: Minimum fraction of tissue in registration ROI.
        skipstep: Step size for the regional window around each tile.

    Returns:
        Displacement map.
    """
    szim = np.array(im_moving.shape)
    m = round(tile_size / 2)
    # pad and blur images and pad masks
    im_moving = im_moving.astype(np.float32)
    im_moving = np.pad(
        im_moving,
        pad_width=n_buffer_pix,
        mode="constant",
        constant_values=scipy.stats.mode(im_moving).mode[0],
    )
    im_moving = cv2.GaussianBlur(im_moving, (0, 0), 3)
    im_ref = im_ref.astype(np.float32)
    im_ref = np.pad(
        im_ref,
        pad_width=n_buffer_pix,
        mode="constant",
        constant_values=scipy.stats.mode(im_ref).mode[0],
    )
    im_ref = cv2.GaussianBlur(im_ref, (0, 0), 3)
    mask_moving = np.pad(
        mask_moving, pad_width=n_buffer_pix, mode="constant", constant_values=0
    )
    mask_ref = np.pad(
        mask_ref, pad_width=n_buffer_pix, mode="constant", constant_values=0
    )
    # make grid for registration points
    # DEBUG: uncomment two lines below for testing without randomization
    # n1 = round(intertile_distance / 2 / 2) + n_buffer_pix + m
    # n2 = round(intertile_distance / 2 / 2) + n_buffer_pix + m
    n1 = random.randint(0, round(intertile_distance / 2)) + n_buffer_pix + m
    n2 = random.randint(0, round(intertile_distance / 2)) + n_buffer_pix + m
    x, y = np.meshgrid(
        np.arange(n1, im_moving.shape[1] - m - n_buffer_pix, intertile_distance),
        np.arange(n2, im_moving.shape[0] - m - n_buffer_pix, intertile_distance),
    )
    x = x.ravel()
    y = y.ravel()
    unique_x_len = len(np.unique(x))
    unique_y_len = len(np.unique(y))
    xgg0 = -5000 * np.ones((unique_y_len, unique_x_len))
    ygg0 = -5000 * np.ones((unique_y_len, unique_x_len))
    # for each window
    for w_i, (x_cent, y_cent) in enumerate(zip(x, y)):
        # get the slice for the indices in the window
        window_slice = np.s_[
            y_cent - m : y_cent + m : skipstep, x_cent - m : x_cent + m : skipstep
        ]
        # check if there is enough tissue in the window
        if np.sum(mask_ref[window_slice]) < cutoff * (tile_size**2) or np.sum(
            mask_moving[window_slice]
        ) < cutoff * (tile_size**2):
            continue
        # calculate registration translation
        displacements_x, displacements_y = _reg_ims_elastic(
            im_ref[window_slice], im_moving[window_slice], 2
        )
        xgg0[w_i // unique_x_len, w_i % unique_x_len] = displacements_x
        ygg0[w_i // unique_x_len, w_i % unique_x_len] = displacements_y
    # smooth registration grid and make interpolated displacement map
    if np.max(szim) > 2000:
        szimout = np.round(szim / 5)
        x = np.round(x / 5)
        y = np.round(y / 5)
        n_buffer_pix = round(n_buffer_pix / 5)
    else:
        szimout = szim
    return _make_final_grids(xgg0, ygg0, n_buffer_pix, x, y, szimout)


def _invert_d(D: np.ndarray) -> np.ndarray:
    """Invert a displacement map by interpolating a downsampled version of it.

    Args:
        D: Displacement map to invert.

    Returns:
        Inverted displacement map.
    """
    # calculate coordinates
    yy, xx = np.meshgrid(np.arange(D.shape[1]), np.arange(D.shape[0]))
    xnew = (xx + D[:, :, 0]).flatten()
    ynew = (yy + D[:, :, 1]).flatten()
    # interpolate D at original position
    D1 = D[:, :, 0].flatten()
    D2 = D[:, :, 1].flatten()
    F1 = scipy.interpolate.LinearNDInterpolator(
        np.column_stack((xnew[::5], ynew[::5])), D1[::5]
    )
    F2 = scipy.interpolate.LinearNDInterpolator(
        np.column_stack((xnew[::5], ynew[::5])), D2[::5]
    )
    yy, xx = np.meshgrid(np.arange(D.shape[1], step=5), np.arange(D.shape[0], step=5))
    D1 = -F1(np.column_stack((xx.flatten(), yy.flatten()))).reshape(xx.shape)
    D2 = -F2(np.column_stack((xx.flatten(), yy.flatten()))).reshape(yy.shape)
    Dnew = np.zeros(D.shape)
    Dnew[:, :, 0] = cv2.resize(
        D1, (D.shape[1], D.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    Dnew[:, :, 1] = cv2.resize(
        D2, (D.shape[1], D.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    return Dnew


def register_coordinates(
    elastic_registration_result_file_path: Union[str, pathlib.Path],
    coordinates: np.ndarray,
    scale: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Apply elastic registration to coordinates.

    Args:
        elastic_registration_result_file_path: Path to the elastic registration result
            file.
        coordinates: Coordinates (x, y) to register.
        scale: Scale factor from the coordinates to the image. For example, if the
            coordinates are drawn on a version of the image downsampled by a factor of
            5, the "scale" here should be 5.
        output_size: Size (height, width) of the full scale image.

    Returns:
        Registered coordinates.

    Raises:
        FileNotFoundError: If the elastic registration result file is not found
    """
    # Load the displacement map from the file
    if isinstance(elastic_registration_result_file_path, str):
        elastic_registration_result_file_path = pathlib.Path(
            elastic_registration_result_file_path
        )
    if not elastic_registration_result_file_path.is_file():
        raise FileNotFoundError(
            f"Elastic registration result file {elastic_registration_result_file_path} "
            "not found."
        )
    data = scipy.io.loadmat(elastic_registration_result_file_path)
    D = data["D"]
    # Invert and resize the displacement map
    D2 = _invert_d(D)
    D2a = D2[:, :, 0].flatten()
    D2b = D2[:, :, 1].flatten()
    # scale the original coordinates and apply the registration
    scaled_coordinates = coordinates * scale
    pp = np.round(scaled_coordinates).astype(int)
    ii = np.ravel_multi_index((pp[:, 1], pp[:, 0]), output_size)
    xmove = np.column_stack((D2a[ii], D2b[ii]))
    xye = (scaled_coordinates + xmove) / scale
    return xye
