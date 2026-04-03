" Functions used in preprocessing images for registration. "

# imports
from typing import Tuple, List, Union
import pathlib
from PIL import Image
import numpy as np
import scipy
import cv2  # pylint: disable=import-error

# constants
MASK_DIR_NAME = "TA"  # name of folder holding tissue area masks

# pylint: disable=no-member


def get_ims(
    dirpath: pathlib.Path, name_stem: str, suffix: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads RGB histological image and tissue area mask from existing files.

    Args:
        dirpath: Path to directory containing image and mask files.
        name_stem: Name stem (no suffix) of image and mask files.
        suffix: Suffix of image and mask files.

    Returns:
        Tuple containing RGB histological image (HxWx3) and tissue area mask (HxW) numpy
            arrays.

    Raises:
        FileNotFoundError: If image or mask file is not found.
    """
    im_path = dirpath / f"{name_stem}{suffix}"
    if not im_path.is_file():
        raise FileNotFoundError(f"Image file not found: {im_path}")
    mask_path = dirpath / MASK_DIR_NAME / f"{name_stem}{suffix}"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    im_array = np.array(Image.open(im_path))
    mask = np.array(Image.open(mask_path))
    mask = (mask > 0).astype(np.uint8)
    return im_array, mask


def _pad_image(
    im_array: np.ndarray,
    pad_to: List[int],
    border_pad: int,
    fillval: Union[np.ndarray, int],
) -> np.ndarray:
    """Pad image to a particular size plus a border, filling in with given values.

    This function was originally called "pad_im_both2" in the MATLAB code.

    Args:
        im_array: Image to pad (HxW or HxWxC).
        pad_to: Size (height, width) to pad image to (before added border).
        border_pad: Amount of border to add in addition to padding to max image size.
        fillval: Value (single integer or per-channel) to fill in padded areas.

    Returns:
        Padded image.

    Raises:
        ValueError: If image is not 2D or 3D.
    """
    # Calculate padding amounts
    to_add = np.array([pad_to[0] - im_array.shape[0], pad_to[1] - im_array.shape[1]])
    left_top_pad = to_add // 2 + border_pad
    right_bot_pad = to_add - to_add // 2 + border_pad
    padded_height_width = (
        (left_top_pad[0], right_bot_pad[0]),
        (left_top_pad[1], right_bot_pad[1]),
    )
    # Pad the image
    if im_array.ndim == 2:
        padded_im = np.pad(
            im_array, padded_height_width, mode="constant", constant_values=fillval
        )
    elif im_array.ndim == 3:
        padded_im = np.stack(
            [
                np.pad(
                    im_array[:, :, i],
                    padded_height_width,
                    mode="constant",
                    constant_values=(
                        fillval
                        if isinstance(fillval, (int, np.integer))
                        else fillval[i]
                    ),
                )
                for i in range(im_array.shape[-1])
            ],
            axis=-1,
        )
    else:
        raise ValueError(f"Image must be 2D or 3D (im_array.ndim = {im_array.ndim})")
    return padded_im


def preprocessing(
    im_array: np.ndarray, mask: np.ndarray, pad_to: List[int], border_pad: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get preprocessed image and tissue mask for registration, plus denoised grayscale
    image and channel fill values.

    Pads image and tissue mask to a maximum size. Removes non-tissue area from image,
    complements, and applies a Gaussian filter for denoising.

    Args:
        im_array: RGB histological image (HxWx3).
        mask: Tissue area mask (HxW).
        pad_to: Size (height, width) to pad image.
        border_pad: Amount of border to add in addition to padding to max image size.

    Returns:
        Tuple containing padded RGB image, denoised grayscale image, padded tissue mask,
            and fill values (mode of each color channel of input image).
    """
    # get mode of each color channel
    fillvals, _ = scipy.stats.mode(im_array, axis=(1, 0))
    # pad image and tissue mask
    im_array = _pad_image(im_array, pad_to, border_pad, fillvals)
    if mask.shape != im_array.shape[:-1]:
        mask = _pad_image(mask, pad_to, border_pad, 0)
    # remove nontissue area, convert to grayscale, and complement image
    impg = np.copy(im_array)
    impg[mask == 0] = 255
    impg = 255 - cv2.cvtColor(impg, cv2.COLOR_RGB2GRAY)
    # apply gaussian filter
    impg = cv2.GaussianBlur(impg, (0, 0), sigmaX=2, sigmaY=2)
    return im_array, impg, mask, fillvals
