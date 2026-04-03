# imports
from typing import Optional, List
import concurrent.futures
from argparse import ArgumentParser
import pathlib
from datetime import datetime
from PIL import Image
import numpy as np
import scipy
import scipy.io
import cv2
from .containers import (
    ElasticRegistrationSettings,
    RegistrationReference,
    GlobalRegistrationResult,
)
from .preprocessing_functions import get_ims, preprocessing
from .global_registration import register_global_im, calculate_global_reg
from .elastic_registration import calculate_elastic_registration

# pylint: disable=no-member

# constants
DEF_PAD_ALL = 250
DEF_HE_RESCALE = 6
DEF_IHC_RESCALE = 2
DEF_MAX_ITERS = 5
DEF_ELASTIC_REG_SETTINGS = ElasticRegistrationSettings()
OUT_TYPE = "jpg"  # Output image type (can be 'jpg' or 'tif')


def calculate_image_registration(
    image_dir: pathlib.Path,
    includes_ihc: bool = False,
    center_ref_index: Optional[int] = None,
    pad_all: int = DEF_PAD_ALL,
    rescale: Optional[int] = None,
    max_iters: int = DEF_MAX_ITERS,
    elastic_reg_settings: Optional[ElasticRegistrationSettings] = None,
):
    """
    Nonlinear registration of a series of 2D tumor sections cut along the z axis.

    Images will be warped into near-alignment with a rigid global registration and
    elastic registration of subimage tiles. Creates a folder structure to save
    registered images and registration metadata.

    Output:
    - A folder "registered" inside "image_dir" containing the globally registered images
        in jpg format and the "elastic_registration" folder
    - A folder "elastic_registration" inside "registered" containing the elastically
        registered images in jpg format, the "save_warps" folder, and the "check" folder
    - The "check" folder inside "elastic_registration" containing lower-resolution 
        elastically registered images in jpg format for visual inspection
    - A folder "save_warps" inside "elastic registration" containing the global
        registration transforms and the "D" folder
    - A folder "D" inside "save_warps" containing the elastic registration transforms

    Usage Notes:
    - If initial results are insufficient and further parameter tuning is needed, first
        delete the "registered" folder that this functions creates inside the folder at
        "image_dir".
    - If elastically-registered images are too jiggly, try reducing "tile_size" and/or
        "intertile_distance".
    - If elastically-registered images are too smeared, try increasing "tile_size"
        and/or "intertile_distance".
    - If registration is taking too long for one image (>5 min), try reducing the
        resolution of the images and/or running on a system with more RAM available.

    Args:
        image_dir: path to folder containing tif or jpg images to register.
        includes_ihc: True if imagestack contains IHC images, False if stack contains
            only H&E images.
        center_ref_index: index of reference image for registration. Default is the
            center index.
        pad_all: padding around all images for global registration.
        rescale: simple whole slide image scale factor for global registration. Separate
            default values for IHC and H&E images.
        max_iters: max number of iterations for global registration calculation.
        elastic_reg_settings: Settings for elastic registration routine.

    Raises:
        ValueError: if less than two tif or jpg images are found in "image_dir".
    """
    # assign default settings
    if rescale is None:
        rescale = DEF_IHC_RESCALE if includes_ihc else DEF_HE_RESCALE
    if elastic_reg_settings is None:
        elastic_reg_settings = DEF_ELASTIC_REG_SETTINGS
    # get list of images
    for sfx in (".tif", ".jpg"):
        image_path_list = sorted(list(image_dir.glob(f"*{sfx}")), key=lambda x: x.name)
        if len(image_path_list) > 0:
            image_suffix = sfx
            break
    if len(image_path_list) == 0:
        raise ValueError(f"No tif or jpg images found in {image_dir}!")
    if len(image_path_list) < 2:
        raise ValueError(
            (
                f"Only {len(image_path_list)} images found in {image_dir}! "
                "At least 2 images are required for registration."
            )
        )
    # calculate center image and order
    if center_ref_index is None:
        center_ref_index = ((len(image_path_list) + 1) // 2) - 1
    ref_indices = list(range(center_ref_index, 0, -1)) + list(
        range(center_ref_index, len(image_path_list) - 1)
    )
    moving_indices = list(range(center_ref_index - 1, -1, -1)) + list(
        range(center_ref_index + 1, len(image_path_list))
    )
    # find max size (height, width) of images in list
    max_size = [0, 0]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        widths, heights = zip(
            *executor.map(
                lambda im_path: Image.open(im_path).size,
                image_path_list,
            )
        )
        max_size[0] = max(heights)
        max_size[1] = max(widths)
    # define outputs
    registration_dir = image_dir / "registered"
    elastic_registration_dir = registration_dir / "elastic registration"
    elastic_check_dir = elastic_registration_dir / "check"
    save_warps_dir = elastic_registration_dir / "save_warps"
    elastic_metadata_dir = save_warps_dir / "D"
    for p in (
        registration_dir,
        elastic_registration_dir,
        elastic_check_dir,
        save_warps_dir,
        elastic_metadata_dir,
    ):
        p.mkdir(exist_ok=True)
    # set up center (reference) image
    global_ref_nm = image_path_list[center_ref_index].stem
    global_ref_im, global_ref_mask = get_ims(image_dir, global_ref_nm, image_suffix)
    global_ref_im, global_ref_im_g, global_ref_mask, _ = preprocessing(
        global_ref_im, global_ref_mask, max_size, pad_all
    )
    print(f"Global reference image: {global_ref_nm}")
    # save reference image outputs
    global_ref_im_pil = Image.fromarray(global_ref_im)
    global_ref_im_pil.save(registration_dir / f"{global_ref_nm}.{OUT_TYPE}")
    global_ref_im_pil.save(elastic_registration_dir / f"{global_ref_nm}.{OUT_TYPE}")
    scipy.io.savemat(
        save_warps_dir / f"{global_ref_nm}.mat", {"center_ref_index": center_ref_index}
    )
    refs = {
        "best": RegistrationReference(global_ref_im_g.copy(), global_ref_mask.copy()),
        "0": RegistrationReference(global_ref_im_g.copy(), global_ref_mask.copy()),
        "00": RegistrationReference(global_ref_im_g.copy(), global_ref_mask.copy()),
        "a": RegistrationReference(global_ref_im_g.copy(), global_ref_mask.copy()),
        "b": RegistrationReference(global_ref_im_g.copy(), global_ref_mask.copy()),
        "c": RegistrationReference(global_ref_im_g.copy(), global_ref_mask.copy()),
    }
    for kk, (ref_index, moving_index) in enumerate(zip(ref_indices, moving_indices)):
        start_time = datetime.now()
        print(
            f"Image {kk+1} of {len(image_path_list)-1}\n"
            f"\treference image: {image_path_list[ref_index].stem}\n"
            f"\tmoving image: {image_path_list[moving_index].stem}"
        )
        # reset reference images when at center
        if ref_index == center_ref_index:
            refs["a"] = RegistrationReference(
                global_ref_im_g.copy(), global_ref_mask.copy()
            )
            refs["b"] = refs["0"]
            refs["c"] = refs["00"]
        # create moving image
        moving_im_name_stem = image_path_list[moving_index].stem
        ref_im_name_stem = image_path_list[ref_index].stem
        moving_im_0, moving_mask = get_ims(image_dir, moving_im_name_stem, image_suffix)
        moving_im, moving_im_g, moving_mask, fillvals = preprocessing(
            moving_im_0, moving_mask, max_size, pad_all
        )
        # load or calculate registration
        elastic_transform_file_path = (
            elastic_metadata_dir / f"{moving_im_name_stem}.mat"
        )
        if elastic_transform_file_path.is_file():
            # load and apply global registration from file
            print("\tRegistration already calculated\n")
            data = scipy.io.loadmat(save_warps_dir / f"{moving_im_name_stem}.mat")
            global_transform = np.squeeze(data["global_transform"])
            flipped = np.squeeze(data["flipped"]) == 1
            moving_im_global_g = register_global_im(
                moving_im_g,
                global_transform,
                flipped,
                scipy.stats.mode(moving_im_g).mode[0],
            )
            moving_mask_global = register_global_im(
                moving_mask, global_transform, flipped, 0
            )
        else:
            ct = 0.8 if includes_ihc else 0.945
            # calculate global registration
            global_regs = {
                "a": {"metric_value": 0.4},
                "b": {"metric_value": 0.4},
                "c": {"metric_value": 0.4},
            }
            # try with registration pairs 1
            tform, flipped, metric_r = calculate_global_reg(
                refs["a"].image, moving_im_g, rescale, max_iters, includes_ihc
            )
            global_regs["a"]["result"] = GlobalRegistrationResult(tform, flipped)
            global_regs["a"]["metric_value"] = metric_r
            # try with registration pairs 2
            if global_regs["a"]["metric_value"] < ct:
                tform, flipped, metric_r = calculate_global_reg(
                    refs["b"].image, moving_im_g, rescale, max_iters, includes_ihc
                )
                global_regs["b"]["result"] = GlobalRegistrationResult(tform, flipped)
                global_regs["b"]["metric_value"] = metric_r
                print("RB")
            if (
                global_regs["a"]["metric_value"] < ct
                and global_regs["b"]["metric_value"] < ct
            ):
                tform, flipped, metric_r = calculate_global_reg(
                    refs["c"].image, moving_im_g, rescale, max_iters, includes_ihc
                )
                global_regs["c"]["result"] = GlobalRegistrationResult(tform, flipped)
                global_regs["c"]["metric_value"] = metric_r
                print("RC")
            # use the best of three global registrations
            max_metric_key = max(
                global_regs, key=lambda k: global_regs[k]["metric_value"]
            )
            print(f"Chose image {max_metric_key.upper()}")
            refs["best"] = refs[max_metric_key]
            best_global_reg_result = global_regs[max_metric_key]["result"]
            global_transform = best_global_reg_result.transformation
            flipped = best_global_reg_result.flipped
            moving_im_global_g = register_global_im(
                moving_im_g, global_transform, flipped, 0
            )
            # save global registration data
            scipy.io.savemat(
                save_warps_dir / f"{moving_im_name_stem}.mat",
                {
                    "global_transform": global_transform,
                    "flipped": flipped,
                    "max_size": max_size,
                    "pad_all": pad_all,
                    "ref_index": ref_index,
                },
            )
            moving_im_global = register_global_im(
                moving_im, global_transform, flipped, fillvals
            )
            moving_mask_global = register_global_im(
                moving_mask, global_transform, flipped, 0
            )
            # write out the "moving_im_global" image
            moving_im_global_pil = Image.fromarray(moving_im_global)
            moving_im_global_pil.save(
                registration_dir / f"{moving_im_name_stem}.{OUT_TYPE}"
            )
            print("elastic")
            # elastic registration: load or calculate Dmv and D
            moving_elastic_metadata_file = (
                elastic_metadata_dir / f"{moving_im_name_stem}.mat"
            )
            ref_elastic_metadata_file = elastic_metadata_dir / f"{ref_im_name_stem}.mat"
            if moving_elastic_metadata_file.is_file():
                data = scipy.io.loadmat(moving_elastic_metadata_file)
                Dmv = data["Dmv"]
            else:
                Dmv = calculate_elastic_registration(
                    refs["best"].image,
                    moving_im_global_g,
                    refs["best"].mask,
                    moving_mask_global,
                    elastic_reg_settings.tile_size,
                    elastic_reg_settings.n_buffer_pixels,
                    elastic_reg_settings.intertile_distance,
                )
                if kk == 0:
                    D = np.zeros_like(Dmv)
                    scipy.io.savemat(ref_elastic_metadata_file, {"D": D})
            data = scipy.io.loadmat(ref_elastic_metadata_file)
            D = data["D"]
            D += Dmv
            scipy.io.savemat(ref_elastic_metadata_file, {"D": D})
            scipy.io.savemat(moving_elastic_metadata_file, {"D": D, "Dmv": Dmv})
            D = cv2.resize(
                D,
                (moving_im_global.shape[1], moving_im_global.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            D = D.astype(np.float32)
            # Create the base coordinate grid
            base_x, base_y = np.meshgrid(
                np.arange(moving_im_global.shape[1]),
                np.arange(moving_im_global.shape[0]),
            )
            # Convert the displacement map to absolute coordinates
            map_x = (base_x + D[..., 0]).astype(np.float32)
            map_y = (base_y + D[..., 1]).astype(np.float32)
            remapped_channels = [
                cv2.remap(
                    channel,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=float(fillvals[i]),
                )
                for i, channel in enumerate(cv2.split(moving_im_global))
            ]
            moving_im_elastic = cv2.merge(remapped_channels)
            # save elastic registration data
            moving_im_elastic_pil = Image.fromarray(moving_im_elastic)
            moving_im_elastic_pil.save(
                elastic_registration_dir / f"{moving_im_name_stem}.{OUT_TYPE}"
            )
            downsampled_moving_im_elastic_pil = Image.fromarray(
                moving_im_elastic[::3, ::3]
            )
            downsampled_moving_im_elastic_pil.save(
                elastic_check_dir / f"{moving_im_name_stem}.{OUT_TYPE}"
            )
        # reset reference images
        refs["c"] = refs["b"]
        refs["b"] = refs["a"]
        refs["a"] = RegistrationReference(
            moving_im_global_g.copy(), moving_mask_global.copy()
        )
        if kk == 0:
            refs["0"] = RegistrationReference(
                moving_im_global_g.copy(), moving_mask_global.copy()
            )
        if kk == 1:
            refs["00"] = RegistrationReference(
                moving_im_global_g.copy(), moving_mask_global.copy()
            )
        print(f"Elapsed time is {(datetime.now()-start_time).total_seconds()} seconds.")


def main(args: Optional[List[str]]=None):
    parser = ArgumentParser()
    parser.add_argument(
        "image_dir",
        type=pathlib.Path,
        help="Path to folder containing images to register",
    )
    parser.add_argument(
        "--includes_ihc",
        action="store_true",
        help="Add this flag if imagestack contains IHC images",
    )
    parser.add_argument(
        "--center_ref_index",
        type=int,
        help="Index of center image (automatically calculated if not provided)",
    )
    global_group = parser.add_argument_group("global registration")
    global_group.add_argument(
        "--pad_all",
        type=int,
        default=DEF_PAD_ALL,
        help=(
            "Padding around all images for global registration "
            f"(default = {DEF_PAD_ALL})"
        ),
    )
    global_group.add_argument(
        "--rescale",
        type=int,
        help=(
            "Simple whole slide image scale factor for global registration "
            f"(default = {DEF_HE_RESCALE} for H&E, {DEF_IHC_RESCALE} for IHC)"
        ),
    )
    global_group.add_argument(
        "--max_iters",
        type=int,
        default=DEF_MAX_ITERS,
        help=(
            "Max iterations of registration calculation for global registration "
            f"(default = {DEF_MAX_ITERS})"
        ),
    )
    elastic_group = parser.add_argument_group("elastic registration")
    elastic_group.add_argument(
        "--tile_size",
        type=int,
        default=DEF_ELASTIC_REG_SETTINGS.tile_size,
        help=(
            "Size of each tile for elastic registration "
            f"(default = {DEF_ELASTIC_REG_SETTINGS.tile_size})"
        ),
    )
    elastic_group.add_argument(
        "--intertile_distance",
        type=int,
        default=DEF_ELASTIC_REG_SETTINGS.intertile_distance,
        help=(
            "Distance between tiles for elastic registration "
            f"(default = {DEF_ELASTIC_REG_SETTINGS.intertile_distance})"
        ),
    )
    elastic_group.add_argument(
        "--n_buffer_pixels",
        type=int,
        default=DEF_ELASTIC_REG_SETTINGS.n_buffer_pixels,
        help=(
            "Number of buffer pixels for elastic registration "
            f"(default = {DEF_ELASTIC_REG_SETTINGS.n_buffer_pixels})"
        ),
    )
    parsed_args = parser.parse_args(args)
    elastic_reg_settings = ElasticRegistrationSettings(
        tile_size=parsed_args.tile_size,
        intertile_distance=parsed_args.intertile_distance,
        n_buffer_pixels=parsed_args.n_buffer_pixels,
    )
    calculate_image_registration(
        parsed_args.image_dir,
        parsed_args.includes_ihc,
        parsed_args.center_ref_index,
        parsed_args.pad_all,
        parsed_args.rescale,
        parsed_args.max_iters,
        elastic_reg_settings,
    )


if __name__ == "__main__":
    main()
