import matplotlib.pyplot as plt
import numpy as np
from stardist.models import StarDist2D, Config2D
import json
import geojson
from typing import List, Tuple
from pathlib import Path
import os
from tifffile import imread
from tqdm import tqdm
import random
from PIL import Image
from stardist import fill_label_holes
import copy
from tensorflow.python.summary.summary_iterator import summary_iterator
import struct
from matplotlib.colors import ListedColormap
import pandas as pd
import cv2
from pathlib import Path
import glob
from PIL import Image
import h5py
import matplotlib.pyplot as plt
import pickle
from scipy.io import savemat, loadmat
import mat73
import re
import time
################ CHANGE THIS TO YOUR LOCAL FOLDER #############
# openslide_path = r'C:\Users\labadmin\Documents\openslide-win64-20230414\bin'
openslide_path = r'C:\Users\Andre\Documents\openslide-win64-20230414\bin'
# ###############################################################
os.environ['PATH'] = openslide_path + ";" + os.environ['PATH']
# # from openslide import OpenSlide
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(openslide_path):
        import openslide
else:
    import openslide
# from openslide import OpenSlide
# import openslide


def extract_and_save_pixel_sizes(pth_WSI_ndpi, output_folder):
    """
    Extracts pixel sizes from NDPI files and saves them in .mat files.

    Args:
    - pth_WSI_ndpi (str): Path to the directory containing NDPI files.
    - output_folder (str): Path to the directory where .mat files will be saved.

    Returns:
    - None
    """
    ndpi_files = sorted([file for file in os.listdir(pth_WSI_ndpi) if (file.endswith('.ndpi') or file.endswith('.svs'))])

    for filename in ndpi_files:
        path = os.path.join(pth_WSI_ndpi, filename)
        print(f"Processing file: {filename}")
        mat_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.mat")
        # Check if the file already exists
        if os.path.exists(mat_path):
            print("Already exists. Skipping...")
            continue
        slide = openslide.OpenSlide(path)

        pix_res = {
            'x': slide.properties['openslide.mpp-x'],
            'y': slide.properties['openslide.mpp-y']
        }

        print(f"Pixel resolution: {pix_res}")
        savemat(mat_path, {'pix_res': pix_res})

# def get_sorted_files(directory, extension, filter_str=None):
#     return sorted(
#         [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension) and (filter_str in f if filter_str else True)]
#     )

def get_sorted_files(directory, *extensions, filter_str=None):
    """
    Returns a sorted list of file paths from a directory that match the specified extensions and optional filter string.

    Args:
    - directory (str): The directory to search for files.
    - *extensions (str): File extensions to filter by (e.g., '.ndpi', '.svs'). Can pass one or more extensions.
    - filter_str (str, optional): A string to filter the file names. Only files containing this string will be included.

    Returns:
    - List[str]: Sorted list of file paths matching the criteria.
    """
    return sorted(
        [os.path.join(directory, f) for f in os.listdir(directory)
         if f.endswith(extensions) and (filter_str in f if filter_str else True)]
    )
#
# def check_file_alignment(jsons: List[str], WSIs: List[str], pixel_res_files: List[str]) -> bool:
#     """
#     Ensure that each file in the lists matches with each other based on the file name without extensions.
#
#     Parameters:
#     - jsons: List of paths to JSON files.
#     - WSIs: List of paths to WSI files.
#     - pixel_res_files: List of paths to pixel resolution files.
#
#     Returns:
#     - True if all files match, False if a mismatch is found.
#     """
#     for i, json_f_name in enumerate(jsons):
#         # Extract the base names (without extensions) for comparison
#         json_name = os.path.splitext(os.path.basename(json_f_name))[0]
#         wsi_name = os.path.splitext(os.path.basename(WSIs[i]))[0]
#         pixel_res_name = os.path.splitext(os.path.basename(pixel_res_files[i]))[0]
#
#         # Check if the names match, and print details if they don't
#         if json_name != wsi_name or json_name != pixel_res_name:
#             print(f"Mismatch at index {i}:")
#             print(f"JSON: {json_name}, WSI: {wsi_name}, Pixel Res: {pixel_res_name}")
#             return False
#
#     return True


def check_file_alignment(jsons: List[str], WSIs: List[str], pixel_res_files: List[str]) -> bool:
    """
    Ensure that each file in the lists matches with each other based on the file name without extensions.

    Parameters:
    - jsons: List of paths to JSON files.
    - WSIs: List of paths to WSI files.
    - pixel_res_files: List of paths to pixel resolution files.

    Returns:
    - True if all files match, False if a mismatch is found.
    """
    # Print total number of files in each list
    print(f"Total WSI files: {len(WSIs)}")
    print(f"Total JSON files: {len(jsons)}")
    print(f"Total Pixel Resolution files: {len(pixel_res_files)}")

    # Convert lists to sets of base filenames
    wsi_names = set(os.path.splitext(os.path.basename(f))[0] for f in WSIs)
    json_names = set(os.path.splitext(os.path.basename(f))[0] for f in jsons)
    pixel_res_names = set(os.path.splitext(os.path.basename(f))[0] for f in pixel_res_files)

    # Find WSIs that were not converted to JSON
    wsi_not_converted_to_json = wsi_names - json_names
    if wsi_not_converted_to_json:
        print(f"\nWSI files not converted to JSON:")
        for name in wsi_not_converted_to_json:
            print(f"WSI: {name}")

    # Find WSIs that were not converted to pixel resolution files
    wsi_not_converted_to_pixel_res = wsi_names - pixel_res_names
    if wsi_not_converted_to_pixel_res:
        print(f"\nWSI files not having corresponding pixel resolution files:")
        for name in wsi_not_converted_to_pixel_res:
            print(f"WSI: {name}")

    # Check if all files match
    all_files_match = True

    # Compare JSON files to corresponding WSI and pixel resolution files
    for json_f_name in jsons:
        json_name = os.path.splitext(os.path.basename(json_f_name))[0]

        if json_name not in wsi_names:
            print(f"\nJSON file {json_name} does not have a corresponding WSI.")
            all_files_match = False

        if json_name not in pixel_res_names:
            print(f"JSON file {json_name} does not have a corresponding pixel resolution file.")
            all_files_match = False

    return all_files_match

# Basic morphology features
def cntarea(cnt: np.ndarray) -> float:
    """Calculate the area of a contour."""
    cnt = np.array(cnt).astype(np.float32)
    area = cv2.contourArea(cnt)
    return area

def cntperi(cnt: np.ndarray) -> float:
    """Calculate the perimeter of a contour."""
    cnt = np.array(cnt).astype(np.float32)
    perimeter = cv2.arcLength(cnt, True)
    return perimeter

def cntMA(cnt: np.ndarray) -> List[float]:
    """Calculate the major axis, minor axis, and orientation of a contour."""
    cnt = np.array(cnt).astype(np.float32)
    [(x, y), (MA, ma), orientation] = cv2.fitEllipse(cnt)
    return [np.max((MA, ma)), np.min((MA, ma)), orientation]

# Functions to adjust format of contours list

def fix_contours(contours: List[np.ndarray]) -> np.ndarray:
    """Adjust the format of a list of contours."""
    contours_fixed = []
    for polygon in contours:
        coords = np.array([list(zip(x, y)) for x, y in [polygon[0]]][0], dtype=np.int32)
        contours_fixed.append(coords)
    contours_fixed = np.array(contours_fixed)
    return contours_fixed

def adjust_contours(contour: np.ndarray, crop_x: int, crop_y: int) -> np.ndarray:
    """Adjust a contour based on cropping coordinates."""
    for i, xy in enumerate(contour):
        x = xy[0] - crop_x
        y = xy[1] - crop_y
        contour[i] = [x, y]
    return contour


def get_rbg_avg(centroid, contour_raw, offset, HE_20x_WSI):
    """gets RBG average intensities inside of a contour given the image and centroid
    It is fast because it crops the image so that the image size is offset*2 width/height.
    Python passes HE_20x_WSI as reference so it shouldn't affect performance passing a
    hugh variable like this."""

    x_low = centroid[0] - offset
    x_high = centroid[0] + offset
    y_low = centroid[1] - offset
    y_high = centroid[1] + offset

    img_shape = HE_20x_WSI.shape

    # if bad shape, return -1 for each intensity mean
    if offset > centroid[0] or offset > centroid[1] or centroid[0] > (img_shape[0] - offset) or centroid[1] > (
            img_shape[1] - offset):
        # print(f'centroid passed: {centroid}')
        r_avg = -1
        g_avg = -1
        b_avg = -1
        return r_avg, g_avg, b_avg

    im_crop = np.array(HE_20x_WSI[x_low:x_high, y_low:y_high], dtype=np.uint16)

    # plt.imshow(im_crop)

    crop_x = centroid[0] - offset - 1
    crop_y = centroid[1] - offset - 1

    contour_adj = adjust_contours(contour_raw, crop_x, crop_y)
    contour_new = contour_adj  # .reshape((-1,1,2)).astype(np.uint16)
    rev_contour = contour_new[:, [1, 0]]  # its backwards for some reason idk why but you need to flip it like this
    # rev_contour = contour_new[:,:, [1, 0]]  # its backwards for some reason idk why but you need to flip it like this
    # print(rev_contour)

    # coords NEEDS to be np.int32 matrix --> 2 columns x y

    # Create a single-channel mask
    mask = np.zeros_like(im_crop[:, :, 0], dtype=np.uint16)  # make black image of same size, will fill with mask

    # Draw contours on the single-channel mask
    # cv2.drawContours(im_crop, [rev_contour], -1, (0,255,0)) #, thickness=cv2.FILLED)  # this one makes it green so that you can see contour
    cv2.drawContours(mask, [rev_contour], 0, (1), thickness=cv2.FILLED)

    # plt.imshow(im_crop)

    r_pixels = im_crop[:, :, 0] * mask  # pixels inside mask are 1, outside == 0
    g_pixels = im_crop[:, :, 1] * mask
    b_pixels = im_crop[:, :, 2] * mask

    num_pixels = np.count_nonzero(mask)

    if num_pixels != 0:

        r_avg = round(np.sum(r_pixels) / num_pixels, 2)
        g_avg = round(np.sum(g_pixels) / num_pixels, 2)
        b_avg = round(np.sum(b_pixels) / num_pixels, 2)

    else:
        print('ZERO PIXEL')
        r_avg = -1
        g_avg = -1
        b_avg = -1

    # plt.imshow(im_crop)

    return r_avg, g_avg, b_avg


def extract_slide_number(nm):
    """
    Extracts the slide number from the filename.
    - If the slide number has a leading underscore, it removes it.
    - If there are additional suffixes after the slide number, it ignores them.
    - Otherwise, it returns the slide number as-is.
    """
    # Match the numeric sequence before any non-numeric suffix following the last underscore
    match = re.search(r'_(\d+)(?=\D*$)', nm)
    if match:
        return match.group(1)  # Return only the numeric part after the last underscore

    # If there's no match with the above pattern, check if the filename ends in a numeric sequence without underscore
    match = re.search(r'(\d+)$', nm)
    if match:
        return match.group(1)  # Return the numeric part at the end

    # Return None or raise an exception if no valid number is found
    return None

def make_features_df_pkl_from_contours_jsons(jsons, WSIs, outpth, inverse=0):
    """
    Processes a list of JSON files, extracts data, and saves the results in pickle files.

    Args:
    - jsons (list): List of JSON file paths.
    - WSIs (list): List of WSI image paths corresponding to the JSON files.
    - outpth (str): Output directory where pickle files will be saved.

    Returns:
    - None
    """
    total_images = len(jsons)
    # Determine the iteration order based on the `inverse` parameter
    if inverse == 1:
        jsons = reversed(jsons)
        start_idx = total_images
        step = -1
    else:
        start_idx = 1
        step = 1
    for i, json_f_name in enumerate(jsons, start=start_idx):
    # for i, json_f_name in enumerate(jsons):

        nm = os.path.splitext(os.path.basename(json_f_name))[0]
        outnm = os.path.join(outpth, f'{nm}.pkl')
        print(f'{nm}  {i+ 1}/{len(jsons)}')

        if not os.path.exists(outnm):
            HE_20x_WSI = imread(WSIs[i])
            print(WSIs[i])
            print(json_f_name)

            try:
                segmentation_data = json.load(open(json_f_name))
            except Exception as e:
                print(f'Error reading JSON: {e}. Skipping {json_f_name}')
                continue

            centroids = [nuc['centroid'][0] for nuc in segmentation_data]
            contours = [nuc['contour'] for nuc in segmentation_data]
            contours_fixed = fix_contours(contours)

            offset = 30  # Radius of image cropped from WSI for RGB intensity

            # r_avg_list, g_avg_list, b_avg_list = [], [], []
            areas, perimeters, circularities, aspect_ratios = [], [], [], []
            compactness_a, eccentricity_a, extent_a, form_factor_a = [], [], [], []
            maximum_radius_a, mean_radius_a, median_radius_a = [], [], []
            minor_axis_length_a, major_axis_length_a, orientation_degrees_a = [], [], []

            np_centroids = np.array(centroids)
            contours_np = np.array(contours)

            for j in range(len(contours_fixed)):
                centroid = centroids[j]
                contour_raw = copy.copy(contours_fixed[j])

                # # Get RGB intensity averages
                # r_avg, g_avg, b_avg = get_rbg_avg(centroid, contour_raw, offset, HE_20x_WSI)
                # r_avg_list.append(r_avg)
                # g_avg_list.append(g_avg)
                # b_avg_list.append(b_avg)

                contour = contours_np[j][0].transpose()
                area = cntarea(contour)
                perimeter = cntperi(contour)
                circularity = 4 * np.pi * area / perimeter ** 2
                MA = cntMA(contour)
                MA, ma, orientation = MA
                aspect_ratio = MA / ma

                compactness = perimeter ** 2 / area
                eccentricity = np.sqrt(1 - (ma / MA) ** 2)
                extent = area / (MA * ma)
                form_factor = (perimeter ** 2) / (4 * np.pi * area)
                major_axis_length = MA
                maximum_radius = np.max(np.linalg.norm(contour - centroid, axis=1))
                mean_radius = np.mean(np.linalg.norm(contour - centroid, axis=1))
                median_radius = np.median(np.linalg.norm(contour - centroid, axis=1))
                minor_axis_length = ma
                orientation_degrees = np.degrees(orientation)

                areas.append(area)
                perimeters.append(perimeter)
                circularities.append(circularity)
                aspect_ratios.append(aspect_ratio)
                compactness_a.append(compactness)
                eccentricity_a.append(eccentricity)
                extent_a.append(extent)
                form_factor_a.append(form_factor)
                maximum_radius_a.append(maximum_radius)
                mean_radius_a.append(mean_radius)
                median_radius_a.append(median_radius)
                minor_axis_length_a.append(minor_axis_length)
                major_axis_length_a.append(major_axis_length)
                orientation_degrees_a.append(orientation_degrees)

            dat = {
                'Centroid_x': np_centroids[:, 1],
                'Centroid_y': np_centroids[:, 0],
                'Area': areas,
                'Perimeter': perimeters,
                'Circularity': circularities,
                'Aspect Ratio': aspect_ratios,
                'compactness': compactness_a,
                'eccentricity': eccentricity_a,
                'extent': extent_a,
                'form_factor': form_factor_a,
                'maximum_radius': maximum_radius_a,
                'mean_radius': mean_radius_a,
                'median_radius': median_radius_a,
                'minor_axis_length': minor_axis_length_a,
                'major_axis_length': major_axis_length_a,
                'orientation_degrees': orientation_degrees_a,
                # 'r_mean_intensity': r_avg_list,
                # 'g_mean_intensity': g_avg_list,
                # 'b_mean_intensity': b_avg_list,
                'slide_num': extract_slide_number(nm)
            }

            df = pd.DataFrame(dat).astype(np.float32)
            df.to_pickle(outnm)
        else:
            print('Already extracted features into pkl from json contours')


def make_RGBfeatures_df_pkl_from_contours_jsons(jsons, WSIs, outpth, inverse=0):
    """
    Processes a list of JSON files, extracts data, and saves the results in pickle files.

    Args:
    - jsons (list): List of JSON file paths.
    - WSIs (list): List of WSI image paths corresponding to the JSON files.
    - outpth (str): Output directory where pickle files will be saved.

    Returns:
    - None
    """

    total_images = len(jsons)
    # total_images = list(reversed(jsons))

    # Determine the iteration order based on the `inverse` parameter
    if inverse == 1:
        # jsons = reversed(jsons)
        jsons = list(reversed(jsons))
        WSIs = list(reversed(WSIs))
        # jsons = reversed(jsons)
        # WSIs = reversed(WSIs)
        # start_idx = total_images-1
        start_idx = 0
        step = -1
    else:
        start_idx = 0
        step = 1

    # make the for loop go in reverse order from start_idx to 0
    for i, json_f_name in enumerate(jsons, start=start_idx):
        # for i, json_f_name in enumerate(jsons):

        nm = os.path.splitext(os.path.basename(json_f_name))[0]
        outnm = os.path.join(outpth, f'{nm}.pkl')
        # ndpinmpth = os.path.join(outpth, f'{nm}.ndpi')
        print(f'{nm}  {i + 1}/{len(jsons)}')
        print(outnm)
        # print(WSIs[i])
        if not os.path.exists(outnm):
            # HE_20x_WSI = imread(WSIs[i])
            # # HE_20x_WSI = imread(ndpinmpth)
            # print(WSIs[i])
            # print(json_f_name)

            try:
                HE_20x_WSI = imread(WSIs[i])
                # HE_20x_WSI = imread(ndpinmpth)
                print(WSIs[i])
                print(json_f_name)
                segmentation_data = json.load(open(json_f_name))
            except Exception as e:
                print(f'Error reading JSON: {e}. Skipping {json_f_name}')
                continue

            centroids = [nuc['centroid'][0] for nuc in segmentation_data]
            contours = [nuc['contour'] for nuc in segmentation_data]
            contours_fixed = fix_contours(contours)

            offset = 30  # Radius of image cropped from WSI for RGB intensity

            r_avg_list, g_avg_list, b_avg_list = [], [], []
            r_std_list, g_std_list, b_std_list = [], [], []
            areas, perimeters, circularities, aspect_ratios = [], [], [], []
            compactness_a, eccentricity_a, extent_a, form_factor_a = [], [], [], []
            maximum_radius_a, mean_radius_a, median_radius_a = [], [], []
            minor_axis_length_a, major_axis_length_a, orientation_degrees_a = [], [], []

            np_centroids = np.array(centroids)
            contours_np = np.array(contours)

            for j in range(len(contours_fixed)):
                centroid = centroids[j]
                contour_raw = copy.copy(contours_fixed[j])

                # Get RGB intensity averages
                # r_avg, g_avg, b_avg = get_rbg_avg(centroid, contour_raw, offset, HE_20x_WSI)
                r_avg, g_avg, b_avg, r_std, g_std, b_std = get_rgb_avg_std(centroid, contour_raw, offset, HE_20x_WSI)

                r_avg_list.append(r_avg)
                g_avg_list.append(g_avg)
                b_avg_list.append(b_avg)
                r_std_list.append(r_std)
                g_std_list.append(g_std)
                b_std_list.append(b_std)

                contour = contours_np[j][0].transpose()
                area = cntarea(contour)
                perimeter = cntperi(contour)
                circularity = 4 * np.pi * area / perimeter ** 2
                MA = cntMA(contour)
                MA, ma, orientation = MA
                aspect_ratio = MA / ma

                compactness = perimeter ** 2 / area
                eccentricity = np.sqrt(1 - (ma / MA) ** 2)
                extent = area / (MA * ma)
                form_factor = (perimeter ** 2) / (4 * np.pi * area)
                major_axis_length = MA
                maximum_radius = np.max(np.linalg.norm(contour - centroid, axis=1))
                mean_radius = np.mean(np.linalg.norm(contour - centroid, axis=1))
                median_radius = np.median(np.linalg.norm(contour - centroid, axis=1))
                minor_axis_length = ma
                orientation_degrees = np.degrees(orientation)

                areas.append(area)
                perimeters.append(perimeter)
                circularities.append(circularity)
                aspect_ratios.append(aspect_ratio)
                compactness_a.append(compactness)
                eccentricity_a.append(eccentricity)
                extent_a.append(extent)
                form_factor_a.append(form_factor)
                maximum_radius_a.append(maximum_radius)
                mean_radius_a.append(mean_radius)
                median_radius_a.append(median_radius)
                minor_axis_length_a.append(minor_axis_length)
                major_axis_length_a.append(major_axis_length)
                orientation_degrees_a.append(orientation_degrees)
            try:
                dat = {
                    'Centroid_x': np_centroids[:, 1],
                    'Centroid_y': np_centroids[:, 0],
                    'Area': areas,
                    'Perimeter': perimeters,
                    'Circularity': circularities,
                    'Aspect Ratio': aspect_ratios,
                    'compactness': compactness_a,
                    'eccentricity': eccentricity_a,
                    'extent': extent_a,
                    'form_factor': form_factor_a,
                    'maximum_radius': maximum_radius_a,
                    'mean_radius': mean_radius_a,
                    'median_radius': median_radius_a,
                    'minor_axis_length': minor_axis_length_a,
                    'major_axis_length': major_axis_length_a,
                    'orientation_degrees': orientation_degrees_a,
                    'r_mean_intensity': r_avg_list,
                    'g_mean_intensity': g_avg_list,
                    'b_mean_intensity': b_avg_list,
                    'r_std': r_std_list,
                    'g_std': g_std_list,
                    'b_std': b_std_list,
                    'slide_num': extract_slide_number(nm)
                }
                df = pd.DataFrame(dat).astype(np.float32)
                df.to_pickle(outnm)
            except:
                print('no cells, maybe slide out of focus, skippinnn...')

        else:
            print('Already extracted features into pkl from json contours')


def featuresdf_pkl2mat(src, outpth, dfs):
    """
    Converts .pkl files containing pandas DataFrames to .mat files.

    Args:
    - src (str): Source directory containing .pkl files.
    - outpth (str): Destination directory where .mat files will be saved.
    - dfs (list): List of .pkl filenames (as strings) to process.

    Returns:
    - None
    """
    print(f'saving here: {outpth}')
    for i,dfnm in enumerate(dfs):
        nm = os.path.splitext(os.path.basename(dfnm))[0]

        print(f"Saving mat file {nm}  {i+1}/{len(dfs)}")
        dst = os.path.join(outpth,nm+ '.mat')

        if os.path.exists(dst):
            print("MAT file already exists, skipping the file ID {}".format(nm))
            continue

        with open(os.path.join(src, dfnm), 'rb') as f:
            df = pickle.load(f)

        # Convert DataFrame to numpy array
        col_names = df.columns.tolist()
        # df_array = df.to_numpy()
        df_array = df.to_numpy().astype(np.float32)
        # print(dst)

        # Save to .mat file
        savemat(dst, {'features': df_array, 'feature_names': col_names})


def featuresdf_pkl2mat_chop(src, outpth, dfs, outpthmatchop, output_pixres, crop_resolution_x, crop_resolution_y, pth_imhe_chop):
    """
    Converts .pkl files containing pandas DataFrames to .mat files, separating cells based on cropping coordinates.

    Args:
    - src (str): Source directory containing .pkl files.
    - outpth (str): Destination directory where .mat files will be saved.
    - dfs (list): List of .pkl filenames (as strings) to process.
    - outpthmatchop (str): Directory containing .mat files with cropping metadata.
    - output_pixres (str): Path to the directory containing pixel resolution files.
    - crop_resolution_x (float): Resolution of the cropping metadata for the x-axis (microns per pixel).
    - crop_resolution_y (float): Resolution of the cropping metadata for the y-axis (microns per pixel).
    - pth_imhe_chop (str): Path to the directory containing chopped HE images.

    Returns:
    - None
    """
    print(f'Saving here: {outpth}')
    total_images = len(dfs)
    target_indices = [0, total_images // 3, total_images // 2, (2 * total_images) // 3]

    for i, dfnm in enumerate(dfs):
        nm = os.path.splitext(os.path.basename(dfnm))[0]  # Get the base name without extension

        print(f"Processing file {nm}  {i+1}/{len(dfs)}")

        # Construct the path to the corresponding .mat file with cropping metadata
        crop_info_matfile = os.path.join(outpthmatchop, f"{nm}.mat")

        # Check if the cropping metadata .mat file exists
        if not os.path.exists(crop_info_matfile):
            print(f"Cropping metadata file {crop_info_matfile} not found. Skipping {dfnm}.")
            continue

        # Load the cropping information from the .mat file
        try:
            print(f"Loading crop info via mat73: {crop_info_matfile}")
            crop_info = mat73.loadmat(crop_info_matfile)
            nms_info = crop_info['nms_info']

        except:
            crop_info = loadmat(crop_info_matfile)
            nms_info = crop_info['nms_info']  # Access the cropping metadata

        # Load the pixel resolution for the current image
        pixres_file = os.path.join(output_pixres, f"{nm}.mat")
        if not os.path.exists(pixres_file):
            print(f"Pixel resolution file {pixres_file} not found. Skipping {dfnm}.")
            continue

        pixres_info = loadmat(pixres_file)
        pix_res = pixres_info['pix_res'][0][0]
        scaling_factor_x = crop_resolution_x / float(pix_res[0][0])
        scaling_factor_y = crop_resolution_y / float(pix_res[1][0])

        # Load the .pkl file
        with open(os.path.join(src, dfnm), 'rb') as f:
            df = pickle.load(f)

        # Convert DataFrame to numpy array
        col_names = df.columns.tolist()
        df_array = df.to_numpy().astype(np.float32)

        if isinstance(nms_info, np.ndarray):
            info = nms_info.tolist()
        elif isinstance(nms_info, (list, tuple)) and len(nms_info) == 3 \
             and not isinstance(nms_info[0], (list, tuple, np.ndarray)):
            # single triple → wrap in a list
            info = [nms_info]
        else:
            info = list(nms_info)

        # Iterate through the cropping information for processing
        for entry in info:
            try:
                source_image = entry[0][0] if isinstance(entry[0], (np.ndarray, list)) else entry[0]
                target_image = entry[1][0] if isinstance(entry[1], (np.ndarray, list)) else entry[1]

                # Ensure they are strings
                source_image = str(source_image)
                target_image = str(target_image)
                # Correct the structure of coords to ensure it is a 1D array
                coords = np.array(entry[2], dtype=float).flatten()  # Flatten the array
            except:
                print('raia')

            # Remove the extension from the source image name
            source_image_nm = os.path.splitext(source_image)[0]
            target_image_nm = os.path.splitext(target_image)[0]

            # Check if the source image matches the current .pkl file
            if source_image_nm == nm:
                # Upscale the cropping coordinates
                xmin, ymin, xwidth, ywidth = coords
                xmin_scaled = xmin * scaling_factor_x
                ymin_scaled = ymin * scaling_factor_y
                xwidth_scaled = xwidth * scaling_factor_x
                ywidth_scaled = ywidth * scaling_factor_y
                xmax_scaled = xmin_scaled + xwidth_scaled
                ymax_scaled = ymin_scaled + ywidth_scaled

                # Filter cells within the cropping coordinates
                mask = (
                    (df_array[:, 0] >= xmin_scaled) & (df_array[:, 0] <= xmax_scaled) &  # Centroid_x
                    (df_array[:, 1] >= ymin_scaled) & (df_array[:, 1] <= ymax_scaled)    # Centroid_y
                )
                filtered_cells = df_array[mask]

                # Adjust centroids relative to the cropping position
                if len(filtered_cells) > 0:
                    # Calculate new centroids relative to the cropped region
                    x_centroid_chop = filtered_cells[:, 0] - xmin_scaled
                    y_centroid_chop = filtered_cells[:, 1] - ymin_scaled

                    # Add the new columns to the filtered cells array
                    filtered_cells = np.hstack((
                        filtered_cells,
                        x_centroid_chop.reshape(-1, 1),
                        y_centroid_chop.reshape(-1, 1)
                    ))

                    # Process the first image, 1/3, 1/2, and 2/3 of the stack
                    if i in target_indices:
                        imhechop = imread(os.path.join(pth_imhe_chop, f"{target_image_nm}.tif"))
                        # plot filtered centroids on top of imhechop
                        plt.figure(figsize=(10, 10))
                        plt.imshow(imhechop)
                        plt.scatter(x_centroid_chop/scaling_factor_x, y_centroid_chop/scaling_factor_y, s=0.5, c='blue', label='Centroids')
                        plt.show()

                    # Update the column names
                    col_names_with_chop = col_names + ['x_centroid_chop', 'y_centroid_chop']

                    # Save the filtered cells to a .mat file
                    dst = os.path.join(outpth, f"{target_image_nm}.mat")
                    savemat(dst, {
                        'features': filtered_cells,
                        'feature_names': col_names_with_chop,
                        'cropping_coords': [xmin_scaled, ymin_scaled, xwidth_scaled, ywidth_scaled]
                    })
                else:
                    print(f"No cells found for {target_image_nm}")


def get_rgb_avg_std(centroid, contour_raw, offset, HE_20x_WSI):
    """gets RBG average intensities inside of a contour given the image and centroid
    It is fast because it crops the image so that the image size is offset*2 width/height.
    Python passes HE_20x_WSI as reference so it shouldn't affect performance passing a
    hugh variable like this."""

    x_low = centroid[0] - offset
    x_high = centroid[0] + offset
    y_low = centroid[1] - offset
    y_high = centroid[1] + offset

    img_shape = HE_20x_WSI.shape

    # if bad shape, return -1 for each intensity mean
    if offset > centroid[0] or offset > centroid[1] or centroid[0] > (img_shape[0] - offset) or centroid[1] > (
            img_shape[1] - offset):
        # print(f'centroid passed: {centroid}')
        r_avg = -1
        g_avg = -1
        b_avg = -1
        r_std = -1
        g_std = -1
        b_std = -1
        return r_avg, g_avg, b_avg, r_std, g_std, b_std

    im_crop = np.array(HE_20x_WSI[x_low:x_high, y_low:y_high], dtype=np.uint16)

    # plt.imshow(im_crop)

    crop_x = centroid[0] - offset - 1
    crop_y = centroid[1] - offset - 1

    contour_adj = adjust_contours(contour_raw, crop_x, crop_y)
    contour_new = contour_adj  # .reshape((-1,1,2)).astype(np.uint16)
    rev_contour = contour_new[:, [1, 0]]  # its backwards for some reason idk why but you need to flip it like this
    # rev_contour = contour_new[:,:, [1, 0]]  # its backwards for some reason idk why but you need to flip it like this
    # print(rev_contour)

    # coords NEEDS to be np.int32 matrix --> 2 columns x y

    # Create a single-channel mask
    mask = np.zeros_like(im_crop[:, :, 0], dtype=np.uint16)  # make black image of same size, will fill with mask

    # Draw contours on the single-channel mask
    # cv2.drawContours(im_crop, [rev_contour], -1, (0,255,0)) #, thickness=cv2.FILLED)  # this one makes it green so that you can see contour
    cv2.drawContours(mask, [rev_contour], 0, (1), thickness=cv2.FILLED)

    # plt.imshow(im_crop)

    r_pixels = im_crop[:, :, 0] * mask  # pixels inside mask are 1, outside == 0
    g_pixels = im_crop[:, :, 1] * mask
    b_pixels = im_crop[:, :, 2] * mask

    num_pixels = np.count_nonzero(mask)

    if num_pixels != 0:

        r_avg = round(np.sum(r_pixels) / num_pixels, 2)
        g_avg = round(np.sum(g_pixels) / num_pixels, 2)
        b_avg = round(np.sum(b_pixels) / num_pixels, 2)

        r_std = np.std(r_pixels)
        g_std = np.std(g_pixels)
        b_std = np.std(b_pixels)

    else:
        print('ZERO PIXEL')
        r_avg = -1
        g_avg = -1
        b_avg = -1

        r_std = -1
        g_std = -1
        b_std = -1

    # plt.imshow(im_crop)

    return r_avg, g_avg, b_avg, r_std, g_std, b_std

