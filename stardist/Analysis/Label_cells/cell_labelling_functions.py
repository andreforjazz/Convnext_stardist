import matplotlib.pyplot as plt
import numpy as np
from stardist.models import StarDist2D, Config2D
import json
import geojson
from typing import List, Dict, Tuple
from pathlib import Path
import os
from tifffile import imread, imwrite
from tqdm import tqdm
import random
from PIL import Image
from stardist import fill_label_holes
import copy
from tensorflow.python.summary.summary_iterator import summary_iterator
import struct
from matplotlib.colors import ListedColormap
import pandas as pd
import gc
import tensorflow as tf
from skimage.transform import rescale, resize
from scipy.io import loadmat
import cv2
import re
import pickle
from scipy.io import savemat
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import tifffile
import json
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Polygon, MultiPolygon
import geojson


os.environ["CV_IO_MAX_IMAGE_PIXELS"] = "9223372036854775807"

def make_geojson_contours_with_classes(
        out_pth_json,
        ds_amt=1,
        ds_mask=1 / 0.4416,
        class_names=None,
        class_colors=None,
        gj=1,
        pth10xsegmented=None,
        step=50
):
    """
    Generates GeoJSON contours from segmented cells, assigning respective classes based on semantic segmentation.

    Args:
    - out_pth_json (str): Path to the directory containing JSON files.
    - semantic_image (numpy array): Resampled semantic segmentation image.
    - ds_amt (float): Downsampling amount, 1 for 20x. Default is 1.
    - class_names (list): List of class names corresponding to semantic segmentation labels.
    - class_colors (list): List of RGB colors corresponding to class names.
    - gj (int): Set to 1 to generate GeoJSON contours, 0 to skip. Default is 1.

    Returns:
    - None
    """
    if gj != 1:
        print("GeoJSON generation is skipped.")
        return

    if class_names is None or class_colors is None:
        raise ValueError("Both class_names and class_colors must be provided.")

    out_pth_contours = os.path.join(out_pth_json, 'geojsons', '32_polys_20x_labels')
    os.makedirs(out_pth_contours, exist_ok=True)
    json_pth_list = [f for f in os.listdir(out_pth_json) if f.endswith('.json')]

    for p, file in enumerate(json_pth_list):
        if p % step != 0:  # Execute every 50 iterations
            #print(f'skippin iteration {p},{file} with {p % step != 0}')
            continue
        nm = os.path.basename(file)
        file_name, _ = os.path.splitext(nm)
        new_fn = os.path.join(out_pth_contours, file_name + '.geojson')
        print(f'{p + 1} / {len(json_pth_list)}')
        print(nm)
        try:
            segmented_image = cv2.imread(os.path.join(pth10xsegmented, file_name + '.tif'), cv2.IMREAD_UNCHANGED)
        except:
            segmented_image = tifffile.imread(os.path.join(pth10xsegmented, file_name + '.tif'))
            print('using tifffile to open large tiff')

        if not os.path.exists(new_fn):
            with open(os.path.join(out_pth_json, file)) as f:
                segmentation_data = json.load(f)

            # Extract downsampled data (centroids and contours)
            data_list = get_ds_data(segmentation_data, ds_amt)

            GEOdata = []

            for j, (centroid, contour) in tqdm(enumerate(data_list), total=len(data_list), desc="Processing contours"):
                y, x = int(round(centroid[0])), int(round(centroid[1]))

                try:
                    # Determine class based on the semantic image
                    label = segmented_image[round(y / ds_mask), round(x / ds_mask)]
                except:
                    print('deu raia, no label')

                # label = semantic_image[x, y]
                if label > 0:  # Ignore background
                    classification_name = class_names[label - 1]
                    classification_color = [int(c * 255) for c in class_colors[label - 1]]
                    # xy coordinates are swapped, so I reverse them here with xy[::-1]
                    # note: add 1 to coords to fix 0 indexing vs 1 index offset
                    contour = [[coord + 0 for coord in xy[::-1]] for xy in contour]  # Convert coordinates to integers

                    # Transform contour to GeoJSON format
                    contour.append(contour[0])  # Close the polygon

                    dict_data = {
                        "type": "Feature",
                        "id": "PathCellObject",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [contour]
                        },
                        "properties": {
                            'objectType': 'annotation',
                            'classification': {
                                'name': classification_name,
                                'color': classification_color
                            }
                        }
                    }

                    GEOdata.append(dict_data)

            with open(new_fn, 'w') as outfile:
                geojson.dump(GEOdata, outfile)
            print('Finished', new_fn)
        else:
            print(f'Skipping {new_fn}')


def get_ds_data(segmentation_data, ds):
    """
    Processes segmentation data and applies downsampling.

    Args:
    - segmentation_data (list): List of segmentation data containing centroids and contours.
    - ds (float): Downsampling factor.

    Returns:
    - list: List of tuples with downsampled centroids and contours.
    """
    data_list = []
    for data in segmentation_data:
        centroid = data['centroid'][0]  # Flatten the centroid
        contour = data['contour'][0]  # Flatten the contour

        # Downsample the centroid
        ds_centroid = [int(c / ds) for c in centroid]

        # Downsample and format the contour
        ds_contour = [[value / ds for value in sublist] for sublist in contour]
        ds_contour = [[round(x, 2), round(y, 2)] for x, y in zip(ds_contour[0], ds_contour[1])]

        # Combine downsampled centroid and contour
        dat = [ds_centroid, ds_contour]
        data_list.append(dat)

    return data_list

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

def make_RGBfeatures_df_pkl_from_contours_jsons_with_classes(
        jsons,
        WSIs,
        outpth,
        segmented_image_dir,
        ds_mask=1 / 0.4416,
        class_names=None,
):
    """
    Processes JSON files, extracts features, determines cell classes using a segmented image, and saves as pickle.

    Args:
    - jsons (list): List of JSON file paths.
    - WSIs (list): List of WSI image paths corresponding to the JSON files.
    - outpth (str): Output directory for saving pickle files.
    - segmented_image_dir (str): Directory containing segmented images.
    - ds_mask (float): Downsampling mask value. Default is 1/0.4416.
    - class_names (list): List of class names corresponding to segmentation labels.

    Returns:
    - None
    """
    if class_names is None:
        raise ValueError("class_names must be provided.")

    # change this so i can skip every 50
    for i, json_f_name in enumerate(jsons,):


        nm = os.path.splitext(os.path.basename(json_f_name))[0]
        outnm = os.path.join(outpth, f'{nm}.pkl')
        print(f'{nm}  {i + 1}/{len(jsons)}')

        if not os.path.exists(outnm):
            HE_20x_WSI = imread(WSIs[i])
            segmented_image_path = os.path.join(segmented_image_dir, f"{nm}.tif")
            print(f"Loading segmented image: {segmented_image_path}")
            # segmented_image = cv2.imread(segmented_image_path, cv2.IMREAD_UNCHANGED)
            try:
                segmented_image = cv2.imread(segmented_image_path, cv2.IMREAD_UNCHANGED)
            except:
                segmented_image = tifffile.imread(segmented_image_path)
                print('using tifffile to open large tiff')

            try:
                segmentation_data = json.load(open(json_f_name))
            except Exception as e:
                print(f"Error reading JSON: {e}. Skipping {json_f_name}")
                continue

            centroids = [nuc['centroid'][0] for nuc in segmentation_data]
            contours = [nuc['contour'] for nuc in segmentation_data]
            contours_fixed = fix_contours(contours)

            classifications = []
            for centroid in centroids:
                y, x = int(round(centroid[0])), int(round(centroid[1]))
                label = segmented_image[round(y / ds_mask), round(x / ds_mask)]
                # class_name = class_names[label - 1] if label > 0 else 'WS'
                classifications.append(label)

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
                'slide_num': extract_slide_number(nm),
                'class_name': classifications  # Add class name column
            }

            df = pd.DataFrame(dat)
            df.to_pickle(outnm)
        else:
            print('Already extracted features into pkl from json contours')


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

    match = re.search(r'[_-](\d+)(?=\D*$)', nm)
    if match:
        return match.group(1)  # e.g., "005" from "6516-005_HE"

    # Return None or raise an exception if no valid number is found
    return None


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






def plot_correlation_heatmap(df, output_dir, cmap="coolwarm", annot=True):
    """
    Creates a correlation heatmap for numerical features in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing numerical features.
        output_dir (str): Directory to save the heatmap image.
        cmap (str): Color map for the heatmap.
        annot (bool): Whether to annotate the heatmap with correlation coefficients.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Select numerical features
    numerical_df = df.select_dtypes(include=['float64', 'int64'])

    # Compute the correlation matrix
    corr_matrix = numerical_df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Generate a custom diverging colormap
    sns.set(style="white")
    heatmap = sns.heatmap(
        corr_matrix,
        annot=annot,
        cmap=cmap,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={"shrink": .5},
        square=True
    )

    # Customize the heatmap
    plt.title('Correlation Heatmap', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save the heatmap
    plot_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Correlation heatmap saved to {plot_path}")


def plot_curve_histogram(df, feature, class_mapping, palette, output_dir):
    """
    Create curve-based histograms for a given feature across classes, excluding class 7,
    sorted by median descending. Ensure consistent color mapping and enlarged font sizes
    for class names on the X-axis.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data with 'class_name' and the feature.
        feature (str): The feature to plot.
        class_mapping (dict): Mapping from class integers to class names.
        palette (dict): Mapping from class names to colors.
        output_dir (str): Path to save the plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Exclude class 7 ("background")
    filtered_df = df[df['class_name'] != 7].copy()

    if filtered_df.empty:
        print("No data available after excluding Class 7 ('background').")
        return

    # 2. Map class integers to class names
    filtered_df['class_name_str'] = filtered_df['class_name'].map(class_mapping)

    # Drop any rows where mapping resulted in NaN (i.e., classes not in class_mapping)
    filtered_df = filtered_df.dropna(subset=['class_name_str'])

    # 3. Calculate median per class and sort classes descending
    median_per_class = filtered_df.groupby('class_name_str')[feature].median().sort_values(ascending=False)
    sorted_classes = median_per_class.index.tolist()

    # 4. Check for missing classes in the palette and assign default colors if needed
    missing_classes = set(sorted_classes) - set(palette.keys())
    if missing_classes:
        print(f"Warning: No colors defined for classes: {missing_classes}. Assigning default color.")
        for class_name in missing_classes:
            palette[class_name] = '#A0A0A0'  # Assign gray color

    # 5. Set the aesthetic style for the plot
    sns.set(style="ticks")
    plt.figure(figsize=(18, 10))  # Increased figure width for better spacing

    # 6. Define the number of points for a smooth curve
    num_points = 1000
    feature_min = filtered_df[feature].min()
    feature_max = filtered_df[feature].max()
    x_values = np.linspace(feature_min, feature_max, num_points)

    # 7. Plot curve histograms for each class
    for class_name in sorted_classes:
        subset = filtered_df[filtered_df['class_name_str'] == class_name]
        scatter_color = palette.get(class_name, 'black')

        # Compute histogram counts with high resolution (many bins)
        counts, bin_edges = np.histogram(subset[feature], bins=num_points, range=(feature_min, feature_max))

        # Interpolate counts to create a smooth curve
        smooth_counts = np.interp(x_values, bin_edges[:-1], counts)

        # Plot the smooth curve
        plt.plot(x_values, smooth_counts, label=class_name, color=scatter_color, linewidth=2.5)

    # 8. Apply despine for a clean look
    sns.despine()

    # 9. Rotate x labels for better readability and increase their font size
    plt.xticks(fontsize=14)  # Increased fontsize for tick labels
    plt.yticks(fontsize=14)  # Optional: Increase fontsize for Y-axis ticks

    # 10. Set titles and labels with separate font sizes
    plt.title(f'Curve Histogram of {feature} by Class', fontsize=18)
    plt.xlabel(feature, fontsize=16)
    plt.ylabel('Count', fontsize=16)

    # 11. Enhance legend
    plt.legend(title='Class Name', fontsize=12, title_fontsize=14, loc='upper right')

    # 12. Adjust layout for better spacing
    plt.tight_layout()

    # 13. Save the plot as PNG
    plot_path = os.path.join(output_dir, f'curve_histogram_{feature}.png')
    try:
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Curve histogram saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")


def plot_density_curves(df, feature, class_mapping, palette, output_dir):
    """
    Create density curves (KDE plots) for a given feature across classes, excluding class 7,
    sorted by median descending. Ensure consistent color mapping and increased spacing
    between curves. Enlarge class names on the X-axis.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data with 'class_name' and the feature.
        feature (str): The feature to plot.
        class_mapping (dict): Mapping from class integers to class names.
        palette (dict): Mapping from class names to colors.
        output_dir (str): Path to save the plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Exclude class 7 ("background")
    filtered_df = df[df['class_name'] != 7].copy()

    if filtered_df.empty:
        print("No data available after excluding Class 7 ('background').")
        return

    # 2. Map class integers to class names
    filtered_df['class_name_str'] = filtered_df['class_name'].map(class_mapping)

    # Drop any rows where mapping resulted in NaN (i.e., classes not in class_mapping)
    filtered_df = filtered_df.dropna(subset=['class_name_str'])

    # 3. Calculate median per class and sort classes descending
    median_per_class = filtered_df.groupby('class_name_str')[feature].median().sort_values(ascending=False)
    sorted_classes = median_per_class.index.tolist()

    # 4. Check for missing classes in the palette and assign default colors if needed
    missing_classes = set(sorted_classes) - set(palette.keys())
    if missing_classes:
        print(f"Warning: No colors defined for classes: {missing_classes}. Assigning default color.")
        for class_name in missing_classes:
            palette[class_name] = '#A0A0A0'  # Assign gray color

    # 5. Set the aesthetic style for the plot
    sns.set(style="ticks")
    plt.figure(figsize=(18, 10))  # Increased figure width for better spacing

    # 6. Create KDE plots for each class
    for class_name in sorted_classes:
        subset = filtered_df[filtered_df['class_name_str'] == class_name]
        scatter_color = palette.get(class_name, 'black')
        sns.kdeplot(
            data=subset,
            x=feature,
            label=class_name,
            color=scatter_color,
            fill=False,
            linewidth=2.5
        )

    # 7. Apply despine for a clean look
    sns.despine()

    # 8. Rotate x labels for better readability and increase their font size
    plt.xticks(fontsize=14)  # Increased fontsize for tick labels
    plt.yticks(fontsize=14)  # Optional: Increase fontsize for Y-axis ticks

    # 9. Set titles and labels with separate font sizes
    plt.title(f'Density Plot of {feature} by Class', fontsize=18)
    plt.xlabel(feature, fontsize=16)
    plt.ylabel('Density', fontsize=16)

    # 10. Enhance legend
    plt.legend(title='Class Name', fontsize=12, title_fontsize=14, loc='upper right')

    # 11. Adjust layout for better spacing
    plt.tight_layout()

    # 12. Save the plot as PNG
    plot_path = os.path.join(output_dir, f'density_{feature}.png')
    try:
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Density plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")


def plot_violin_shifted(df, feature, class_mapping, palette, output_dir):
    """
    Create a violin plot for a given feature across classes, excluding class 7,
    sorted by median descending. Plot data points shifted to the right of violins.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data with 'class_name' and the feature.
        feature (str): The feature to plot.
        class_mapping (dict): Mapping from class integers to class names.
        palette (dict): Mapping from class names to colors.
        output_dir (str): Path to save the plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Exclude class 7 ("background")
    filtered_df = df[df['class_name'] != 7].copy()

    if filtered_df.empty:
        print("No data available after excluding Class 7 ('background').")
        return

    # 2. Map class integers to class names
    filtered_df['class_name_str'] = filtered_df['class_name'].map(class_mapping)

    # Drop any rows where mapping resulted in NaN (i.e., classes not in class_mapping)
    filtered_df = filtered_df.dropna(subset=['class_name_str'])

    # 3. Calculate median per class and sort classes descending
    median_per_class = filtered_df.groupby('class_name_str')[feature].median().sort_values(ascending=False)
    sorted_classes = median_per_class.index.tolist()

    # 4. Set the order for the plot
    sns.set(style="ticks")
    plt.figure(figsize=(12, 8))

    # 5. Create violin plot
    ax = sns.violinplot(
        x='class_name_str',
        y=feature,
        data=filtered_df,
        order=sorted_classes,
        palette=palette,
        inner='quartile',
        width=0.35
    )

    # 6. Shift data points to the right
    shift = 0.5  # Amount to shift data points to the right
    for i, class_name in enumerate(sorted_classes):
        subset = filtered_df[filtered_df['class_name_str'] == class_name]
        # Generate jittered x positions shifted to the right
        x = np.random.normal(loc=i, scale=0.05, size=len(subset)) + shift
        scatter_color = palette.get(class_name, 'black')
        plt.scatter(x, subset[feature], color=scatter_color, s=0.5, alpha=0.3)

    # 7. Apply despine for a clean look
    sns.despine()

    # 8. Rotate x labels for better readability
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)

    # 9. Set titles and labels
    plt.title(f'Violin Plot of {feature} by Class')
    plt.xlabel('Class Name', fontsize=18)
    plt.ylabel(feature, fontsize=18)

    # 10. Adjust layout for better spacing
    plt.tight_layout()

    # 11. Save the plot as PNG
    plot_path = os.path.join(output_dir, f'violin_{feature}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Violin plot saved to {plot_path}")


def rgb_to_hex(rgb_list):
    """
    Convert a list of RGB values (0-255) to HEX format.

    Parameters:
        rgb_list (list): List of RGB lists.

    Returns:
        list: List of HEX color strings.
    """
    hex_colors = ['#%02x%02x%02x' % tuple(int(round(c)) for c in color) for color in rgb_list]
    return hex_colors

def load_data(data_dir):
    """
    Load all .pkl files from the specified directory and concatenate them into a single DataFrame.

    Parameters:
        data_dir (str): Path to the directory containing .pkl files.

    Returns:
        pd.DataFrame: Combined DataFrame containing data from all images.
    """
    # List all .pkl files in the directory
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]

    if not all_files:
        raise FileNotFoundError(f"No .pkl files found in directory: {data_dir}")

    # Load each .pkl file into a DataFrame and store in a list
    df_list = []
    # Initialize tqdm progress bar with total count
    for file in tqdm(all_files, desc=f"Loading {len(all_files)} .pkl files", unit="file"):
        try:
            df = pd.read_pickle(file)
            df_list.append(df)
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")

    # Concatenate all DataFrames into one
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    return combined_df

# Extract measurements and convert to a dataframe
def extract_measurements(geometry_data):
    # Check if the measurements field is not NaN
    if geometry_data['measurements'] and isinstance(geometry_data['measurements'], str):
        # Convert the measurements string to a dictionary
        measurements_dict = json.loads(geometry_data['measurements'])
        return measurements_dict
    return {}


def save_GEOJSON_with_clusters(out_pth_json, df, class_names, class_colors=None, ds_amt=1, ds_mask=1/0.4416):
    """
    Save GeoJSON with cluster IDs, class names, and assigned colors.
    """
    # Ensure class_names is provided
    if class_names is None:
        raise ValueError("class_names must be provided.")

    # Generate class_colors if not provided
    if class_colors is None:
        # Use a colormap to generate distinct colors for each cluster
        cmap = plt.get_cmap('tab20')  # Access colormap using plt.get_cmap()
        norm = mcolors.Normalize(vmin=0, vmax=len(class_names) - 1)
        class_colors = [mcolors.rgb2hex(cmap(norm(i))) for i in range(len(class_names))]

    # Create a GeoJSON dictionary
    GEOdata = []

    for index, row in df.iterrows():
        cluster_id = row['Cluster']
        contour = row['geometry']

        if cluster_id > 0:  # Ensure it's not the background
            # Get classification name and color
            classification_name = class_names[cluster_id - 1]  # Assuming cluster_id starts from 1

            # Convert hex color to RGB if it is not already in RGB format
            if isinstance(class_colors[cluster_id - 1], str):  # If it's a hex string
                classification_color = mcolors.hex2color(class_colors[cluster_id - 1])
            else:
                classification_color = class_colors[cluster_id - 1]  # Already in RGB

            # Convert to a list of integers for RGB format
            classification_color = [int(c * 255) for c in classification_color]

            # Extract coordinates from contour, handling both Polygon and MultiPolygon geometries
            if isinstance(contour, Polygon):
                contour_coords = list(contour.exterior.coords)  # For Polygon
                # Force the polygon to close by checking first and last points
                if contour_coords[0] != contour_coords[-1]:
                    contour_coords.append(contour_coords[0])  # Ensure the polygon is closed
                # Wrap coordinates in a list of lists for GeoJSON Polygon
                contour_coords = [contour_coords]
            elif isinstance(contour, MultiPolygon):
                contour_coords = []
                for polygon in contour.geoms:  # Access the individual polygons in MultiPolygon
                    coords = list(polygon.exterior.coords)
                    # Force the polygon to close by checking first and last points
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])  # Ensure the polygon is closed
                    # Wrap each polygon's coordinates in a list for GeoJSON MultiPolygon
                    contour_coords.append([coords])
            else:
                continue  # Skip invalid geometry types

            # Create GeoJSON feature
            dict_data = {
                "type": "Feature",
                "id": "PathCellObject",
                "geometry": {
                    "type": "Polygon" if isinstance(contour, Polygon) else "MultiPolygon",
                    "coordinates": contour_coords
                },
                "properties": {
                    'objectType': 'annotation',
                    'classification': {
                        'name': classification_name,
                        'color': classification_color
                    }
                }
            }

            GEOdata.append(dict_data)

    # Save GeoJSON to file
    geojson_path = os.path.join(out_pth_json, 'updated_segmentation.geojson')
    with open(geojson_path, 'w') as outfile:
        geojson.dump(GEOdata, outfile)
    print('GeoJSON file saved to', geojson_path)


def add_class_labels_to_existing_pkl_files(
    pkl_files_needing_classes,
    outpthpkl_with_classes,
    pth10xsegmented,
    pixelres,
    class_names,
    class_colors,
    vis_check_interval=50,
    out_pth_analysis=None
):
    """
    Add class labels to existing pkl files by loading segmented images and assigning classes to centroids.
    
    Parameters:
        pkl_files_needing_classes (list): List of tuples (basename, pkl_path) for files needing class labels
        outpthpkl_with_classes (str): Output directory for pkl files with classes
        pth10xsegmented (str): Directory containing segmented images
        pixelres (float): Pixel resolution for coordinate conversion
        class_names (list): List of class names
        vis_check_interval (int): Interval for visualization checks (default: 50)
        out_pth_analysis (str): Optional path for saving visualizations
    
    Returns:
        int: Number of files successfully processed
    """
    os.makedirs(outpthpkl_with_classes, exist_ok=True)
    ds_mask = 1 / pixelres
    processed_count = 0
    
    for i, (nm, pkl_path) in enumerate(pkl_files_needing_classes):
        print(f"Processing {nm} ({i+1}/{len(pkl_files_needing_classes)})...")
        
        # Load existing pkl file
        df = pd.read_pickle(pkl_path)
        
        # Check if class_name already exists
        if 'class_name' in df.columns:
            print(f"  Warning: {nm} already has class_name column. Skipping.")
            continue
        
        # Load corresponding segmented image
        segmented_image_path = os.path.join(pth10xsegmented, f"{nm}.tif")
        if not os.path.exists(segmented_image_path):
            print(f"  Warning: Segmented image not found for {nm}. Skipping.")
            continue
        
        try:
            segmented_image = cv2.imread(segmented_image_path, cv2.IMREAD_UNCHANGED)
        except:
            segmented_image = tifffile.imread(segmented_image_path)
            print('  Using tifffile to open large tiff')
        
        # Get centroids from dataframe
        centroids = df[['Centroid_x', 'Centroid_y']].values
        
        # Assign class labels based on segmented image
        classifications = []
        for centroid_x, centroid_y in centroids:
            y, x = int(round(centroid_y)), int(round(centroid_x))
            try:
                label = segmented_image[round(y / ds_mask), round(x / ds_mask)]
                classifications.append(label)
            except:
                classifications.append(0)  # Default to background if out of bounds
        
        # Add class_name column
        df['class_name'] = classifications
        
        # Save to classes folder
        output_pkl_path = os.path.join(outpthpkl_with_classes, f'{nm}.pkl')
        df.to_pickle(output_pkl_path)
        print(f"  ✓ Saved to {output_pkl_path}")
        
        # Visualization check
        if ((i + 1) % vis_check_interval == 0) or (i == 0):
            visualize_class_assignment(
                df, segmented_image, nm, class_names, class_colors, 
                ds_mask, out_pth_analysis
            )
        
        processed_count += 1
    
    return processed_count


def visualize_class_assignment(
    df, segmented_image, nm, class_names, class_colors, ds_mask, out_pth_analysis=None
):
    """
    Create visualization of class assignment: segmented image and centroids with class labels.
    
    Parameters:
        df (pd.DataFrame): DataFrame with centroids and class_name column
        segmented_image (np.ndarray): Segmented image array
        nm (str): Image basename
        class_names (list): List of class names
        class_colors (list): List of RGB color tuples for classes
        ds_mask (float): Downsampling factor for coordinate conversion
        out_pth_analysis (str): Optional directory to save visualization
    """
    import matplotlib.pyplot as plt
    
    print(f"  → Visualization check for {nm}...")
    
    # Create colormap for segmented image
    seg_colored = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)
    for label_id in range(1, len(class_names) + 1):
        mask = segmented_image == label_id
        seg_colored[mask] = class_colors[label_id - 1]
    
    # Get centroids and their classes
    centroids_x = df['Centroid_x'].values
    centroids_y = df['Centroid_y'].values
    classes = df['class_name'].values
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Segmented image
    axes[0].imshow(seg_colored)
    axes[0].set_title(f'Segmented Image: {nm}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot 2: Segmented image with overlaid centroids colored by class
    axes[1].imshow(seg_colored)
    axes[1].set_title(f'Centroids with Class Labels: {nm}\n(Total cells: {len(df)})', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Plot centroids with colors matching their assigned classes
    for class_id in range(1, len(class_names) + 1):
        mask = classes == class_id
        if np.any(mask):
            # Convert centroids to segmented image coordinates
            centroids_x_scaled = (centroids_x[mask] / ds_mask).astype(int)
            centroids_y_scaled = (centroids_y[mask] / ds_mask).astype(int)
            # Filter out-of-bounds points
            valid = (centroids_x_scaled >= 0) & (centroids_x_scaled < segmented_image.shape[1]) & \
                    (centroids_y_scaled >= 0) & (centroids_y_scaled < segmented_image.shape[0])
            if np.any(valid):
                # Convert RGB color to normalized [0,1] range for matplotlib
                color = tuple(np.array(class_colors[class_id - 1]) / 255.0)
                colors_list = [color] * np.sum(valid)
                axes[1].scatter(centroids_x_scaled[valid], centroids_y_scaled[valid], 
                              c=colors_list, s=15, alpha=0.7, 
                              label=f'{class_names[class_id - 1]} (n={np.sum(valid)})', 
                              edgecolors='white', linewidths=0.3)
    
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    plt.tight_layout()
    
    # Save visualization if output directory provided
    if out_pth_analysis:
        vis_output_dir = os.path.join(out_pth_analysis, 'class_assignment_checks')
        os.makedirs(vis_output_dir, exist_ok=True)
        vis_path = os.path.join(vis_output_dir, f'{nm}_class_check.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualization saved to {vis_path}")
    
    # Display visualization in notebook output
    plt.show()
    plt.close()


def process_pkl_files_with_class_labels(
    outpthpkl_no_classes,
    outpthpkl_with_classes,
    pth10xsegmented,
    pixelres,
    class_names,
    class_colors,
    json_basenames,
    pthjsonfiles,
    WSIs,
    vis_check_interval=50,
    out_pth_analysis=None
):
    """
    Main function to process pkl files: add class labels to existing files or extract features with classes.
    
    Parameters:
        outpthpkl_no_classes (str): Directory with pkl files without classes
        outpthpkl_with_classes (str): Output directory for pkl files with classes
        pth10xsegmented (str): Directory containing segmented images
        pixelres (float): Pixel resolution
        class_names (list): List of class names
        class_colors (list): List of RGB color tuples
        json_basenames (dict): Dictionary mapping basenames to json file paths
        pthjsonfiles (list): List of all json file paths
        WSIs (list): List of WSI file paths
        vis_check_interval (int): Interval for visualization checks
        out_pth_analysis (str): Optional path for saving visualizations
    
    Returns:
        tuple: (files_with_classes_added, remaining_json_files)
    """
    os.makedirs(outpthpkl_with_classes, exist_ok=True)
    
    # Check if pkl files exist without classes
    if os.path.exists(outpthpkl_no_classes):
        existing_pkl_files = get_sorted_files(outpthpkl_no_classes, '.pkl')
        existing_pkl_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in existing_pkl_files}
        
        # Find which ones need class labels added
        pkl_files_needing_classes = []
        for nm, pkl_path in existing_pkl_basenames.items():
            output_pkl_path = os.path.join(outpthpkl_with_classes, f'{nm}.pkl')
            if not os.path.exists(output_pkl_path):
                pkl_files_needing_classes.append((nm, pkl_path))
        
        if len(pkl_files_needing_classes) > 0:
            print(f"\nFound {len(pkl_files_needing_classes)} pkl files without classes. Adding class labels...")
            print(f"This is faster than re-extracting all features!")
            
            # Add class labels to existing pkl files
            processed = add_class_labels_to_existing_pkl_files(
                pkl_files_needing_classes,
                outpthpkl_with_classes,
                pth10xsegmented,
                pixelres,
                class_names,
                class_colors,
                vis_check_interval=vis_check_interval,
                out_pth_analysis=out_pth_analysis
            )
            
            print(f"\n✓ Added class labels to {processed} pkl files!")
            
            # Find remaining JSON files that need full feature extraction
            remaining_json_files = [json_basenames[nm] for nm in json_basenames 
                                   if nm not in existing_pkl_basenames]
            
            return processed, remaining_json_files
        else:
            print("\nAll pkl files already have class labels. Skipping class assignment.")
            remaining_json_files = [json_basenames[nm] for nm in json_basenames 
                                   if nm not in existing_pkl_basenames]
            return 0, remaining_json_files
    else:
        # No existing pkl files
        remaining_json_files = list(json_basenames.values())
        return 0, remaining_json_files


def calculate_statistics(df, feature, class_mapping, exclude_class=None):
    """
    Calculate statistics (mean, median, std, etc.) for a given feature across classes.
    Also computes the coefficient of variation (CV = std/mean).

    Parameters:
        df (pd.DataFrame): DataFrame containing the data with 'class_name' and the feature.
        feature (str): The feature to calculate statistics for.
        class_mapping (dict): Mapping from class integers to class names.
        exclude_class (int): Optional class ID to exclude (default: None, excludes class 7 if not specified).

    Returns:
        pd.DataFrame: A DataFrame containing statistics for each class.
    """
    if exclude_class is None:
        exclude_class = 7  # Default: exclude background
    
    # Exclude specified class and map class names
    filtered_df = df[df['class_name'] != exclude_class].copy()
    filtered_df['class_name_str'] = filtered_df['class_name'].map(class_mapping)
    filtered_df = filtered_df.dropna(subset=['class_name_str'])

    # Group by class and calculate statistics
    stats_df = filtered_df.groupby('class_name_str')[feature].agg(
        mean='mean',
        median='median',
        std='std',
        min='min',
        max='max',
        count='count'
    ).reset_index()

    # Calculate coefficient of variation (CV = std/mean), avoiding division by zero
    stats_df['coef_var'] = stats_df.apply(
        lambda row: row['std'] / row['mean'] if row['mean'] != 0 else np.nan, 
        axis=1
    )

    return stats_df


def plot_violin_publication(
    df, feature, class_mapping, palette, output_dir, 
    resolution=0.504, exclude_class=None, save_stats=True, 
    figsize=(10, 6), dpi=300
):
    """
    Create a publication-ready violin plot for a given feature across classes.
    Scientific formatting: box off (only x and y axes), clean appearance.

    Parameters:
        df (pd.DataFrame): DataFrame with 'class_name' and the feature.
        feature (str): The feature to plot.
        class_mapping (dict): Mapping from class integers to class names.
        palette (dict): Mapping from class names to colors.
        output_dir (str): Directory to save the plot and statistics files.
        resolution (float): Microns per pixel for unit conversion (default: 0.504).
        exclude_class (int): Optional class ID to exclude (default: None, excludes class 7).
        save_stats (bool): Whether to save statistics to CSV/Excel (default: True).
        figsize (tuple): Figure size (width, height) in inches (default: (10, 6)).
        dpi (int): Resolution for saved figure (default: 300).
    
    Returns:
        str: Path to saved plot file.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if exclude_class is None:
        exclude_class = 7  # Default: exclude background
    
    os.makedirs(output_dir, exist_ok=True)

    # Filter and map class names
    filtered_df = df[df['class_name'] != exclude_class].copy()
    filtered_df['class_name_str'] = filtered_df['class_name'].map(class_mapping)
    filtered_df = filtered_df.dropna(subset=['class_name_str'])

    # Convert pixel-based metrics to microns if applicable
    pixel_features = ['Area', 'Perimeter', 'maximum_radius', 'mean_radius',
                      'median_radius', 'minor_axis_length', 'major_axis_length']
    if feature in pixel_features:
        if feature == 'Area':
            filtered_df[feature] = filtered_df[feature] * (resolution ** 2)
        else:
            filtered_df[feature] = filtered_df[feature] * resolution

    # Calculate statistics
    stats_df = calculate_statistics(df, feature, class_mapping, exclude_class=exclude_class)
    if feature in pixel_features:
        factor = resolution ** 2 if feature == 'Area' else resolution
        stats_df['mean'] = stats_df['mean'] * factor
        stats_df['std'] = stats_df['std'] * factor
        stats_df['median'] = stats_df['median'] * factor
        stats_df['min'] = stats_df['min'] * factor
        stats_df['max'] = stats_df['max'] * factor

    # Save statistics if requested
    if save_stats:
        stats_csv_path = os.path.join(output_dir, f'stats_{feature}.csv')
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"Statistics saved to {stats_csv_path}")

        stats_excel_path = os.path.join(output_dir, f'stats_{feature}.xlsx')
        stats_df.to_excel(stats_excel_path, index=False)
        print(f"Statistics saved to {stats_excel_path}")

    # Sort classes by median descending for plotting
    median_per_class = filtered_df.groupby('class_name_str')[feature].median().sort_values(ascending=False)
    sorted_classes = median_per_class.index.tolist()

    # Create figure with scientific formatting
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Create violin plot with proper color mapping
    # Ensure palette only contains classes that exist in the data
    available_palette = {cls: palette.get(cls, '#000000') for cls in sorted_classes if cls in palette}
    
    sns.violinplot(
        x='class_name_str',
        y=feature,
        data=filtered_df,
        order=sorted_classes,
        hue='class_name_str',
        palette=available_palette if available_palette else palette,
        inner='quartile',
        width=0.7,
        linewidth=1.5,
        ax=ax,
        legend=False
    )

    # Scientific formatting: box off (remove top and right spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Format axes
    ax.set_xlabel('Class Name', fontsize=14, fontweight='normal')
    unit = 'µm²' if feature == 'Area' else 'µm' if feature in pixel_features else ''
    ylabel = f'{feature} ({unit})' if unit else feature
    ax.set_ylabel(ylabel, fontsize=14, fontweight='normal')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Remove title for publication (clean look)
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f'violin_{feature}.png')
    plt.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Violin plot saved to {plot_path}")
    
    # Create and save log-scale version
    # Filter out zero/negative values for log scale
    filtered_df_log = filtered_df[filtered_df[feature] > 0].copy()
    
    if len(filtered_df_log) > 0:
        # Re-sort classes by median for log plot
        median_per_class_log = filtered_df_log.groupby('class_name_str')[feature].median().sort_values(ascending=False)
        sorted_classes_log = median_per_class_log.index.tolist()
        
        # Create figure with scientific formatting
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        # Create violin plot with log scale
        sns.violinplot(
            x='class_name_str',
            y=feature,
            data=filtered_df_log,
            order=sorted_classes_log,
            hue='class_name_str',
            palette=available_palette if available_palette else palette,
            inner='quartile',
            width=0.7,
            linewidth=1.5,
            ax=ax,
            legend=False
        )
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Scientific formatting: box off (remove top and right spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        # Format axes with log scale notation (using LaTeX for proper subscript rendering)
        ax.set_xlabel('Class Name', fontsize=14, fontweight='normal')
        unit = 'µm²' if feature == 'Area' else 'µm' if feature in pixel_features else ''
        # Use LaTeX notation for log scale to avoid font issues
        if unit:
            ylabel = f'$\\log_{{10}}$({feature} ({unit}))'
        else:
            ylabel = f'$\\log_{{10}}$({feature})'
        ax.set_ylabel(ylabel, fontsize=14, fontweight='normal')
        
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Remove title for publication (clean look)
        plt.tight_layout()
        
        # Save log-scale plot
        plot_path_log = os.path.join(output_dir, f'violin_{feature}_log.png')
        plt.savefig(plot_path_log, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Log-scale violin plot saved to {plot_path_log}")
    else:
        print(f"  Warning: No positive values found for {feature}, skipping log-scale plot")
    
    return plot_path


def plot_mean_std_and_cv(
    csv_directory, palette, output_svg_dir=None,
    numerical_features=None, figsize=(10, 6), dpi=500
):
    """
    Plot mean ± standard deviation and coefficient of variation from saved statistics CSV files.

    Parameters:
        csv_directory (str): Directory containing the CSV files.
        palette (dict): Mapping from class names to colors.
        output_svg_dir (str): Optional directory for SVG output (default: csv_directory/svg_plots).
        numerical_features (list): List of features to plot (default: None, uses all CSV files found).
        figsize (tuple): Figure size (width, height) in inches (default: (10, 6)).
        dpi (int): Resolution for saved figures (default: 500).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if output_svg_dir is None:
        output_svg_dir = os.path.join(csv_directory, 'svg_plots')
    os.makedirs(output_svg_dir, exist_ok=True)

    if numerical_features is None:
        # Find all CSV files in directory
        csv_files = [f for f in os.listdir(csv_directory) if f.startswith('stats_') and f.endswith('.csv')]
        numerical_features = [f.replace('stats_', '').replace('.csv', '') for f in csv_files]

    for feature in numerical_features:
        csv_path = os.path.join(csv_directory, f'stats_{feature}.csv')
        if not os.path.exists(csv_path):
            print(f"CSV file for {feature} not found at {csv_path}. Skipping.")
            continue

        # Read the saved statistics
        stats_df = pd.read_csv(csv_path)

        # Determine unit
        pixel_features = ['Area', 'Perimeter', 'maximum_radius', 'mean_radius',
                          'median_radius', 'minor_axis_length', 'major_axis_length']
        unit = 'µm²' if feature == 'Area' else 'µm' if feature in pixel_features else ''

        # Plot Mean ± Standard Deviation
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        for _, row in stats_df.iterrows():
            class_name = row['class_name_str']
            # Get color from palette, ensuring it's a valid color
            color = palette.get(class_name, '#000000')
            # If palette contains hex colors, use them directly; otherwise convert
            if isinstance(color, str) and color.startswith('#'):
                color = color
            elif isinstance(color, (list, tuple)) and len(color) == 3:
                # Convert RGB tuple to hex if needed
                color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
            
            ax.errorbar(
                class_name,
                row['mean'],
                yerr=row['std'],
                fmt='o',
                capsize=18,
                linestyle='none',
                markersize=18,
                color=color,
                label=f'{class_name}'
            )
        
        ax.set_xlabel('Class Name', fontsize=18)
        ylabel = f'{feature} ({unit})' if unit else feature
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(f'Mean {feature} with Standard Deviation', fontsize=14)
        plt.xticks(rotation=60, fontsize=18)
        
        # Scientific formatting: box off
        sns.despine()
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.tick_params(axis='both', which='both', width=2)
        plt.tick_params(axis='y', labelsize=18)
        plt.legend(fontsize=18, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        svg_path = os.path.join(output_svg_dir, f'mean_{feature}_with_std.svg')
        plt.savefig(svg_path, format='svg', dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Mean ± Std plot saved to {svg_path}")

        # Plot Coefficient of Variation (CV) as bar plot
        plt.figure(figsize=figsize)
        ax = plt.gca()
        
        # Get colors for each class with proper color handling
        class_names_list = stats_df['class_name_str'].values
        colors_list = []
        for name in class_names_list:
            color = palette.get(name, '#000000')
            # Handle different color formats
            if isinstance(color, str) and color.startswith('#'):
                colors_list.append(color)
            elif isinstance(color, (list, tuple)) and len(color) == 3:
                # Convert RGB tuple to hex
                colors_list.append('#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2])))
            else:
                colors_list.append('#000000')
        
        # Create bar plot
        bars = ax.bar(
            class_names_list,
            stats_df['coef_var'].values,
            color=colors_list,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        
        ax.set_xlabel('Class Name', fontsize=18)
        ax.set_ylabel('Coefficient of Variation', fontsize=18)
        ax.set_title(f'Coefficient of Variation of {feature}', fontsize=14)
        plt.xticks(rotation=60, ha='right', fontsize=18)
        
        # Scientific formatting: box off
        sns.despine()
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.tick_params(axis='both', which='both', width=2)
        plt.tick_params(axis='y', labelsize=18)
        plt.tight_layout()
        
        cv_svg_path = os.path.join(output_svg_dir, f'cv_{feature}.svg')
        plt.savefig(cv_svg_path, format='svg', dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Coefficient of Variation plot saved to {cv_svg_path}")


def add_log_scale_violin_plot(
    filtered_df, feature, sorted_classes, palette, output_dir,
    pixel_features, figsize=(12, 8), dpi=300
):
    """
    Create and save a log-scale version of a violin plot.
    
    Parameters:
        filtered_df (pd.DataFrame): Filtered dataframe with class_name_str and feature
        feature (str): Feature name
        sorted_classes (list): List of class names in sorted order
        palette (dict): Color palette mapping class names to colors
        output_dir (str): Output directory
        pixel_features (list): List of pixel-based features for unit conversion
        figsize (tuple): Figure size
        dpi (int): Resolution
    
    Returns:
        str: Path to saved log-scale plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Filter out zero/negative values for log scale
    filtered_df_log = filtered_df[filtered_df[feature] > 0].copy()
    
    if len(filtered_df_log) == 0:
        print(f"  Warning: No positive values found for {feature}, skipping log-scale plot")
        return None
    
    # Re-sort classes by median for log plot
    median_per_class_log = filtered_df_log.groupby('class_name_str')[feature].median().sort_values(ascending=False)
    sorted_classes_log = median_per_class_log.index.tolist()
    
    sns.set(style="ticks")
    plt.figure(figsize=figsize)
    
    # Create violin plot with log scale
    ax = sns.violinplot(
        x='class_name_str',
        y=feature,
        data=filtered_df_log,
        order=sorted_classes_log,
        hue='class_name_str',
        palette=palette,
        inner='quartile',
        width=0.35,
        linewidth=2.5,
        legend=False
    )
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Plot shifted data points
    shift = 0.5
    for i, class_name in enumerate(sorted_classes_log):
        subset = filtered_df_log[filtered_df_log['class_name_str'] == class_name]
        if len(subset) > 0:
            x = np.random.normal(loc=i, scale=0.05, size=len(subset)) + shift
            scatter_color = palette.get(class_name, 'black')
            plt.scatter(x, subset[feature], color=scatter_color, s=0.5, alpha=0.3)
    
    sns.despine()
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f'Violin Plot of {feature} by Class (Log Scale)', fontsize=20)
    plt.xlabel('Class Name', fontsize=20)
    
    # Scientific notation for y-axis label with log scale indication (using LaTeX)
    unit = "µm²" if feature == "Area" else "µm" if feature in pixel_features else ""
    if unit:
        ylabel = f'$\\log_{{10}}$({feature} ({unit}))'
    else:
        ylabel = f'$\\log_{{10}}$({feature})'
    plt.ylabel(ylabel, fontsize=20)
    plt.tight_layout()
    
    plot_path_log = os.path.join(output_dir, f'violin_{feature}_log.png')
    plt.savefig(plot_path_log, dpi=dpi)
    plt.close()
    print(f"Log-scale violin plot saved to {plot_path_log}")
    
    return plot_path_log


def invert_D(D):
    """
    Invert displacement field D.
    In MATLAB, this typically negates the displacement field.
    
    Parameters:
        D (np.ndarray): Displacement field of shape (H, W, 2) where D[:,:,0] is x-displacement
                       and D[:,:,1] is y-displacement
    
    Returns:
        np.ndarray: Inverted displacement field
    """
    # Invert by negating the displacement (standard approach)
    D_inv = -D.copy()
    return D_inv


def register_cell_coordinates_pointbased_monkey_centroids(pth0, pthcoords, scale):
    """
    Register cell coordinates using point-based registration with rough and elastic transformations.
    Python version of the MATLAB function.
    
    Parameters:
        pth0 (str): Path containing images where registration was calculated
        pthcoords (str): Path to cell coordinates (.mat files)
        scale (float): Scale between coordinate images and registration images (>1)
    
    Returns:
        None (saves registered coordinates to output directory)
    """
    import cv2
    
    # Define paths using pth0
    pthimG = os.path.join(pth0, 'registered')
    pthimE = os.path.join(pthimG, 'elastic registration')
    outpth = os.path.join(pthimE, 'cell_coordinates_registered', 'centroids_coords')
    os.makedirs(outpth, exist_ok=True)
    
    datapth = os.path.join(pthimE, 'save_warps')
    pthD = os.path.join(datapth, 'D')
    
    # Set up padding information and find registered images
    matlist = [f for f in os.listdir(pthcoords) if f.endswith('.mat')]
    if not matlist:
        print(f"No .mat files found in {pthcoords}")
        return
    
    # Determine image extension
    first_mat_name = os.path.splitext(matlist[0])[0]
    test_jpg = os.path.join(pthimE, f'{first_mat_name}.jpg')
    test_tif = os.path.join(pthimE, f'{first_mat_name}.tif')
    
    if os.path.exists(test_jpg):
        tp = '.jpg'
        tp2 = '.tif'
    else:
        tp = '.tif'
        tp2 = '.jpg'
    
    # Load padding information from first file
    first_mat_path = os.path.join(datapth, matlist[0])
    if not os.path.exists(first_mat_path):
        print(f"First mat file not found: {first_mat_path}")
        return
    
    first_data = loadmat(first_mat_path)
    padall = first_data.get('padall', np.array([[0]]))[0, 0] if 'padall' in first_data else 0
    szz = first_data.get('szz', np.array([[0, 0]])).flatten() if 'szz' in first_data else np.array([0, 0])
    
    szz2 = szz + (2 * padall)
    szz3 = np.round(szz2 * scale).astype(int)
    pad2 = 1
    
    # Register coordinates for each image
    for kk, matfile in enumerate(matlist, 1):
        matfile_path = os.path.join(pthcoords, matfile)
        D_path = os.path.join(pthD, matfile)
        out_path = os.path.join(outpth, matfile)
        
        if not os.path.exists(D_path):
            continue
        if os.path.exists(out_path):
            continue
        
        # Load coordinates
        try:
            coord_data = loadmat(matfile_path)
            xy = coord_data.get('xy', np.array([]))
            if xy.size == 0:
                print(f"No xy coordinates found in {matfile}")
                continue
            xy = xy.astype(float)
            xy0 = xy.copy()
        except Exception as e:
            print(f"Error loading {matfile}: {e}")
            continue
        
        # Scale coordinates
        xy = xy / scale
        
        # Check unregistered image (for visualization on first and middle images)
        if kk == 1 or kk == (len(matlist) // 2) + 5:
            img_path_tp = os.path.join(pth0, f'{os.path.splitext(matfile)[0]}{tp}')
            img_path_tp2 = os.path.join(pth0, f'{os.path.splitext(matfile)[0]}{tp2}')
            if os.path.exists(img_path_tp):
                im0 = cv2.imread(img_path_tp)
            elif os.path.exists(img_path_tp2):
                im0 = cv2.imread(img_path_tp2)
            else:
                im0 = None
            
            if im0 is not None:
                plt.figure(31)
                plt.imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
                plt.scatter(xy[:, 0], xy[:, 1], marker='*', c='y', s=10)
                plt.title(f'Unregistered: {matfile}')
                plt.show()
        
        # Rough register points
        xy = xy + padall
        
        if pad2 == 1:
            # Get image dimensions
            img_path_tp = os.path.join(pth0, f'{os.path.splitext(matfile)[0]}{tp}')
            img_path_tp2 = os.path.join(pth0, f'{os.path.splitext(matfile)[0]}{tp2}')
            
            if os.path.exists(img_path_tp):
                img_info = cv2.imread(img_path_tp)
                if img_info is not None:
                    a = np.array([img_info.shape[0], img_info.shape[1]])  # [Height, Width]
                else:
                    a = np.array([szz[0], szz[1]])
            elif os.path.exists(img_path_tp2):
                img_info = cv2.imread(img_path_tp2)
                if img_info is not None:
                    a = np.array([img_info.shape[0], img_info.shape[1]])
                else:
                    a = np.array([szz[0], szz[1]])
            else:
                a = np.array([szz[0], szz[1]])
            
            szim = np.array([szz[0] - a[0], szz[1] - a[1]])
            szA = np.floor(szim / 2).astype(int)
            szA = np.array([szA[1], szA[0]])  # [x, y] format
            xy = xy + szA
        
        # Apply rough registration (affine transform)
        try:
            warp_data = loadmat(os.path.join(datapth, matfile))
            tform_matrix = warp_data.get('tform', None)
            cent = warp_data.get('cent', np.array([[0], [0]]))
            # Handle different cent formats (MATLAB can store as column vector or array)
            if isinstance(cent, np.ndarray):
                cent = cent.flatten()
            if len(cent) == 0 or cent.size == 0:
                cent = np.array([0, 0])
            elif len(cent) == 1:
                cent = np.array([cent[0], cent[0]])
            
            szz_warp = warp_data.get('szz', szz)
            if isinstance(szz_warp, np.ndarray) and szz_warp.ndim > 1:
                szz_warp = szz_warp.flatten()
            
            if tform_matrix is not None:
                # Transform points: xyr = transformPointsForward(tform, xy - cent) + cent
                xy_centered = xy - cent
                
                # Apply affine transformation
                # tform_matrix is typically a 3x3 or 2x3 matrix
                if isinstance(tform_matrix, np.ndarray) and tform_matrix.ndim == 2:
                    if tform_matrix.shape == (3, 3):
                        # 3x3 homogeneous transformation matrix
                        tform_2x3 = tform_matrix[:2, :]
                        xy_homogeneous = np.column_stack([xy_centered, np.ones(len(xy_centered))])
                        xyr = (tform_2x3 @ xy_homogeneous.T).T
                    elif tform_matrix.shape == (2, 3):
                        # 2x3 affine transformation matrix
                        xy_homogeneous = np.column_stack([xy_centered, np.ones(len(xy_centered))])
                        xyr = (tform_matrix @ xy_homogeneous.T).T
                    else:
                        # Unknown format, try direct application
                        print(f"Warning: Unknown tform shape {tform_matrix.shape} for {matfile}")
                        xyr = xy_centered.copy()
                else:
                    # tform is not a standard matrix, skip transformation
                    print(f"Warning: tform is not a 2D array for {matfile}")
                    xyr = xy_centered.copy()
                
                xyr = xyr + cent
                
                # Filter points within bounds
                cc = (xyr[:, 0] > 1) & (xyr[:, 1] > 1) & (xyr[:, 0] < szz2[1]) & (xyr[:, 1] < szz2[0])
                xyr = xyr[cc, :]
            else:
                xyr = xy
        except Exception as e:
            print(f'No tform for {matfile}: {e}')
            xyr = xy
        
        # Visualization after rough registration
        if kk == 1 or kk == (len(matlist) // 2) + 5:
            img_path_tp = os.path.join(pthimG, f'{os.path.splitext(matfile)[0]}{tp}')
            img_path_tp2 = os.path.join(pthimG, f'{os.path.splitext(matfile)[0]}{tp2}')
            if os.path.exists(img_path_tp):
                imG = cv2.imread(img_path_tp)
            elif os.path.exists(img_path_tp2):
                imG = cv2.imread(img_path_tp2)
            else:
                imG = None
            
            if imG is not None:
                plt.figure(32)
                plt.imshow(cv2.cvtColor(imG, cv2.COLOR_BGR2RGB))
                plt.scatter(xyr[:, 0], xyr[:, 1], marker='*', c='y', s=10)
                plt.title(f'Rough registered: {matfile}')
                plt.show()
        
        # Elastic register points
        try:
            xytmp = xyr * scale
            
            # Load displacement field
            D_data = loadmat(D_path)
            D = D_data.get('D', None)
            
            if D is None:
                raise ValueError(f"No D field found in {matfile}")
            
            # Resize D (standard is 5x)
            D = resize(D, (D.shape[0] * 5, D.shape[1] * 5), order=1, preserve_range=True, anti_aliasing=False)
            
            # Invert displacement field
            Dnew = invert_D(D)
            Dnew = resize(Dnew, szz3, order=1, preserve_range=True, anti_aliasing=False) * scale
            
            D2a = Dnew[:, :, 0]  # x-displacement
            D2b = Dnew[:, :, 1]  # y-displacement
            
            # Get displacement at point locations
            pp = np.round(xytmp).astype(int)
            
            # Ensure points are within bounds
            valid_mask = (pp[:, 1] >= 0) & (pp[:, 1] < szz3[0]) & (pp[:, 0] >= 0) & (pp[:, 0] < szz3[1])
            pp_valid = pp[valid_mask]
            xytmp_valid = xytmp[valid_mask]
            
            if len(pp_valid) > 0:
                # Get displacement values at point locations
                # Note: D2a and D2b are indexed as [row, col] = [y, x]
                xmove = np.column_stack([
                    D2a[pp_valid[:, 1], pp_valid[:, 0]],
                    D2b[pp_valid[:, 1], pp_valid[:, 0]]
                ])
                
                xye_valid = xytmp_valid + xmove
                
                # Reconstruct full array with NaN for invalid points
                xye = np.full((len(xytmp), 2), np.nan)
                xye[valid_mask] = xye_valid
                xye = xye[~np.isnan(xye).any(axis=1)]  # Remove NaN rows
            else:
                xye = xyr
        except Exception as e:
            print(f'Error in elastic registration for {matfile}: {e}')
            xye = xyr
        
        # Visualization after elastic registration
        if kk == 1 or kk == (len(matlist) // 2) + 5:
            img_path_tp = os.path.join(pthimE, f'{os.path.splitext(matfile)[0]}{tp}')
            img_path_tp2 = os.path.join(pthimE, f'{os.path.splitext(matfile)[0]}{tp2}')
            if os.path.exists(img_path_tp):
                imE = cv2.imread(img_path_tp)
            elif os.path.exists(img_path_tp2):
                imE = cv2.imread(img_path_tp2)
            else:
                imE = None
            
            if imE is not None:
                plt.figure(33)
                plt.imshow(cv2.cvtColor(imE, cv2.COLOR_BGR2RGB))
                if len(xye) > 0:
                    plt.scatter(xye[:, 0] / scale, xye[:, 1] / scale, marker='*', c='y', s=10)
                plt.title(f'Elastic registered: {matfile}')
                plt.show()
                plt.pause(1)
        
        # Save registered points (in the resolution of input points)
        savemat(out_path, {'xyr': xyr, 'xye': xye})
        print(f'{kk}/{len(matlist)}: {matfile} - Input: {len(xy0)}, Output: {len(xye)} points')


def register_centroids_from_pkl_files(
    pth0, pthcoords_pkl, scale, outpthpkl_registered,
    pthcoords_mat=None, temp_mat_dir=None
):
    """
    Register centroids from pkl files (dataframes) and append registered coordinates.
    
    This function:
    1. Loads pkl files containing dataframes with 'Centroid_x' and 'Centroid_y'
    2. Extracts centroids and saves them as temporary .mat files (or uses existing)
    3. Calls register_cell_coordinates_pointbased_monkey_centroids to register them
    4. Loads registered coordinates and appends them to dataframes
    5. Saves updated dataframes to new folder
    
    Parameters:
        pth0 (str): Path containing images where registration was calculated
        pthcoords_pkl (str): Path to pkl files with centroids (dataframes)
        scale (float): Scale between coordinate images and registration images (>1)
        outpthpkl_registered (str): Output path for registered pkl files
        pthcoords_mat (str, optional): Path to existing .mat coordinate files. 
                                      If None, will create temporary .mat files from pkl
        temp_mat_dir (str, optional): Temporary directory for .mat files if creating them
    
    Returns:
        None
    """
    import tempfile
    import shutil
    
    os.makedirs(outpthpkl_registered, exist_ok=True)
    
    # Get list of pkl files
    pkl_files = get_sorted_files(pthcoords_pkl, '.pkl')
    if not pkl_files:
        print(f"No .pkl files found in {pthcoords_pkl}")
        return
    
    # If pthcoords_mat is provided, use it; otherwise create temp directory
    use_temp = pthcoords_mat is None
    if use_temp:
        if temp_mat_dir is None:
            temp_mat_dir = tempfile.mkdtemp(prefix='centroid_registration_')
        pthcoords_mat = temp_mat_dir
        os.makedirs(pthcoords_mat, exist_ok=True)
        print(f"Creating temporary .mat files in {pthcoords_mat}")
    
    # Step 1: Extract centroids from pkl files and save as .mat (if needed)
    if use_temp:
        print("Extracting centroids from pkl files...")
        for pkl_file in pkl_files:
            pkl_path = os.path.join(pthcoords_pkl, pkl_file) if not os.path.isabs(pkl_file) else pkl_file
            mat_name = os.path.splitext(os.path.basename(pkl_file))[0] + '.mat'
            mat_path = os.path.join(pthcoords_mat, mat_name)
            
            if os.path.exists(mat_path) and not use_temp:
                continue  # Skip if mat file already exists
            
            try:
                df = pd.read_pickle(pkl_path)
                # Extract centroids
                if 'Centroid_x' in df.columns and 'Centroid_y' in df.columns:
                    xy = df[['Centroid_x', 'Centroid_y']].values
                    savemat(mat_path, {'xy': xy})
                else:
                    print(f"Warning: {pkl_file} does not have Centroid_x/Centroid_y columns")
            except Exception as e:
                print(f"Error processing {pkl_file}: {e}")
                continue
    
    # Step 2: Register coordinates using the MATLAB-equivalent function
    print("Registering coordinates...")
    register_cell_coordinates_pointbased_monkey_centroids(pth0, pthcoords_mat, scale)
    
    # Step 3: Load registered coordinates and update dataframes
    pthimE = os.path.join(pth0, 'registered', 'elastic registration')
    registered_coords_path = os.path.join(pthimE, 'cell_coordinates_registered', 'centroids_coords')
    
    print("Updating dataframes with registered coordinates...")
    for pkl_file in pkl_files:
        pkl_path = os.path.join(pthcoords_pkl, pkl_file) if not os.path.isabs(pkl_file) else pkl_file
        mat_name = os.path.splitext(os.path.basename(pkl_file))[0] + '.mat'
        registered_mat_path = os.path.join(registered_coords_path, mat_name)
        output_pkl_path = os.path.join(outpthpkl_registered, pkl_file)
        
        if not os.path.exists(registered_mat_path):
            print(f"Warning: Registered coordinates not found for {mat_name}, copying original pkl")
            try:
                df = pd.read_pickle(pkl_path)
                df.to_pickle(output_pkl_path)
            except:
                pass
            continue
        
        try:
            # Load original dataframe
            df = pd.read_pickle(pkl_path)
            
            # Load registered coordinates
            registered_data = loadmat(registered_mat_path)
            xyr = registered_data.get('xyr', np.array([]))
            xye = registered_data.get('xye', np.array([]))
            
            # Use elastic registered coordinates (xye) if available, otherwise use rough (xyr)
            if xye.size > 0:
                registered_xy = xye
            elif xyr.size > 0:
                registered_xy = xyr
            else:
                print(f"Warning: No registered coordinates found for {mat_name}")
                df.to_pickle(output_pkl_path)
                continue
            
            # Match registered coordinates to dataframe rows
            # Since registration may filter out some points, we need to match them
            original_xy = df[['Centroid_x', 'Centroid_y']].values
            
            # If sizes match, assume direct correspondence
            if len(registered_xy) == len(original_xy):
                df['Centroid_x_registered'] = registered_xy[:, 0]
                df['Centroid_y_registered'] = registered_xy[:, 1]
            else:
                # Use nearest neighbor matching for filtered points
                from scipy.spatial.distance import cdist
                if len(registered_xy) > 0:
                    # Find closest matches
                    distances = cdist(original_xy, registered_xy)
                    min_indices = np.argmin(distances, axis=1)
                    min_distances = np.min(distances, axis=1)
                    
                    # Only assign if distance is reasonable (within 10 pixels)
                    threshold = 10.0
                    valid_mask = min_distances < threshold
                    
                    df['Centroid_x_registered'] = np.nan
                    df['Centroid_y_registered'] = np.nan
                    df.loc[valid_mask, 'Centroid_x_registered'] = registered_xy[min_indices[valid_mask], 0]
                    df.loc[valid_mask, 'Centroid_y_registered'] = registered_xy[min_indices[valid_mask], 1]
                else:
                    df['Centroid_x_registered'] = np.nan
                    df['Centroid_y_registered'] = np.nan
            
            # Save updated dataframe
            df.to_pickle(output_pkl_path)
            print(f"Saved registered dataframe: {pkl_file} ({len(df)} cells, {np.sum(~df['Centroid_x_registered'].isna())} registered)")
            
        except Exception as e:
            print(f"Error updating {pkl_file}: {e}")
            continue
    
    # Clean up temporary directory if created
    if use_temp and temp_mat_dir and os.path.exists(temp_mat_dir):
        try:
            shutil.rmtree(temp_mat_dir)
            print(f"Cleaned up temporary directory: {temp_mat_dir}")
        except:
            print(f"Warning: Could not remove temporary directory: {temp_mat_dir}")
    
    print("Registration complete!")



