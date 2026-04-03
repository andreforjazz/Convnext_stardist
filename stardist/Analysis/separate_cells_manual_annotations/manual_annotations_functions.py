import pickle
import os
import h5py
import numpy as np
from scipy.io import savemat


def add_newlabel_to_df_pkl(pthpkl, cluster1_pth, outpklpth, target_class=None):
    """
    Converts .pkl files containing pandas DataFrames to .mat files.

    Args:
    - src (str): Source directory containing .pkl files.
    - outpth (str): Destination directory where .mat files will be saved.
    - dfs (list): List of .pkl filenames (as strings) to process.

    Returns:
    - None
    """
    pkl_pthlist = [os.path.join(pthpkl, f) for f in os.listdir(pthpkl) if f.endswith(".pkl")]
    # pkl_classes_pthlist = [os.path.join(json_path, os.path.basename(f)[:-8] + '.pkl') for f in pkl_pthlist]
    pthmatfiles = os.path.join(outpklpth, 'mat')
    os.makedirs(pthmatfiles, exist_ok=True)

    print(f'saving here: {outpklpth}')
    for i, dfnm in enumerate(pkl_pthlist):
        nm = os.path.splitext(os.path.basename(dfnm))[0]

        print(f"Saving mat file {nm}  {i + 1}/{len(pkl_pthlist)}")
        dst = os.path.join(outpklpth, nm + '.pkl')
        matnm = os.path.join(cluster1_pth, nm + '.mat')

        if os.path.exists(dst):
            print("MAT file already exists, skipping the file ID {}".format(nm))
            continue

        with open(os.path.join(pthpkl, dfnm), 'rb') as f:
            df = pickle.load(f)

        # Load the corresponding .mat file and extract the cluster1 variable using h5py
        with h5py.File(os.path.join(matnm), 'r') as mat_data:
            # Read and flatten the index list
            idx_list = np.array(mat_data['cluster1']).flatten()
            print("Extracted cluster1 data")

        df['cluster1'] = idx_list

        # Store the original class_name values in a new column
        df['previous_class_name'] = df['class_name']

        # Find the maximum class_name value
        max_class_name = df['class_name'].max()

        # Determine the new class number
        new_class_X = max_class_name + 1

        # Update rows based on the target_class or cluster1 == 1
        if target_class is not None:
            # Subdivide the target_class based on cluster1 values
            df.loc[(df['class_name'] == target_class) & (df['cluster1'] == 1), 'class_name'] = new_class_X
        else:
            # Update all rows with cluster1 == 1 to the new class number
            df.loc[df['cluster1'] == 1, 'class_name'] = new_class_X

        # Add a new column to track the new class_name
        df['new_class_name'] = df['class_name']
        df = df.drop(columns=['class_name'])
        df = df.rename(columns={'new_class_name': 'class_name'})

        # Save the updated DataFrame to a new .pkl file
        new_pkl_path = os.path.join(outpklpth, f"{nm}.pkl")
        with open(new_pkl_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"Saved updated DataFrame")

        # Optionally, save the updated DataFrame to a .mat file
        new_mat_path = os.path.join(pthmatfiles, f"{nm}.mat")
        col_names = df.columns.tolist()
        df_array = df.to_numpy().astype(np.float32)
        savemat(new_mat_path, {'features': df_array, 'feature_names': col_names})
        print(f"Saved matfile")


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


def get_ds_data(segmentation_data, ds):
    data_list = []
    for data in segmentation_data:
        centroid = data['centroid'][0]
        contour = data['contour'][0]

        # print(centroid)
        # print(contour)

        ds_centroid = [int(c / ds) for c in centroid]
        ds_contour = [[value / ds for value in sublist] for sublist in contour]
        ds_contour = [[round(x, 2), round(y, 2)] for x, y in zip(ds_contour[0], ds_contour[1])]
        # ds_contour = ds_contour[0:-1:4]  # make shape have 8 points instead of 32

        # print(ds_centroid)
        # print(ds_contour)

        dat = [ds_centroid, ds_contour]
        data_list.append(dat)
    return data_list


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
from skimage.transform import rescale
from scipy.io import loadmat
import cv2
import re
import pickle
from scipy.io import savemat
import seaborn as sns
import matplotlib.pyplot as plt


def make_geojson_contours_with_classes_immune(
        out_pth_json,
        ds_amt=1,
        ds_mask=1 / 0.4416,
        class_names=None,
        class_colors=None,
        gj=1,
        pth_pkl_classes=None
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

    out_pth_contours = os.path.join(out_pth_json, 'geojsons', '32_polys_20x_labels_immune')
    os.makedirs(out_pth_contours, exist_ok=True)
    json_pth_list = [f for f in os.listdir(out_pth_json) if f.endswith('.json')]

    for p, file in enumerate(json_pth_list):
        nm = os.path.basename(file)
        file_name, _ = os.path.splitext(nm)
        new_fn = os.path.join(out_pth_contours, file_name + '.geojson')
        print(f'{p + 1} / {len(json_pth_list)}')
        print(nm)

        # skip to next image if new_fn already exists
        if os.path.exists(new_fn):
            print(f'Skipping {new_fn}')
            continue

        # load pickle file with the class names
        pth_pkl_classes_nm = os.path.join(pth_pkl_classes, file_name + '.pkl')
        with open(pth_pkl_classes_nm, 'rb') as f:
            df = pickle.load(f)

        # segmented_image = cv2.imread(os.path.join(pth10xsegmented, file_name + '.tif'), cv2.IMREAD_UNCHANGED)

        if not os.path.exists(new_fn):
            with open(os.path.join(out_pth_json, file)) as f:
                segmentation_data = json.load(f)

            # Extract downsampled data (centroids and contours)
            data_list = get_ds_data(segmentation_data, ds_amt)

            GEOdata = []

            for j, (centroid, contour) in tqdm(enumerate(data_list), total=len(data_list), desc="Processing contours"):
                y, x = int(round(centroid[0])), int(round(centroid[1]))

                # Get the label from the dataframe
                label = df.iloc[j]['class_name']

                # Determine class based on the semantic image
                # label = segmented_image[round(y / ds_mask), round(x / ds_mask)]

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