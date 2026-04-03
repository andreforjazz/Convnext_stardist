import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, csgraph
import os
from tifffile import imread
import matplotlib.pyplot as plt
from tifffile import imread
import os
from matplotlib import pyplot as plt
import json
from scipy.io import loadmat
from scipy.io import savemat
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, csgraph
import pandas as pd

def get_geojson_centroids(pth):
    """Returns a list of xy coordinates of centroids from contours in a geojson file"""
    geo_data = json.load(open(pth))
    centroids_ann = []

    for i in range(len(geo_data['features'])):
        data = geo_data['features'][i]
        coords = data['geometry']['coordinates'][0]
        x_cent = 0
        y_cent = 0
        for pair in coords:
            x_cent += pair[1]
            y_cent += pair[0]
        x_cent /= len(coords)
        y_cent /= len(coords)
        centroids_ann.append([x_cent, y_cent])
    
    centroids_ann = np.array(centroids_ann)
    
    return centroids_ann

def get_json_centroids(pth):
    """Returns a list of centroids and contours from a stardist custom output json file"""
    segmentation_data = json.load(open(pth))

    centroids = np.array([nuc['centroid'][0] for nuc in segmentation_data])
    contours = np.array([nuc['contour'] for nuc in segmentation_data])

    return centroids, contours

def colocalize_points(points_a: np.ndarray, points_b: np.ndarray, r: int):
    """ Find pairs that minimize global distance. Filters out anything outside radius `r` """

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(points_b)
    distances, b_indices = neigh.radius_neighbors(points_a, radius=r)

    # flatten and get indices for A. This will also drop points in A with no matches in range
    d_flat = np.hstack(distances) + 1
    b_flat = np.hstack(b_indices)
    a_flat = np.array([i for i, neighbors in enumerate(distances) for n in neighbors])

    # filter out A points that cannot be matched
    sm = csr_matrix((d_flat, (a_flat, b_flat)))
    a_matchable = csgraph.maximum_bipartite_matching(sm, perm_type='column')
    sm_filtered = sm[a_matchable != -1]

    # now run the distance minimizing matching
    row_match, col_match = csgraph.min_weight_full_bipartite_matching(sm_filtered)
    return row_match, col_match

def adjust_contours_match(contours_matched, x, y):

    contours_matched_adjusted = []
    for i in range(len(contours_matched)):
        contour = contours_matched[i][0]
        x_coords = contour[0]
        y_coords = contour[1]
        x_coords = [point-x for point in x_coords]
        y_coords = [point-y for point in y_coords]
        
        shape = list(zip(x_coords, y_coords))
        contours_matched_adjusted.append(shape)
    
    return contours_matched_adjusted

def plot_results(ndpi_pth, cropping, centroids, contours, matching):
    crop_x, crop_y, tile_size = cropping

    indices_not_matched = np.setdiff1d(range(len(centroids)), matching[1])
    indices_matched = matching[1]

    centroids_matched = centroids[indices_matched]
    adj_centroids_matched = [[pair[0] - crop_y, pair[1] - crop_x] for pair in centroids_matched]

    contours_matched = contours[indices_matched]
    contours_not_matched = contours[indices_not_matched]

    contours_matched_adjusted = adjust_contours_match(contours_matched, crop_x, crop_y)
    contours_not_matched_adjusted = adjust_contours_match(contours_not_matched, crop_x, crop_y)

    # flip x and y   >:(   # <- face
    reversed_contours = [[(y, x) for x, y in polygon] for polygon in contours_matched_adjusted]
    reversed_contours_negative = [[(y, x) for x, y in polygon] for polygon in contours_not_matched_adjusted]

    fig, ax = plt.subplots(figsize=(16, 8))

    img = imread(os.path.join(ndpi_pth))

    #print([crop_x, crop_x+tile_size, crop_y, crop_y+tile_size])
    ax.imshow(img[crop_x:crop_x+tile_size,crop_y:crop_y+tile_size])
    ax.set_axis_off()

    # Plot each reversed polygon on the same image
    for polygon in reversed_contours:
        x_coords, y_coords = zip(*polygon)
        x_coords = list(x_coords) + [x_coords[0]]  # Close the polygon
        y_coords = list(y_coords) + [y_coords[0]]  # Close the polygon

        color = 'yellow'

        skip = False
        for x in x_coords:
            if x < 0 or x > (tile_size - 1):
                skip = True
                break
        for y in y_coords:
            if y < 0 or y > (tile_size - 1):
                skip = True
                break
        
        if not skip:

            ax.plot(x_coords, y_coords, alpha=0.3, color=color)
            ax.fill(x_coords, y_coords, alpha=0.3, color=color)  # Fill the polygon
        
    # Plot each reversed polygon on the same image
    for polygon in reversed_contours_negative:
        x_coords, y_coords = zip(*polygon)
        x_coords = list(x_coords) + [x_coords[0]]  # Close the polygon
        y_coords = list(y_coords) + [y_coords[0]]  # Close the polygon

        color = 'red'
        
        skip = False
        for x in x_coords:
            if x < 0 or x > (tile_size - 1):
                skip = True
                break
        for y in y_coords:
            if y < 0 or y > (tile_size - 1):
                skip = True
                break
        
        if not skip:

            ax.plot(x_coords, y_coords, alpha=0.4, color=color)
            ax.fill(x_coords, y_coords, alpha=0.4, color=color)  # Fill the polygon

    # Set labels and title for the plot
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Qupath selected nuclei vs unselected')

    plt.show()

# def get_matched_inds(ndpi_pth, centroids, contours, matching):
#     centroids_all = [centroids[i] for i in centroids]
#     centroids_matched = [pair.tolist() for pair in matching[1]]
#
#     indices_not_matched = np.setdiff1d(matching[1], range(len(centroids)))
#     indices_matched = matching[1]
#
#     return indices_matched

def get_matched_inds(ndpi_pth, centroids, contours, matching):
    # Ensure that indices in matching[1] are within the valid range of centroids
    valid_indices = [i for i in matching[1] if i < len(centroids)]

    # If valid indices are found, proceed
    centroids_all = [centroids[i] for i in valid_indices]
    centroids_matched = [pair.tolist() for pair in valid_indices]

    # Handle the not matched case by finding the difference between all indices and valid ones
    indices_not_matched = np.setdiff1d(matching[1], range(len(centroids)))

    return centroids_all, centroids_matched, indices_not_matched
def save_json_data_from_selected(coords, points, out_pth, name):
    """Saves a json file with centroids and contours for StarDist output."""

    # new_fn = name[:-5] + '.json'
    new_fn = name + '.json'
    pthjson = os.path.join(out_pth, 'json')
    os.makedirs(pthjson, exist_ok=True)

    out_nm = os.path.join(pthjson, new_fn)

    if not os.path.exists(out_nm):

        json_data = []

        for i in range(len(points)):
            point = points[i]
            contour = coords[i]
            #print(contour)
            centroid = [int(point[0]), int(point[1])]  # TODO: FIX
            contour = [[coord for coord in xy] for xy in contour][0]

            # Create a new dictionary for each contour
            dict_data = {
                "centroid": [centroid],
                "contour": [contour]
            }

            json_data.append(dict_data)

        with open(out_nm,'w') as outfile:
            json.dump(json_data, outfile)
        print('Finished',new_fn)
    else:
        print(f'{os.path.basename(out_nm)} already exists, skipping...')


def show_selected_unselected_nuclei_zoomin(um_adjust=1, cropping_center=[5356, 2969], tile_sz=1024, geojson_pth_list=None,
                       ndpi_pth_list=None, json_pth_list=None):
    """
    Function to process NDPI tiles based on given cropping center, tile size, and file lists.

    Args:
    - um_adjust (float): Scaling factor for the resolution adjustment.
    - cropping_center (list): List containing x and y coordinates in microns.
    - tile_sz (int): Size of the cropping tile.
    - geojson_pth_list (list): List of geojson file paths.
    - ndpi_pth_list (list): List of NDPI file paths.
    - json_pth_list (list): List of json file paths.

    Returns:
    - None: The function processes and plots results for the first item in the lists.
    """

    if geojson_pth_list is None or ndpi_pth_list is None or json_pth_list is None:
        raise ValueError("File lists for geojson, ndpi, and json paths must be provided.")

    # Adjust cropping center based on resolution adjustment
    cropping_center = [int(i * um_adjust) for i in cropping_center]
    cropping = [cropping_center[1] - tile_sz // 2, cropping_center[0] - tile_sz // 2]
    cropping.append(tile_sz)  # Append the tile size to the cropping coordinates
    print(f"Cropping coordinates: {cropping}")

    # Process the first set of paths (change to process all by removing 'break' below)
    for i in range(len(geojson_pth_list)):
        ndpi_pth_f = ndpi_pth_list[i]
        geojson_file_f = geojson_pth_list[i]
        json_file_f = json_pth_list[i]

        # Get centroids from both geojson and json files
        centroids_ann = get_geojson_centroids(geojson_file_f)
        centroids_json, contours_json = get_json_centroids(json_file_f)

        # Perform colocalization of points
        matching = colocalize_points(centroids_ann, centroids_json, r=20)

        # Plot the results using the specified NDPI path and matching data
        plot_results(ndpi_pth_f, cropping, centroids_json, contours_json, matching)

        # Break after the first iteration (remove this to process all items)
        # break


def match_selection_with_stardistcontours(geojson_pth_list, ndpi_pth_list, json_pth_list, selection_path, r=20):
    """
    Process files to match centroids, annotate contours, and save the results as JSON.

    Args:
    - geojson_pth_list (list): List of geojson file paths.
    - ndpi_pth_list (list): List of NDPI file paths.
    - json_pth_list (list): List of json file paths.
    - selection_path (str): Directory path to save the selected annotations.
    - r (int, optional): Radius for colocalization matching. Defaults to 20.

    Returns:
    - None: The function processes files and saves the annotated results.
    """

    if not os.path.exists(selection_path):
        os.makedirs(selection_path)  # Ensure the output directory exists

    for i in range(len(geojson_pth_list)):
        ndpi_pth_f = ndpi_pth_list[i]
        geojson_file_f = geojson_pth_list[i]
        json_file_f = json_pth_list[i]

        # check if json file already exists in the json_file_f path
        if os.path.exists(json_file_f):
            print(f"Skipping {json_file_f} as it already exists.")
            continue

        # Get centroids and contours from StarDist output and selected nuclei
        centroids_ann = get_geojson_centroids(geojson_file_f)  # Nuclei centroids from geojson
        centroids_json, contours_json = get_json_centroids(json_file_f)  # StarDist output

        # Match centroids from selected nuclei with StarDist output
        matching = colocalize_points(centroids_ann, centroids_json, r=r)

        # Get matched indices, assume this could be a list of lists or tuples
        indices_matched = get_matched_inds(ndpi_pth_f, centroids_json, contours_json, matching)

        # Flatten indices_matched if it's a list of lists or tuples
        if isinstance(indices_matched[0], (list, tuple)):
            indices_matched = [item for sublist in indices_matched for item in sublist]

        # Ensure valid indices within bounds of centroids_json
        indices_matched = [i for i in indices_matched if isinstance(i, int) and i < len(centroids_json)]

        # Index the matched centroids and contours using valid indices
        centroids_annotated = np.array([centroids_json[i] for i in indices_matched])
        contours_annotated = np.array([contours_json[i] for i in indices_matched])

        # Prepare to save the data
        nm = os.path.basename(geojson_file_f).replace('.geojson', '')  # Get the file name without extension
        out_centroids = centroids_annotated.tolist()  # Convert to Python list for JSON serialization
        out_contours = contours_annotated.tolist()

        # Save JSON file with selected annotations
        save_json_data_from_selected(out_contours, out_centroids, selection_path, nm)


# def match_selection_with_stardistcontours(geojson_pth_list, ndpi_pth_list, json_pth_list, selection_path, r=20):
#     """
#     Process files to match centroids, annotate contours, and save the results as JSON.
#
#     Args:
#     - geojson_pth_list (list): List of geojson file paths.
#     - ndpi_pth_list (list): List of NDPI file paths.
#     - json_pth_list (list): List of json file paths.
#     - selection_path (str): Directory path to save the selected annotations.
#     - r (int, optional): Radius for colocalization matching. Defaults to 20.
#
#     Returns:
#     - None: The function processes files and saves the annotated results.
#     """
#
#     if not os.path.exists(selection_path):
#         os.makedirs(selection_path)  # Ensure the output directory exists
#
#     for i in range(len(geojson_pth_list)):
#         ndpi_pth_f = ndpi_pth_list[i]
#         geojson_file_f = geojson_pth_list[i]
#         json_file_f = json_pth_list[i]
#
#         # Get centroids and contours from StarDist output and selected nuclei
#         centroids_ann = get_geojson_centroids(geojson_file_f)
#         centroids_json, contours_json = get_json_centroids(json_file_f)
#
#         # Match centroids from selected nuclei with StarDist output
#         matching = colocalize_points(centroids_ann, centroids_json, r=r)
#         indices_matched = get_matched_inds(ndpi_pth_f, centroids_json, contours_json, matching)
#         centroids_annotated = centroids_json[indices_matched]
#         contours_annotated = contours_json[indices_matched]
#
#         # Prepare to save
#         nm = os.path.basename(geojson_file_f[:-3])  # Get the file name without extension
#         out_centroids = centroids_annotated.tolist()  # Convert to Python list for JSON
#         out_contours = contours_annotated.tolist()
#
#         # Save JSON file with selected annotations
#         save_json_data_from_selected(out_contours, out_centroids, selection_path, nm)


def convert_pkl_to_mat(pthpkl, pthpklmat):
    """
    Converts pickle files in a specified directory to MAT files.

    Args:
    - pthpkl (str): Directory containing the .pkl files.
    - pthpklmat (str): Directory to save the converted .mat files.

    Returns:
    - None: The function processes and saves .mat files for each .pkl file.
    """

    # Create list of full paths for each .pkl file in the directory
    df_full_path_list = [os.path.join(pthpkl, f) for f in os.listdir(pthpkl) if f.endswith(".pkl")]

    # Iterate through each pickle file
    for dfnm in df_full_path_list:
        # Create output .mat file name
        outnm = os.path.join(pthpklmat, f"{os.path.basename(dfnm)[:-4]}.mat")

        print(f"Saving: {dfnm}")

        # Load the .pkl file as a DataFrame
        with open(dfnm, 'rb') as f:
            df = pd.read_pickle(f)

        # Extract column names and convert the DataFrame to a NumPy array
        col_names = df.columns.tolist()
        df_array = df.to_numpy()

        # Save the DataFrame as a .mat file
        savemat(outnm, {'features': df_array, 'feature_names': col_names})
def make_geojson_contours(out_pth_json, gj=1, ds_amt=1, classification_name='Nuclei', classification_color=[97, 214, 59]) -> None:
    """
    Generates GeoJSON contours from segmented cells.

    Args:
    - out_pth_json (str): Path to the directory containing JSON files.
    - gj (int): Set to 1 to generate GeoJSON contours, 0 to skip. Default is 1.
    - ds_amt (float): Downsampling amount, 1 for 20x. Default is 1.
    - classification_name (str): Name for the classification. Default is 'Nuclei'.
    - classification_color (list): Color for the classification. Default is [97, 214, 59].

    Returns:
    - None
    """
    if gj != 1:
        print("GeoJSON generation is skipped.")
        return

    out_pth_contours = os.path.join(out_pth_json, 'geojsons', '32_polys_20x')
    os.makedirs(out_pth_contours, exist_ok=True)
    json_pth_list = get_sorted_files(out_pth_json, '.json')

    for p, file in enumerate(json_pth_list):
        nm = os.path.basename(file)
        file_name, _ = os.path.splitext(nm)
        new_fn = os.path.join(out_pth_contours, file_name + '.geojson')
        print(f'{p + 1} / {len(json_pth_list)}')
        print(nm)

        if not os.path.exists(new_fn):
            with open(file) as f:
                segmentation_data = json.load(f)

            data_list = get_ds_data(segmentation_data, ds_amt)

            GEOdata = []

            # Using tqdm for a progress bar
            for j, (centroid, contour) in tqdm(enumerate(data_list), total=len(data_list), desc="Processing contours"):
                centroid = [centroid[0], centroid[1]]
                contour = [[coord for coord in xy[::-1]] for xy in contour]
                contour.append(contour[0])

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
