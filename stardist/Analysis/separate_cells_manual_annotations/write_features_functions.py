import os
import numpy as np
import cv2
import json
from scipy.io import loadmat
import h5py
import pandas as pd
from tifffile import imread
import matplotlib.pyplot as plt
import copy
import re

# basic morphology features
def cntarea(cnt):
    cnt = np.array(cnt).astype(np.float32)
    area = cv2.contourArea(cnt)
    return area

def cntperi(cnt):
    cnt = np.array(cnt).astype(np.float32)
    perimeter = cv2.arcLength(cnt,True)
    return perimeter

def cntMA(cnt):
    cnt = np.array(cnt).astype(np.float32)
    #Orientation, Aspect_ratio
    [(x,y),(MA,ma),orientation] = cv2.fitEllipse(cnt)
    return [np.max((MA,ma)),np.min((MA,ma)),orientation]


# functions to adjust format of contours list
def fix_contours(contours):
    contours_fixed = []
    for polygon in contours:
        coords = np.array([list(zip(x,y)) for x,y in [polygon[0]]][0], dtype=np.int32)
        contours_fixed.append(coords)
    contours_fixed = np.array(contours_fixed)
    return contours_fixed

def adjust_contours(contour, crop_x, crop_y):

    for i, xy in enumerate(contour):
        x = xy[0] - crop_x
        y = xy[1] - crop_y
        
        
        contour[i] = [x, y]
    return contour

def get_rgb_avg(centroid, contour_raw, offset, HE_20x_WSI):
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
    if offset > centroid[0] or offset > centroid[1] or centroid[0] > (img_shape[0] - offset) or centroid[1] > (img_shape[1] - offset):
        print(f'centroid passed: {centroid}')
        r_avg = -1
        g_avg = -1
        b_avg = -1
        r_std = -1
        g_std = -1
        b_std = -1
        return r_avg, g_avg, b_avg, r_std, g_std, b_std
    
    im_crop = np.array(HE_20x_WSI[x_low:x_high, y_low:y_high], dtype=np.uint16)
    
    #plt.imshow(im_crop)
    
    crop_x = centroid[0]-offset-1
    crop_y = centroid[1]-offset-1
    
    contour_adj = adjust_contours(contour_raw, crop_x, crop_y)
    contour_new = contour_adj# .reshape((-1,1,2)).astype(np.uint16)
    rev_contour = contour_new[:, [1, 0]]  # its backwards for some reason idk why but you need to flip it like this
    #rev_contour = contour_new[:,:, [1, 0]]  # its backwards for some reason idk why but you need to flip it like this
    # print(rev_contour)
    
    # coords NEEDS to be np.int32 matrix --> 2 columns x y
    
    # Create a single-channel mask
    mask = np.zeros_like(im_crop[:, :, 0], dtype=np.uint16)  # make black image of same size, will fill with mask

    # Draw contours on the single-channel mask
    #cv2.drawContours(im_crop, [rev_contour], -1, (0,255,0)) #, thickness=cv2.FILLED)  # this one makes it green so that you can see contour
    cv2.drawContours(mask, [rev_contour], 0, (1), thickness=cv2.FILLED)
    
    # plt.imshow(im_crop)
    
    r_pixels = im_crop[:,:,0] * mask  # pixels inside mask are 1, outside == 0
    g_pixels = im_crop[:,:,1] * mask
    b_pixels = im_crop[:,:,2] * mask
    
    num_pixels = np.count_nonzero(mask)
    
    if num_pixels != 0:
        
        r_avg = round(np.sum(r_pixels)/num_pixels,2)
        g_avg = round(np.sum(g_pixels)/num_pixels,2)
        b_avg = round(np.sum(b_pixels)/num_pixels,2)
        
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
    
    #plt.imshow(im_crop)
    
    return r_avg, g_avg, b_avg, r_std, g_std, b_std

def write_df_features_pkl(json_full_pth_list, WSI_full_pth_list, outpth):
    # json_full_pth_list: a list of the full paths to each json file that was outputted from custom stardist output in this module
    # outpth: path to save the resulting .pkl dataframes of features

    for i, json_f_name in enumerate(json_full_pth_list):
    
        nm = json_f_name.split('\\')[-1].split('.')[0]
        
        outnm = os.path.join(outpth, f'{nm}.pkl')
        print(outnm)
        
        if not os.path.exists(outnm):
            
            # delete, just for testing
            #i = 400
            #print(i)
            #json_f_name = jsons[400]
            
            HE_20x_WSI = imread(WSI_full_pth_list[i])
            
            print(WSI_full_pth_list[i])
            print(json_f_name)
            
            try:
                segmentation_data = json.load(open(json_f_name))
            except:
                print(f'error reading json... Skipping {json_f_name}')
                continue
        
            centroids = [nuc['centroid'][0] for nuc in segmentation_data]
            contours = [nuc['contour'] for nuc in segmentation_data]
            contours_fixed = fix_contours(contours)
            
            offset = 30  # radius of image that gets cropped from WSI, used for getting rgb intensity average inside nuc contour
            
            centroids_np = np.array(centroids)  # for other formatting
            contours_np = np.array(contours)
            
            r_avg_list, g_avg_list, b_avg_list, r_std_list, g_std_list, b_std_list = [],[],[],[],[],[]
            
            areas = []
            perimeters = []
            circularities = []
            aspect_ratios = []
            image_ids = []
            classes = []
            
            compactness_a, eccentricity_a, euler_number_a, extent_a, form_factor_a, maximum_radius_a, mean_radius_a, median_radius_a, minor_axis_length_a, major_axis_length_a, orientation_degrees_a = [], [], [], [], [], [], [], [], [], [], []
            
            np_centroids = np.array(centroids)
            
            for j in range(len(contours_fixed)):
                #break
                
                centroid = centroids[j]
                # print(f'centroid: {centroid}')
                contour_raw = copy.copy(contours_fixed[j])  # used for rgb intensities
                
                # get rbg intensity averages
                r_avg, g_avg, b_avg, r_std, g_std, b_std = get_rgb_avg(centroid, contour_raw, offset, HE_20x_WSI)
                # print(r_avg, g_avg, b_avg)
                
                r_avg_list.append(r_avg)
                g_avg_list.append(g_avg)
                b_avg_list.append(b_avg)
                r_std_list.append(r_std)
                g_std_list.append(g_std)
                b_std_list.append(b_std)
                
                contour = contours_np[j][0].transpose()  # used for other stuff, too lazy to make formatting the same
                area = cntarea(contour)
                perimeter = cntperi(contour)
                circularity = 4 * np.pi * area / perimeter ** 2
                MA = cntMA(contour)
                [MA, ma, orientation] = MA
                aspect_ratio = MA / ma
                #center_x = centroid[0]
                #center_y = centroid[1]
                
                cent_x = np_centroids[j,0]
                cent_y = np_centroids[j,1]
                
                #compactness and form_factor are stupid because they are basically same as circularity, maybe extent too
                
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
        
                # additional features
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
                
                
            # exit loop
                
            dat = {
                'Centroid_x': np_centroids[:,1],
                'Centroid_y': np_centroids[:,0],
                'Area': areas,
                'Perimeter': perimeters,
                'Circularity': circularities,
                'Aspect Ratio': aspect_ratios,
                'compactness' : compactness_a,
                'eccentricity' : eccentricity_a,
                'extent' : extent_a,
                'form_factor' : form_factor_a,
                'maximum_radius' : maximum_radius_a,
                'mean_radius' : mean_radius_a,
                'median_radius' : median_radius_a,
                'minor_axis_length' : minor_axis_length_a,
                'major_axis_length' : major_axis_length_a,
                'orientation_degrees' : orientation_degrees_a,
                'r_mean_intensity' : r_avg_list,
                'g_mean_intensity' : g_avg_list,
                'b_mean_intensity' : b_avg_list,
                'r_std' : r_std_list,
                'g_std' : g_std_list,
                'b_std' : b_std_list,
                'slide_num': nm[-4:]  # fix this for your own needs, this gets slide number for my monkey fetus
            }
        
            df = pd.DataFrame(dat).astype(np.float32)  # save a little space with float16 type -> Edit 2 months later, this did not save time.
            
            df.to_pickle(outnm)
            # break
        else:
            print('skipping')

def write_df_features_pkl_single(json_file_pth, WSI_file_pth, outpth):

    # single slide version of above function

    # json_file_pth: 
    # WSI_file_pth: 
    # outpth: path to save the resulting .pkl dataframes of features
    
    nm = os.path.basename(json_file_pth)[:-5]
    
    outnm = os.path.join(outpth, f'{nm}.pkl')
    print(outnm)
    
    if not os.path.exists(outnm):

        try:
            HE_20x_WSI = imread(WSI_file_pth)
        except:
            # change the path so it tries to read as svs
            HE_20x_WSI = imread(WSI_file_pth[:-5] + '.svs')

        segmentation_data = json.load(open(json_file_pth))

        centroids = [nuc['centroid'][0] for nuc in segmentation_data]
        contours = [nuc['contour'] for nuc in segmentation_data]
        contours_fixed = fix_contours(contours)
        
        offset = 30  # radius of image that gets cropped from WSI, used for getting rgb intensity average inside nuc contour
        
        centroids_np = np.array(centroids)  # for other formatting
        contours_np = np.array(contours)
        
        r_avg_list, g_avg_list, b_avg_list, r_std_list, g_std_list, b_std_list = [],[],[],[],[],[]
        
        areas = []
        perimeters = []
        circularities = []
        aspect_ratios = []
        image_ids = []
        classes = []
        
        compactness_a, eccentricity_a, euler_number_a, extent_a, form_factor_a, maximum_radius_a, mean_radius_a, median_radius_a, minor_axis_length_a, major_axis_length_a, orientation_degrees_a = [], [], [], [], [], [], [], [], [], [], []
        
        np_centroids = np.array(centroids)
        
        for j in range(len(contours_fixed)):
            #break
            
            centroid = centroids[j]
            # print(f'centroid: {centroid}')
            contour_raw = copy.copy(contours_fixed[j])  # used for rgb intensities
            
            # get rbg intensity averages
            r_avg, g_avg, b_avg, r_std, g_std, b_std = get_rgb_avg(centroid, contour_raw, offset, HE_20x_WSI)
            # print(r_avg, g_avg, b_avg)
            
            r_avg_list.append(r_avg)
            g_avg_list.append(g_avg)
            b_avg_list.append(b_avg)
            r_std_list.append(r_std)
            g_std_list.append(g_std)
            b_std_list.append(b_std)
            
            contour = contours_np[j][0].transpose()  # used for other stuff, too lazy to make formatting the same
            area = cntarea(contour)
            perimeter = cntperi(contour)
            circularity = 4 * np.pi * area / perimeter ** 2
            MA = cntMA(contour)
            [MA, ma, orientation] = MA
            aspect_ratio = MA / ma
            #center_x = centroid[0]
            #center_y = centroid[1]
            
            cent_x = np_centroids[j,0]
            cent_y = np_centroids[j,1]
            
            #compactness and form_factor are stupid because they are basically same as circularity, maybe extent too
            
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
    
            # additional features
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
            
            
        # exit loop
            
        dat = {
            'Centroid_x': np_centroids[:,1],
            'Centroid_y': np_centroids[:,0],
            'Area': areas,
            'Perimeter': perimeters,
            'Circularity': circularities,
            'Aspect Ratio': aspect_ratios,
            'compactness' : compactness_a,
            'eccentricity' : eccentricity_a,
            'extent' : extent_a,
            'form_factor' : form_factor_a,
            'maximum_radius' : maximum_radius_a,
            'mean_radius' : mean_radius_a,
            'median_radius' : median_radius_a,
            'minor_axis_length' : minor_axis_length_a,
            'major_axis_length' : major_axis_length_a,
            'orientation_degrees' : orientation_degrees_a,
            'r_mean_intensity' : r_avg_list,
            'g_mean_intensity' : g_avg_list,
            'b_mean_intensity' : b_avg_list,
            'r_std' : r_std_list,
            'g_std' : g_std_list,
            'b_std' : b_std_list,
            'slide_num': extract_slide_number(nm) #nm[-4:]  # fix this for your own needs, this gets slide number for my monkey fetus
        }
    
        df = pd.DataFrame(dat).astype(np.float32)  # save a little space with float16 type -> Edit 2 months later, this did not save time.
        
        df.to_pickle(outnm)

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

def save_features_selected_nuclei(geojson_pth_list, selection_json_path, selection_path, pthpkl):
    """
    Process geojson files to generate JSON and NDPI paths and save the extracted features.

    Args:
    - geojson_pth_list (list): List of geojson file paths.
    - selection_json_path (str): Directory where the selection JSON files are stored.
    - selection_path (str): Directory where the NDPI files are stored.
    - pthpkl (str): Path to save the features as a pickle file.

    Returns:
    - None: The function processes and saves features for each file.
    """

    for geo_json_f_nm in geojson_pth_list:
        # Get the base name without the last 8 characters (e.g., without extension and extra details)
        nm = os.path.basename(geo_json_f_nm[:-8])

        # Construct the corresponding JSON and NDPI file paths
        json_file = os.path.join(selection_json_path, f"{nm}.json")
        try:
            WSI_file = os.path.join(selection_path, f"{nm}.ndpi")
        except:
            WSI_file = os.path.join(selection_path, f"{nm}.svs")

        # Save features using the provided function
        write_df_features_pkl_single(json_file, WSI_file, pthpkl)
        # make_RGBfeatures_df_pkl_from_contours_jsons(json_file, WSI_file, pthpkl)
