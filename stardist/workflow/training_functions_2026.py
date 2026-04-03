# from numba.core.cgutils import printf
from glob import glob

from stardist import random_label_cmap

from functions import *


def create_directory(path):
    os.makedirs(path, exist_ok=True)
    return path

def pths_folders(pth_ndpis, date, nm, folderds):
    # [1] input for cropping out tiles:
    pth2x = os.path.join(pth_ndpis, folderds)
    output_pixres = create_directory(os.path.join(pth_ndpis, 'segmentation_analysis', 'pix_res_info'))

    # [2] segmenting tiles
    tiles_folder = create_directory(os.path.join(pth_ndpis, "20x_tiff_tiles"))
    training_tiles_folder = create_directory(os.path.join(tiles_folder, "training"))
    pth_seg_train_tiles = create_directory(os.path.join(training_tiles_folder, 'geojsons'))
    project_train = create_directory(os.path.join(training_tiles_folder, 'project'))

    testing_tiles_folder = create_directory(os.path.join(tiles_folder, "testing"))
    pth_seg_test_tiles = create_directory(os.path.join(testing_tiles_folder, 'geojsons'))
    project_test = create_directory(os.path.join(testing_tiles_folder, 'project'))

    # [3] training
    training_masks_path = os.path.join(training_tiles_folder, 'project', 'ground_truth', 'masks')
    outpthmodel = create_directory(os.path.join(pth_ndpis, 'stardist_models', f'{nm}_{date}'))
    offshoot_model_pth = create_directory(os.path.join(outpthmodel, 'offshoot_model'))

    # [4] testing
    testing_tiles_path = testing_tiles_folder
    testing_masks_path = os.path.join(testing_tiles_folder, 'project', 'ground_truth', 'masks')

    return (
        pth2x, output_pixres, tiles_folder, training_tiles_folder, pth_seg_train_tiles, project_train,
        testing_tiles_folder, pth_seg_test_tiles, project_test, offshoot_model_pth, training_masks_path,
        outpthmodel, testing_masks_path
    )

# def load_selected_models_AF(model_name)-> StarDist2D:
#     base_path = r'\\10.162.80.16\Andre\data\Stardist\models'
#     model_paths = {
#         "pdac": os.path.join(base_path, "lea_model"),
#         "panin": os.path.join(base_path, "Big_PANIN_model_8_10_24_lr_0.001_epochs_400_pt_40"),
#         "monkey": os.path.join(base_path, "monkey_12_12_2023_lr_0.001_epochs_400_pt_10_gaus_ratio_0"),
#         "fallopian_tube": os.path.join(base_path, "fallopian_tube_12_7_2023_lr_0.001_epochs_400_pt_40")
#     }
#
#     if model_name in model_paths:
#         model = load_model(model_paths[model_name])
#         print(f"Loaded {model_name} model successfully.")
#         return model
#     else:
#         raise ValueError("Invalid model name provided. Choose from 'lea_model', 'panin', 'monkey', or 'fallopian tube'.")
# def load_model(model_path: str) -> StarDist2D:
#     # TODO: remove offshoot thing
#     with open(model_path + '\\config.json', 'r') as f:
#         config = json.load(f)
#     with open(model_path + '\\thresholds.json', 'r') as f:
#         thresh = json.load(f)
#     model = StarDist2D(config=Config2D(**config), basedir=model_path, name='offshoot_model')
#     model.thresholds = thresh
#     print('Overriding defaults:', model.thresholds, '\n')
#     model.load_weights(model_path + '\\weights_best.h5')
#     return model

import json
from stardist.models import StarDist2D


def load_model(model_path: str, offshoot_model_pth: str) -> StarDist2D:
    """
    Loads a StarDist2D model from the specified path and sets the offshoot model path.

    Args:
        model_path (str): Path to the directory containing the model files (config.json, thresholds.json, weights_best.h5).
        offshoot_model_pth (str): Path to the directory where the offshoot model should be saved or loaded from.

    Returns:
        StarDist2D: The loaded StarDist2D model.
    """
    # Load the model configuration
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)

    # Load the thresholds
    with open(os.path.join(model_path, 'thresholds.json'), 'r') as f:
        thresh = json.load(f)

    # Create the StarDist2D model with the specified offshoot model path
    model = StarDist2D(config=Config2D(**config), basedir=offshoot_model_pth, name='offshoot_model')
    print('offshoot model path:', offshoot_model_pth, '\n')

    # Set the thresholds
    model.thresholds = thresh
    print('Overriding defaults:', model.thresholds, '\n')

    # Load the model weights
    model.load_weights(os.path.join(model_path, 'weights_best.h5'))

    return model

def load_selected_models_folder_AF(model_name) -> StarDist2D:
    """Load StarDist model from the models folder in the repository."""
    # Get the path to the models folder relative to this file
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__S)), 'models')
    model_paths = {
        "pdac": os.path.join(base_path, "PDAC"),
        "panin": os.path.join(base_path, "Big_PANIN"),
        "panin_healthy": os.path.join(base_path, "Big_Panin_healthy_nPOD"),
        "fallopian_tube": os.path.join(base_path, "fallopian_tube"),
        "cross_fetal_species": os.path.join(base_path, "cross_fetal_species"),
        "cross_fetal_species_alcian_blue": os.path.join(base_path, "cross_fetal_species_alcian_blue")
    }

    if model_name in model_paths:
        model = load_model(model_paths[model_name])
        print(f"Loaded {model_name} model successfully from repo.")
        return model
    else:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from: {list(model_paths.keys())}")

def load_selected_models_AF(model_name,offshoot_model_pth)-> StarDist2D:
    base_path = r'\\10.162.80.16\Andre\data\Stardist\models'
    model_paths = {
        "pdac": os.path.join(base_path, "lea_model"),
        "panin": os.path.join(base_path, "Big_PANIN_model_8_10_24_lr_0.001_epochs_400_pt_40"),
        "monkey": os.path.join(base_path, "monkey_12_12_2023_lr_0.001_epochs_400_pt_10_gaus_ratio_0"),
        "fallopian_tube": os.path.join(base_path, "fallopian_tube_12_7_2023_lr_0.001_epochs_400_pt_40")
    }

    if model_name in model_paths:
        model = load_model(model_paths[model_name], offshoot_model_pth)
        print(f"Loaded {model_name} model successfully.")
        return model
    else:
        raise ValueError("Invalid model name provided. Choose from 'lea_model', 'panin', 'monkey', or 'fallopian tube'.")

#this date param refers to the date entered in file3 input
def hyperparam_setup(date,nm,lr=0.001, epochs=400, patience=40, ratio_validation_tiles=0.4):
    outnm = nm + date + '_lr_' + str(lr) + '_epochs_' + str(epochs) + '_pt_' + str(patience)
    print(outnm)
    return outnm

def tile_setup_details(HE_train_aug, masks_train_aug, tiles_val, masks_val):
    print(f'{len(HE_train_aug)}')
    print(f'{len(masks_train_aug)}')
    print(f'{len(tiles_val)}')
    print(f'{len(masks_val)}')

# Check for GPU availability
def GPU_availability():
    print("GPU is available" if tf.config.list_physical_devices('GPU') else "GPU is not available")
    # print(tf.__version__)


def read_tiles_masks(tiles_pth, mask_pth):
    # Get list of tile file names (assuming files in the tile directory have extensions like .png, .jpg, etc.)
    tile_files = sorted(os.listdir(tiles_pth))
    # Initialize empty lists to hold the tiles and masks
    tiles = []
    masks = []
    # Loop through tile files and load the corresponding mask with the same name
    for tile_file in tile_files:
        if tile_file.endswith(('.png', '.jpg', '.jpeg','.tif','.tiff')):
            # Read the tile and the corresponding mask
            tile = imread(os.path.join(tiles_pth, tile_file))
            mask = imread(os.path.join(mask_pth, tile_file))
            # Append to the lists
            tiles.append(tile)
            masks.append(mask)
    return tiles, masks

def set_up_training(date, nm, outpthmodel, training_tiles_folder, training_masks_path, additional_tiles_paths=None,
                    additional_masks_paths=None):
    """
    Prepares the StarDist training setup, including loading tiles and masks, splitting into training/validation sets,
    augmenting the data, and normalizing the images. Allows for multiple additional tiles and masks from other datasets.

    Args:
        date (str): The date for naming the output folder.
        nm (str): A name identifier for the model.
        outpthmodel (str): The output path for the model.
        training_tiles_folder (str): The folder containing the training tiles.
        training_masks_path (str): The folder containing the training masks.
        additional_tiles_paths (list of str, optional): List of folders containing additional training tiles. Defaults to None.
        additional_masks_paths (list of str, optional): List of folders containing additional training masks. Defaults to None.

    Returns:
        tuple: A tuple containing the training and validation tiles and masks, and the path to the log folder.
    """
    # Set up the output folder for training logs
    outnm = hyperparam_setup(date, nm)
    pth_new_trained_model = os.path.join(outpthmodel, outnm)
    os.makedirs(pth_new_trained_model, exist_ok=True)

    # Load the primary training tiles and masks
    # HE_tiles = read_tiles(training_tiles_folder)
    # masks = read_masks(training_masks_path)

    HE_tiles,masks=read_tiles_masks(training_tiles_folder,training_masks_path)

    # Load additional tiles and masks if provided
    if additional_tiles_paths is not None and additional_masks_paths is not None:
        if len(additional_tiles_paths) != len(additional_masks_paths):
            raise ValueError("The number of additional tiles paths must match the number of additional masks paths.")

        for tiles_path, masks_path in zip(additional_tiles_paths, additional_masks_paths):
            additional_tiles = read_tiles(tiles_path)
            additional_masks = read_masks(masks_path)

            # Combine the primary and additional tiles/masks
            HE_tiles.extend(additional_tiles)
            masks.extend(additional_masks)

    # Split the combined dataset into training and validation sets
    tiles_train, masks_train, tiles_val, masks_val = split_train_val_set(HE_tiles, masks, 0.4)

    # Augment the training tiles and masks
    HE_train_aug, masks_train_aug = augment_tiles(tiles_train, masks_train)

    # Normalize the H&E images by dividing by 255
    HE_train_aug = normalize_images(HE_train_aug)
    tiles_val = normalize_images(tiles_val)

    # Print details about the tile setup
    tile_setup_details(HE_train_aug, masks_train_aug, tiles_val, masks_val)

    # Visualize a sample from the training data
    for kk in range(0, len(HE_train_aug), 8):
        show_HE_and_segmented(HE_train_aug[kk], masks_train_aug[kk])

    # Check GPU availability
    GPU_availability()
    print("Training setup finished")

    return tiles_train, HE_train_aug, masks_train_aug, tiles_val, masks_val, pth_new_trained_model

# def set_up_training(date, nm, outpthmodel, training_tiles_folder, training_masks_path):
#     outnm = hyperparam_setup(date,nm)
#     pth_log_train = fr"{outpthmodel}\{outnm}"
#     os.makedirs(pth_log_train, exist_ok=True)
#     HE_tiles = read_tiles(training_tiles_folder)
#     masks = read_masks(training_masks_path)
#     tiles_train, masks_train, tiles_val, masks_val = split_train_val_set(HE_tiles, masks, 0.4)
#     # add flips and rotations to the images/masks pairs
#     HE_train_aug, masks_train_aug = augment_tiles(tiles_train, masks_train)
#     # normalize H&E images by dividing by 255
#     HE_train_aug = normalize_images(HE_train_aug)
#     tiles_val = normalize_images(tiles_val)
#     tile_setup_details(HE_train_aug, masks_train_aug, tiles_val, masks_val)
#     #look at the training data
#     #todo: this number may need to change depend on #of tiles
#     i = 5
#     show_HE_and_segmented(HE_train_aug[i], masks_train_aug[i])
#     # check GPU availability
#     GPU_availability()
#     print("training setup finished")
#     return tiles_train, HE_train_aug, masks_train_aug, tiles_val, masks_val, pth_log_train

import shutil
from stardist.models import Config2D, StarDist2D


def train_model(tiles_train, model, offshoot_model_pth, pth_new_trained_model, HE_train_aug, masks_train_aug, tiles_val,
                masks_val, epochs=400, lr=0.001, patience=40):
    """
    Trains the StarDist model with GPU acceleration and ensures the best nuclear segmentation masks.

    Args:
        tiles_train (list): List of training tiles.
        model: The StarDist model.
        offshoot_model_pth (str): Path to the offshoot model.
        pth_log_train (str): Path to the training logs.
        HE_train_aug (list): Augmented training tiles.
        masks_train_aug (list): Augmented training masks.
        tiles_val (list): Validation tiles.
        masks_val (list): Validation masks.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        patience (int): Patience for learning rate reduction.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(pth_new_trained_model, exist_ok=True)

    # Define the configuration, set up the model
    conf = Config2D(
        n_rays=32,
        grid=(2, 2),
        use_gpu=True,  # Enable GPU usage
        n_channel_in=1 if tiles_train[0].ndim == 2 else tiles_train[0].shape[-1]
    )

    # TRAIN THE MODEL
    model.config.train_learning_rate = lr
    model.config.train_patch_size = (256, 256)
    model.config.train_reduce_lr = {'factor': 0.5, 'patience': patience, 'min_delta': 0}

    # Train the model
    history = model.train(
        HE_train_aug,
        masks_train_aug,
        validation_data=(tiles_val, masks_val),
        epochs=epochs,
        steps_per_epoch=100
    )

    # Optimize thresholds for better segmentation
    model.optimize_thresholds(tiles_val, masks_val)

    # Save the trained model
    # model.export_weights(pth_log_train)  # Correct method to save the model
    # model.save(pth_log_train)  # Correct method to save the model
    shutil.copytree(offshoot_model_pth, pth_new_trained_model, dirs_exist_ok=True)

    # Save training logs
    pth_log = glob.glob(os.path.join(offshoot_model_pth, 'logs', 'train', '*.v2'))[0]
    # log_dir = os.path.join(offshoot_model_pth, 'logs', 'train')
    # os.makedirs(log_dir, exist_ok=True)
    with open(pth_log, 'w') as f:
        for epoch, loss, val_loss in zip(range(epochs), history.history['loss'], history.history['val_loss']):
            f.write(f"Epoch {epoch + 1}: loss={loss}, val_loss={val_loss}\n")

    # # Analyze the loss data
    # loss = get_loss_data(str(pth_log), pth_new_trained_model)

    # Print the final loss and validation metrics
    print(f"Final training loss: {history.history['loss'][-1]}")
    print(f"Final validation loss: {history.history['val_loss'][-1]}")

# def train_model(tiles_train, model, offshoot_model_pth, pth_log_train, HE_train_aug, masks_train_aug, tiles_val,
#                 masks_val,
#                 epochs=400, lr=0.001, patience=40):
#     # Define the configuration, set up the model
#     conf = Config2D(
#         n_rays=32,
#         grid=(2, 2),
#         use_gpu=gputools_available(),
#         n_channel_in=1 if tiles_train[0].ndim == 2 else tiles_train[0].shape[-1]
#     )
#
#     # TRAIN THE MODEL
#     # model = load_published_he_model(outpthmodel, outnm)
#     # model = load_model(model_path)
#     model.config.train_learning_rate = lr
#     model.config.train_patch_size = (256, 256)
#     model.config.train_reduce_lr = {'factor': 0.5, 'patience': patience, 'min_delta': 0}
#     model.train(HE_train_aug, masks_train_aug, validation_data=(tiles_val, masks_val), epochs=epochs,
#                 steps_per_epoch=100)
#     model.optimize_thresholds(tiles_val, masks_val)
#     shutil.copytree(offshoot_model_pth, pth_log_train, dirs_exist_ok=True)
#
#     pth_log = glob(os.path.join(pth_log_train, 'logs', 'train', '*.v2'))[0]
#     loss = get_loss_data(str(pth_log), pth_log_train)


def display_case(testing_tiles, testing_masks, predictions, lbl_cmap):
    # i = -2
    i=0
    tile = testing_tiles[i]
    gt_mask = testing_masks[i]
    pred_mask = predictions[i]
    plot_predictions_vs_gt(tile, gt_mask, pred_mask, lbl_cmap)

    i = 1
    tile = testing_tiles[i]
    gt_mask = testing_masks[i]
    pred_mask = predictions[i]
    plot_predictions_vs_gt(tile, gt_mask, pred_mask, lbl_cmap)

    i = 2
    tile = testing_tiles[i]
    gt_mask = testing_masks[i]
    pred_mask = predictions[i]
    plot_predictions_vs_gt(tile, gt_mask, pred_mask, lbl_cmap)

def plot_predictions_vs_gt(tile: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, cmap: ListedColormap) -> None:
    """Visualizes predictions and ground truth for a given tile."""
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].imshow(tile)
    ax[0].axis('off')
    ax[0].set_title('H&E')

    ax[1].imshow(tile)
    ax[1].imshow(gt_mask, cmap=cmap, alpha=0.5)
    ax[1].axis('off')
    ax[1].set_title('Ground Truth')

    ax[2].imshow(tile)
    ax[2].imshow(pred_mask, cmap=cmap, alpha=0.5)
    ax[2].axis('off')
    ax[2].set_title('Predicted')
    plt.show()

from PIL import ImageDraw, ImageTk
import tkinter as tk

def mouse_click_coordinate(image_path, window_sf, img_size, square_size, ds):
    """
    Displays an image in a Tkinter window, allows the user to click on it, captures the mouse click coordinates,
    and draws a circle and square around the click. Asks if the user wants to reselect the coordinate.

    Args:
        image_path (str): The file path to the image to be displayed.
        window_sf (int): The scaling factor to resize the window and image to fit the display.
        img_size (tuple): The size of the original image as a tuple (height, width).
        square_size (int): The size of the square to be drawn.
        ds (int): downsample factor.

    Returns:
        list: A list containing the x, y coordinates of the mouse click on the image.
    """
    coordinate = []
    zoom_factor = 1.0

    def update_image():
        nonlocal tkimage
        zoomed_width = int(window_width * zoom_factor)
        zoomed_height = int(window_height * zoom_factor)
        zoomed_image = image_copy.resize((zoomed_width, zoomed_height), Image.LANCZOS)
        tkimage = ImageTk.PhotoImage(zoomed_image)

        canvas.delete("all")  # Clear previous images
        canvas.create_image(0, 0, anchor="nw", image=tkimage)
        canvas.config(scrollregion=canvas.bbox("all"))  # Update scroll region

    def click_event(event):
        x = int(canvas.canvasx(event.x) / zoom_factor)
        y = int(canvas.canvasy(event.y) / zoom_factor)

        # Draw a circle at the clicked position
        draw = ImageDraw.Draw(image_copy)
        dot_radius = 1
        draw.ellipse([(x - dot_radius, y - dot_radius), (x + dot_radius, y + dot_radius)], fill="red")

        # Draw a square centered on the click
        half_square = int(square_size / window_sf / ds / 2 / 2)
        draw.rectangle([(x - half_square, y - half_square), (x + half_square, y + half_square)], outline="blue",
                       width=1)

        # Display coordinates
        draw.text((x + 10, y + 10), f"({x}, {y})", fill="red")

        # Update the displayed image
        update_image()

        # Store the coordinates
        coordinate.clear()
        coordinate.extend([x, y])

        # Show Yes/No buttons to confirm selection
        confirm_window = tk.Toplevel(root)
        confirm_window.title("Confirm Selection")
        tk.Label(confirm_window, text="Do you want to proceed with these coordinates?").pack()
        tk.Button(confirm_window, text="Yes", command=lambda: [confirm_window.destroy(), root.quit()]).pack(side=tk.LEFT)
        tk.Button(confirm_window, text="No, reselect coordinates", command=lambda: [confirm_window.destroy(), reset_selection()]).pack(side=tk.RIGHT)

    def reset_selection():
        nonlocal image_copy
        image_copy = image.copy()
        update_image()  # Preserve zoom factor when resetting selection

    def zoom(event):
        nonlocal zoom_factor

        # Get cursor position before zoom
        mouse_x = canvas.canvasx(event.x)
        mouse_y = canvas.canvasy(event.y)
        prev_zoom_factor = zoom_factor

        # Apply zoom
        if event.delta > 0 or event.num == 4:  # Scroll up
            zoom_factor *= 1.1
        elif event.delta < 0 or event.num == 5:  # Scroll down
            zoom_factor /= 1.1

        # Update image with new zoom factor
        update_image()

        # Calculate new scroll position to keep cursor at same location
        new_mouse_x = mouse_x * (zoom_factor / prev_zoom_factor)
        new_mouse_y = mouse_y * (zoom_factor / prev_zoom_factor)

        canvas.xview_moveto((new_mouse_x - event.x) / (window_width * zoom_factor))
        canvas.yview_moveto((new_mouse_y - event.y) / (window_height * zoom_factor))

    # Open and resize the image
    image = Image.open(image_path)
    window_width = img_size[1] // window_sf
    window_height = img_size[0] // window_sf
    image = image.resize((window_width, window_height), Image.LANCZOS)
    image_copy = image.copy()

    # Create the main Tkinter window
    root = tk.Tk()
    root.geometry("1280x720")  # Fixed window size

    # Create canvas and scrollbars
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(canvas_frame, bg="white")
    h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    v_scroll.grid(row=0, column=1, sticky="ns")
    h_scroll.grid(row=1, column=0, sticky="ew")

    canvas_frame.grid_rowconfigure(0, weight=1)
    canvas_frame.grid_columnconfigure(0, weight=1)

    tkimage = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor="nw", image=tkimage)
    canvas.config(scrollregion=canvas.bbox("all"))

    canvas.bind("<Button-1>", click_event)
    canvas.bind("<MouseWheel>", zoom)  # For Windows and MacOS
    canvas.bind("<Button-4>", zoom)  # For Linux scroll up
    canvas.bind("<Button-5>", zoom)  # For Linux scroll down

    # Start the Tkinter event loop
    root.mainloop()
    root.destroy()

    return coordinate


from PIL import Image
import pickle as pkl


def crop_20x_ndpi(img_2x, cdn_20x_y, cdn_20x_x, tilesfolder, SQUARE_SIZE, folderds):
    """
    Crops a 20x NDPI or SVS image based on the given coordinates and saves the cropped tile,
    ensuring the selected coordinate is in the middle of the tile.

    Args:
        img_2x (str): The file path of the 2.5x image.
        cdn_20x_y (int): The y-coordinate for cropping in the 20x image.
        cdn_20x_x (int): The x-coordinate for cropping in the 20x image.
        tilesfolder (str): The directory path where the cropped 20x tiles and pkl files will be saved.
        SQUARE_SIZE (int): The size of the cropped tile (default: 512).
        folderds (str): The folder containing downsampled images.
    """

    # Function to transform the 2.5x image path to the 20x image path
    def transform_path(path, folderds):
        base_dir = os.path.dirname(os.path.dirname(path))
        base_name = os.path.basename(path).replace(f"{folderds}\\", "").replace(".tif", "")

        # Check if the corresponding .ndpi or .svs file exists
        ndpi_path = os.path.join(base_dir, base_name + ".ndpi")
        svs_path = os.path.join(base_dir, base_name + ".svs")

        if os.path.exists(ndpi_path):
            return ndpi_path
        elif os.path.exists(svs_path):
            return svs_path
        else:
            raise FileNotFoundError(f"Neither .ndpi nor .svs file found for {base_name}")

    # Get the 20x image path
    img_pth = transform_path(img_2x, folderds)  # 20x image path
    img_20x = imread(img_pth)  # Read 20x image

    # Ensure the coordinates are integers
    cdn_20x_x = int(cdn_20x_x)
    cdn_20x_y = int(cdn_20x_y)

    # Calculate the top-left corner of the tile so that the selected coordinate is in the middle
    half_size = SQUARE_SIZE // 2
    top_left_x = cdn_20x_x - half_size
    top_left_y = cdn_20x_y - half_size

    # Ensure the tile stays within the image boundaries
    img_height, img_width = img_20x.shape[:2]
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    top_left_x = min(top_left_x, img_width - SQUARE_SIZE)
    top_left_y = min(top_left_y, img_height - SQUARE_SIZE)

    # Slice the 20x image using the adjusted coordinates
    sliced_20x = img_20x[top_left_y:top_left_y + SQUARE_SIZE, top_left_x:top_left_x + SQUARE_SIZE]

    # Name the 20x tile
    tile_20x = os.path.basename(img_2x.replace('.tif', '.tif'))
    print("Image name is:", tile_20x)

    # If the image has 4 channels, remove the alpha channel and keep RGB only
    if sliced_20x.shape[-1] == 4:
        sliced_20x = sliced_20x[..., :3]  # Keep only the RGB channels
        print("Alpha channel removed")

    # Save the sliced 20x image to the tiles folder
    Image.fromarray(sliced_20x).save(os.path.join(tilesfolder, tile_20x))

    # Create full path for the click coordinates folder
    cdn_folder_path = os.path.join(tilesfolder, "click_coordinate_20x")

    # If the folder doesn't exist, create it
    if not os.path.exists(cdn_folder_path):
        os.makedirs(cdn_folder_path, exist_ok=True)

    # Create full path for the pkl file
    pkl_file_path = os.path.join(cdn_folder_path, os.path.basename(img_2x.replace('.tif', '.pkl')))

    # Save the data (tile size and coordinates) to the pkl file
    with open(pkl_file_path, mode='wb') as file:
        pkl.dump([SQUARE_SIZE, cdn_20x_x, cdn_20x_y], file)

import os
import random
import glob
from scipy.io import loadmat
from skimage.io import imread

def select_crop_tiles(num_tiles_need, twox_im_pth_list, tilesfolder, folderds, output_pixres,ds, SQUARE_SIZE=512, used_ndpi_list=None):
    """
    Randomly selects a subset of 2.5x HE images, allows the user to interactively select crop regions with a mouse click,
    and saves the cropped 20x NDPI tiles.

    Args:
        num_tiles_need (int): The number of 2.5x images to select and crop.
        twox_im_pth_list (list): A list of file paths to the 2.5x images.
        tilesfolder (str): The directory path where the cropped 20x tiles and associated data will be saved.
        folderds (str): The folder containing downsampled images.
        ds (float): Downsample factor.
        output_pixres (str): Path to the directory containing pixel resolution data.
        SQUARE_SIZE (int): Size of the cropped tiles (default: 512).
        used_ndpi_list (list): List of NDPI files already used for cropping (default: None).

    Returns:
        list: A list of paths to the selected 2.5x images.
    """
    if used_ndpi_list is None:
        used_ndpi_list = []

    # 1. Get list of equally spaced images from the 2.5x list of images
    num_total_images = len(twox_im_pth_list)
    num_per = num_total_images // num_tiles_need
    rand_img_list = []  # List to hold the selected 2.5x image paths

    # Get list of existing tiles in the tilesfolder
    existing_tiles_list = glob.glob(f"{tilesfolder}/*.tif")
    num_existing_tiles = len(existing_tiles_list)

    # If fewer tiles exist than needed, select additional images
    if num_existing_tiles < num_tiles_need:
        start_num = 0
        end_num = num_per - 1

        for _ in range(num_tiles_need - num_existing_tiles):
            while True:
                rand_img_idx = random.randint(start_num, end_num)
                if twox_im_pth_list[rand_img_idx] not in used_ndpi_list:
                    break
            rand_img_list.append(twox_im_pth_list[rand_img_idx])
            used_ndpi_list.append(twox_im_pth_list[rand_img_idx])
            start_num = end_num + 1
            end_num += num_per

        # 2. Cropping task: Go through the list of selected images, pick a location to crop, and save the tiles
        for i, img_path in enumerate(rand_img_list):
            img = imread(img_path)
            img_size = img.shape

            # Get the mouse click coordinate
            win_sf = 3
            click_coordinate = mouse_click_coordinate(img_path, win_sf, img_size, SQUARE_SIZE, ds)
            click_coordinate[0] *= win_sf
            click_coordinate[1] *= win_sf
            print('Got the click coordinate:', click_coordinate)

            # Get the 20x click coordinate
            mat_file_path = os.path.join(output_pixres, os.path.basename(img_path).split('.')[0] + '.mat')
            mat_data = loadmat(mat_file_path)
            pix_res = mat_data['pix_res']['x'][0, 0]  # Extract pixel resolution
            pix_res_float = float(pix_res[0])
            cdn_20x_x = click_coordinate[0] * (ds / pix_res_float)
            cdn_20x_y = click_coordinate[1] * (ds / pix_res_float)

            # Crop the 20x NDPI image
            crop_20x_ndpi(img_path, cdn_20x_y, cdn_20x_x, tilesfolder, SQUARE_SIZE=SQUARE_SIZE,folderds=folderds)
            # crop_20x_ndpi(img_path, cdn_20x_x, cdn_20x_y, tilesfolder, SQUARE_SIZE=SQUARE_SIZE,folderds=folderds)
            print(f"Select & Cropping Progress: {i + 1} / {len(rand_img_list)}")

    return rand_img_list


def segment_save_tiles(tiles_folder, model, pth_save_to):
    # Segment tiles
    tiles = normalize_images(read_tiles(tiles_folder))
    predictions = segment_tiles(tiles, model)
    # make a random cmap to show the predicted results
    np.random.seed(0)
    cmap = random_label_cmap()
    # show first segmentation, recommended that you check a few to make sure they look alright
    show_HE_and_segmented(tiles[0], predictions[0])

    # OPTIONAL: ONLY DO THIS IF YOU NEED TO FIX THE TILES FOR TRAINING
    # saves a .geojson file with segmentation information. If dragged and dropped into QuPath over the tile image, it shows results.
    save_geojson_from_segmentation(tiles_folder, model,
                                   pth_save_to)  # last variable originally holds: #path to old model segmentation result for training tiles

def test_model_performance(outpthmodel, testing_tiles_path, testing_masks_path):
    # trained_model = load_model(outpthmodel)
    offshoot_model_pth = create_directory(os.path.join(outpthmodel, 'offshoot_model'))
    trained_model = load_model(outpthmodel,offshoot_model_pth)
    testing_tiles = normalize_images(read_tiles(testing_tiles_path))
    testing_masks = read_masks(testing_masks_path)

    predictions = segment_tiles(testing_tiles, trained_model)
    # Random color map labels
    np.random.seed(42)
    lbl_cmap = random_label_cmap()
    # display a case
    display_case(testing_tiles, testing_masks, predictions, lbl_cmap)

    taus = [0.6]
    results = get_stats(testing_tiles_path, testing_masks, predictions, taus)
    make_f1_plot(testing_tiles_path, results, taus)



def get_stats(HE_tiles_pth: str, mask_gt_tiles: List[np.ndarray], mask_pred_tiles: List[np.ndarray], taus: list) -> pd.DataFrame:
    """Return a df with stats about each tile."""
    nms = [os.path.basename(file) for file in os.listdir(HE_tiles_pth) if file.endswith('.tif')]
    scores = TileSetScorer(nms, mask_gt_tiles, mask_pred_tiles, taus)
    results = scores.score_set()
    return results


def make_f1_plot(HE_tiles_pth: str, results: pd.DataFrame, taus: list) \
        -> None:
    """idk yet"""
    nms = [os.path.basename(file) for file in os.listdir(HE_tiles_pth) if file.endswith('.tif')]
    names = results['Image']

    names = [name.split(".")[0][21:] for name in names]  # this should be a list of the numbers at end of file names

    for i in range(len(names)):
        if len(names[i]) > 6:
            names[i] = names[i][:5]

    f1_scores = results['F1 Score']

    index = np.arange(len(nms))

    # Plotting the bars
    fig = plt.figure(figsize=(25, 10))
    fig.set_facecolor('white')

    plt.bar(index, f1_scores, color='darksalmon')

    plt.xlabel("Tile Name", fontsize=20)
    plt.ylabel("F1 Score", fontsize=20)
    plt.title("F1 Scores in Testing Tiles (tau = 0.7)", fontsize=28)
    plt.axhline(y=0.7, linestyle='--', color='red', label=f'Target F1 = {taus[0]}')
    plt.ylim(0, 1)
    plt.xticks(index, names)  # Set x-axis labels to tile names
    plt.legend(fontsize=20)
    plt.show()

    print(f1_scores)

    return f1_scores

