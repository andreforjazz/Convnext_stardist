import os
import glob
import tifffile
from math import ceil
from PIL import Image
import scipy.io as sio
import numpy as np, cv2
from openslide import OpenSlide
import matplotlib.pyplot as plt
os.add_dll_directory(r"C:\Program Files\MATLAB\R2025a\bin\win64")
import matlab.engine

Image.MAX_IMAGE_PIXELS = None  # Disable the limit

def save_ome_tif(image, output_name, pixelsize):
    """Exports an image in ome-tif format with metadata compatible with Qupath.

        Each ome-tif file will contain pyramidal copies saved at downsample factors of
        [1, 2, 4, 8, 16, 32] and tiled at a size of [1024 x 1024] for rapid loading
        """

    # some settings for the ome-tif file
    tile_size = 1024
    compression_quality = 95
    scale_factors = [1, 2, 4, 8, 16]  # Define downsampling factors to save in each ome-tif

    # Ensure shape matches expected format (Y, X, Channels)
    image = np.array(image)
    shape = image.shape
    axes = 'YXS' if shape[-1] == 1 else 'YXC'  # 'YXC' for RGB, 'YXS' for single-channel

    metadata = {
        'axes': axes,
        'SignificantBits': 8,
        'PhysicalSizeX': pixelsize,
        'PhysicalSizeXUnit': 'µm',
        'PhysicalSizeY': pixelsize,
        'PhysicalSizeYUnit': 'µm',
        'Software': 'tifffile',
    }

    options = dict(
        photometric='rgb' if shape[-1] == 3 else 'minisblack',
        tile=(tile_size, tile_size),
        compression='jpeg',
        compressionargs={"level": compression_quality},
        resolutionunit=3,  # 3 = Centimeter
    )

    with tifffile.TiffWriter(output_name, bigtiff=True) as tif:
        subifds_data = []  # List to store downsampled images for SubIFDs

        # Generate downsampled images
        for scale in scale_factors[1:]:  # Start from the second level (skip 1x)
            new_size = (shape[1] // scale, shape[0] // scale)
            if min(new_size) < 1:
                break  # Stop if downsampling is too small

            downsampled = Image.fromarray(image).resize(new_size, resample=Image.Resampling.LANCZOS)
            subifds_data.append(np.array(downsampled, dtype=np.uint8))

        # Save main image with SubIFDs for pyramidal TIFF structure
        tif.write(
            image.astype(np.uint8),
            subifds=len(subifds_data),  # Define how many SubIFDs will follow
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **options
        )

        # Write the SubIFDs (pyramidal levels)
        for idx, sub_image in enumerate(subifds_data):
            scale = scale_factors[idx + 1]  # Get scale factor
            res_val = 1e4 / scale / pixelsize

            tif.write(
                sub_image,
                subfiletype=1,  # Mark as pyramid level
                resolution=(res_val, res_val),
                **options
            )

        # Add a thumbnail image for QuPath and ImageScope
        thumbnail = image[::8, ::8]  # Downsample by factor of 8
        tif.write(thumbnail.astype(np.uint8), metadata={'Name': 'thumbnail'})

def load_mat_file(path):
    """
    Load variables from a MATLAB .mat file (handles v7.3 HDF5 and older).
    Returns a dict mapping variable names to numpy arrays / Python objects.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    # Try legacy MAT (<= v7.2) via SciPy
    try:
        try:
            data = sio.loadmat(path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
        except TypeError:
            # simplify_cells not available on older SciPy
            data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        # drop metadata keys
        return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception:
        # Fall back to HDF5 (v7.3) via h5py
        import h5py

        def _read_h5(obj):
            import h5py
            if isinstance(obj, h5py.Dataset):
                arr = obj[()]
                # convert bytes -> str where appropriate
                if isinstance(arr, bytes):
                    return arr.decode("utf-8", "ignore")
                if hasattr(arr, "dtype") and arr.dtype.kind == "S":
                    return arr.astype(str)
                return arr
            elif isinstance(obj, h5py.Group):
                return {k: _read_h5(obj[k]) for k in obj.keys()}
            else:
                return obj

        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = _read_h5(f[k])
        return out

def register_image_elastic(image, displacement_field, scale, image_elastic, tile=8192):
    """
    Apply a dense displacement field with tiling to avoid OpenCV SHRT_MAX (32767) limits.
    - image: HxWxC (uint8/uint16)
    - displacement_field: HxWx2 (dx, dy) at some base scale
    - scale: multiply the displacement after resizing to the image size
    - tile: tile size for remap (keep << 32767)
    """
    # Ensure types
    image = np.ascontiguousarray(image)
    H, W = image.shape[:2]
    C = 1 if image.ndim == 2 else image.shape[2]
    if C == 1:
        image = image[..., None]

    # Resize displacement to image size & scale
    displacement_field = np.asarray(displacement_field)
    #dy, dx = displacement_field[1], displacement_field[0]
    #dy, dx = dy.T, dx.T
    #dy = cv2.resize(dy.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * scale
    #dx = cv2.resize(dx.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * scale
    print(f"      displacement_field shape: {np.asarray(displacement_field).shape}")
    if displacement_field.shape[-1] == 2:  # (H, W, 2)
        dx, dy = displacement_field[..., 0], displacement_field[..., 1]
        dy = cv2.resize(dy.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * scale
        dx = cv2.resize(dx.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * scale
    elif displacement_field.shape[0] == 2: # (2, H, W)
        dx, dy = displacement_field[0, ...], displacement_field[1, ...]
        dy, dx = dy.T, dx.T
        dy = cv2.resize(dy.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * scale
        dx = cv2.resize(dx.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * scale
    else:
        raise ValueError(f"Unexpected shape for displacement field: {displacement_field.shape}")


    # Output
    out = np.empty_like(image)
    fill = (241, 241, 241)

    # Process tiles
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            ht = min(tile, H - y0)
            wt = min(tile, W - x0)

            dx_tile = dx[y0:y0+ht, x0:x0+wt]
            dy_tile = dy[y0:y0+ht, x0:x0+wt]

            # Compute a minimal source ROI that covers all mapped coords in this tile
            # (nearest-neighbor => 0.5px margin not needed)
            dx_min = float(np.floor(dx_tile.min()))
            dx_max = float(np.ceil(dx_tile.max()))
            dy_min = float(np.floor(dy_tile.min()))
            dy_max = float(np.ceil(dy_tile.max()))

            xs0 = max(0, int(x0 + dx_min))
            ys0 = max(0, int(y0 + dy_min))
            xs1 = min(W, int(x0 + wt + dx_max))
            ys1 = min(H, int(y0 + ht + dy_max))

            # Local source chunk (keeps src size < 32767)
            src = image[ys0:ys1, xs0:xs1, :]

            # Build local maps (float32)
            # Absolute base coords for this tile:
            xs = (x0 + np.arange(wt, dtype=np.float32))[None, :].repeat(ht, axis=0)
            ys = (y0 + np.arange(ht, dtype=np.float32))[:, None].repeat(wt, axis=1)
            map_x = xs + dx_tile
            map_y = ys + dy_tile

            # Shift maps into the local src-chunk coordinate system
            map_x_local = (map_x - xs0).astype(np.float32)
            map_y_local = (map_y - ys0).astype(np.float32)

            # Remap per channel
            for c in range(C):
                out[y0:y0+ht, x0:x0+wt, c] = cv2.remap(
                    src[..., c], map_x_local, map_y_local,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=float(fill[c if C > 1 else 0])
                )

    # Drop single-channel axis if needed
    out = out[..., 0] if C == 1 else out
    out = np.ascontiguousarray(out)

    #overlay = overlay_greyscale(image_elastic, out, 1)
    return out

def transform_image(image, tform_T, flip, cent=None, scale=1.0, fill=(241, 241, 241), tile=4096):
    """
    MATLAB-equivalent:
        cent = cent * scale;
        tform.T(3,1:2) = tform.T(3,1:2) * scale;
        Rin = imref2d(size(IM));
        Rin.XWorldLimits -= cent(1); Rin.YWorldLimits -= cent(2);
        IM = imwarp(IM, Rin, tform, 'nearest', 'outputview', Rin, 'fillvalues', fill);
    """
    import numpy as np, cv2

    img = np.ascontiguousarray(image)
    h, w = img.shape[:2]
    C = img.shape[2] if img.ndim == 3 else 1
    if C == 1:
        img = img[..., None]

    # (1) optional vertical flip (MATLAB flips before register_IM)
    if int(flip) == 1:
        img = img[::-1, ...]

    # (2) center shift (MATLAB: [cent_x, cent_y], scaled)
    if cent is None:
        cx = cy = 0.0
    else:
        cx = float(cent[0]) * float(scale)  # X (cols)
        cy = float(cent[1]) * float(scale)  # Y (rows)

    # (3) MATLAB affine2d.T is row-vector form; scale translation, then transpose
    T_row = np.array(tform_T, dtype=np.float64, copy=True)
    if T_row.shape != (3, 3):
        raise ValueError("tform_T must be 3x3 (MATLAB affine2d.T)")
    T_row[2, 0:2] *= float(scale)
    T_col = T_row.T  # convert to column-vector convention

    # (4) imref2d world=index-cent
    S     = np.array([[1, 0, -cx],
                      [0, 1, -cy],
                      [0, 0,   1]], dtype=np.float64)
    S_inv = np.array([[1, 0,  cx],
                      [0, 1,  cy],
                      [0, 0,   1]], dtype=np.float64)

    # forward (src→dst) in index space
    M_full = S_inv @ T_col @ S

    # OpenCV needs inverse (dst→src)
    Minv = np.linalg.inv(M_full)
    M_cv2 = Minv[:2, :].astype(np.float32)

    # Small images: single warp (matches MATLAB: nearest + constant fill)
    #if h < 32767 and w < 32767 and img.shape[0] < 32767 and img.shape[1] < 32767:
    #    out = cv2.warpAffine(img, M_cv2, (w, h),
    #                         flags=cv2.INTER_NEAREST,
    #                         borderMode=cv2.BORDER_CONSTANT,
    #                         borderValue=fill)
    #    return np.ascontiguousarray(out[..., 0] if C == 1 else out)

    # Large images: tile the DESTINATION, but compute maps in GLOBAL coords,
    # then remap from a cropped SOURCE ROI so neither src nor dst exceed SHRT_MAX.
    a, b, tx = float(M_cv2[0, 0]), float(M_cv2[0, 1]), float(M_cv2[0, 2])
    c, d, ty = float(M_cv2[1, 0]), float(M_cv2[1, 1]), float(M_cv2[1, 2])
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    out = np.empty_like(img)

    # 1-px guard ensures ROI fully covers all sampled coords
    GUARD = 1

    for y0 in range(0, h, tile):
        ht = min(tile, h - y0)
        Y = ys[y0:y0+ht][:, None]                # (ht,1)
        for x0 in range(0, w, tile):
            wt = min(tile, w - x0)
            X = xs[x0:x0+wt][None, :]            # (1,wt)

            # Global dest->source maps for this dest tile
            map_x = (a * X + b * Y + tx).astype(np.float32)  # (ht,wt)
            map_y = (c * X + d * Y + ty).astype(np.float32)  # (ht,wt)

            # Compute minimal SOURCE ROI that covers all mapped coords (+guard)
            xs0 = max(0, int(np.floor(map_x.min())) - GUARD)
            ys0 = max(0, int(np.floor(map_y.min())) - GUARD)
            xs1 = min(w, int(np.ceil(map_x.max())) + 1 + GUARD)
            ys1 = min(h, int(np.ceil(map_y.max())) + 1 + GUARD)

            if xs1 <= xs0 or ys1 <= ys0:
                # Everything maps outside the source → fill
                out[y0:y0+ht, x0:x0+wt, :] = np.array(fill, dtype=img.dtype)
                continue

            src_roi = img[ys0:ys1, xs0:xs1, :]  # guaranteed < SHRT_MAX

            # Shift maps into ROI-local coords
            map_x_local = (map_x - xs0).astype(np.float32)
            map_y_local = (map_y - ys0).astype(np.float32)

            # Remap per channel with constant fill (matches MATLAB 'fillvalues')
            for ch in range(C):
                out[y0:y0+ht, x0:x0+wt, ch] = cv2.remap(
                    src_roi[..., ch], map_x_local, map_y_local,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=float(fill[ch if C > 1 else 0])
                )

    return np.ascontiguousarray(out[..., 0] if C == 1 else out)

def pad_image(image, refsize, padall):
    """pads a pixmap image to a defined width and height.
    Used in multiple tabs
    Args:
        image: image array
        refsize: maximum size of images in the registration project
        padall: additional sizes to add to the registered image
    Returns:
        image_pad: padded image array
        image_pad_grey: greyscale padded image array
    """

    fillval = (241, 241, 241)
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image must be an (H, W, 3) array")

    refsize = np.asarray(refsize, dtype=int).reshape(2)  # [rows, cols]
    H, W = img.shape[:2]
    szim = refsize - np.array([H, W], dtype=int)  # how much to add overall

    if (szim < 0).any():
        raise ValueError(f"refsize {refsize.tolist()} is smaller than image size {[H, W]}")

    padall_arr = np.asarray(padall)
    if padall_arr.size == 1:
        padall_arr = np.array([int(padall_arr), int(padall_arr)], dtype=int)
    else:
        padall_arr = padall_arr.astype(int).reshape(2)

    szA = (szim // 2).astype(int)  # pre pad (base)
    szB = (szim - szA + padall_arr).astype(int)  # post pad (base + padall)
    szA = (szA + padall_arr).astype(int)  # pre pad (+ padall)

    # Build pad widths: ((pre_rows, post_rows), (pre_cols, post_cols))
    pads_2d = ((int(szA[0]), int(szB[0])), (int(szA[1]), int(szB[1])))

    # Per-channel constant padding (MATLAB pads each channel separately with its own fill)
    ch0 = np.pad(img[..., 0], pads_2d, mode='constant', constant_values=int(fillval[0]))
    ch1 = np.pad(img[..., 1], pads_2d, mode='constant', constant_values=int(fillval[1]))
    ch2 = np.pad(img[..., 2], pads_2d, mode='constant', constant_values=int(fillval[2]))

    image_pad = np.stack((ch0, ch1, ch2), axis=2).astype(img.dtype, copy=False)
    return image_pad

def overlay_rgb(image_a, image_b, view_image, alpha=0.5, filename=None):
    """
    Overlay image_b on image_a.
    """
    image_a = np.asarray(image_a, dtype=np.uint8)
    image_b = np.asarray(image_b, dtype=np.uint8)
    assert image_a.ndim == 3 and image_a.shape[2] == 3
    assert image_b.ndim == 3 and image_b.shape[2] == 3

    h, w = image_a.shape[:2]
    b_resized = cv2.resize(image_b, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(image_a, 1.0 - float(alpha), b_resized, float(alpha), 0)

    if view_image:
        plt.imshow(overlay), plt.axis('off'), plt.show()

    if filename:
        Image.fromarray(overlay).save(filename, quality=95, subsampling=0, optimize=True, dpi=(300, 300))

    return overlay

def overlay_greyscale(image_a, image_b, view_image, filename=None):
    """
    Overlay image_b on image_a.
    """
    image_a = np.asarray(image_a, dtype=np.uint8)
    image_b = np.asarray(image_b, dtype=np.uint8)
    assert image_a.ndim == 3 and image_a.shape[2] == 3
    assert image_b.ndim == 3 and image_b.shape[2] == 3

    h, w = image_a.shape[:2]
    image_b = cv2.resize(image_b, (w, h), interpolation=cv2.INTER_NEAREST)
    image_a = cv2.bitwise_not(cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY))
    image_b = cv2.bitwise_not(cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY))
    overlay = np.dstack([image_a, image_a, image_b])

    if view_image:
        plt.imshow(overlay), plt.axis('off'), plt.show()

    if filename:
        Image.fromarray(overlay).save(filename, quality=95, subsampling=0, optimize=True, dpi=(300, 300))

    return overlay

def apply_CODA_registration(image, image_name, pthdata, scale, view_image):

    # Load the registration metadata
    global_registration_file = os.path.join(pthdata, image_name+'.mat')
    elastic_registration_file = os.path.join(pthdata, 'D', image_name+'.mat')

    # load the original registered images
    original_image_file = os.path.join(pthdata, '../../../', image_name + '.tif')
    image_original = Image.open(original_image_file)
    global_registration_image_file = os.path.join(pthdata, '../../', image_name + '.jpg')
    image_global = Image.open(global_registration_image_file)
    elastic_registration_image_file = os.path.join(pthdata, '../', image_name + '.jpg')
    image_elastic = Image.open(elastic_registration_image_file)
    #scale = image.size[0] / image_original.size[0]

    # get the padding information
    mat_files = sorted(glob.glob(os.path.join(pthdata, "*.mat")))
    image_size_file = mat_files[0]
    vars_dict = load_mat_file(image_size_file)
    szz = vars_dict.get('szz')
    refsize = np.ceil(szz * scale).astype(np.int64)
    padall = ceil(np.asarray(vars_dict.get('padall', 0)).squeeze().item() * float(scale))
    image = pad_image(image, refsize, padall)

    # account for the situation where the mat file exists but tform is missing (reference image
    moving_image = 0
    if os.path.isfile(global_registration_file):
        vars_dict = load_mat_file(global_registration_file)
        moving_image = int('tform_python' in vars_dict) or int('f' in vars_dict)

    # account for reference image
    if not moving_image:
        print("         ...reference image, registration not required. save padded image.")
        overlay = overlay_greyscale(image_elastic, image, view_image)
        return image, overlay

    # load image-specific registration information
    vars_dict = load_mat_file(global_registration_file)
    vars_dict_elastic = load_mat_file(elastic_registration_file)

    # get tform from affine2D object and calculate global registration
    eng = matlab.engine.start_matlab()
    eng.eval(f"load('{global_registration_file.replace('\\', '/')}')", nargout=0)
    eng.eval("T = tform.T;", nargout=0)  # extract the 3x3 from the object
    tform = np.array(eng.workspace['T'])  # -> NumPy (3,3)
    eng.quit()
    flip = int(np.asarray(vars_dict.get('f')).squeeze().item())
    cent_arr = np.asarray(vars_dict['cent']).squeeze()
    cy, cx = (int(round(v)) for v in cent_arr.tolist())  # or swap to (cx, cy) if your code expects x,y
    cent = (cy, cx)
    image = transform_image(image, tform, flip, cent, scale)
    #overlay_greyscale(image_global, image, view_image)

    # elastic registration
    D = vars_dict_elastic.get('D')
    image = register_image_elastic(image, D, scale, image_elastic)

    overlay = overlay_greyscale(image_elastic, image, view_image)
    print("         ...registration successful!")

    return image, overlay

def process_images(pthim, image_list, pthdata, sx, out_folder, view_image=0, ome=1):
    """Process missing images by converting .ndpi, .svs, .scn, or .tif files to .tif or .ome.tif."""

    # Ensure the image directory exists
    out_folder_v = os.path.join(out_folder, 'validation_overlay')
    if not os.path.isdir(out_folder_v):
        os.makedirs(out_folder_v)

    for idx, image_in_list in enumerate(image_list):
        print(f"  Starting image {idx + 1} of {len(image_list)}: {image_in_list}...")

        # check if the image is already registered
        image_name = image_in_list.rsplit('.', 1)[0]
        if ome==1:
            output_name = os.path.join(out_folder, image_name + '.ome.tif')
        else:
            output_name = os.path.join(out_folder, image_name + '.tif')
        if os.path.exists(output_name):
            print(f"    ...already saved this file")
            continue

        # Read the image
        slide_path = os.path.join(pthim, image_in_list)
        try:
            # Get file extension
            file_ext = os.path.splitext(slide_path)[-1].lower()
            if file_ext in ['.ndpi', '.svs', '.scn']:
                print(f"    ...reading {slide_path} with OpenSlide")
                wsi = OpenSlide(slide_path)
                image = wsi.read_region(location=(0, 0), level=0, size=wsi.level_dimensions[0]).convert('RGB')
                w, h = image.width, image.height
                mppx, mppy = float(wsi.properties['openslide.mpp-x']), float(wsi.properties['openslide.mpp-y'])
            elif file_ext in ['.tif', '.png', '.jpg']:
                print(f"    ...reading {slide_path} with PIL")
                image = Image.open(slide_path)
                w, h = image.size[:2]
                mppx, mppy = 1, 1

            print(f"       ...image read successfully - file parameters: resolution of {mppx} and size of ({w}, {h})")

        except Exception as e:
            print(f"       ...ERROR reading {image_in_list}: {e}")
            continue

        # get the scale between the high-resolution and registered images
        scale = sx / mppx

        # register the image
        image, overlay = apply_CODA_registration(image, image_name, pthdata, scale, view_image)
        filename = os.path.join(out_folder_v, image_name + '.jpg')
        Image.fromarray(overlay).save(filename, quality=95, subsampling=0, optimize=True, dpi=(300, 300))

        # save the file as either a normal or an ome-tif
        if ome == 1:
            print("             ...saving as an ome tif")
            save_ome_tif(image, output_name, mppx)
        else:
            try: # save as normal tif
                image.save(output_name, resolution=1, resolution_unit=1, quality=100, compression=None)
            except Exception as e: # save as ome-tif
                print(f"          ...error saving {image_in_list} as tif: {e}, try saving this image as an ome-tif")
                continue

        print("  Image save successful!")
        print("  ")

def apply_registration_to_20x(pthim, pthdata, sx, validate_imgs=0, out_folder=None, ome=1):
    print('Applying CODA registration to svs images:')

    # Get the .ndpi and .svs image names, sorted alphabetically
    patterns = ['*.ndpi', '*.svs', '*.scn', '*.czi', '*.tif', '*.png', '*.jpg']
    image_list = [file for pattern in patterns for file in glob.glob(os.path.join(pthim, pattern))]
    image_list = sorted(image_list)
    image_list = [os.path.basename(file) for file in image_list]
    if not image_list:
        print("  No image files found.")
        return
    #else:
    #    print(f"  Of {len(image_list)} image files:")

    # --- remove images from the list that don't have a corresponding .mat in pthdata ---
    mat_paths = glob.glob(os.path.join(pthdata, "D", "*.mat"))
    mat_names = {os.path.splitext(os.path.basename(p))[0] for p in mat_paths}  # basenames without .mat

    filtered, dropped = [], []
    for img in image_list:
        base = os.path.splitext(img)[0]
        if base in mat_names:
            filtered.append(img)
        else:
            dropped.append(img)

    #if filtered:
    #    print(f"     - found {len(filtered)} image file(s) with matching .mat files.")
    #if dropped:
    #    print(f"     - skipping {len(dropped)} image file(s) without matching .mat files")

    image_list = filtered
    if not image_list:
        print("  No images have matching .mat files; nothing to process.")
        return

    if not out_folder:
        out_folder = os.path.join(pthim, 'registeredE')
    process_images(pthim, image_list, pthdata, sx, out_folder, validate_imgs, ome)

if __name__ == '__main__':

    which_folder = 1

    if which_folder == 1:
        pthimA = r'\\10.99.134.183\kiemen-lab-data\Yu Shen\T1D\HE2IHC\all images combined\IHC'
        pthimB = r'\\10.99.134.183\kiemen-lab-data\Yu Shen\T1D\HE2IHC\all images combined\HE'
        out_folder = r'\\10.99.134.183\kiemen-lab-data\Yu Shen\T1D\HE2IHC\make 20x dataset\registered 20x b'
        sx = 5  # micron / pixel resolution of the images used for registration
        view_image = 0

        pth0 = r'\\10.99.134.183\kiemen-lab-data\Yu Shen\T1D\HE2IHC\make 20x dataset\correct_registration_metadata'
        subfolders = [d for d in os.listdir(pth0) if os.path.isdir(os.path.join(pth0, d))]
        subfolders = [d for d in subfolders if d.isdigit() and len(d) == 3]

        for sub in subfolders:
            sub_root = os.path.join(pth0, sub)
            registered_dirs = [
                d for d in os.listdir(sub_root)
                if os.path.isdir(os.path.join(sub_root, d)) and d.lower().startswith('registered')
            ]

            if len(registered_dirs) == 0:
                print(f"[WARN] No 'registered*' folder found in {sub_root}; skipping.")
                continue
            pthdata = os.path.join(sub_root, registered_dirs[0], 'elastic registration', 'save_warps')
            if not os.path.isdir(pthdata):
                print(f"[WARN] Expected path not found: {pthdata}; skipping.")
                continue

            print(f"Using registration data: {pthdata}")
            apply_registration_to_20x(pthimA, pthdata, sx, view_image, out_folder)
            apply_registration_to_20x(pthimB, pthdata, sx, view_image, out_folder)