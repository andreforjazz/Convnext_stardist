import json
import os
from pathlib import Path
import numpy as np
import cv2
import tifffile
from PIL import Image
from tqdm import tqdm

def main():
    # Directories
    dataset_root = Path(r"\\kittyserverdw\Andre_kit\data\students\Diogo\data\fetal\GS40\cellvit_training\data_for_cellvit_GS40_balanced")
    
    json_dir = dataset_root / "train" / "images" / "StarDist_4_3_2026_cross_fetal_species" / "json"
    images_dir = dataset_root / "train" / "images"
    
    # Destination
    out_train_labels = dataset_root / "stardist_multitask_ready" / "train_instance_labels"
    out_val_labels = dataset_root / "stardist_multitask_ready" / "val_instance_labels"
    
    out_train_labels.mkdir(parents=True, exist_ok=True)
    out_val_labels.mkdir(parents=True, exist_ok=True)
    
    # We'll just put them all in train_instance_labels since train_multitask_GS40_paths.ipynb 
    # uses MT_TRAIN_INSTANCE_LABELS for the training data (and potentially validation as well, 
    # or you can copy the val ones over). To be safe, we will just dump everything into one folder, 
    # and the notebook will fetch the ones it needs.
    dest_dir = out_train_labels

    # Get all jsons
    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to rasterize.")
    
    success = 0
    for j_path in tqdm(json_files):
        stem = j_path.stem
        
        # Find corresponding image to get shape
        img_path = None
        for ext in [".png", ".jpg", ".tif"]:
            if (images_dir / f"{stem}{ext}").exists():
                img_path = images_dir / f"{stem}{ext}"
                break
                
        if img_path is None:
            continue
            
        with Image.open(img_path) as img:
            w, h = img.size
            
        # Create blank mask (16-bit to allow >255 nuclei)
        mask = np.zeros((h, w), dtype=np.uint16)
        
        with open(j_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Draw each polygon
        for i, obj in enumerate(data, start=1):
            if "contour" in obj and len(obj["contour"]) > 0:
                # StarDist format: obj["contour"][0] is [ [y_coords], [x_coords] ]
                y_coords = obj["contour"][0][0]
                x_coords = obj["contour"][0][1]
                
                # Combine into shape (N, 1, 2) which cv2.fillPoly expects as [X, Y]
                pts = np.column_stack((x_coords, y_coords)).astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                cv2.fillPoly(mask, [pts], color=i)
                
        # Save as PNG (16-bit works seamlessly with cv2)
        out_path = dest_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)
        success += 1
        
    print(f"Done! Successfully rasterized {success} instance masks into:")
    print(f"  {dest_dir}")

if __name__ == "__main__":
    main()
