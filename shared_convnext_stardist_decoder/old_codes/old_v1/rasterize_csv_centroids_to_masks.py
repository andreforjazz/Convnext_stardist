import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

def main():
    # Directories
    dataset_root = Path(r"\\kittyserverdw\Andre_kit\data\students\Diogo\data\fetal\GS40\cellvit_training\data_for_cellvit_GS40_balanced")
    
    images_dir = dataset_root / "train" / "images"
    csv_labels_dir = dataset_root / "train" / "labels"
    
    # Destination
    out_train_labels = dataset_root / "stardist_multitask_ready" / "train_instance_labels"
    out_train_labels.mkdir(parents=True, exist_ok=True)
    
    # Get all csv files
    csv_files = list(csv_labels_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files with centroids.")
    
    success = 0
    radius = 8  # Average radius of a nucleus
    
    for csv_path in tqdm(csv_files):
        stem = csv_path.stem
        
        # Find corresponding image to get shape
        img_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif"]:
            if (images_dir / f"{stem}{ext}").exists():
                img_path = images_dir / f"{stem}{ext}"
                break
                
        if img_path is None:
            continue
            
        with Image.open(img_path) as img:
            w, h = img.size
            
        # Create blank mask (16-bit to allow >255 nuclei)
        mask = np.zeros((h, w), dtype=np.uint16)
        
        try:
            # Read the GS40 CSV (no header, cols are x, y, class)
            df = pd.read_csv(csv_path, header=None)
        except Exception:
            continue
            
        if len(df) > 0 and len(df.columns) >= 3:
            df = df.iloc[:, :3]
            df.columns = ["x", "y", "class"]
            
            # Draw each centroid as a circle
            for i, row in df.iterrows():
                try:
                    cx = int(float(row["x"]))
                    cy = int(float(row["y"]))
                    # ID starts from 1, background is 0
                    cv2.circle(mask, (cx, cy), radius, int(i + 1), -1)
                except ValueError:
                    pass
                
        # Save as PNG
        out_path = out_train_labels / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)
        success += 1
        
    print(f"Done! Successfully rasterized {success} instance masks into:")
    print(f"  {out_train_labels}")

if __name__ == "__main__":
    main()
