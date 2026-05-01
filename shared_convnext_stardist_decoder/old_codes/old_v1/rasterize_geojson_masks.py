import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def main():
    dataset_root = Path(r"\\kittyserverdw\Andre_kit\data\students\Diogo\data\fetal\GS40\cellvit_training\data_for_cellvit_GS40_balanced")
    
    geojsons_dir = dataset_root / "train" / "images" / "StarDist_4_3_2026_cross_fetal_species" / "json" / "geojsons" / "32_polys_20x"
    images_dir = dataset_root / "train" / "images"
    
    out_train_labels = dataset_root / "stardist_multitask_ready" / "train_instance_labels"
    out_train_labels.mkdir(parents=True, exist_ok=True)
    
    geojson_files = list(geojsons_dir.glob("*.geojson"))
    print(f"Found {len(geojson_files)} GeoJSON files to rasterize.")
    
    success = 0
    for geo_path in tqdm(geojson_files):
        stem = geo_path.stem
        
        img_path = None
        for ext in [".png", ".jpg", ".tif"]:
            if (images_dir / f"{stem}{ext}").exists():
                img_path = images_dir / f"{stem}{ext}"
                break
                
        if img_path is None:
            continue
            
        with Image.open(img_path) as img:
            w, h = img.size
            
        mask = np.zeros((h, w), dtype=np.uint16)
        
        with open(geo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Draw each polygon
        # GeoJSON is usually a list of features, or a FeatureCollection
        features = data if isinstance(data, list) else data.get("features", [])
        
        inst_id = 1
        for feat in features:
            if feat.get("geometry", {}).get("type") == "Polygon":
                coords = feat["geometry"]["coordinates"][0]
                
                # coordinates are [x, y]
                pts = np.array(coords, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                cv2.fillPoly(mask, [pts], color=inst_id)
                inst_id += 1
                
        out_path = out_train_labels / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)
        success += 1
        
    print(f"Done! Successfully rasterized {success} instance masks into:")
    print(f"  {out_train_labels}")

if __name__ == "__main__":
    main()
