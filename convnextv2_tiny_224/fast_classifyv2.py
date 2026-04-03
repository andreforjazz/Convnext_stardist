#!/usr/bin/env python3
"""
TILE-BASED Nucleus Classification Pipeline
- Correct patch size (256 to match training)
- Optimized for speed
- QuPath-compatible output
"""

import argparse
import json
import gzip
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
from openslide import OpenSlide
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Try fast JSON
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

# CUDA optimizations
USE_AMP = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_geojson(path):
    """Load nuclei from GeoJSON"""
    print(f"[INFO] Loading GeoJSON: {path}")
    with open(path, 'rb' if USE_ORJSON else 'r') as f:
        data = orjson.loads(f.read()) if USE_ORJSON else json.load(f)
    
    nuclei = []
    features = data.get('features', data) if isinstance(data, dict) else data
    
    for feat in features:
        geom = feat.get('geometry', {})
        coords = geom.get('coordinates', [[]])[0]
        
        if len(coords) >= 3:
            coords_arr = np.array(coords)
            centroid = coords_arr.mean(axis=0)
            nuclei.append({
                'feature': feat,
                'centroid': centroid,
            })
    
    print(f"[INFO] Loaded {len(nuclei)} nuclei")
    return nuclei


def group_nuclei_by_tile(nuclei, tile_size=2048):
    """Group nuclei by tile"""
    tile_to_nuclei = defaultdict(list)
    
    for i, nuc in enumerate(nuclei):
        cx, cy = nuc['centroid']
        tile_x = int(cx // tile_size) * tile_size
        tile_y = int(cy // tile_size) * tile_size
        tile_to_nuclei[(tile_x, tile_y)].append(i)
    
    return tile_to_nuclei


def extract_patches_from_tile(tile_np, nuclei, indices, tile_x, tile_y, 
                               patch_size=256, half_pad=256):
    """Extract all patches from a loaded tile"""
    patches = []
    valid_indices = []
    half = patch_size // 2
    
    h, w = tile_np.shape[:2]
    
    for idx in indices:
        cx, cy = nuclei[idx]['centroid']
        
        # Convert to tile-local coordinates
        local_x = int(cx - tile_x + half_pad)
        local_y = int(cy - tile_y + half_pad)
        
        # Check bounds
        if (half <= local_x < w - half and half <= local_y < h - half):
            patch = tile_np[local_y - half:local_y + half, 
                           local_x - half:local_x + half]
            
            if patch.shape == (patch_size, patch_size, 3):
                patches.append(patch)
                valid_indices.append(idx)
    
    return patches, valid_indices


@torch.no_grad()
def batch_classify(patches, model, processor, batch_size=256):
    """Classify patches in batches"""
    if not patches:
        return []
    
    all_preds = []
    
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs['pixel_values'].cuda()
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            logits = model(pixel_values=pixel_values).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
        
        all_preds.extend(preds)
    
    return all_preds


def classify_tile_based(slide_path, nuclei, model, processor, id2label,
                        tile_size=2048, patch_size=256, batch_size=256):
    """Tile-based classification with correct patch size"""
    slide = OpenSlide(slide_path)
    
    # Padding must accommodate patch_size
    half_pad = patch_size
    
    # Group nuclei by tile
    tile_to_nuclei = group_nuclei_by_tile(nuclei, tile_size)
    print(f"[INFO] Grouped into {len(tile_to_nuclei)} tiles")
    
    total_classified = 0
    pbar = tqdm(tile_to_nuclei.items(), desc="Processing tiles", unit="tile")
    
    for (tile_x, tile_y), indices in pbar:
        # Read tile with padding
        padded_size = tile_size + 2 * half_pad
        
        try:
            tile = slide.read_region(
                (tile_x - half_pad, tile_y - half_pad),
                0,
                (padded_size, padded_size)
            ).convert('RGB')
            tile_np = np.array(tile)
        except Exception as e:
            print(f"[WARN] Failed to read tile at ({tile_x}, {tile_y}): {e}")
            continue
        
        # Extract patches
        patches, valid_indices = extract_patches_from_tile(
            tile_np, nuclei, indices, tile_x, tile_y, patch_size, half_pad
        )
        
        # Classify
        if patches:
            predictions = batch_classify(patches, model, processor, batch_size)
            
            for idx, pred in zip(valid_indices, predictions):
                label = id2label.get(pred, f"class_{pred}")
                nuclei[idx]['classification'] = {
                    'name': label,
                    'index': int(pred)
                }
            
            total_classified += len(predictions)
        
        pbar.set_postfix({'classified': total_classified})
    
    return nuclei, total_classified


def save_geojson(nuclei, output_path, id2label, precision=1):
    """Save classified nuclei to GeoJSON - QuPath compatible, optimized"""
    print(f"[INFO] Saving to {output_path}")
    
    # Color map for classes
    CLASS_COLORS = {
        "bladder": [255, 0, 0],
        "bone": [255, 128, 0],
        "brain": [255, 255, 0],
        "collagen": [128, 255, 0],
        "ear": [0, 255, 0],
        "eye": [0, 255, 128],
        "gi": [0, 255, 255],
        "heart": [0, 128, 255],
        "kidney": [0, 0, 255],
        "liver": [128, 0, 255],
        "lungs": [255, 0, 255],
        "mesokidney": [255, 0, 128],
        "nontissue": [255, 255, 255],
        "pancreas": [128, 128, 128],
        "skull": [64, 64, 64],
        "spleen": [255, 192, 203],
        "spleen2": [255, 182, 193],
        "thymus": [173, 216, 230],
        "thyroid": [144, 238, 144],
    }
    DEFAULT_COLOR = [200, 200, 200]
    
    features = []
    
    for nuc in nuclei:
        orig_feat = nuc['feature']
        orig_geom = orig_feat.get('geometry', {})
        orig_coords = orig_geom.get('coordinates', [[]])[0]
        
        # Round coordinates to reduce file size
        rounded_coords = [[round(x, precision), round(y, precision)] 
                          for x, y in orig_coords]
        
        # Build minimal feature
        if 'classification' in nuc:
            class_name = nuc['classification']['name']
            class_idx = nuc['classification']['index']
            color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
            
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [rounded_coords]
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": class_name,
                        "color": color
                    }
                }
            }
        else:
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [rounded_coords]
                },
                "properties": {
                    "objectType": "annotation"
                }
            }
        
        features.append(feat)
    
    output = {
        "type": "FeatureCollection",
        "features": features
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with compact JSON
    if USE_ORJSON:
        with open(output_path, 'wb') as f:
            f.write(orjson.dumps(output))
    else:
        with open(output_path, 'w') as f:
            json.dump(output, f, separators=(',', ':'))
    
    # Print size info
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[INFO] Saved {len(features)} features ({size_mb:.1f} MB)")


def load_model(ckpt_path):
    """Load ConvNeXt model and processor"""
    print(f"[INFO] Loading model from: {ckpt_path}")
    
    processor = AutoImageProcessor.from_pretrained(ckpt_path)
    model = AutoModelForImageClassification.from_pretrained(ckpt_path)
    model = model.cuda().eval()
    
    # Get label mapping
    id2label = model.config.id2label
    print(f"[INFO] Classes: {len(id2label)}")
    for i, label in id2label.items():
        print(f"  {i}: {label}")
    
    # Print expected input size
    size = processor.size
    print(f"[INFO] Processor expects size: {size}")
    
    return model, processor, id2label


def main():
    parser = argparse.ArgumentParser(description="Tile-based nucleus classification")
    parser.add_argument("--ndpi", required=True, help="Path to NDPI/SVS slide")
    parser.add_argument("--geojson", required=True, help="Path to input GeoJSON")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--out", required=True, help="Output GeoJSON path")
    parser.add_argument("--tile_size", type=int, default=2048, help="Tile read size")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size (MUST match training!)")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--precision", type=int, default=1, help="Coordinate decimals")
    parser.add_argument("--max_n", type=int, default=None, help="Max nuclei (debug)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TILE-BASED NUCLEUS CLASSIFICATION")
    print("=" * 60)
    print(f"Slide:      {args.ndpi}")
    print(f"GeoJSON:    {args.geojson}")
    print(f"Tile size:  {args.tile_size} (I/O optimization)")
    print(f"Patch size: {args.patch_size} (CNN input - must match training!)")
    print(f"Batch size: {args.batch}")
    print(f"Precision:  {args.precision} decimals")
    print("=" * 60)
    
    # Load model
    model, processor, id2label = load_model(args.ckpt)
    
    # Load nuclei
    nuclei = load_geojson(args.geojson)
    
    if args.max_n:
        nuclei = nuclei[:args.max_n]
        print(f"[INFO] Limited to {len(nuclei)} nuclei")
    
    # Classify
    nuclei, total = classify_tile_based(
        args.ndpi, nuclei, model, processor, id2label,
        tile_size=args.tile_size,
        patch_size=args.patch_size,
        batch_size=args.batch
    )
    
    # Save
    save_geojson(nuclei, args.out, id2label, precision=args.precision)
    
    print("=" * 60)
    print(f"DONE! Classified {total} nuclei")
    print("=" * 60)


if __name__ == "__main__":
    main()