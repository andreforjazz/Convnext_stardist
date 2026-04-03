#!/usr/bin/env python3
"""
MAXIMUM PERFORMANCE Nucleus Classification Pipeline
Fixed version - Windows compatible
"""

import argparse
import json
import numpy as np
from pathlib import Path
from threading import Thread, Event
from queue import Queue
import time
import gc
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from openslide import OpenSlide
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Optional fast JSON
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class FastGPUPreprocessor:
    """GPU preprocessing"""
    
    def __init__(self, processor, patch_size=256, target_size=224):
        self.patch_size = patch_size
        self.target_size = target_size
        self.mean = torch.tensor(processor.image_mean, device='cuda', dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor(processor.image_std, device='cuda', dtype=torch.float16).view(1, 3, 1, 1)
    
    def __call__(self, patches_np):
        if isinstance(patches_np, list):
            patches_np = np.stack(patches_np)
        
        x = torch.from_numpy(patches_np).cuda(non_blocking=True)
        x = x.permute(0, 3, 1, 2).to(torch.float16) / 255.0
        
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size), 
                             mode='bilinear', align_corners=False)
        
        x = (x - self.mean) / self.std
        return x


class RAMSlideCache:
    """Load slide into RAM"""
    
    def __init__(self, slide_path, nuclei_bounds=None, padding=512):
        print(f"[RAM] Opening: {slide_path}")
        slide = OpenSlide(slide_path)
        dims = slide.dimensions
        print(f"[RAM] Dimensions: {dims[0]:,} x {dims[1]:,}")
        
        if nuclei_bounds is not None:
            x_min, y_min, x_max, y_max = nuclei_bounds
            x_min = max(0, int(x_min) - padding)
            y_min = max(0, int(y_min) - padding)
            x_max = min(dims[0], int(x_max) + padding)
            y_max = min(dims[1], int(y_max) + padding)
        else:
            x_min, y_min = 0, 0
            x_max, y_max = dims[0], dims[1]
        
        self.offset_x = x_min
        self.offset_y = y_min
        width = x_max - x_min
        height = y_max - y_min
        
        mem_gb = (width * height * 3) / (1024**3)
        print(f"[RAM] Region: {width:,} x {height:,} ({mem_gb:.1f} GB)")
        
        t0 = time.time()
        chunk_height = 8192
        chunks = []
        
        for y in tqdm(range(y_min, y_max, chunk_height), desc="Loading slide"):
            h = min(chunk_height, y_max - y)
            region = slide.read_region((x_min, y), 0, (width, h)).convert('RGB')
            chunks.append(np.array(region, dtype=np.uint8))
        
        self.image = np.vstack(chunks)
        print(f"[RAM] Loaded in {time.time()-t0:.1f}s")
        slide.close()
        gc.collect()
    
    def extract_patches_batch(self, centroids, patch_size):
        half = patch_size // 2
        h, w = self.image.shape[:2]
        
        valid_patches = []
        valid_indices = []
        
        for i, (cx, cy) in enumerate(centroids):
            local_x = int(cx) - self.offset_x
            local_y = int(cy) - self.offset_y
            
            y1, y2 = local_y - half, local_y + half
            x1, x2 = local_x - half, local_x + half
            
            if 0 <= y1 and y2 <= h and 0 <= x1 and x2 <= w:
                patch = self.image[y1:y2, x1:x2]
                if patch.shape == (patch_size, patch_size, 3):
                    valid_patches.append(patch)
                    valid_indices.append(i)
        
        if valid_patches:
            return np.stack(valid_patches), valid_indices
        return None, []


class ParallelBatchProducer:
    """Multi-threaded batch producer"""
    
    def __init__(self, slide_cache, centroids, patch_size, batch_size, num_workers=8):
        self.slide_cache = slide_cache
        self.centroids = centroids
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = Queue(maxsize=6)
        self.stop_event = Event()
        self.total_nuclei = len(centroids)
        self.num_batches = (self.total_nuclei + batch_size - 1) // batch_size
        self.workers = []
    
    def _worker(self, batch_indices):
        for batch_idx in batch_indices:
            if self.stop_event.is_set():
                break
            
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.total_nuclei)
            
            patches_np, valid_local = self.slide_cache.extract_patches_batch(
                self.centroids[start_idx:end_idx], self.patch_size
            )
            
            if patches_np is not None:
                valid_global = [start_idx + i for i in valid_local]
                self.queue.put((patches_np, valid_global, batch_idx))
            else:
                self.queue.put((None, [], batch_idx))
    
    def start(self):
        assignments = [[] for _ in range(self.num_workers)]
        for i in range(self.num_batches):
            assignments[i % self.num_workers].append(i)
        
        for assignment in assignments:
            if assignment:
                t = Thread(target=self._worker, args=(assignment,), daemon=True)
                t.start()
                self.workers.append(t)
        
        return self.num_batches
    
    def get_batch(self, timeout=120):
        return self.queue.get(timeout=timeout)
    
    def stop(self):
        self.stop_event.set()


@torch.no_grad()
def classify_fast(slide_path, nuclei, model, gpu_preprocessor, id2label,
                  patch_size=256, batch_size=128, num_workers=8):
    """Main classification loop"""
    
    centroids = np.array([nuc['centroid'] for nuc in nuclei], dtype=np.float32)
    print(f"[INFO] Total nuclei: {len(centroids):,}")
    
    x_min, y_min = centroids.min(axis=0)
    x_max, y_max = centroids.max(axis=0)
    bounds = (x_min, y_min, x_max, y_max)
    
    slide_cache = RAMSlideCache(slide_path, nuclei_bounds=bounds, padding=patch_size)
    
    producer = ParallelBatchProducer(
        slide_cache, centroids, patch_size, batch_size, num_workers
    )
    
    num_batches = producer.start()
    print(f"[INFO] {num_batches} batches, batch_size={batch_size}")
    
    results = {}
    total_classified = 0
    t_start = time.time()
    
    pbar = tqdm(total=num_batches, desc="Classifying", unit="batch")
    
    for _ in range(num_batches):
        try:
            patches_np, valid_indices, batch_idx = producer.get_batch(timeout=120)
        except Exception as e:
            print(f"[WARN] {e}")
            pbar.update(1)
            continue
        
        if patches_np is None:
            pbar.update(1)
            continue
        
        pixel_values = gpu_preprocessor(patches_np)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(pixel_values=pixel_values).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
        
        for idx, pred in zip(valid_indices, preds):
            results[idx] = (id2label.get(int(pred), f"class_{pred}"), int(pred))
        
        total_classified += len(preds)
        
        # Fixed: use time.time() instead of pbar.elapsed
        elapsed = time.time() - t_start
        rate = total_classified / max(elapsed, 0.001)
        pbar.update(1)
        pbar.set_postfix({'done': f'{total_classified:,}', 'rate': f'{rate:,.0f}/s'})
    
    pbar.close()
    producer.stop()
    
    for idx, (label, pred_idx) in results.items():
        nuclei[idx]['classification'] = {'name': label, 'index': pred_idx}
    
    del slide_cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return nuclei, total_classified


def save_geojson(nuclei, output_path, precision=1):
    """Save results"""
    print(f"[INFO] Saving to {output_path}")
    
    CLASS_COLORS = {
        "bladder": [255, 0, 0], "bone": [255, 128, 0], "brain": [255, 255, 0],
        "collagen": [128, 255, 0], "ear": [0, 255, 0], "eye": [0, 255, 128],
        "gi": [0, 255, 255], "heart": [0, 128, 255], "kidney": [0, 0, 255],
        "liver": [128, 0, 255], "lungs": [255, 0, 255], "mesokidney": [255, 0, 128],
        "nontissue": [255, 255, 255], "pancreas": [128, 128, 128],
        "skull": [64, 64, 64], "spleen": [255, 192, 203], "spleen2": [255, 182, 193],
        "thymus": [173, 216, 230], "thyroid": [144, 238, 144],
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    with open(output_path, 'w', buffering=64*1024*1024) as f:
        f.write('{"type":"FeatureCollection","features":[\n')
        
        first = True
        for nuc in tqdm(nuclei, desc="Writing", mininterval=2.0):
            coords = nuc['feature'].get('geometry', {}).get('coordinates', [[]])[0]
            coords_str = ",".join(f"[{round(x,precision)},{round(y,precision)}]" for x, y in coords)
            
            if 'classification' in nuc:
                cn = nuc['classification']['name']
                color = CLASS_COLORS.get(cn, [200, 200, 200])
                props = f'{{"objectType":"annotation","classification":{{"name":"{cn}","color":{color}}}}}'
            else:
                props = '{"objectType":"annotation"}'
            
            line = f'{{"type":"Feature","geometry":{{"type":"Polygon","coordinates":[[{coords_str}]]}},"properties":{props}}}'
            
            if not first:
                f.write(',\n')
            first = False
            f.write(line)
        
        f.write('\n]}')
    
    size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"[INFO] Saved {len(nuclei):,} features ({size_mb:.1f} MB) in {time.time()-t0:.1f}s")


def load_geojson(path):
    """Load nuclei"""
    print(f"[INFO] Loading: {path}")
    t0 = time.time()
    
    if USE_ORJSON:
        with open(path, 'rb') as f:
            data = orjson.loads(f.read())
    else:
        with open(path, 'r') as f:
            data = json.load(f)
    
    nuclei = []
    features = data.get('features', data) if isinstance(data, dict) else data
    
    for feat in features:
        coords = feat.get('geometry', {}).get('coordinates', [[]])[0]
        if len(coords) >= 3:
            centroid = np.array(coords, dtype=np.float32).mean(axis=0)
            nuclei.append({'feature': feat, 'centroid': centroid})
    
    print(f"[INFO] Loaded {len(nuclei):,} nuclei in {time.time()-t0:.1f}s")
    return nuclei


def load_model(ckpt_path):
    """Load model"""
    print(f"[INFO] Loading model: {ckpt_path}")
    
    processor = AutoImageProcessor.from_pretrained(ckpt_path, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(ckpt_path)
    model = model.cuda().eval().half()
    
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"[INFO] Classes: {len(id2label)}")
    
    size = processor.size
    target_size = size.get('height', size.get('shortest_edge', 224)) if isinstance(size, dict) else size
    
    return model, processor, id2label, target_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndpi", required=True)
    parser.add_argument("--geojson", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", type=int, default=1)
    parser.add_argument("--max_n", type=int, default=None)
    args = parser.parse_args()
    
    print("=" * 60)
    print("  FAST NUCLEUS CLASSIFICATION")
    print("=" * 60)
    
    t_start = time.time()
    
    model, processor, id2label, target_size = load_model(args.ckpt)
    gpu_preprocessor = FastGPUPreprocessor(processor, args.patch_size, target_size)
    
    # Warmup
    print("[INFO] Warming up...")
    dummy = np.zeros((args.batch, args.patch_size, args.patch_size, 3), dtype=np.uint8)
    for _ in range(3):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            _ = model(pixel_values=gpu_preprocessor(dummy))
    torch.cuda.synchronize()
    
    nuclei = load_geojson(args.geojson)
    if args.max_n:
        nuclei = nuclei[:args.max_n]
    
    t_cls = time.time()
    nuclei, total = classify_fast(
        args.ndpi, nuclei, model, gpu_preprocessor, id2label,
        args.patch_size, args.batch, args.workers
    )
    t_cls_end = time.time()
    
    save_geojson(nuclei, args.out, args.precision)
    
    t_total = time.time() - t_start
    rate = total / (t_cls_end - t_cls)
    
    print("\n" + "=" * 60)
    print(f"  DONE: {total:,} nuclei in {t_total:.1f}s ({rate:,.0f}/sec)")
    print("=" * 60)


if __name__ == "__main__":
    main()