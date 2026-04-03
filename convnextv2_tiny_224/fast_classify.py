"""
FAST nucleus classification pipeline for GS40 ConvNeXt.
Target: 10-20x speedup over sequential version.

Key optimizations:
1. Multi-threaded patch extraction (OpenSlide releases GIL)
2. Async prefetching with queues
3. Batched GPU inference with AMP
4. torch.compile for fused kernels
5. Parallel JSON writing
6. Memory-mapped slide reading where possible
"""

import argparse
import json
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import openslide
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Optional: even faster JSON
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    print("[WARN] orjson not found, using stdlib json (pip install orjson for 2-3x faster JSON)")

try:
    import ijson
except ImportError:
    raise ImportError("pip install ijson")

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# CONFIG
# ============================================================

NUM_EXTRACT_WORKERS = 8      # Threads for patch extraction
PREFETCH_BATCHES = 4         # Number of batches to prefetch
BATCH_SIZE = 128             # Larger batch = better GPU utilization
USE_COMPILE = False           # torch.compile (PyTorch 2.0+)
USE_CHANNELS_LAST = True     # Memory format optimization
USE_AMP = True               # Mixed precision


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class NucleusJob:
    """Lightweight container for a single nucleus to classify."""
    idx: int
    feature: dict
    cx: float
    cy: float


@dataclass  
class PatchResult:
    """Extracted patch ready for inference."""
    idx: int
    feature: dict
    cx: float
    cy: float
    patch: np.ndarray  # RGB uint8 [H, W, 3]


@dataclass
class PredictionResult:
    """Final prediction for a nucleus."""
    idx: int
    feature: dict
    cx: float
    cy: float
    pred_id: int
    pred_name: str
    confidence: float


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def centroid_of_ring(ring) -> Optional[tuple]:
    """Fast centroid calculation."""
    if not ring or len(ring) < 3:
        if ring:
            arr = np.array(ring, dtype=np.float64)
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())
        return None
    
    arr = np.array(ring, dtype=np.float64)
    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    
    n = len(arr)
    if n < 3:
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    
    # Shoelace formula vectorized
    x = arr[:, 0]
    y = arr[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    
    cross = x * y_next - x_next * y
    A = cross.sum() * 0.5
    
    if abs(A) < 1e-9:
        return float(x.mean()), float(y.mean())
    
    Cx = ((x + x_next) * cross).sum() / (6.0 * A)
    Cy = ((y + y_next) * cross).sum() / (6.0 * A)
    
    return float(Cx), float(Cy)


def centroid_from_geom(geom) -> Optional[tuple]:
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None
    if gtype == "Polygon":
        return centroid_of_ring(coords[0])
    if gtype == "MultiPolygon":
        return centroid_of_ring(coords[0][0])
    return None


# ============================================================
# FAST JSON HELPERS
# ============================================================

def fast_json_dumps(obj) -> str:
    """Use orjson if available, else stdlib."""
    if USE_ORJSON:
        return orjson.dumps(obj, default=_orjson_default).decode('utf-8')
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, Decimal):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"Not serializable: {type(o)}")


def _orjson_default(o):
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Not serializable: {type(o)}")


# ============================================================
# GEOJSON STREAMING
# ============================================================

def iter_geojson_features(geojson_path: Path):
    """Stream features from GeoJSON without loading entire file."""
    with geojson_path.open("rb") as f:
        # Detect format
        first_char = None
        while True:
            ch = f.read(1)
            if not ch:
                return
            if ch in b" \t\r\n":
                continue
            first_char = ch
            break
        
        f.seek(0)
        
        if first_char == b"{":
            yield from ijson.items(f, "features.item")
        elif first_char == b"[":
            yield from ijson.items(f, "item")
        else:
            raise RuntimeError(f"Unknown GeoJSON format: {geojson_path}")


# ============================================================
# PARALLEL PATCH EXTRACTOR
# ============================================================

class ParallelPatchExtractor:
    """
    Multi-threaded patch extraction from OpenSlide.
    OpenSlide releases GIL during read_region, so threading works well.
    """
    
    def __init__(
        self,
        slide_path: Path,
        patch_size: int,
        coord_scale: float = 1.0,
        swap_xy: bool = False,
        num_workers: int = 8,
    ):
        self.slide_path = str(slide_path)
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.coord_scale = coord_scale
        self.swap_xy = swap_xy
        self.num_workers = num_workers
        
        # Thread-local slides (OpenSlide is not thread-safe for single instance)
        self._local = threading.local()
    
    def _get_slide(self) -> openslide.OpenSlide:
        if not hasattr(self._local, 'slide'):
            self._local.slide = openslide.OpenSlide(self.slide_path)
        return self._local.slide
    
    def extract_one(self, job: NucleusJob) -> PatchResult:
        """Extract single patch (called in worker thread)."""
        slide = self._get_slide()
        
        x = job.cx * self.coord_scale
        y = job.cy * self.coord_scale
        if self.swap_xy:
            x, y = y, x
        
        x0 = int(round(x - self.half))
        y0 = int(round(y - self.half))
        
        rgba = slide.read_region((x0, y0), 0, (self.patch_size, self.patch_size))
        rgb = np.array(rgba.convert("RGB"), dtype=np.uint8)
        
        return PatchResult(
            idx=job.idx,
            feature=job.feature,
            cx=job.cx,
            cy=job.cy,
            patch=rgb,
        )
    
    def extract_batch(self, jobs: List[NucleusJob]) -> List[PatchResult]:
        """Extract batch of patches in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.extract_one, jobs))
        return results


# ============================================================
# ASYNC PIPELINE
# ============================================================

class AsyncPipeline:
    """
    Coordinates async reading → extraction → inference → writing.
    Uses queues to overlap I/O and compute.
    """
    
    def __init__(
        self,
        geojson_path: Path,
        extractor: ParallelPatchExtractor,
        model: torch.nn.Module,
        processor: AutoImageProcessor,
        id2label: Dict[int, str],
        batch_size: int = 128,
        prefetch_batches: int = 4,
        device: str = "cuda",
        class_colors: Optional[Dict[str, List[int]]] = None,
        max_n: int = 0,
    ):
        self.geojson_path = geojson_path
        self.extractor = extractor
        self.model = model
        self.processor = processor
        self.id2label = id2label
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.device = device
        self.class_colors = class_colors or {}
        self.max_n = max_n
        
        # Queues
        self.job_queue = queue.Queue(maxsize=prefetch_batches * batch_size)
        self.patch_queue = queue.Queue(maxsize=prefetch_batches)
        self.result_queue = queue.Queue(maxsize=prefetch_batches)
        
        # Signals
        self.stop_event = threading.Event()
        self.total_processed = 0
        self.lock = threading.Lock()
    
    def _reader_thread(self):
        """Read GeoJSON and enqueue nucleus jobs."""
        idx = 0
        for feat in iter_geojson_features(self.geojson_path):
            if self.stop_event.is_set():
                break
            
            geom = feat.get("geometry")
            if geom is None:
                continue
            
            c = centroid_from_geom(geom)
            if c is None:
                continue
            
            job = NucleusJob(idx=idx, feature=feat, cx=c[0], cy=c[1])
            self.job_queue.put(job)
            idx += 1
            
            if self.max_n and idx >= self.max_n:
                break
        
        # Signal end
        self.job_queue.put(None)
    
    def _extractor_thread(self):
        """Batch extract patches and enqueue."""
        batch = []
        
        while not self.stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if job is None:
                # Flush remaining
                if batch:
                    results = self.extractor.extract_batch(batch)
                    self.patch_queue.put(results)
                self.patch_queue.put(None)
                break
            
            batch.append(job)
            
            if len(batch) >= self.batch_size:
                results = self.extractor.extract_batch(batch)
                self.patch_queue.put(results)
                batch = []
    
    @torch.inference_mode()
    def _inference_thread(self):
        """Run batched inference on GPU."""
        while not self.stop_event.is_set():
            try:
                patches = self.patch_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if patches is None:
                self.result_queue.put(None)
                break
            
            # Stack patches
            images = [Image.fromarray(p.patch) for p in patches]
            
            # Process batch
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
            
            if USE_CHANNELS_LAST:
                pixel_values = pixel_values.to(memory_format=torch.channels_last)
            
            # Inference with AMP
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = self.model(pixel_values=pixel_values).logits
            
            # Post-process
            probs = F.softmax(logits, dim=1)
            confidences, pred_ids = probs.max(dim=1)
            
            pred_ids = pred_ids.cpu().numpy()
            confidences = confidences.cpu().numpy()
            
            # Build results
            results = []
            for p, pid, conf in zip(patches, pred_ids, confidences):
                pred_name = self.id2label.get(int(pid), f"class_{pid}")
                results.append(PredictionResult(
                    idx=p.idx,
                    feature=p.feature,
                    cx=p.cx,
                    cy=p.cy,
                    pred_id=int(pid),
                    pred_name=pred_name,
                    confidence=float(conf),
                ))
            
            self.result_queue.put(results)
    
    def run(self, out_path: Path, chunk_size: int = 50000) -> int:
        """Run full pipeline and write results."""
        
        # Start threads
        reader = threading.Thread(target=self._reader_thread, daemon=True)
        extractor = threading.Thread(target=self._extractor_thread, daemon=True)
        inferencer = threading.Thread(target=self._inference_thread, daemon=True)
        
        reader.start()
        extractor.start()
        inferencer.start()
        
        # Write results
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        part_idx = 0
        wrote_total = 0
        wrote_chunk = 0
        
        stem = out_path.stem
        suffix = out_path.suffix
        
        def open_chunk(idx):
            if chunk_size > 0:
                p = out_path.parent / f"{stem}_part{idx:03d}{suffix}"
            else:
                p = out_path
            f = p.open("w", encoding="utf-8")
            f.write('{"type":"FeatureCollection","features":[\n')
            return f, p
        
        def close_chunk(f):
            f.write("\n]}\n")
            f.close()
        
        f_out, current_path = open_chunk(part_idx)
        first_in_chunk = True
        
        pbar = tqdm(desc="Classifying nuclei", dynamic_ncols=True, unit="nuc")
        t0 = time.time()
        
        try:
            while True:
                try:
                    results = self.result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if results is None:
                    break
                
                for r in results:
                    # Update feature properties
                    props = r.feature.setdefault("properties", {})
                    props["class_id_pred"] = r.pred_id
                    props["class_name_pred"] = r.pred_name
                    props["pred_conf"] = r.confidence
                    props["centroid_x"] = r.cx
                    props["centroid_y"] = r.cy
                    
                    # QuPath classification
                    color = self.class_colors.get(r.pred_name.lower(), [255, 255, 255])
                    props["classification"] = {
                        "name": r.pred_name,
                        "color": color,
                    }
                    
                    # Write
                    if not first_in_chunk:
                        f_out.write(",\n")
                    first_in_chunk = False
                    
                    f_out.write(fast_json_dumps(r.feature))
                    
                    wrote_total += 1
                    wrote_chunk += 1
                    pbar.update(1)
                    
                    # Chunk rotation
                    if chunk_size > 0 and wrote_chunk >= chunk_size:
                        close_chunk(f_out)
                        print(f"\n[WROTE] {current_path} | n={wrote_chunk:,}")
                        part_idx += 1
                        f_out, current_path = open_chunk(part_idx)
                        wrote_chunk = 0
                        first_in_chunk = True
                
                # Update speed
                elapsed = time.time() - t0
                if elapsed > 0:
                    pbar.set_postfix(nuc_s=f"{wrote_total/elapsed:.0f}")
        
        finally:
            self.stop_event.set()
            close_chunk(f_out)
            pbar.close()
            
            reader.join(timeout=2)
            extractor.join(timeout=2)
            inferencer.join(timeout=2)
        
        print(f"\n[WROTE] {current_path} | n={wrote_chunk:,}")
        print(f"[DONE] Total: {wrote_total:,} nuclei in {time.time()-t0:.1f}s")
        
        return wrote_total


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_processor(ckpt_dir: Path, device: str = "cuda"):
    """Load model with optimizations."""
    
    processor = AutoImageProcessor.from_pretrained(ckpt_dir, use_fast=False)
    
    model = AutoModelForImageClassification.from_pretrained(
        ckpt_dir,
        use_safetensors=True,
        torch_dtype=torch.float16 if USE_AMP else torch.float32,
    )
    
    model.eval()
    model.to(device)
    
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    
    # Compile for speed (PyTorch 2.0+)
    if USE_COMPILE and hasattr(torch, 'compile'):
        print("[INFO] Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    return model, processor


def get_id2label(model, label_space_path: Optional[Path] = None) -> Dict[int, str]:
    """Extract id2label from model config or label file."""
    
    # Try model config first
    raw = getattr(model.config, "id2label", {})
    if raw:
        return {int(k): str(v) for k, v in raw.items()}
    
    # Fall back to label file
    if label_space_path and label_space_path.exists():
        suffix = label_space_path.suffix.lower()
        
        if suffix in [".yaml", ".yml"]:
            import yaml
            d = yaml.safe_load(label_space_path.read_text())
            labels = d.get("labels", {})
            return {int(k): str(v) for k, v in labels.items()}
        
        d = json.loads(label_space_path.read_text())
        if "id2label" in d:
            return {int(k): str(v) for k, v in d["id2label"].items()}
        return {int(k): str(v) for k, v in d.items()}
    
    raise RuntimeError("Could not determine id2label mapping")


# ============================================================
# MAIN
# ============================================================

CLASS_COLORS = {
    "bladder": [230, 25, 75], "bone": [245, 130, 48], "brain": [255, 225, 25],
    "collagen": [210, 245, 60], "ear": [60, 180, 75], "eye": [70, 240, 240],
    "gi": [0, 130, 200], "heart": [0, 0, 128], "kidney": [145, 30, 180],
    "liver": [240, 50, 230], "lungs": [128, 128, 128], "mesokidney": [170, 110, 40],
    "nontissue": [255, 255, 255], "pancreas": [128, 0, 0], "skull": [170, 255, 195],
    "spleen": [128, 128, 0], "spleen2": [255, 215, 180], "thymus": [0, 0, 0],
    "thyroid": [250, 190, 190],
}


def main():
    ap = argparse.ArgumentParser(description="Fast nucleus classification pipeline")
    ap.add_argument("--ndpi", required=True, help="Path to NDPI/SVS slide")
    ap.add_argument("--geojson", required=True, help="Input GeoJSON with nuclei")
    ap.add_argument("--ckpt", required=True, help="ConvNeXt checkpoint directory")
    ap.add_argument("--label_space", default="", help="Optional label_space.json or label_map.yaml")
    ap.add_argument("--out", required=True, help="Output GeoJSON path")
    
    ap.add_argument("--patch", type=int, default=256, help="Patch size")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    ap.add_argument("--workers", type=int, default=NUM_EXTRACT_WORKERS, help="Extraction threads")
    ap.add_argument("--chunk", type=int, default=50000, help="Features per output file (0=single file)")
    ap.add_argument("--coord_scale", type=float, default=1.0)
    ap.add_argument("--swap_xy", action="store_true")
    ap.add_argument("--max_n", type=int, default=0, help="Max nuclei to process (0=all)")
    
    args = ap.parse_args()
    
    ndpi_path = Path(args.ndpi)
    geojson_path = Path(args.geojson)
    ckpt_dir = Path(args.ckpt)
    out_path = Path(args.out)
    label_space_path = Path(args.label_space) if args.label_space else None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("FAST NUCLEUS CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Slide: {ndpi_path}")
    print(f"GeoJSON: {geojson_path}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Batch size: {args.batch}")
    print(f"Extract workers: {args.workers}")
    print(f"torch.compile: {USE_COMPILE}")
    print(f"AMP: {USE_AMP}")
    print(f"orjson: {USE_ORJSON}")
    print("=" * 60)
    
    # Load model
    model, processor = load_model_and_processor(ckpt_dir, device)
    id2label = get_id2label(model, label_space_path)
    
    print(f"Labels: {len(id2label)}")
    for k in sorted(id2label):
        print(f"  {k}: {id2label[k]}")
    
    # Create extractor
    extractor = ParallelPatchExtractor(
        slide_path=ndpi_path,
        patch_size=args.patch,
        coord_scale=args.coord_scale,
        swap_xy=args.swap_xy,
        num_workers=args.workers,
    )
    
    # Create and run pipeline
    pipeline = AsyncPipeline(
        geojson_path=geojson_path,
        extractor=extractor,
        model=model,
        processor=processor,
        id2label=id2label,
        batch_size=args.batch,
        prefetch_batches=PREFETCH_BATCHES,
        device=device,
        class_colors=CLASS_COLORS,
        max_n=args.max_n,
    )
    
    t0 = time.time()
    total = pipeline.run(out_path, chunk_size=args.chunk)
    elapsed = time.time() - t0
    
    print("\n" + "=" * 60)
    print(f"FINISHED: {total:,} nuclei in {elapsed:.1f}s")
    print(f"Throughput: {total/elapsed:.0f} nuclei/sec")
    print(f"Output: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()