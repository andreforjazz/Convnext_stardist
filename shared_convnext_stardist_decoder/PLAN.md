# Shared ConvNeXt encoder + StarDist-like decoder (multi-task)

## Goal

One **PyTorch** forward pass per image tile produces:

1. **StarDist-style** outputs: object probability map + `n_rays` radial distance maps (star-convex nuclei).
2. **Per-pixel class logits** for embryo / tissue classes; each detected nucleus gets a class by **majority vote** (or argmax mean) inside its polygon.

Output artifact: **GeoJSON** with polygon geometry and `properties` (class name, confidence, optional probabilities).

This subproject is the **largest training effort** in the repo: it replaces chained StarDist (TensorFlow) + ConvNeXt classifier with a **joint** model so the encoder cost is paid **once** per tile.

---

## Implementation plan (what the code does)

| Step | Deliverable |
|------|-------------|
| 1 | **Targets**: instance label image → `prob` target (EDT-based, StarDist-compatible) + `star_dist` targets (`n_rays` channels) + **class map** (per-pixel class id from `instance_id → class_id`). |
| 2 | **Backbone**: `timm` ConvNeXt v2 Tiny (ImageNet-22k fine-tune) with `features_only=True`, 4 feature levels. |
| 3 | **Decoder**: UNet-style fusion + two extra upsample steps to restore **full input resolution** (ConvNeXt stride-4 highest res → need ×4 total to match 256×256 tiles). |
| 4 | **Heads**: `prob` (1× sigmoid), `dist` (`n_rays`, positive via softplus), `cls` (`num_classes` logits). |
| 5 | **Loss**: weighted BCE (prob) + masked L1 (dist, foreground only) + CE on nucleus pixels (class). |
| 6 | **Postprocess**: peaks on `prob` + **soft** NMS + `dist_to_coord` polygons (StarDist-style). |
| 7 | **Train / infer CLI**: YAML config, checkpointing, WSI-style tiling hooks (offsets, stitching). |

---

## Pitfalls

1. **StarDist target speed** — Pure Python `star_dist` is correct but slow; production data pipelines should use StarDist’s **C++** `star_dist(..., mode='cpp')` in a preprocessing job, or cache `.npz` targets per patch.
2. **Class–segmentation alignment** — If class labels are noisy or instance masks leak across boundaries, the **semantic head** learns texture outside true nuclei; mitigate with **erosion** of instance masks for CE, or higher `loss_cls` weight only on centers (future extension).
3. **Small objects / crowding** — Dense embryonic tissue: peak NMS and prob threshold are **coupled**; tune on val with IoU + classification accuracy jointly.
4. **Tile boundaries** — Nuclei cut by tile edges need **overlap + merge** (same as classical StarDist WSI pipelines); this repo’s `infer.py` stitches by offset; cross-tile duplicate suppression is left as a **TODO** (union of IoU-based merging).
5. **Domain shift** — Pretrained ConvNeXt is RGB natural images; H&E may need **strong augmentations** and/or staged freezing (freeze backbone early epochs).
6. **Framework split** — Existing TF StarDist models are **not** loadable into this network; this is **train-from-scratch (with encoder init)** or **distillation** (optional future).

---

## Bottlenecks

| Bottleneck | Why | Mitigation |
|------------|-----|------------|
| Target generation | `star_dist` per patch CPU-heavy | Offline cache; StarDist cpp; smaller `n_rays` for ablations |
| Memory | Full-res heads at 512²+ | Reduce batch size; gradient checkpointing (future); `n_rays`=32 not 96 |
| Postprocess on CPU | Many peaks per tile | Torch GPU peak find (future); larger `prob_thresh`; vectorized NumPy |
| Annotation cost | Need **instance masks + class per instance** | Use existing StarDet/StarDist labels + CODA/GS40 class tables |

---

## Recommended workflow

1. **Phase A — Sanity**: Train on a **few hundred** labeled tiles with frozen backbone (optional) to verify losses decrease and polygons look reasonable.
2. **Phase B — Full**: Unfreeze, tune LR, augment heavily, validate with **panoptic-style** metrics (mask IoU + nucleus-level accuracy).
3. **Phase C — WSI**: Run `infer.py` with overlap; add **QuPath**-style export; refine duplicate merging using IoU matching.

---

## Files in this folder

- `config.example.yaml` — hyperparameters and paths.
- `model.py` — `StardistMultitaskNet`.
- `targets.py` — EDT prob + Python `star_dist` + class maps.
- `geometry.py` — polar ↔ polygon, peak suppression.
- `losses.py` — combined loss.
- `dataset.py` — image + instance label (+ optional JSON id→class).
- `train.py` — training loop.
- `infer.py` — tile inference + GeoJSON.
- `requirements.txt` — Python deps for this subproject.

License note: StarDist geometry routines follow the **BSD-3-Clause** ideas (ray marching fallback); for fastest target generation install official `stardist` and generate targets in a separate script using `mode='cpp'`.

## Commands (from repository root `Convnext_stardist/`)

```bash
pip install -r shared_convnext_stardist_decoder/requirements.txt
# optionally install a CUDA wheel for torch first

copy shared_convnext_stardist_decoder\\config.example.yaml shared_convnext_stardist_decoder\\config.yaml
# edit paths, then:

py -3 -m shared_convnext_stardist_decoder.train --config shared_convnext_stardist_decoder/config.yaml
py -3 -m shared_convnext_stardist_decoder.infer --config shared_convnext_stardist_decoder/config.yaml --weights runs/.../best.pt --image tile.png --out pred.geojson
```
