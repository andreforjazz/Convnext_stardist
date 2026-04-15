# ConvNeXt-StarDist Multitask — Codebase Reference

## What this does
Nuclear instance segmentation + 19-class tissue classification on fetal monkey H&E WSIs
at 20× magnification (0.4416 µm/px). Outputs StarDist star-polygon nuclei with per-nucleus
tissue class (bone, brain, eye, heart, lungs, gi, liver, spleen, pancreas, kidney,
mesokidney, collagen, ear, nontissue, thymus, thyroid, bladder, skull, spleen2).

## Architecture (model_v2.py)
- Backbone: ConvNeXt-V2-Tiny (facebook/convnextv2-tiny-22k-224, ~28M params)
  - Stage channels: [96, 192, 384, 768]; outputs at H/4, H/8, H/16, H/32
- Decoder: UNet, 128ch, 5 progressive upsamplings with encoder skip connections
- Heads (all pixel-wise Conv2d at full resolution):
  - head_prob:  Conv2d(128→1)  → sigmoid → foreground probability
  - head_dist:  Conv2d(128→32) → softplus+1e-3 → 32 ray distances
  - head_cls:   2-layer MLP (Conv→BN→GELU→Conv, 256→128→19) → class logits
- Semantic skip: stage-4 (768ch) → Conv2d(128ch) + BN + GELU → upsample to full res
  → concatenated with decoder output before cls head (256ch total input to cls head)
- Key params: n_rays=32, num_classes=19, decoder_channels=128, cls_semantic_dim=128,
  head_cls_layers=2

## Loss (losses_v2.py)
total = 2.0×BCE + 0.15×dist_L1(fg) + 1.0×(cls_pixel_CE + 0.25×cls_inst_CE)
- cls_inst_CE: scatter-average logits over each instance mask → CE; matches vote_class() at inference
- class_weights: inv_sqrt_freq, normalized to mean=1.0 (computed from inst2class JSONs)

## Training (train_v2.py + train_utils.py)
- 256×256 patches, batch=8, AdamW lr=5e-5, weight_decay=0.03, AMP, grad_clip=1.0
- CosineAnnealingLR over unfrozen epochs (T_max=epochs−freeze_backbone_epochs)
- freeze_backbone_epochs=10 (main) / 0 (finetune) — backbone frozen first N epochs
- cls_balanced_sampler: WeightedRandomSampler 2× weight for tiles with cls supervision
- Logs: train_log.csv; checkpoints: best.pt + epoch_XXX.pt

## Dataset (dataset_v2.py + targets.py)
- Multi-root: GS40 (~38k train tiles) + GS55 (~16k train tiles)
- Labels: uint16 instance masks + _inst2class.json sidecars {inst_id: tissue_name}
- Targets per tile: EDT prob map, star-polygon distances (ray marching or stardist C++),
  class map (int64, -100=ignore), fg_mask, inst map (for inst_CE loss)
- Augmentation: 8 deterministic (flip/rot90 variants) applied during dataset generation
  (NOT online during training — tiles are pre-augmented on disk)
- ImageNet normalization: mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)

## Inference (inference_utils.py + geometry.py)
- WSI tiling: configurable tile_size + overlap, stride=tile_size−overlap
- Batch forward: pad to 32× multiple, FP16 optional, unpad after
- Post-process per tile: local_peaks → ray sampling → polygon vertices → vote_class
- vote_class: rasterize polygon interior (skimage.draw.polygon), average cls logits
  over interior pixels, softmax → class_id + probs
- Dedup: centroid NMS (cKDTree) + polygon overlap NMS (shapely)
- Export: GeoJSON FeatureCollection (classified or seg-only for QuPath)
- Backward compat: auto-detects old (head_cls.weight) vs new (head_cls.0.weight) checkpoint

## Dataset Generation (make_training_dataset/)
- Stage 0: Assign class from CODA organ masks to StarDist polygons (centroid→mask lookup)
  Scale: geojson_x × (mpp_20x / mpp_mask); mpp_20x=0.4416, mpp_mask=2.0
- Stage 1: Class distribution stats
- Stage 2: Tile extraction with 4-tier class-balanced sampling + 8-fold augmentation
  → uint16 instance masks + inst2class JSON sidecars

## Config files
- config_gs40_gs55_multitask.yaml  — main training (35 epochs, freeze 10, lr 5e-5→1e-6)
  experiment: convnext_stardist_mt_gs40_gs55_v2_cls2L_skip128
- config_finetune_gs40_gs55.yaml   — fine-tune from best.pt (15 epochs, freeze 0, lr 1e-5→1e-7)
  experiment: convnext_stardist_mt_gs40_gs55_v2_finetune
- config_gs40_multitask.yaml       — GS40-only baseline

## Key train commands
```bash
# Full training
python -m shared_convnext_stardist_decoder.train_v2 \
  --config shared_convnext_stardist_decoder/config_gs40_gs55_multitask.yaml

# Fine-tune from best checkpoint
python -m shared_convnext_stardist_decoder.train_v2 \
  --config shared_convnext_stardist_decoder/config_finetune_gs40_gs55.yaml \
  --resume <out_dir>/convnext_stardist_mt_gs40_gs55_v2_cls2L_skip128/best.pt
```

## Known issues / history
- GS55 labels were pointing to wrong path (stardist_multitask_ready/train_instance_labels)
  instead of train/labels and val/labels — FIXED in all configs and notebooks (2026-04)
- Old checkpoints: single-layer cls head (head_cls.weight), cls_semantic_dim=64
  Auto-detected in load_model_and_classes() via state_dict key inspection
- cls head overfits with backbone frozen at lr=5e-5; val improves sharply when backbone
  unfreezes (observed at epoch 11 in v2 training log)
- v2 architecture (head_cls_layers=2, cls_semantic_dim=128) not backward-compatible with
  v1 checkpoints; use --resume_strict=False or load_v1_weights_into_v2() for warm-start