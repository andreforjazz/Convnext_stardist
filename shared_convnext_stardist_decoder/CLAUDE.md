# ConvNeXt-StarDist Multitask — Codebase Reference

## What this does
Nuclear instance segmentation + 19-class tissue classification on fetal monkey H&E WSIs
at 20× magnification (0.4416 µm/px). Outputs StarDist star-polygon nuclei with per-nucleus
tissue class (bone, brain, eye, heart, lungs, gi, liver, spleen, pancreas, kidney,
mesokidney, collagen, ear, nontissue, thymus, thyroid, bladder, skull, spleen2).

**Scope:** Fetal macaque (Macaca mulatta), gestational stages GS33–GS55, H&E staining,
0.4416 µm/px (20×). Planned expansion: alligator, chicken, mouse (CODA labels available
for all species). Embryonic-stage morphology is similar enough across species that the
19-class taxonomy is expected to transfer with minor adjustments.

**Target hardware:** RTX 4090 (24 GB VRAM). BATCH_TILES=96, FP16, torch.compile.
Model is ~32 M params / ~64 MB in FP16 — intentionally kept at ConvNeXt-V2-Tiny for
inference speed, not due to memory constraints.

## ⚠ CLASS NAMING BUG (do not fix until public release — keep for backward compat)
The class named `spleen`  (index 7)  is actually the **GONAD**.
The class named `spleen2` (index 18) is actually the **SPLEEN**.
This must be renamed (`gonad` / `spleen`) before public release across:
  config class_names, CODA label mapping, LABELS_VIZ in inference notebooks, CLAUDE.md.

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

## Evaluation
- eval_classification.ipynb — per-class P/R/F1 + confusion matrix on any GS40/GS55/GS33 slide
  against CODA-class ground truth GeoJSON. Set PRED_GEOJSON to load existing inference
  output, or leave None to run inference from scratch.
- **Action required:** run eval on held-out slides (never seen in any training run) with
  current best.pt BEFORE adding GS33 to training. Weighted F1 drop after GS33 = hurting.
- Matching: greedy bijective centroid KDTree, MATCH_DIST_PX=8px (20x pixels).
- VOTE_WINDOW_PX=9 fast approximation in infer_wsi_v4 has NOT been validated against full
  polygon vote_class — add a comparison cell to eval_classification.ipynb before release.

## Stain normalisation (planned, not yet implemented)
- Use Macenko normalisation via torchstain (GPU-batched, <1 ms/tile).
- Store one reference 256×256 tile alongside model weights.
- Add as preprocessing step in batch_forward_fast / forward_batch_with_perm before
  _normalize_chw, with an opt-out flag for slides already matching training distribution.
- Also add torchstain.augmentors.RandomStainAugmentor to training augmentation when
  retraining with new species — teaches stain invariance rather than relying on correction.

## Adding GS33 / new species
- GS33 dataset pipeline follows same structure as GS40/GS55 (CODA labels available).
- Risk: GS33 labels not yet validated for consistency with GS40+GS55 taxonomy.
  Protocol: eval on held-out GS40+GS55 slides before and after adding GS33; if weighted
  F1 drops, GS33 is hurting. Fine-tune from GS40+GS55 checkpoint rather than training
  from scratch jointly.
- Multi-species (alligator, chicken, mouse): CODA labels available for all. Embryonic
  morphology similarity justifies shared 19-class taxonomy. Finalize class list before
  collecting new species data (see CLASS NAMING BUG above — fix naming first).

## Inference speed optimisation notes

### ConvNeXt-V2 Base vs Tiny
Base has ~3.4× more FLOPs (15.4 vs 4.5 GFLOPs at 224²) but practical slowdown is
~1.5–2× on RTX 4090 because inference is memory-bandwidth bound, not compute bound.
Base weights: ~178 MB FP16 vs ~64 MB — fits in 24 GB easily.
Decision rule: run eval_classification.ipynb first. If rare-class F1 is low, the cause
is more likely data imbalance or label quality than backbone capacity — Base won't fix it.

### FP8 inference (RTX 4090 — Ada Lovelace has native FP8 Tensor Cores)
Expected speed gain over FP16: ~1.5–2× in practice (not the theoretical 4× peak ratio,
because inference is already memory-bandwidth bound at BATCH_TILES=96).

Accuracy risk:
- Segmentation (prob + dist heads): <1% F1 change with proper calibration. Smooth spatial
  maps average out FP8 quantisation error.
- Classification (cls head): higher risk. FP8 E4M3 has only 3 mantissa bits vs 10 in FP16.
  Small logit margins between similar classes (kidney/mesokidney, thymus/brain) flip more
  easily. Rare classes (skull, thyroid, bladder) most exposed. ~1–3% drop on common
  classes with calibration; unpredictable without it.

Implementation:
- Best path: TensorRT FP8 with per-layer calibration on ~5 representative slides.
  Native PyTorch torchao.float8 is less mature for CNNs than for transformers (as of 2026).
- Do NOT attempt without baseline F1 numbers from eval_classification.ipynb — need a
  reference to measure accuracy degradation against.

Prerequisites before attempting FP8:
1. Profile CPU postprocessing per batch — if CPU post is the bottleneck, FP8 GPU speedup
   gives zero WSI throughput improvement (GPU is already sitting idle).
2. Have per-class F1 baseline from eval notebook.
3. Export model to TensorRT; calibrate on held-out slides.

## Pre-release checklist (public model release)
1. [ ] Rename spleen→gonad, spleen2→spleen in all configs and code
2. [ ] Add Macenko stain normalisation (torchstain) to inference pipeline
3. [ ] Run eval_classification.ipynb on held-out slides; document baseline F1 per class
4. [ ] Validate VOTE_WINDOW_PX=9 vs full polygon vote_class classification accuracy
5. [ ] Profile CPU postprocessing per batch (add timer inside postprocess_batch_v4)
       to confirm GPU is the bottleneck, not CPU post
6. [ ] Write clean CLI inference script (infer.py --slide ... --out ...) replacing notebook
7. [ ] Write model card: species, stages, magnification, scanner, class definitions,
       known limitations, citation

## Known issues / history
- GS55 labels were pointing to wrong path (stardist_multitask_ready/train_instance_labels)
  instead of train/labels and val/labels — FIXED in all configs and notebooks (2026-04)
- Old checkpoints: single-layer cls head (head_cls.weight), cls_semantic_dim=64
  Auto-detected in load_model_and_classes() via state_dict key inspection
- cls head overfits with backbone frozen at lr=5e-5; val improves sharply when backbone
  unfreezes (observed at epoch 11 in v2 training log). Affects base training only
  (freeze_backbone_epochs=10); finetune config uses freeze=0 so is unaffected.
  Fix: reduce freeze_backbone_epochs to 5, or add lower lr group for cls head during
  frozen phase. Not urgent until next full base training run.
- v2 architecture (head_cls_layers=2, cls_semantic_dim=128) not backward-compatible with
  v1 checkpoints; use --resume_strict=False or load_v1_weights_into_v2() for warm-start