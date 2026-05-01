"""
model_v2.py  —  StardistMultitaskNetV2

Changes vs model.py (v1):
  Fix #1 — Semantic skip: encoder stage-4 features (768ch, 1/32 scale) are
            projected and bilinearly upsampled to full resolution, then
            concatenated with the shared decoder output *before* head_cls.
            This gives the classification head access to tile-global, semantic
            representations (texture / color / tissue pattern) that are largely
            erased by the geometry-optimised UNet decoder.

  Unchanged: segmentation decoder (bridge → dec_extra), head_prob, head_dist.
  Incompatible with v1 checkpoints (new layers in cls path).  To warm-start from
  a v1 checkpoint use strict=False — the seg decoder transfers; cls head re-initialises.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# Stage output channels per ConvNeXt V2 variant. Used to size the decoder skip
# connections and the cls semantic-skip projection. Add an entry here if you
# train with a new backbone, or rely on the HF-config fallback below.
_BACKBONE_STAGE_CHANNELS: dict[str, list[int]] = {
    "facebook/convnextv2-tiny-22k-224":  [96, 192, 384, 768],
    "facebook/convnextv2-base-22k-224":  [128, 256, 512, 1024],
    "facebook/convnextv2-large-22k-224": [192, 384, 768, 1536],
}


def _stage_channels(backbone_name: str, backbone_module) -> list[int]:
    """Return the 4 stage-output channel widths for the loaded backbone."""
    if backbone_name in _BACKBONE_STAGE_CHANNELS:
        return list(_BACKBONE_STAGE_CHANNELS[backbone_name])
    hidden_sizes = getattr(getattr(backbone_module, "config", None), "hidden_sizes", None)
    if hidden_sizes is not None and len(hidden_sizes) == 4:
        return list(hidden_sizes)
    raise ValueError(
        f"Unknown backbone {backbone_name!r}: cannot determine stage channels. "
        f"Either add it to _BACKBONE_STAGE_CHANNELS in model_v2.py, or ensure "
        f"the backbone exposes `config.hidden_sizes` with 4 values."
    )


class StardistMultitaskNetV2(nn.Module):
    """
    ConvNeXt V2 encoder + UNet segmentation decoder + semantic-skip cls head.

    Input:  RGB (B,3,H,W), H,W divisible by 32 (e.g. 256).
    Output: prob logits (B,1,H,W), dist (B,n_rays,H,W), cls logits (B,C,H,W).

    Key difference from V1:
        head_cls reads  (decoder_out || semantic_ctx)  where semantic_ctx comes
        directly from encoder stage-4 (bilinear ×32 upsample + 1×1 projection).
        Stage-4 has a full-tile receptive field and encodes tissue texture/colour
        rather than instance geometry — exactly what per-nucleus classification needs.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/convnextv2-tiny-22k-224",
        *,
        pretrained: bool = True,
        n_rays: int = 32,
        num_classes: int = 19,
        decoder_channels: int = 128,
        cls_semantic_dim: int = 128,  # channels from stage-4 semantic skip
        head_cls_layers: int = 2,     # 1 = single Conv2d (old), 2 = two-layer MLP (new)
    ) -> None:
        super().__init__()
        self.n_rays = int(n_rays)
        self.num_classes = int(num_classes)
        self.head_cls_layers = int(head_cls_layers)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(backbone_name, use_safetensors=True)
        else:
            from transformers import AutoConfig
            self.backbone = AutoModel.from_config(
                AutoConfig.from_pretrained(backbone_name)
            )

        ch = _stage_channels(backbone_name, self.backbone)
        dc = int(decoder_channels)
        S  = int(cls_semantic_dim)

        # ── Segmentation decoder (identical to V1) ────────────────────────────
        self.bridge   = ConvBlock(ch[3], dc)
        self.up3      = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec3     = ConvBlock(dc + ch[2], dc)
        self.up2      = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec2     = ConvBlock(dc + ch[1], dc)
        self.up1      = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec1     = ConvBlock(dc + ch[0], dc)
        self.up0      = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec0     = ConvBlock(dc, dc)
        self.up_extra = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec_extra= ConvBlock(dc, dc)

        # ── Segmentation heads (identical to V1) ─────────────────────────────
        self.head_prob = nn.Conv2d(dc, 1, 1)
        self.head_dist = nn.Conv2d(dc, self.n_rays, 1)

        # ── Semantic context branch: stage-4 → S channels ────────────────────
        # Lightweight 1×1 projection keeps added cost negligible (~50K params).
        self.cls_ctx_proj = nn.Sequential(
            nn.Conv2d(ch[3], S, 1, bias=False),
            nn.BatchNorm2d(S),
            nn.GELU(),
        )

        # ── Classification head: decoder_out (dc) + semantic skip (S) → C ────
        if self.head_cls_layers == 1:
            # Legacy single-layer head — used for loading old checkpoints
            self.head_cls = nn.Conv2d(dc + S, self.num_classes, 1)
        else:
            # Two-layer MLP head — default for new training runs
            self.head_cls = nn.Sequential(
                nn.Conv2d(dc + S, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, self.num_classes, 1),
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.backbone(x, output_hidden_states=True)
        # hidden_states: [stem, stage1, stage2, stage3, stage4]
        feats = outputs.hidden_states[1:]   # (B,96,H/4), (192,H/8), (384,H/16), (768,H/32)

        # Segmentation decoder
        t = self.bridge(feats[3])
        t = self.up3(t);   t = torch.cat([t, feats[2]], dim=1);  t = self.dec3(t)
        t = self.up2(t);   t = torch.cat([t, feats[1]], dim=1);  t = self.dec2(t)
        t = self.up1(t);   t = torch.cat([t, feats[0]], dim=1);  t = self.dec1(t)
        t = self.up0(t);   t = self.dec0(t)
        t = self.up_extra(t); t = self.dec_extra(t)   # (B, dc, H, W)

        # Segmentation heads
        prob_logit = self.head_prob(t)
        dist       = F.softplus(self.head_dist(t)) + 1e-3

        # Semantic skip: project stage-4, upsample to match decoder resolution
        sem = self.cls_ctx_proj(feats[3])                              # (B, S, H/32, W/32)
        sem = F.interpolate(sem, size=t.shape[2:], mode="bilinear", align_corners=False)

        # Classification head: richer input = geometry (t) + semantics (sem)
        cls_logit = self.head_cls(torch.cat([t, sem], dim=1))

        return prob_logit, dist, cls_logit


def build_model_v2(cfg: dict) -> StardistMultitaskNetV2:
    m = cfg.get("model", cfg)
    return StardistMultitaskNetV2(
        backbone_name    = str(m.get("backbone", "facebook/convnextv2-tiny-22k-224")),
        pretrained       = bool(m.get("pretrained", True)),
        n_rays           = int(m.get("n_rays", 32)),
        num_classes      = int(m.get("num_classes", 19)),
        decoder_channels = int(m.get("decoder_channels", 128)),
        cls_semantic_dim = int(m.get("cls_semantic_dim", 128)),
        head_cls_layers  = int(m.get("head_cls_layers", 2)),
    )


def load_v1_weights_into_v2(
    model_v2: StardistMultitaskNetV2,
    v1_state_dict: dict,
    device: torch.device,
) -> None:
    """
    Warm-start V2 from a V1 checkpoint.
    All segmentation decoder weights transfer; new cls layers are re-initialised.
    """
    missing, unexpected = model_v2.load_state_dict(v1_state_dict, strict=False)
    new_layers = [k for k in missing if "cls_ctx_proj" in k or "head_cls" in k]
    other_missing = [k for k in missing if k not in new_layers]
    print(f"Transferred {len(v1_state_dict) - len(unexpected)} / {len(v1_state_dict)} V1 params")
    print(f"New V2 layers (re-init): {new_layers}")
    if other_missing:
        print(f"WARNING — unexpected missing keys: {other_missing}")
