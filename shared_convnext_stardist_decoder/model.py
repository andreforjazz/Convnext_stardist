from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


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


class StardistMultitaskNet(nn.Module):
    """
    ConvNeXt (timm) encoder + UNet-style decoder.

    Input:  RGB (B,3,H,W), H,W divisible by 32 (e.g. 256).
    Output: prob logits (B,1,H,W), dist (B,n_rays,H,W) positive, cls logits (B,C,H,W).
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k_224",
        *,
        pretrained: bool = True,
        n_rays: int = 32,
        num_classes: int = 8,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.n_rays = int(n_rays)
        self.num_classes = int(num_classes)

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        ch = self.backbone.feature_info.channels()
        dc = int(decoder_channels)

        self.bridge = ConvBlock(ch[3], dc)
        self.up3 = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(dc + ch[2], dc)
        self.up2 = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(dc + ch[1], dc)
        self.up1 = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(dc + ch[0], dc)
        self.up0 = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(dc, dc)
        self.up_extra = nn.ConvTranspose2d(dc, dc, kernel_size=2, stride=2)
        self.dec_extra = ConvBlock(dc, dc)

        self.head_prob = nn.Conv2d(dc, 1, 1)
        self.head_dist = nn.Conv2d(dc, self.n_rays, 1)
        self.head_cls = nn.Conv2d(dc, self.num_classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        t = self.bridge(feats[3])
        t = self.up3(t)
        t = torch.cat([t, feats[2]], dim=1)
        t = self.dec3(t)
        t = self.up2(t)
        t = torch.cat([t, feats[1]], dim=1)
        t = self.dec2(t)
        t = self.up1(t)
        t = torch.cat([t, feats[0]], dim=1)
        t = self.dec1(t)
        t = self.up0(t)
        t = self.dec0(t)
        t = self.up_extra(t)
        t = self.dec_extra(t)

        prob_logit = self.head_prob(t)
        dist_raw = self.head_dist(t)
        cls_logit = self.head_cls(t)
        dist = F.softplus(dist_raw) + 1e-3
        return prob_logit, dist, cls_logit


def build_model(cfg: dict) -> StardistMultitaskNet:
    m = cfg.get("model", cfg)
    return StardistMultitaskNet(
        backbone_name=str(m.get("backbone", "convnextv2_tiny.fcmae_ft_in22k_in1k_224")),
        pretrained=bool(m.get("pretrained", True)),
        n_rays=int(m.get("n_rays", 32)),
        num_classes=int(m.get("num_classes", 8)),
        decoder_channels=int(m.get("decoder_channels", 128)),
    )
