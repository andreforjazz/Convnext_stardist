"""Shared ConvNeXt backbone + StarDist-like multitask decoder (segment + classify)."""

from .aux_codes.model_v2 import StardistMultitaskNetV2, build_model_v2

__all__ = ["StardistMultitaskNetV2", "build_model_v2"]
