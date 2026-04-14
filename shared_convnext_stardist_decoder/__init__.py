"""Shared ConvNeXt backbone + StarDist-like multitask decoder (segment + classify)."""

from .model_v2 import StardistMultitaskNetV2, build_model_v2

__all__ = ["StardistMultitaskNetV2", "build_model_v2"]
