"""Shared ConvNeXt backbone + StarDist-like multitask decoder (segment + classify)."""

from .model import StardistMultitaskNet, build_model

__all__ = ["StardistMultitaskNet", "build_model"]
