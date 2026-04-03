" Some simple dataclass containers for registration settings and results. "

# imports
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ElasticRegistrationSettings:
    "Settings for elastic registration of subimage tiles."
    tile_size: int = 250
    intertile_distance: int = 100
    n_buffer_pixels: int = 200


@dataclass(frozen=True)
class RegistrationReference:
    "Reference image (grayscale) with mask for registration."
    image: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class GlobalRegistrationResult:
    "Result of global registration."
    transformation: np.ndarray
    flipped: bool
