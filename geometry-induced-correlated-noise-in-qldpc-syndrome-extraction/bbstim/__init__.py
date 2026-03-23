# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
"""BB geometry-induced correlated-noise project."""

__all__ = [
    'BBCodeSpec',
    'build_bb72',
    'build_bb144',
    'MonomialColumnEmbedding',
    'IBMToricBiplanarEmbedding',
    'IBMBiplanarSurrogateEmbedding',
    'IBMBiplanarEmbedding',
    'JsonPolylineEmbedding',
    'crossing_kernel',
    'regularized_power_law_kernel',
    'exponential_kernel',
]
__version__ = '0.1.0'

from .bbcode import BBCodeSpec, build_bb72, build_bb144
from .embeddings import (
    MonomialColumnEmbedding,
    IBMToricBiplanarEmbedding,
    IBMBiplanarSurrogateEmbedding,
    IBMBiplanarEmbedding,
    JsonPolylineEmbedding,
)
from .geometry import crossing_kernel, regularized_power_law_kernel, exponential_kernel
