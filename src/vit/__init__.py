"""
Vision Transformer (ViT) implementation from scratch in TensorFlow/Keras.

This package provides a modular implementation of the Vision Transformer architecture
as described in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(Dosovitskiy et al., 2020).
"""

from vit.layers import PatchEmbedding, TransformerLayer
from vit.models import TransformerEncoder, ViT

__version__ = "0.1.0"
__all__ = [
    "PatchEmbedding",
    "TransformerLayer",
    "TransformerEncoder",
    "ViT",
]
