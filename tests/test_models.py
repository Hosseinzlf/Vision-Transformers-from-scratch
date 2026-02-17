"""
Basic tests for ViT model and layers.
"""

import numpy as np
import pytest
import tensorflow as tf

from vit import PatchEmbedding, TransformerLayer, TransformerEncoder, ViT


def test_patch_embedding_shape():
    layer = PatchEmbedding(patch_size=8, num_patches=16, projection_dim=64)
    x = tf.random.normal((2, 32, 32, 3))
    out = layer(x)
    assert out.shape == (2, 16 + 1, 64)  # num_patches + 1 for CLS


def test_transformer_layer_shape():
    layer = TransformerLayer(d_model=64, num_heads=2, mlp_ratio=2.0, dropout_rate=0.0)
    x = tf.random.normal((2, 10, 64))
    out = layer(x, training=False)
    assert out.shape == x.shape


def test_vit_build_and_call():
    model = ViT(
        num_classes=10,
        patch_size=8,
        num_patches=16,
        d_model=64,
        num_heads=2,
        num_layers=2,
        mlp_ratio=2.0,
        dropout_rate=0.0,
    )
    x = tf.random.normal((2, 32, 32, 3))
    logits = model(x, training=False)
    assert logits.shape == (2, 10)
    assert tf.reduce_all(tf.math.abs(tf.reduce_sum(logits, axis=1) - 1.0) < 1e-5)


def test_vit_cifar_config():
    """Config used in notebook/scripts: 72x72, patch 6 -> 144 patches."""
    model = ViT(
        num_classes=10,
        patch_size=6,
        num_patches=(72 // 6) ** 2,
        d_model=128,
        num_heads=2,
        num_layers=4,
        mlp_ratio=2.0,
        dropout_rate=0.1,
    )
    x = tf.random.normal((4, 72, 72, 3))
    logits = model(x, training=False)
    assert logits.shape == (4, 10)
