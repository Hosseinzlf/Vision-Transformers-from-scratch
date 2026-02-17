"""
Core building blocks for the Vision Transformer: patch embedding and transformer layers.
"""

from typing import Optional

import tensorflow as tf
from tensorflow import keras


class PatchEmbedding(keras.layers.Layer):
    """
    Splits an image into patches, linearly projects them, and adds learnable
    [CLS] token and positional embeddings (ViT-style).

    Args:
        patch_size: Height and width of each patch (square).
        num_patches: Number of patches (excluding [CLS]). Typically (H/patch_size)².
        projection_dim: Output dimension of the linear projection (d_model).
    """

    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        projection_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        # +1 for [CLS] token
        self.num_patches = num_patches + 1
        self.projection_dim = projection_dim

        self.projection = keras.layers.Dense(projection_dim)
        self.cls_token = tf.Variable(
            keras.initializers.GlorotNormal()(shape=(1, 1, projection_dim)),
            trainable=True,
        )
        self.positional_embedding = keras.layers.Embedding(
            self.num_patches, projection_dim
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        # Extract non-overlapping patches
        patches = tf.image.extract_patches(
            inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        # Flatten patch spatial dims: (B, n_h, n_w, p*p*C) -> (B, n_patches, p*p*C)
        patches = tf.reshape(
            patches,
            (batch_size, -1, self.patch_size * self.patch_size * 3),
        )
        patches = self.projection(patches)

        cls_tokens = tf.repeat(self.cls_token, batch_size, axis=0)
        patches = tf.concat([cls_tokens, patches], axis=1)

        positions = tf.range(0, self.num_patches, 1)[tf.newaxis, ...]
        pos_emb = self.positional_embedding(positions)
        return patches + pos_emb

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "num_patches": self.num_patches - 1,  # store without CLS
            "projection_dim": self.projection_dim,
        })
        return config


class TransformerLayer(keras.layers.Layer):
    """
    Single Transformer block: pre-norm multi-head self-attention + pre-norm MLP.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate

        self.layernorm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.layernorm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = keras.Sequential([
            keras.layers.Dense(mlp_dim, activation="gelu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout_rate),
        ])

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        **kwargs,
    ) -> tf.Tensor:
        # Pre-norm self-attention
        x = self.layernorm_1(inputs)
        x = self.mha(x, x, training=training)
        x = inputs + x

        # Pre-norm MLP (no activation on last Dense)
        y = self.layernorm_2(x)
        y = self.mlp(y, training=training)
        return x + y

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
        })
        return config
