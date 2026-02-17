"""
Vision Transformer (ViT) model and encoder stack.
"""

from typing import Optional

import tensorflow as tf
from tensorflow import keras

from vit.layers import PatchEmbedding, TransformerLayer


class TransformerEncoder(keras.layers.Layer):
    """Stack of Transformer blocks."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_layers = [
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        **kwargs,
    ) -> tf.Tensor:
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return x


class ViT(keras.Model):
    """
    Vision Transformer for image classification.

    Architecture: PatchEmbedding -> TransformerEncoder -> [CLS] -> MLP head.

    Args:
        num_classes: Number of output classes.
        patch_size: Patch size (square).
        num_patches: Number of patches, e.g. (image_size // patch_size) ** 2.
        d_model: Transformer hidden dimension (embedding dim).
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        mlp_ratio: MLP hidden dim = d_model * mlp_ratio.
        dropout_rate: Dropout rate in transformer and head.
    """

    def __init__(
        self,
        num_classes: int,
        patch_size: int,
        num_patches: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            num_patches=num_patches,
            projection_dim=d_model,
        )
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
        head_hidden = int(d_model * mlp_ratio)
        self.head = keras.Sequential([
            keras.layers.Dropout(0.3),
            keras.layers.Dense(head_hidden, activation="gelu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ])

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None,
        **kwargs,
    ) -> tf.Tensor:
        x = self.patch_embedding(inputs)
        x = self.encoder(x, training=training)
        cls_output = x[:, 0, :]
        return self.head(cls_output, training=training)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "num_patches": self.num_patches,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
        })
        return config
