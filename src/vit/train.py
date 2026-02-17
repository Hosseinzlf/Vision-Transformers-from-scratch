"""
Training utilities: strategy selection (CPU/GPU/TPU) and model compilation.
"""

from typing import Optional, List, Any

import tensorflow as tf
from tensorflow import keras

from vit.models import ViT


def get_strategy(use_tpu: bool = False, tpu_address: Optional[str] = None) -> tf.distribute.Strategy:
    """
    Return a distribution strategy for training.

    Args:
        use_tpu: If True, attempt to connect to TPU.
        tpu_address: TPU address (e.g. '' for Colab, 'local' for local TPU). Ignored if use_tpu=False.
    """
    if use_tpu:
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address or "")
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            return tf.distribute.TPUStrategy(resolver)
        except ValueError:
            pass
    if len(tf.config.list_physical_devices("GPU")) > 0:
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()


def build_and_compile_vit(
    num_classes: int,
    image_size: int,
    patch_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    mlp_ratio: float = 2.0,
    dropout_rate: float = 0.1,
    optimizer: Any = "adam",
    metrics: Optional[List[Any]] = None,
) -> ViT:
    """
    Build ViT and compile with cross-entropy loss and given optimizer/metrics.

    num_patches is set to (image_size // patch_size) ** 2.
    """
    num_patches = (image_size // patch_size) ** 2
    model = ViT(
        num_classes=num_classes,
        patch_size=patch_size,
        num_patches=num_patches,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_ratio=mlp_ratio,
        dropout_rate=dropout_rate,
    )
    if metrics is None:
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy"),
        ]
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=metrics,
    )
    return model
