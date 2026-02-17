"""
Data loading, preprocessing, and augmentation for ViT training.
"""

from typing import Tuple, Optional

import tensorflow as tf
from tensorflow import keras


def build_preprocessing(image_size: Tuple[int, int], adapt_data: Optional[tf.Tensor] = None) -> keras.Sequential:
    """
    Build a preprocessing model: normalization + resizing.

    Args:
        image_size: (height, width) after resize.
        adapt_data: Optional tensor to adapt Normalization layer (e.g. x_train).
    """
    model = keras.Sequential([
        keras.layers.Normalization(),
        keras.layers.Resizing(image_size[0], image_size[1]),
    ])
    if adapt_data is not None:
        model.layers[0].adapt(adapt_data)
    return model


def build_augmentation() -> keras.Sequential:
    """Build augmentation pipeline (horizontal flip, rotation, zoom)."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomZoom(width_factor=0.2, height_factor=0.2),
    ])


def create_dataset(
    x: tf.Tensor,
    y: tf.Tensor,
    batch_size: int,
    preprocessing: keras.Sequential,
    augmentation: Optional[keras.Sequential] = None,
    shuffle: bool = False,
    shuffle_buffer_size: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset with preprocessing and optional augmentation.

    Args:
        x: Images tensor (N, H, W, C).
        y: Labels tensor (N,) or (N, 1).
        batch_size: Batch size.
        preprocessing: Model that normalizes and resizes.
        augmentation: Optional augmentation model (used when shuffle=True if provided).
        shuffle: Whether to shuffle the dataset.
        shuffle_buffer_size: Buffer size for shuffle; default min(len(x), 10000).
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    def preprocess(x_batch, y_batch):
        return preprocessing(x_batch), y_batch

    dataset = dataset.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        buffer = shuffle_buffer_size or min(int(tf.shape(x)[0].numpy()), 10000)
        dataset = dataset.shuffle(buffer)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    if augmentation is not None and shuffle:
        def augment(x_batch, y_batch):
            return augmentation(x_batch, training=True), y_batch
        dataset = dataset.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


def load_cifar10() -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """Load CIFAR-10 and return (x_train, y_train), (x_test, y_test)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)
