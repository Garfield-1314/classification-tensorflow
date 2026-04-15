"""Data augmentation utilities.

This module provides simple preprocessing helpers used by the training
pipeline. It follows Google Python style: module-level constants, small
functions with type annotations and clear docstrings.
"""

from typing import Tuple

import tensorflow as tf


# Data augmentation pipeline applied during training.
DATA_AUGMENTATION = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=(-0.1, 0.1), fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.10, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
        tf.keras.layers.RandomBrightness(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ]
)


def preprocess_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Return the image/label tuple without augmentation.

    Args:
        image: Input image tensor.
        label: Corresponding label tensor.

    Returns:
        The (image, label) tuple unchanged.
    """

    return image, label


def preprocess_image_aug(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply augmentation pipeline to ``image`` and return (image, label).

    Args:
        image: Input image tensor.
        label: Corresponding label tensor.

    Returns:
        A tuple of (augmented_image, label).
    """

    image = DATA_AUGMENTATION(image)
    return image, label


__all__ = ["DATA_AUGMENTATION", "preprocess_image", "preprocess_image_aug"]
