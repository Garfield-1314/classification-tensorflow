"""Utilities to display a sample grid of images from a dataset."""

from typing import Optional

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def plot_images(dataset: tf.data.Dataset,
                title: str,
                augmentation: Optional[tf.keras.layers.Layer] = None,
                rows: int = 3,
                cols: int = 3) -> None:
    """Plot a grid of images taken from the first batch of ``dataset``.

    Args:
        dataset: A ``tf.data.Dataset`` yielding (images, labels).
        title: Title for the figure.
        augmentation: Optional augmentation layer to apply for visualization.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
    """

    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=16)

    for images, labels in dataset.take(1):
        for i in range(rows * cols):
            ax = plt.subplot(rows, cols, i + 1)
            image = images[i].numpy()

            if augmentation is not None:
                image = augmentation(tf.expand_dims(image, axis=0))[0]

            # If the image is float in [0, 1], convert to uint8 for display.
            if image.dtype == np.float32:
                image = np.clip(image * 255, 0, 255).astype("uint8")

            plt.imshow(image)
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis("off")

    plt.tight_layout()
    plt.show()


__all__ = ["plot_images"]
