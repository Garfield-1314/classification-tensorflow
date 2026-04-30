"""Plot utilities for combined training curves.

This module provides a single helper to combine multiple training
``History`` objects and plot/save the aggregated loss and accuracy
curves. The function accepts an iterable of Keras History objects.
"""

from pathlib import Path
from typing import Iterable, List, Union

import matplotlib.pyplot as plt
from datetime import datetime


MODEL_DIR = Path("model")


def plot_combined_curves_improved(history_list: Iterable, save_dir: Union[str, Path] = MODEL_DIR) -> Path:
    """Plot combined loss and accuracy curves and save to save_dir.

    Args:
        history_list: Iterable of Keras ``History`` objects.
        save_dir: Directory to save the resulting plot.

    Returns:
        The filesystem ``Path`` to the saved image.
    """
    all_train_loss: List[float] = []
    all_val_loss: List[float] = []
    all_train_accuracy: List[float] = []
    all_val_accuracy: List[float] = []
    all_epochs: List[int] = []

    global_epoch = 0

    for history in history_list:
        length = len(history.history["loss"])
        epochs = range(global_epoch + 1, global_epoch + 1 + length)
        all_train_loss.extend(history.history["loss"])  # type: ignore[index]
        all_val_loss.extend(history.history["val_loss"])  # type: ignore[index]
        all_train_accuracy.extend(history.history["accuracy"])  # type: ignore[index]
        all_val_accuracy.extend(history.history["val_accuracy"])  # type: ignore[index]
        all_epochs.extend(epochs)
        global_epoch += length

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "training_curves.png"

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(all_epochs, all_train_loss, label="Train Loss", color="blue")
    plt.plot(all_epochs, all_val_loss, label="Validation Loss", color="orange", linestyle="--")
    plt.title("Combined Loss Curves")
    plt.xlabel("Global Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(all_epochs, all_train_accuracy, label="Train Accuracy", color="green")
    plt.plot(all_epochs, all_val_accuracy, label="Validation Accuracy", color="red", linestyle="--")
    plt.title("Combined Accuracy Curves")
    plt.xlabel("Global Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Image saved to: {save_path}")
    return save_path


__all__ = ["plot_combined_curves_improved"]
