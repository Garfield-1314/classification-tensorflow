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
    """Plot combined loss and accuracy curves and save to save_dir."""
    metrics = {
        "loss": ([], [], "blue", "orange"),
        "accuracy": ([], [], "green", "red")
    }
    all_epochs = []
    global_epoch = 0

    for history in history_list:
        h = history.history
        length = len(h["loss"])
        all_epochs.extend(range(global_epoch + 1, global_epoch + 1 + length))
        for key, (train_vals, val_vals, _, _) in metrics.items():
            train_vals.extend(h[key])
            val_vals.extend(h[f"val_{key}"])
        global_epoch += length

    save_path = Path(save_dir) / "training_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    for i, (key, (train_vals, val_vals, c1, c2)) in enumerate(metrics.items(), 1):
        plt.subplot(1, 2, i)
        plt.plot(all_epochs, train_vals, label=f"Train {key.capitalize()}", color=c1)
        plt.plot(all_epochs, val_vals, label=f"Val {key.capitalize()}", color=c2, linestyle="--")
        plt.title(f"Combined {key.capitalize()} Curves")
        plt.xlabel("Global Epochs")
        plt.ylabel(key.capitalize())
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Image saved to: {save_path}")
    return save_path


__all__ = ["plot_combined_curves_improved"]
