"""Refactored training script converted from train.ipynb.

This script follows Google Python style: ordered imports, constants,
small functions with docstrings and type annotations, and a ``main``
entrypoint. It reproduces the notebook's two-stage training and TFLite
export pipeline.
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

# Ensure project root is reachable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from lib import AU, polt_improved  # noqa: E402


# Constants
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
IMG_SHAPE = IMG_SIZE + (3,)
MODEL_DIR = "model"
CACHE_DIR = "cache"
BATCH_SIZE = 32


def prepare_datasets(base_dir: str, augment: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset, list]:
    """Prepare train/validation datasets from a directory.

    Args:
        base_dir: Root directory that contains `train` and `val` subfolders.
        augment: Whether to apply augmentation to training pipeline.

    Returns:
        A tuple (train_dataset, validation_dataset, class_names).
    """

    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "val")

    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )
    validation_raw = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    )

    class_names = train_raw.class_names

    train_cache = os.path.join(CACHE_DIR, "train_cache")
    val_cache = os.path.join(CACHE_DIR, "val_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    preprocess_fn = AU.preprocess_image_aug if augment else AU.preprocess_image

    train_ds = (
        train_raw.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
        .cache(train_cache)
        .shuffle(1000, reshuffle_each_iteration=True)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        validation_raw.map(AU.preprocess_image, num_parallel_calls=AUTOTUNE)
        .cache(val_cache)
        .prefetch(AUTOTUNE)
    )

    # Warm-up caches
    for _ in train_ds:
        break
    for _ in val_ds:
        break

    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    """Build and return a MobileNetV2-based classifier."""

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE, include_top=False, pooling="avg", alpha=0.35, weights="imagenet"
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            base_model,
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.build((None, *IMG_SIZE, 3))
    return model


def train_model(model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int = 20) -> tf.keras.callbacks.History:
    """Compile and train the model, returning the history."""

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1.5e-5, decay_steps=len(train_ds), decay_rate=0.99, staircase=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping])
    return history


def export_tflite(model: tf.keras.Model, validation_raw: tf.data.Dataset, output_dir: str = MODEL_DIR) -> str:
    """Export the provided Keras model to a quantized TFLite file.

    Returns the path to the saved tflite model.
    """

    def representative_dataset():
        calibration = (
            validation_raw.map(AU.preprocess_image, num_parallel_calls=AUTOTUNE).take(500).cache().prefetch(AUTOTUNE)
        )
        for images, _ in tqdm(calibration, desc="Calibration"):
            yield [tf.cast(images, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(output_dir, f"model_{timestamp}.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    return out_path


def evaluate_confusion_matrix(model: tf.keras.Model, val_ds: tf.data.Dataset, class_names: Iterable[str]) -> None:
    """Compute and plot confusion matrix on validation dataset."""

    y_pred = np.argmax(model.predict(val_ds), axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in val_ds])

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def main() -> None:
    """Main training pipeline reproducing the notebook workflow."""

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Example: set this to your dataset root which contains `train` and `val`.
    base_dir = "your_dataset_path"
    epochs1 = 20
    epochs2 = 5
    # Stage 1: train without augmentation
    train_ds, val_ds, class_names = prepare_datasets(base_dir, augment=False)
    model = build_model(len(class_names))
    history1 = train_model(model, train_ds, val_ds, epochs=epochs1)
    model.save(os.path.join(MODEL_DIR, "stage1_model.h5"))

    # Stage 2: fine-tune with augmentation
    train_ds2, val_ds2, _ = prepare_datasets(base_dir, augment=True)
    # reload model if needed
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "stage1_model.h5"))
    history2 = train_model(model, train_ds2, val_ds2, epochs=epochs2)

    # Export and evaluation
    tflite_path = export_tflite(model, val_ds)
    print(f"TFLite model saved to: {tflite_path}")

    evaluate_confusion_matrix(model, val_ds, class_names)

    # Combined curves
    polt_improved.plot_combined_curves_improved([history1, history2])

    # Cleanup cache directory
    if os.path.exists(CACHE_DIR) and os.path.isdir(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
            print(f"Removed directory: {CACHE_DIR}")
        except Exception as exc:
            print(f"Failed to remove {CACHE_DIR}: {exc}")



if __name__ == "__main__":
    main()