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
from typing import Tuple, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

import model_test

# Ensure project root is reachable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from lib import AU, polt_improved  # noqa: E402


# Constants
IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE
IMG_SHAPE = IMG_SIZE + (3,)
MODEL_DIR = "model"
CACHE_DIR = "cache"
BATCH_SIZE = 16


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
            tf.keras.layers.Dropout(0.8),
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


def export_tflite(model: tf.keras.Model, validation_raw: tf.data.Dataset, class_names: list, output_dir: str = MODEL_DIR) -> Tuple[str, str]:
    """Export the provided Keras model to a quantized TFLite file and save labels.

    Returns a tuple (tflite_path, model_folder_path).
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
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    # Create folder named with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_folder_name = f"model_{timestamp}"
    model_folder_path = os.path.join(output_dir, model_folder_name)
    os.makedirs(model_folder_path, exist_ok=True)

    # Save TFLite model
    tflite_name = f"{model_folder_name}.tflite"
    out_path = os.path.join(model_folder_path, tflite_name)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    # Save labels.txt
    labels_path = os.path.join(model_folder_path, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")

    return out_path, model_folder_path


def evaluate_confusion_matrix(model: tf.keras.Model, val_ds: tf.data.Dataset, class_names: Iterable[str], save_path: Union[str, None] = None) -> None:
    """Compute and plot confusion matrix on validation dataset."""

    y_pred_probs = model.predict(val_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in val_ds])

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def main() -> None:
    """Main training pipeline reproducing the notebook workflow."""

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Example: set this to your dataset root which contains `train` and `val`.
    base_dir = '../Datasets/smartcar26_dataset' 
    epochs1 = 30
    epochs2 = 10
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
    tflite_path, model_folder_path = export_tflite(model, val_ds, class_names)
    print(f"TFLite model saved to: {tflite_path}")

    # Save stage 1 model to the model folder as well
    stage1_h5_dest = os.path.join(model_folder_path, "stage1_model.h5")
    shutil.copy(os.path.join(MODEL_DIR, "stage1_model.h5"), stage1_h5_dest)
    print(f"Stage 1 model saved to: {stage1_h5_dest}")

    cm_save_path = os.path.join(model_folder_path, "confusion_matrix.png")
    evaluate_confusion_matrix(model, val_ds, class_names, save_path=cm_save_path)

    # Combined curves
    polt_improved.plot_combined_curves_improved([history1, history2], save_dir=model_folder_path)

    # Test the model using model_test
    # Assuming test directory is at '../Datasets/smartcar26_dataset/test'
    test_dir = os.path.join(os.path.dirname(base_dir), 'smartcar26_dataset', 'test')
    if os.path.exists(test_dir):
        print(f"\nStarting model test using model_test.py...")
        model_test.main(model_path=tflite_path, test_dir=test_dir)
    else:
        # Fallback if specific dataset structure is different
        test_dir_alt = os.path.join(base_dir, 'test')
        if os.path.exists(test_dir_alt):
            print(f"\nStarting model test using model_test.py...")
            model_test.main(model_path=tflite_path, test_dir=test_dir_alt)

    # Cleanup cache directory
    if os.path.exists(CACHE_DIR) and os.path.isdir(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
            print(f"Removed directory: {CACHE_DIR}")
        except Exception as exc:
            print(f"Failed to remove {CACHE_DIR}: {exc}")



if __name__ == "__main__":
    main()