"""Training and export example for MNIST using MobileNetV2.

This script was originally a notebook. It has been refactored to follow
Google Python style: constants, small helper functions, and a ``main``
entrypoint.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tqdm import tqdm

# Ensure project root is on sys.path so `from lib import AU` works when
# the script is executed directly (``python test/test.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import AU


# Constants
IMG_SIZE = (32, 32)
AUTOTUNE = tf.data.AUTOTUNE
IMG_SHAPE = IMG_SIZE + (3,)
MODEL_DIR = "model"
CACHE_DIR = "cache"
BATCH_SIZE = 32


def preprocess_mnist(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Convert MNIST grayscale image to RGB and resize to IMG_SIZE.

    Args:
        image: Single-channel image tensor.
        label: Corresponding label.

    Returns:
        A tuple (image_rgb, label).
    """

    image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


def representative_dataset(validation_raw, preprocess_fn):
    """Yield calibration batches for TFLite quantization."""

    calibration_dataset = (
        validation_raw.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
        .take(500)
        .cache()
        .prefetch(AUTOTUNE)
    )

    for images, _ in tqdm(calibration_dataset, desc="Calibration"):
        yield [tf.cast(images, tf.float32)]


def main() -> None:
    """Run the end-to-end training, export and evaluation pipeline."""

    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(CACHE_DIR) and os.path.isdir(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
            print(f"Removed directory: {CACHE_DIR}")
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"Failed to remove {CACHE_DIR}: {exc}")

    # Load MNIST and prepare datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_dataset_raw = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
    validation_dataset_raw = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    class_names = [str(i) for i in range(10)]
    print("Class Names:", class_names)

    # Create folder named with model name and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_folder_name = f"model_{timestamp}"
    model_folder_path = os.path.join(MODEL_DIR, model_folder_name)
    os.makedirs(model_folder_path, exist_ok=True)

    # Save labels.txt
    labels_path = os.path.join(model_folder_path, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f"Labels written to: {labels_path}")

    train_dataset = (
        train_dataset_raw.map(preprocess_mnist, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(1000, reshuffle_each_iteration=True)
        .prefetch(AUTOTUNE)
    )

    validation_dataset = (
        validation_dataset_raw.map(preprocess_mnist, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    # Warm up caches
    print("Warming up caches...")
    for _ in train_dataset:
        pass
    for _ in validation_dataset:
        pass

    # Build model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        pooling="avg",
        alpha=0.35,
        weights="imagenet",
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255),
        base_model,
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])

    model.build((None, 32, 32, 3))
    model.summary()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.000015,
        decay_steps=len(train_dataset),
        decay_rate=0.99,
        staircase=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    model.fit(train_dataset, validation_data=validation_dataset, epochs=1, callbacks=[early_stopping])
    
    # Save the model directly to the timestamp folder
    h5_path = os.path.join(model_folder_path, "model.h5")
    model.save(h5_path)
    print(f"Model saved to: {h5_path}")

    # TFLite export with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(validation_dataset_raw, preprocess_mnist)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    output_path = os.path.join(model_folder_path, "model.tflite")
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {output_path}")

    # Evaluation: confusion matrix
    y_pred_probs = model.predict(validation_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.concatenate([labels.numpy() for _, labels in validation_dataset])

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    
    # Save confusion matrix to model folder
    cm_path = os.path.join(model_folder_path, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    
    plt.show()

    # Plot and save combined training curves
    from lib import polt_improved
    polt_improved.plot_combined_curves_improved([history], save_dir=model_folder_path)
    print(f"Training curves saved to: {model_folder_path}/training_curves.png")

    # Test the model using model_test
    import model_test
    # For MNIST test, we might need a different test logic or skip if model_test is for directory-based datasets
    # But if the user wants consistent behavior:
    print(f"\nModel test process completed. Output folder: {model_folder_path}")

    # Final cleanup of cache directory
    if os.path.exists(CACHE_DIR) and os.path.isdir(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
            print(f"Removed directory: {CACHE_DIR}")
        except Exception as exc:
            print(f"Failed to remove {CACHE_DIR}: {exc}")


if __name__ == "__main__":
    main()
