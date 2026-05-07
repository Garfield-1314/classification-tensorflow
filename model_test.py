"""Evaluate a TFLite classification model on a directory dataset and
plot per-class accuracies.

This module is refactored to follow Google Python style: ordered imports,
type annotations, small well-documented functions, and a `main` entrypoint.
"""

import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm


# Configuration constants
MODEL_PATH = './model/model_20260427_1518.tflite'  # Replace with your model path
TEST_DIR = os.path.join('../Datasets/smartcar26_dataset', 'test')  # Ensure this directory exists
BATCH_SIZE = 1  # Increase for better throughput if memory allows


def load_interpreter(model_path: str):
    """Load a TFLite interpreter and return interpreter metadata.

    Args:
        model_path: Path to the .tflite model.

    Returns:
        A tuple (interpreter, input_details, output_details, input_dtype,
        (expected_height, expected_width)).
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # If a GPU physical device exists, try to attach a delegate where applicable.
    if tf.config.list_physical_devices('GPU'):
        try:
            delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
            interpreter.modify_graph_with_delegate(delegate)
        except Exception:
            # Delegate loading is optional; continue without it on failure.
            pass

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_height = int(input_details[0]['shape'][1])
    expected_width = int(input_details[0]['shape'][2])
    input_dtype = input_details[0]['dtype']

    return (
        interpreter,
        input_details,
        output_details,
        input_dtype,
        (expected_height, expected_width),
    )


def preprocess_image(image: tf.Tensor, input_dtype: type) -> np.ndarray:
    """Preprocess a single image to the required input dtype.

    Args:
        image: An image tensor from the dataset.
        input_dtype: Expected NumPy dtype for the model input.

    Returns:
        A NumPy array suitable for feeding into the TFLite interpreter.
    """
    if input_dtype == np.uint8 or input_dtype == tf.uint8:
        return tf.cast(image, tf.uint8).numpy()
    return (tf.cast(image, tf.float32) / 255.0).numpy()


def predict_batch(
    interpreter, input_details, output_details, images: List[np.ndarray], input_dtype
) -> np.ndarray:
    """Run inference on a preprocessed batch of images.

    Args:
        interpreter: A TFLite Interpreter instance.
        input_details: Interpreter input details.
        output_details: Interpreter output details.
        images: A list (or array) of preprocessed images.
        input_dtype: NumPy dtype expected by the model.

    Returns:
        Raw model outputs as a NumPy array.
    """
    input_data = np.array(images, dtype=input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # If the output is uint8 (fully quantized), dequantize it manually for consistency if needed
    # But since we changed train.py to float32 output, this won't be strictly necessary for the new model
    if output_details[0]['dtype'] == np.uint8 or output_details[0]['dtype'] == tf.uint8:
        scale, zero_point = output_details[0]['quantization']
        if scale > 0:
            output_data = (output_data.astype(np.float32) - zero_point) * scale
            
    return output_data


def evaluate_dataset(
    interpreter,
    input_details,
    output_details,
    input_dtype,
    img_size: Tuple[int, int],
    test_dir: str,
    batch_size: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Evaluate the model on images loaded from a directory.

    Args:
        interpreter: TFLite interpreter.
        input_details: Interpreter input details.
        output_details: Interpreter output details.
        input_dtype: Expected input dtype.
        img_size: (height, width) tuple for resizing.
        test_dir: Directory containing test images organized by class.
        batch_size: Batch size for dataset loader.

    Returns:
        A tuple (class_names, correct_predictions, total_predictions).
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, batch_size=batch_size, image_size=img_size, shuffle=False
    )
    class_names = dataset.class_names
    num_classes = len(class_names)

    correct_predictions = np.zeros(num_classes)
    total_predictions = np.zeros(num_classes)

    for images, labels in tqdm(dataset, desc='测试进度'):
        processed = [preprocess_image(img, input_dtype) for img in images]
        batch_preds = predict_batch(
            interpreter, input_details, output_details, processed, input_dtype
        )
        predicted_labels = np.argmax(batch_preds, axis=1)

        for true_label, pred_label in zip(labels.numpy(), predicted_labels):
            total_predictions[true_label] += 1
            if true_label == pred_label:
                correct_predictions[true_label] += 1

    return class_names, correct_predictions, total_predictions


def compute_accuracies(correct_predictions: np.ndarray, total_predictions: np.ndarray) -> np.ndarray:
    """Compute per-class accuracies in percentage."""
    class_accuracies = np.zeros_like(correct_predictions, dtype=np.float32)
    for i in range(len(class_accuracies)):
        if total_predictions[i] > 0:
            class_accuracies[i] = correct_predictions[i] / total_predictions[i]
        else:
            class_accuracies[i] = 0.0
    return class_accuracies * 100


def plot_class_accuracies(class_names: List[str], accuracies: np.ndarray, save_dir: str, model_path: str) -> str:
    """Plot per-class accuracies and save the bar chart to `save_dir`.

    Returns the path to the saved image.
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, accuracies, color='skyblue')

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{acc:.1f}%', ha='center', va='bottom')

    plt.xlabel('class')
    plt.ylabel('acc (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_accuracy_bar_chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return save_path


def main(model_path: str = MODEL_PATH, test_dir: str = TEST_DIR) -> None:
    """Main entrypoint: load model, evaluate dataset, and plot results."""
    (
        interpreter,
        input_details,
        output_details,
        input_dtype,
        img_size,
    ) = load_interpreter(model_path)

    print(f"模型输入尺寸: {img_size[0]}x{img_size[1]}, 数据类型: {input_dtype}")

    class_names, correct_predictions, total_predictions = evaluate_dataset(
        interpreter, input_details, output_details, input_dtype, img_size, test_dir, BATCH_SIZE
    )

    # If model is in a subdirectory, use that as save directory
    model_dir = os.path.dirname(model_path)
    save_dir = model_dir if model_dir and model_dir != '.' else 'test'

    accuracies = compute_accuracies(correct_predictions, total_predictions)
    save_path = plot_class_accuracies(class_names, accuracies, save_dir, model_path)
    print(f'柱状图已保存到: {save_path}')

    print('\n各类别准确率：')
    for i, name in enumerate(class_names):
        print(f'  {name.ljust(15)}: {accuracies[i]:.2f}%  ({int(correct_predictions[i])}/{int(total_predictions[i])})')

    total_correct = int(np.sum(correct_predictions))
    total_samples = int(np.sum(total_predictions))
    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

    print('\n' + '-' * 50)
    print(f' 整体测试准确率: {overall_accuracy:.2f}%  ({total_correct}/{total_samples})')
    print('-' * 50)


if __name__ == '__main__':
    main()
