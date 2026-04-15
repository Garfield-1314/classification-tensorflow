"""Utilities to check GPU availability for TensorFlow.

This small module prints TensorFlow version and available physical
devices. It is intended as a diagnostic helper.
"""

from typing import Sequence

import tensorflow as tf


def check_gpu_availability() -> Sequence[tf.config.PhysicalDevice]:
    """Print TensorFlow GPU information and return the list of devices.

    Returns:
        The list of physical devices discovered by TensorFlow.
    """

    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")

    gpu_devices = tf.config.list_physical_devices("GPU")
    print("\nGPU available:", "Yes" if gpu_devices else "No")

    if gpu_devices:
        print("\nGPU devices:")
        for i, gpu in enumerate(gpu_devices):
            details = tf.config.experimental.get_device_details(gpu)
            print(f"[GPU {i}]")
            print(f"  name: {gpu.name}")
            print(f"  type: {gpu.device_type}")
            print(f"  compute capability: {details.get('compute_capability', 'unknown')}")

            # Backwards compatible memory info retrieval
            if hasattr(gpu, "memory_limit"):
                print(f"  memory: {gpu.memory_limit // 1024 // 1024} MB")
            else:
                from tensorflow.python.client import device_lib

                local_devices = device_lib.list_local_devices()
                gpu_info = [x for x in local_devices if x.device_type == "GPU"][i]
                print(f"  memory: {int(gpu_info.memory_limit // 1024 // 1024)} MB (compat)")

            print(f"  device_name: {details.get('device_name', 'unknown')}")
    else:
        print("\nHint: check CUDA/cuDNN installation and TensorFlow GPU build.")

    print("\nAll visible devices:")
    devices = tf.config.list_physical_devices()
    for device in devices:
        print(f"- {device.name} ({device.device_type})")

    return gpu_devices


if __name__ == "__main__":
    check_gpu_availability()
    print("\n" + "=" * 50)
