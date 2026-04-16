[![](https://img.shields.io/badge/lang-English-blue.svg)](README.md) [![](https://img.shields.io/badge/lang-中文-red.svg)](README_CN.md)

# TensorFlow 图像分类与 TFLite 量化项目

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://python.org)
[![TFLite](https://img.shields.io/badge/TFLite-INT8_Quantized-green.svg)](https://www.tensorflow.org/lite)

这是一个基于 TensorFlow 的图像分类项目，专注于高效模型训练与 TFLite 全整型 (INT8) 量化，旨在将深度学习模型部署到边缘设备。

---

# TensorFlow Image Classification and TFLite Quantization
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://python.org)

This repository provides reproducible training pipelines, model export utilities, and a TFLite INT8 quantization workflow for deploying image classification models on edge devices.
---

## Key updates (recent refactor)

- Added `train.py` — a command-line training and export script refactored from `train.ipynb`.
- Improved `model_test.py` — evaluate `.tflite` models and plot per-class accuracy bar charts.
- Kept `test/test.py` as an end-to-end MNIST example (label file generation and TFLite export).

---


## Requirements

- OS: Windows or Linux
- Python 3.8 (virtual environment recommended)
- For GPU: compatible CUDA and cuDNN versions for the installed TensorFlow build

Install dependencies:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## Repository layout

```text
classification-tensorflow/
├── train.ipynb          # Jupyter notebook (interactive)
├── train.py             # CLI training and export script (refactored)
├── model_test.py        # Evaluate .tflite models and plot per-class accuracies
├── requirements.txt     # Python dependencies
├── test/                # Example & test scripts
│   ├── GPU_test.py      # GPU availability check
│   └── test.py          # MNIST end-to-end example (train -> export -> evaluate)
├── lib/                 # Utilities (preprocessing, plotting, etc.)
├── model/               # Output models
└── cache/               # Runtime caches
```

---

## Quick start

Check GPU availability:

```bash
python test/GPU_test.py
```

Run the MNIST end-to-end example:

```bash
python test/test.py
```

This example converts MNIST to 32x32 RGB, trains a small MobileNetV2 model, exports a TFLite model and writes labels to `model/labels_<timestamp>.txt`.

Train with `train.py` (command line):

```bash
# Edit `base_dir` in train.py to point to your dataset root containing `train/` and `val/`
python train.py
```

`train.py` runs a two-stage training process: stage 1 saves `stage1_model.h5`, stage 2 fine-tunes and exports a timestamped `model_YYYYMMDD_HHMM.tflite`.

Evaluate a .tflite model:

```bash
# Modify MODEL_PATH and TEST_DIR in model_test.py or update variables in the script
python model_test.py
```

`model_test.py` loads the specified `.tflite`, evaluates images organized by class under `TEST_DIR`, and saves a per-class accuracy bar chart (to the `test/` directory by default).

---

## Output conventions

- TFLite export example: `model/model_YYYYMMDD_HHMM.tflite`
- Label file example: `model/labels_YYYYMMDD_HHMM.txt`

---

## References

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

---



