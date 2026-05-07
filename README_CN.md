
[![](https://img.shields.io/badge/lang-中文-red.svg)](README_CN.md) [![](https://img.shields.io/badge/lang-English-blue.svg)](README.md)



# TensorFlow 图像分类与 TFLite 量化项目

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://python.org)
[![TFLite](https://img.shields.io/badge/TFLite-INT8_Quantized-green.svg)](https://www.tensorflow.org/lite)

本仓库提供可复现的训练流程、模型导出工具，以及 TFLite INT8 量化流程，便于将图像分类模型部署到边缘设备。

---

## 运行环境

- 操作系统：Windows 或 Linux
- TensorFlow：
  - Windows 环境下最大支持 TensorFlow 2.10（> 2.10 版本无法在 Windows 下使用 GPU 训练）。
  - Linux 环境下无具体版本限制。
- Python 3.8（推荐使用虚拟环境）
- 若需 GPU：请准备匹配的 CUDA 与 cuDNN（依赖 TensorFlow 版本）


安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 目录结构

```text
classification-tensorflow/
├── train.ipynb          # Jupyter notebook（交互式，已更新导出流程）
├── train.py             # 命令行训练与导出脚本（已更新导出流程）
├── model_test.py        # 评估 .tflite 模型并绘制每类准确率
├── requirements.txt     # 依赖列表
├── test/                # 示例与测试脚本
│   ├── GPU_test.py      # 检查 GPU 可用性
│   └── test.py          # MNIST 端到端示例（已更新导出流程）
├── lib/                 # 工具库（预处理、绘图等）
├── model/               # 输出模型（每个导出任务拥有独立子文件夹）
└── cache/               # 运行时缓存
```

---

## 快速开始

检查 GPU 可用性：

```bash
python test/GPU_test.py
```

运行 MNIST 端到端示例：

```bash
python test/test.py
```

该示例会将 MNIST 转为 32x32 RGB，训练一个小型 MobileNetV2 模型，并将所有产物（TFLite、标签、图表）导出到 `model/` 下带时间戳的文件夹中。

命令行训练（使用 `train.py`）：

```bash
# 编辑 train.py 中的 base_dir 路径，指向包含 `train/` 和 `val/` 的数据集根目录
python train.py
```

`train.py` 会执行两阶段训练，并将所有相关文件（tflite, h5, labels, 曲线图, 混淆矩阵）导出到专用文件夹。

评估已导出的 TFLite 模型：

```bash
# 修改 model_test.py 中的 MODEL_PATH。测试结果将直接保存到模型所在文件夹。
python model_test.py
```

`model_test.py` 会加载指定 `.tflite`，进行精度评估，并直接在模型子目录下生成准确率柱状图。

---

## 输出约定

模型将导出至 `model/model_YYYYMMDD_HHMM/` 结构下，包含：
- `model_YYYYMMDD_HHMM.tflite`: 量化后的 TFLite 模型。
- `stage1_model.h5`: 阶段一中间模型。
- `stage2_model.h5`: 阶段二精调后的原始模型（推荐用于断点续训）。
- `labels.txt`: 类别标签文件。
- `confusion_matrix.png`: 验证集混淆矩阵。
- `training_curves.png`: 训练损失与准确率曲线。
- `*_accuracy_bar_chart.png`: 测试集评估结果（由 `model_test.py` 生成）。

---

## 更新日志

详情请参阅 [CHANGELOG.md](CHANGELOG.md)。

---

## 参考链接

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
