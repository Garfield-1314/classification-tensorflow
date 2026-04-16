
[![](https://img.shields.io/badge/lang-中文-red.svg)](README_CN.md) [![](https://img.shields.io/badge/lang-English-blue.svg)](README.md)



# TensorFlow 图像分类与 TFLite 量化项目

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://python.org)
[![TFLite](https://img.shields.io/badge/TFLite-INT8_Quantized-green.svg)](https://www.tensorflow.org/lite)

本仓库提供可复现的训练流程、模型导出工具，以及 TFLite INT8 量化流程，便于将图像分类模型部署到边缘设备。

---

# 主要变化（最近一次重构）

- 新增 `train.py` —— 由 `train.ipynb` 重构的命令行训练与导出脚本。
- 改进 `model_test.py` —— 支持评估 `.tflite` 模型并绘制每类准确率柱状图。
- 保留 `test/test.py` 作为 MNIST 端到端示例（标签文件生成与 TFLite 导出）。

---

## 运行环境

- 操作系统：Windows 或 Linux
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
├── train.ipynb          # Jupyter notebook（交互式）
├── train.py             # 命令行训练与导出脚本（重构自 notebook）
├── model_test.py        # 评估 .tflite 模型并绘制每类准确率
├── requirements.txt     # 依赖列表
├── test/                # 示例与测试脚本
│   ├── GPU_test.py      # 检查 GPU 可用性
│   └── test.py          # MNIST 端到端示例（训练 -> 导出 -> 评估）
├── lib/                 # 工具库（预处理、绘图等）
├── model/               # 输出模型
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

该示例会将 MNIST 转为 32x32 RGB，训练一个小型 MobileNetV2 模型，导出 TFLite，并将类别标签写入 `model/labels_<timestamp>.txt`。

命令行训练（使用 `train.py`）：

```bash
# 编辑 train.py 中的 base_dir 路径，指向包含 `train/` 和 `val/` 的数据集根目录
python train.py
```

`train.py` 会执行两阶段训练：阶段 1 保存 `stage1_model.h5`，阶段 2 微调并导出带时间戳的 `model_YYYYMMDD_HHMM.tflite`。

评估已导出的 TFLite 模型：

```bash
# 在 model_test.py 中修改 MODEL_PATH 和 TEST_DIR，或在脚本中按需替换路径
python model_test.py
```

`model_test.py` 会加载指定 `.tflite`，对 `TEST_DIR` 下按类别组织的图片进行评估，并将每类准确率柱状图保存到 `test/` 目录。

---

## 输出约定

- TFLite 导出示例：`model/model_YYYYMMDD_HHMM.tflite`
- 标签文件示例：`model/labels_YYYYMMDD_HHMM.txt`

---

## 参考链接

- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
