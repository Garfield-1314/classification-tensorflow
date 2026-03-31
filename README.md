# TensorFlow 图像分类与 TFLite 量化项目

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://python.org)
[![TFLite](https://img.shields.io/badge/TFLite-INT8_Quantized-green.svg)](https://www.tensorflow.org/lite)

这是一个基于 TensorFlow 的图像分类项目，专注于高效模型训练与 TFLite 全整型 (INT8) 量化，旨在将深度学习模型部署到边缘设备。

---

## 🚀 项目特性

- **轻量化主干网络**: 默认采用 **MobileNetV2**，并支持多阶段微调（Fine-tuning）。
- **全整型量化**: 包含完整的 TFLite INT8 量化流程，大幅降低模型体积并加速推理。
- **自动化工具**: 
  - 自动生成 `labels.txt`。
  - 自动化可视化训练曲线与混淆矩阵。
  - GPU 环境检测脚本 `GPU_test.py`。
- **代码结构清晰**: 逻辑清晰的 `train.ipynb` 操作手册。

---

## 🛠️ 环境要求

### 系统环境
- **操作系统**: Windows / Linux
- **Python**: 3.8
- **CUDA**: 11.8
- **cuDNN**: 8.9.7

### 依赖安装
推荐使用清华镜像源加速安装：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📂 目录结构

```text
classification-tensorflow/
├── train.ipynb          # 主训练脚本 (Jupyter Notebook)
├── model_test.py        # TFLite 模型测试与推理脚本
├── requirements.txt     # 项目依赖列表
├── test/                # 测试与验证目录
│   ├── GPU_test.py      # GPU 环境可用性检测
│   └── test.py          # 流程完整性测试脚本 (支持 MNIST 自动测试)
├── lib/                 # 自定义工具库
│   ├── AU.py            # 数据预处理与增强
│   ├── polt_improved.py # 曲线绘制工具
│   └── show_img.py      # 图像可视化工具
├── model/               # 模型输出目录 (自动生成)
└── cache/               # 数据缓存目录 (自动生成)
```

---

## 📖 使用指南

### 1. 环境验证与流程测试
在开始大规模训练前，建议先进行环境验证。

**检测 GPU 状态**:
```bash
python test/GPU_test.py
```

**运行自动化流程测试**:
项目提供了 `test/test.py` 脚本，该脚本会自动加载 **MNIST 数据集**，并将其转换为 32x32 RGB 格式，以测试从训练到 TFLite INT8 量化导出的全流程：
```bash
python test/test.py
```

### 2. 数据准备
将你的数据集按照以下结构存放：
```text
yourdataset/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```
并在 `train.ipynb` 中修改 `base_dir` 变量。

### 3. 模型训练
打开 `train.ipynb`，按照以下步骤运行：
- **阶段一**: 模型基础训练与收敛。
- **阶段二**: 二次微调或特定数据处理。
- **量化导出**: 自动运行 `representative_dataset` 进行校准，并导出全整型量化模型。

### 4. 模型评估
使用 `model_test.py` 对导出的 `.tflite` 模型进行快速测试。修改脚本中的 `model_path` 指向生成的模型文件即可。

---

## 📊 结果展示

项目会自动生成：
- **训练曲线**: 准确率与损失值的交互对比图。
- **混淆矩阵**: 用于衡量模型在各个类别上的表现。
- **测试报告**: 包含每个类别的 Top-1 准确率统计柱状图。

---

## 🔗 相关项目与参考
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)




