
[![](https://img.shields.io/badge/lang-中文-red.svg)](README_CN.md) [![](https://img.shields.io/badge/lang-English-blue.svg)](README.md)

 # TensorFlow 图像分类与 TFLite 量化项目

 基于 TensorFlow 的图像分类仓库，侧重于可重复的训练流程、模型导出及 TFLite INT8 量化，支持在边缘设备上高效推理。

 ---

 ## 主要变化（于最近一次代码重构）

 - 提供了由 `train.ipynb` 重构得到的脚本 `train.py`，用于命令行运行训练与导出流程。
 - 新增/改进了 `model_test.py`：用于评估 `.tflite` 模型并绘制每类准确率柱状图。
 - `test/test.py` 保留并作为 MNIST 的端到端示例（包含标签文件生成与 TFLite 导出）。

 ---

 ## 运行环境

 - 操作系统：Windows / Linux
 - Python：3.8（推荐使用虚拟环境）
 - 若需 GPU 支持，请准备匹配的 CUDA 与 cuDNN（视 TensorFlow 版本而定）

 安装依赖：

 ```bash
 pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
 ```

 ---

 ## 目录概览

 ```text
 classification-tensorflow/
 ├── train.ipynb          # Jupyter notebook（交互式演示）
 ├── train.py             # 命令行可运行的训练与导出脚本（重构自 notebook）
 ├── model_test.py        # 对 .tflite 模型进行评估并绘制每类准确率
 ├── requirements.txt     # 依赖列表示例
 ├── test/                # 小示例与测试
 │   ├── GPU_test.py      # GPU 环境可用性检测
 │   └── test.py          # MNIST 端到端示例（训练 -> 导出 -> 评估）
 ├── lib/                 # 自定义工具库（预处理、绘图工具等）
 ├── model/               # 模型输出目录（训练/导出结果）
 └── cache/               # 数据缓存目录（运行时生成）
 ```

 ---

 ## 快速使用指南

 - 环境检查（GPU）：

 ```bash
 python test/GPU_test.py
 ```

 - 运行 MNIST 示例（端到端）：

 ```bash
 python test/test.py
 ```

 该脚本会将 MNIST 转为 32x32 RGB、训练一个 MobileNetV2 小模型、导出 TFLite，并把类别标签写入 `model/labels_<timestamp>.txt`。

 - 命令行训练（使用 `train.py`）

 ```bash
 # 编辑 train.py 中的 base_dir 为你的数据集路径（包含 train/ 和 val/ 子目录）
 python train.py
 ```

 `train.py` 会执行两阶段训练：阶段 1 保存 stage1_model.h5，阶段 2 微调后导出带时间戳的 `model_YYYYMMDD_HHMM.tflite`。

 - 评估已导出的 TFLite 模型：

 ```bash
 # 在 model_test.py 中修改 MODEL_PATH 和 TEST_DIR 常量，或在脚本中按需替换路径
 python model_test.py
 ```

 `model_test.py` 会加载指定的 `.tflite`，对 `TEST_DIR` 中的按类别组织图像进行评估，并输出每类准确率柱状图（保存到 `test/` 目录下）。

 ---

 ## 输出约定

 - 导出的 TFLite 文件命名示例： `model/model_YYYYMMDD_HHMM.tflite`
 - 标签文件示例（MNIST 示例）： `model/labels_YYYYMMDD_HHMM.txt`

 ---

 ## 参考链接

 - [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide)
 - [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
