# 更新日志 / Change Log

---

## [2026-05-14] 图形界面完善与评估分离 / GUI Enhancements & Evaluation Isolation

### 中文 (CN)
- **GUI 交互体验优化**：优化 PyQt5 训练界面布局，默认使用 1080p 缩放适配，修复了按钮黑色文字不可见的样式问题，统一了终端绿字输出，优化了底部进度条表现。
- **评估隔离重构**：将 GUI 测试专用的模型评估代码抽离至 `modules/model_test_utils.py`，与 CLI 核心执行代码分离，提高了项目的向后兼容性和代码健壮性。
- **配置与参数扩展**：模型配置界面新增第二阶段微调专属的学习率调整选项。
- **修复 TensorFlow 评估警告**：修复了在 TFLite 导出预热阶段数据集迭代抓取的缓存截断警告；修复了 GUI 在模型评估时因为返回值类型不匹配造成的程序崩溃；优雅释放 TFLite 解释器，消除了底层的 `Delegate ` 错误输出。

### English (EN)
- **GUI Interaction Optimizations**: Refined the PyQt5 training interface layout with 1080p scaling by default, fixed invisible black text in buttons via robust QSS, unified the terminal output font color to green, and improved progress bar stability.
- **Evaluation Isolation**: Extracted GUI-specific model test code into `modules/model_test_utils.py`, separating it from the main CLI workflows for improved backward compatibility.
- **Expanded Configurations**: Added specific learning rate input option for the Phase 2 fine-tuning within the GUI settings.
- **TF Evaluation Warnings Resolved**: Fixed Dataset pipeline caching truncation warnings during TFLite extraction, rectified tuple unpacking crashes in GUI evaluations, and cleanly garbage-collected the TFLite Interpreter to prevent missing Delegate library spam.

---

## [2026-05-09] 深度代码精简与工作流一致性优化 / Deep Code Streamlining & Workflow Consistency

### 中文 (CN)
- **遵循 Karpathy 准则**：对全库代码（`train.py`, `train.ipynb`, `test/test.py`, `lib/`）进行了深度精简，减少冗余操作，保持“外科手术式”修改。
- **文件管理简化**：移除所有冗余的中间文件移动与删除逻辑。现在所有输出（h5, TFLite, 日志, 图表）直接保存至 `train/train_<timestamp>` 文件夹。
- **内存管理优化**：在 Notebook 和脚本中移除了冗余的模型重载操作，复用内存中的模型对象，减少磁盘 I/O。
- **目录结构变更**：根目录下的模型存储文件夹由 `model/` 改为 `train/`。
- **输出名称标准化**：所有导出文件夹内的 TFLite 文件统一命名为 `model.tflite`。
- **绘图逻辑重构**：`polt_improved.py` 逻辑抽象化，通过指标映射减少重复代码。

### English (EN)
- **Adhered to Karpathy Guidelines**: Performed deep code streamlining across the entire repository (`train.py`, `train.ipynb`, `test/test.py`, `lib/`) to minimize redundancy.
- **Simplified File Management**: Removed redundant intermediate file moving and deletion logic. All outputs (h5, TFLite, logs, plots) are now saved directly to the target timestamped folder.
- **Memory Management**: Eliminated redundant model reloads in both Notebook and scripts, reusing in-memory objects to reduce disk I/O.
- **Standardized Output Naming**: All exported TFLite files are now consistently named `model.tflite` within their respective folders.
- **Plotting Logic Refactor**: Abstracted logic in `polt_improved.py` using metric mapping to reduce code duplication.

---

## [2026-05-07] 工作流优化与脚本重构 / Workflow Optimization & Script Refactor

### 中文 (CN)
- **量化工具优化**：`train.py` 和 `train.ipynb` 现在自动归档 `stage1` 和 `stage2` 的 `.h5` 模型文件到独立导出文件夹中，并自动删除临时文件。
- **鲁棒的 TFLite 评估**：`model_test.py` 现在支持自动检测并处理 `uint8` 或 `float32` 输出类型的量化模型。
- **流程精简**：重构了导出管道，确保工作区目录整洁。
- **新增脚本**：新增 `train.py` —— 由 `train.ipynb` 重构的命令行训练与导出脚本。

### English (EN)
- **Quantization Tooling**: `train.py` and `train.ipynb` now automatically archive `stage1` and `stage2` `.h5` model files to export folders and clean up temporary files.
- **Robust TFLite Evaluation**: `model_test.py` now supports automatic detection and handling of both `uint8` and `float32` output types for quantized models.
- **Workflow Streamlining**: Refactored export pipeline to ensure a clean workspace directory.
- **New Scripts**: Added `train.py` — a command-line training and export script refactored from `train.ipynb`.

---

## [2026-04-30] 模型导出与测试流程优化 / Model Export & Testing Optimization

### 中文 (CN)
- **统一导出工作流**：模型现在被导出到以时间戳命名的独立子文件夹中，方便版本管理。
- **产物自动化管理**：自动将 `labels.txt`、中间阶段 `.h5` 文件、训练曲线和混淆矩阵保存至同一模型文件夹。
- **测试工具改进**：`model_test.py` 现在支持参数化调用，并自动将测试结果保存至模型目录。
- **Notebook 增强**：`train.ipynb` 新增测试单元格，可直接调用 `model_test.py` 进行逻辑统一的验证。
- **兼容性修复**：修复了 Python 3.8 不支持 `|` 类型联合语法的错误。
- **性能与稳定性**：确保数据预热时完整遍历，消除了 TensorFlow 的缓存截断警告。

### English (EN)
- **Unified Export Workflow**: Models are now exported into dedicated sub-folders named with timestamps.
- **Artifact Management**: Automatically saves `labels.txt`, intermediate `.h5` files, training curves, and confusion matrices into the same model folder.
- **Improved Testing**: `model_test.py` now supports parameter-based calls and automatically saves results to the model's directory.
- **Notebook Enhancements**: `train.ipynb` now includes a full testing cell calling `model_test.py` directly.
- **Compatibility**: Fixed type hinting issues for Python 3.8.
- **Stability**: Fixed dataset caching warnings by ensuring full iteration during warm-up.
