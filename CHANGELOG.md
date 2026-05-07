# 更新日志 / Change Log

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
