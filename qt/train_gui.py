import sys
import os
import threading

# --- 核心改进：解决打包为带界面的 EXE 模式下 sys.stdout 为 None 的问题 ---
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar, 
                             QLineEdit, QComboBox, QGridLayout, QStackedWidget, QCheckBox, 
                             QListWidget, QListWidgetItem, QScrollArea, QMessageBox, QGroupBox)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QSize
from PyQt5.QtGui import QTextCursor, QFont, QPixmap
from pathlib import Path
import random
import datetime

# 将项目根目录添加到 python 路径
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf
from pathlib import Path
from lib import AU, polt_improved
# 取消顶部直接引入全局 model_test，防止发生污染，让验证按钮自身触发 qt 的解耦副本
# import model_test

# 从根目录导入抽离的模块
from modules import model_utils, export_utils, plot_utils

# --- GUI 类定义开始 ---

class ScalableLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self._original_pixmap = None
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap):
        self._original_pixmap = pixmap
        self._update_pixmap()

    def resizeEvent(self, event):
        self._update_pixmap()
        super().resizeEvent(event)

    def _update_pixmap(self):
        if self._original_pixmap and not self._original_pixmap.isNull():
            sz = self.size()
            if sz.width() > 0 and sz.height() > 0:
                super().setPixmap(self._original_pixmap.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation))

class TrainSignals(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    export_progress_signal = pyqtSignal(int) # 导出百分比信号
    test_progress_signal = pyqtSignal(int)   # 测试百分比信号
    finished_signal = pyqtSignal(bool, str)
    test_log_signal = pyqtSignal(str)
    test_img_signal = pyqtSignal(str)
    live_img_signal = pyqtSignal(str) # 新增：实时曲线信号

class DummyHistory:
    def __init__(self):
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

class LogCallback(tf.keras.callbacks.Callback):
    def __init__(self, signals, histories, save_dir, total_epochs=1, epoch_offset=0):
        super().__init__()
        self.signals = signals
        self.histories = histories
        self.save_dir = save_dir
        self.total_epochs = total_epochs
        self.epoch_offset = epoch_offset
        self.current_epoch = 0
        self.current_history = DummyHistory()
        self.histories.append(self.current_history)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if not self.params: return
        steps = self.params.get('steps', 1)
        if steps is None: steps = 1
        
        # 避免 steps 为 0 的异常
        steps = max(steps, 1)
        
        total_steps = self.total_epochs * steps
        current_step = (self.current_epoch + self.epoch_offset) * steps + batch
        
        pct = int((current_step / total_steps) * 100)
        pct = min(max(pct, 0), 100) # 限制在 0-100 范围
        self.signals.progress_signal.emit(pct)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # 确保实时绘图能获取到最新数据
        for k in ["loss", "accuracy", "val_loss", "val_accuracy"]:
            if k in logs:
                self.current_history.history[k].append(logs[k])
            
        msg = f"Epoch {epoch + 1}: " + ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.signals.log_signal.emit(msg)
        
        # epoch结束时更新100%的情况可能需要补偿（尤其最后一个epoch的最后一个batch结束时）
        total_steps = self.total_epochs
        current_step = self.current_epoch + self.epoch_offset + 1
        pct = int((current_step / total_steps) * 100)
        self.signals.progress_signal.emit(min(pct, 100))
        
        # 改进：如果 histories 里是 history 对象，它们会自动随训练更新
        try:
            from lib import polt_improved
            # 强制清空 plt 缓存防止重叠，并生成新图
            import matplotlib.pyplot as plt
            plt.close('all') 
            valid_histories = [h for h in self.histories if h is not None]
            temp_path = polt_improved.plot_combined_curves_improved(valid_histories, save_dir=self.save_dir)
            self.signals.live_img_signal.emit(str(temp_path))
        except Exception as e:
            print(f"Live plot error: {e}")

class TrainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorFlow 图像分类全流程工具")
        self.resize(1920, 1080)
        
        self.params = {
            "base_dir": "",
            "model_type": "MobileNetV2",
            "alpha": 0.35,
            "img_size": (160, 160),
            "batch_size": 16,
            "learning_rate": 1.5e-4,
            "learning_rate2": 1.5e-5,
            "patience": 3,
            "patience2": 3,
            "dropout_rate": 0.8,
            "epochs_stage1": 10,
            "epochs_stage2": 10,
            "run_stage2": True,
            "model_path": "",
            "split_ratio": (0.75, 0.2, 0.05)
        }
        
        self.signals = TrainSignals()
        self.signals.log_signal.connect(self.append_log)
        self.signals.test_log_signal.connect(lambda t: self.test_log.append(t))
        self.signals.test_img_signal.connect(lambda p: self.test_res_display.setPixmap(QPixmap(p)))
        self.signals.live_img_signal.connect(lambda p: self.live_curve_display.setPixmap(QPixmap(p))) # 实时信号连接动态缩放
        self.signals.finished_signal.connect(self.on_train_finished)
        
        # 新增连接：进度条信号
        self.signals.progress_signal.connect(self.update_progress_bar)
        self.signals.export_progress_signal.connect(lambda v: self.export_progress.setValue(v))
        self.signals.test_progress_signal.connect(lambda v: self.test_progress.setValue(v))
        
        self.init_ui()

    def update_progress_bar(self, val):
        self.progress_bar.setValue(val)

    def init_ui(self):
        # 整体切换为明亮风格
        self.setStyleSheet("""
            /* 全局字体和颜色 */
            * {
                color: #333333;
                font-family: "Microsoft YaHei", "Segoe UI";
            }
            
            QMainWindow, QWidget#MainContent {
                background-color: #f5f6f7;
            }
            
            QLabel {
                font-size: 13px;
            }

            /* 左侧导航栏 */
            QListWidget {
                background-color: #ffffff;
                border: none;
                border-right: 1px solid #e0e0e0;
                outline: none;
                padding-top: 10px;
            }
            QListWidget::item {
                padding: 12px 20px;
                border-left: 4px solid transparent;
            }
            QListWidget::item:selected {
                background-color: #e8f2fb;
                color: #007acc;
                border-left: 4px solid #007acc;
                font-weight: bold;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }

            /* 按钮样式 */
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #dcdfe6;
                padding: 8px 16px;
                border-radius: 4px;
                color: #606266;
            }
            QPushButton:hover:enabled {
                background-color: #ecf5ff;
                border-color: #c6e2ff;
                color: #409eff;
            }
            QPushButton:pressed:enabled {
                background-color: #d9ecff;
                border-color: #b3d8ff;
            }
            QPushButton:disabled {
                background-color: #f5f7fa;
                border-color: #e4e7ed;
                color: #c0c4cc;
            }

            /* 主要操作按钮 */
            QPushButton#PrimaryBtn {
                background-color: #409eff;
                border: 1px solid #409eff;
                border-radius: 4px;
                color: #000000;
            }
            QPushButton#PrimaryBtn:hover:enabled {
                background-color: #66b1ff;
                border-color: #66b1ff;
                color: #000000;
            }
            QPushButton#PrimaryBtn:pressed:enabled {
                background-color: #3a8ee6;
                border-color: #3a8ee6;
                color: #000000;
            }
            QPushButton#PrimaryBtn:disabled {
                background-color: #f5f7fa;
                border-color: #e4e7ed;
                color: #000000; /* 纯黑色字体确保绝对可见 */
            }

            /* 输入控件 */
            QLineEdit, QComboBox, QSpinBox {
                background-color: #ffffff;
                border: 1px solid #dcdfe6;
                border-radius: 4px;
                padding: 6px;
                selection-background-color: #409eff;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #409eff;
            }

            /* 进度条 */
            QProgressBar {
                border: none;
                background-color: #ebeef5;
                border-radius: 5px;
                text-align: center;
                height: 10px;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #409eff;
                border-radius: 5px;
            }

            /* 文本框 */
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #e4e7ed;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                padding: 10px;
            }

            /* 滚动条 */
            QScrollBar:vertical {
                border: none;
                background: #f1f1f1;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c1c1c1;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a8a8a8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        main_content = QWidget()
        main_content.setObjectName("MainContent")
        self.setCentralWidget(main_content)
        main_layout = QHBoxLayout(main_content)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 左侧导航栏
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(200)
        self.nav_list.setStyleSheet("""
            QListWidget {
                background-color: #252526;
                color: #cccccc;
                border: none;
                outline: none;
                font-size: 14px;
            }
            QListWidget::item {
                height: 50px;
                padding-left: 15px;
                border-left: 3px solid transparent;
            }
            QListWidget::item:selected {
                background-color: #37373d;
                color: white;
                border-left: 3px solid #007acc;
            }
            QListWidget::item:hover {
                background-color: #2a2d2e;
            }
        """)
        
        steps = [
            "1. 数据集",
            "2. 模型配置",
            "3. 训练",
            "4. 结果与导出",
            "5. 模型测试"
        ]
        for step in steps:
            item = QListWidgetItem(step)
            item.setTextAlignment(Qt.AlignVCenter)
            self.nav_list.addItem(item)
        
        self.nav_list.currentRowChanged.connect(self.on_nav_change)
        main_layout.addWidget(self.nav_list)

        # 右侧内容区
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # 创建各页面
        self.page_data = self.create_data_page()
        self.page_config = self.create_config_page()
        self.page_train = self.create_train_page()
        self.page_combined = self.create_combined_result_page() # 合并后的页面
        self.page_test = self.create_test_page()

        self.stacked_widget.addWidget(self.page_data)
        self.stacked_widget.addWidget(self.page_config)
        self.stacked_widget.addWidget(self.page_train)
        self.stacked_widget.addWidget(self.page_combined) # 索引 3
        self.stacked_widget.addWidget(self.page_test)     # 索引 4

        self.nav_list.setCurrentRow(0)

    def on_nav_change(self, index):
        # 切换页面时，静默同步所有界面的输入参数，防止未按“下一步”导致参数未更新
        if hasattr(self, 'path_input'):
            self.params["base_dir"] = self.path_input.text()
        if hasattr(self, 'model_combo'):
            try:
                self.params.update({
                    "model_type": self.model_combo.currentText(),
                    "alpha": float(self.alpha_input.currentText()),
                    "img_size": (int(self.size_input.text()), int(self.size_input.text())),
                    "batch_size": int(self.batch_input.text()),
                    "learning_rate": float(self.lr_input.text()),
                    "learning_rate2": float(self.lr2_input.text()),
                    "patience": int(self.patience_input.text()),
                    "patience2": int(self.patience2_input.text()),
                    "dropout_rate": float(self.dropout_input.text()),
                    "epochs_stage1": int(self.epoch1_input.text()),
                    "epochs_stage2": int(self.epoch2_input.text()),
                    "run_stage2": self.stage2_cb.isChecked()
                })
            except Exception:
                pass
        self.stacked_widget.setCurrentIndex(index)

    def _on_model_changed(self, model_name):
        self.alpha_input.clear()
        if model_name == "MobileNetV1":
            self.alpha_input.addItems(["0.25", "0.50", "0.75", "1.0"])
            self.alpha_input.setCurrentText("0.25")
        else: # MobileNetV2
            self.alpha_input.addItems(["0.35", "0.50", "0.75", "1.0", "1.3", "1.4"])
            self.alpha_input.setCurrentText("0.35")

    # --- 页面创建方法 ---
    
    def create_data_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel("步骤 1: 数据集")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(label)
        
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("选择数据集根目录 (可包含 train/val 文件夹，或直接放类别文件夹)")
        btn_browse = QPushButton("浏览")
        btn_browse.clicked.connect(self.browse_folder)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(btn_browse)
        layout.addLayout(path_layout)

        # 划分比例设置
        split_group = QGroupBox("数据集自动划分配置")
        split_group.setToolTip("若根目录下没有 train/val 文件夹，将自动按照此比例进行划分。")
        split_layout = QHBoxLayout(split_group)
        
        split_layout.addWidget(QLabel("训练集:"))
        self.split_train = QLineEdit("0.75")
        self.split_train.setFixedWidth(50)
        split_layout.addWidget(self.split_train)
        
        split_layout.addWidget(QLabel("验证集:"))
        self.split_val = QLineEdit("0.2")
        self.split_val.setFixedWidth(50)
        split_layout.addWidget(self.split_val)
        
        split_layout.addWidget(QLabel("测试集:"))
        self.split_test = QLineEdit("0.05")
        self.split_test.setFixedWidth(50)
        split_layout.addWidget(self.split_test)
        
        split_desc = QLabel("(总和应为 1.0)")
        split_desc.setStyleSheet("color: #666; font-size: 11px;")
        split_layout.addWidget(split_desc)
        split_layout.addStretch()
        
        layout.addWidget(split_group)
        
        layout.addWidget(QLabel("数据集预览:"))
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(250)
        self.preview_content = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_content)
        self.scroll_area.setWidget(self.preview_content)
        layout.addWidget(self.scroll_area)
        
        self.info_label = QLabel("请先加载数据集...")
        layout.addWidget(self.info_label)
        
        btn_next = QPushButton("下一步：配置模型")
        btn_next.setFixedHeight(40)
        btn_next.clicked.connect(self.go_to_config)
        layout.addWidget(btn_next)
        return page

    def create_config_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel("步骤 2: 模型配置")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(label)
        
        # 全局与第一阶段配置（横向排列）
        group1 = QGroupBox("全局与第一阶段配置")
        group1.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 4px; padding-top: 15px; margin-top: 10px; }")
        v_group1 = QVBoxLayout(group1)
        
        h_row1 = QHBoxLayout()
        h_row1.addWidget(QLabel("模型类型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["MobileNetV2", "MobileNetV1"])
        h_row1.addWidget(self.model_combo)
        
        h_row1.addWidget(QLabel("宽度系数 (Alpha):"))
        self.alpha_input = QComboBox()
        # 默认 V2 的选项
        self.alpha_input.addItems(["0.35", "0.50", "0.75", "1.0", "1.3", "1.4"])
        self.alpha_input.setCurrentText("0.35")
        h_row1.addWidget(self.alpha_input)
        
        # 根据模型类型动态改变可选的 alpha
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        
        h_row1.addWidget(QLabel("输入尺寸:"))
        self.size_input = QLineEdit("160")
        h_row1.addWidget(self.size_input)
        
        h_row1.addWidget(QLabel("Batch Size:"))
        self.batch_input = QLineEdit("16")
        h_row1.addWidget(self.batch_input)
        v_group1.addLayout(h_row1)

        h_row2 = QHBoxLayout()
        h_row2.addWidget(QLabel("学习率 (LR):"))
        self.lr_input = QLineEdit("1.5e-4")
        h_row2.addWidget(self.lr_input)
        
        h_row2.addWidget(QLabel("Dropout 率:"))
        self.dropout_input = QLineEdit("0.8")
        h_row2.addWidget(self.dropout_input)
        
        h_row2.addWidget(QLabel("第一阶段 Epochs:"))
        self.epoch1_input = QLineEdit("10")
        h_row2.addWidget(self.epoch1_input)
        
        h_row2.addWidget(QLabel("第一阶段早停:"))
        self.patience_input = QLineEdit("3")
        h_row2.addWidget(self.patience_input)
        v_group1.addLayout(h_row2)
        
        layout.addWidget(group1)

        # 第二阶段开关
        self.stage2_cb = QCheckBox("启用第二阶段微调训练")
        self.stage2_cb.setChecked(False)  # 默认不勾选
        layout.addWidget(self.stage2_cb)

        # 第二阶段配置
        self.group2 = QGroupBox("第二阶段微调配置")
        self.group2.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ccc; border-radius: 4px; padding-top: 15px; margin-top: 5px; }")
        h_row3 = QHBoxLayout(self.group2)
        
        h_row3.addWidget(QLabel("第二阶段学习率:"))
        self.lr2_input = QLineEdit("1.5e-5")
        h_row3.addWidget(self.lr2_input)
        
        h_row3.addWidget(QLabel("第二阶段 Epochs:"))
        self.epoch2_input = QLineEdit("10")
        h_row3.addWidget(self.epoch2_input)
        
        h_row3.addWidget(QLabel("第二阶段早停:"))
        self.patience2_input = QLineEdit("3")
        h_row3.addWidget(self.patience2_input)
        
        h_row3.addStretch()
        layout.addWidget(self.group2)
        
        self.group2.setVisible(False)  # 默认隐藏

        # 勾选框事件，控制第二阶段面板显隐
        self.stage2_cb.stateChanged.connect(lambda state: self.group2.setVisible(state == Qt.Checked))

        layout.addStretch()
        
        nav = QHBoxLayout()
        btn_back = QPushButton("上一步")
        btn_back.clicked.connect(lambda: self.nav_list.setCurrentRow(0))
        btn_next = QPushButton("下一步：开始训练")
        btn_next.clicked.connect(self.go_to_train)
        nav.addWidget(btn_back)
        nav.addWidget(btn_next)
        layout.addLayout(nav)
        return page

    def create_train_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # 实时曲线展示区
        self.live_curve_display = ScalableLabel("训练开始后将实时更新曲线...")
        self.live_curve_display.setMinimumHeight(400)
        self.live_curve_display.setStyleSheet("background-color: transparent; border: none;")
        layout.addWidget(self.live_curve_display)

        self.btn_run = QPushButton("启动训练任务")
        self.btn_run.setObjectName("PrimaryBtn")
        self.btn_run.setFixedHeight(40)
        self.btn_run.clicked.connect(self.start_train_thread)
        layout.addWidget(self.btn_run)
        
        # 日志区缩小到下方
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(260)
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 14px;")
        layout.addWidget(self.log_output)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False) # 太窄不显示带数值的文字更美观
        layout.addWidget(self.progress_bar)
        
        return page

    def create_combined_result_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 上半部分：导出(左) + 混淆矩阵(右)
        top_layout = QHBoxLayout()
        
        # 导出区域 (左) - 极简极窄设计
        export_box = QWidget()
        export_box.setFixedWidth(180)
        export_box.setStyleSheet("background: #fff; border: 1px solid #ddd; border-radius: 4px;")
        export_v = QVBoxLayout(export_box)
        export_v.setContentsMargins(5, 5, 5, 5)
        
        self.btn_export = QPushButton("导出 TFLite")
        self.btn_export.setObjectName("PrimaryBtn")
        self.btn_export.setFixedHeight(30)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.start_export_task)
        export_v.addWidget(self.btn_export)
        
        self.export_info = QLabel("等待训练...")
        self.export_info.setStyleSheet("font-size: 10px; color: #666; border: none;")
        self.export_info.setWordWrap(True)
        self.export_info.setMaximumHeight(30)
        export_v.addWidget(self.export_info)
        
        self.btn_open_folder = QPushButton("打开文件夹")
        self.btn_open_folder.setFixedHeight(30)
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(lambda: os.startfile(self.export_dir_path) if hasattr(self, "export_dir_path") else None)
        export_v.addWidget(self.btn_open_folder)
        
        self.export_progress = QProgressBar()
        self.export_progress.setFixedHeight(10)
        self.export_progress.setRange(0, 100) # 百分比进度
        self.export_progress.setTextVisible(False)
        self.export_progress.setValue(0)
        export_v.addWidget(self.export_progress)
        
        export_v.addStretch()
        top_layout.addWidget(export_box)
        
        # 混淆矩阵区域 (右) - 撑满
        cm_box = QWidget()
        cm_box.setStyleSheet("background: #fff; border: 1px solid #ddd; border-radius: 4px;")
        cm_v = QVBoxLayout(cm_box)
        cm_v.setContentsMargins(2, 2, 2, 2)

        cm_top_h = QHBoxLayout()
        cm_top_h.setContentsMargins(5, 0, 5, 0)
        cm_top_h.addWidget(QLabel("混淆矩阵", styleSheet="border: none; font-weight: bold; color: #555;"))
        cm_top_h.addStretch()
        self.btn_enlarge_cm = QPushButton("🔍 放大")
        self.btn_enlarge_cm.setFixedHeight(25)
        self.btn_enlarge_cm.setStyleSheet("border: none; color: #007acc; background: transparent; font-weight: bold; padding: 0 5px;")
        self.btn_enlarge_cm.clicked.connect(self.show_large_cm)
        self.btn_enlarge_cm.setEnabled(False)
        cm_top_h.addWidget(self.btn_enlarge_cm)
        cm_v.addLayout(cm_top_h)
        
        self.cm_display = ScalableLabel()
        self.cm_display.setStyleSheet("border: none;")
        cm_v.addWidget(self.cm_display, 1)
        top_layout.addWidget(cm_box, 1) 
        
        layout.addLayout(top_layout, 2) # 上方分配更多权重
        
        # 下半部分：训练曲线 (撑满)
        curve_box = QWidget()
        curve_box.setStyleSheet("background: #fff; border: 1px solid #ddd; border-radius: 4px;")
        curve_v = QVBoxLayout(curve_box)
        curve_v.setContentsMargins(2, 2, 2, 2)
        
        self.curve_display = ScalableLabel()
        self.curve_display.setStyleSheet("border: none;")
        curve_v.addWidget(self.curve_display)
        layout.addWidget(curve_box, 3) # 下方分配最多权重
        
        return page
        
        self.btn_open = QPushButton("打开导出目录")
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(lambda: os.startfile(self.export_dir_path) if hasattr(self, "export_dir_path") else None)
        layout.addWidget(self.btn_open)
        layout.addStretch()
        return page

    def create_test_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel("步骤 5: 模型测试")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(label)
        
        h = QHBoxLayout()
        self.test_path_input = QLineEdit()
        self.test_path_input.setPlaceholderText("默认使用数据集 test 目录")
        btn = QPushButton("选择目录")
        btn.clicked.connect(self.browse_test_folder)
        h.addWidget(self.test_path_input)
        h.addWidget(btn)
        layout.addLayout(h)
        
        # 重点展示图片：加大高度
        self.test_res_display = ScalableLabel("测试结果图表将在此展示")
        self.test_res_display.setMinimumHeight(450) # 加大核心展示区
        self.test_res_display.setStyleSheet("border: 1px solid #444; background: #1a1a1a;")
        layout.addWidget(self.test_res_display)

        # 终端输出改为小窗，仅供参考详情
        self.test_log = QTextEdit()
        self.test_log.setReadOnly(True)
        self.test_log.setMaximumHeight(200) # 限制高度，突出图表
        self.test_log.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 14px;")
        layout.addWidget(self.test_log)
        
        self.btn_run_test = QPushButton("运行测试集评估")
        self.btn_run_test.setObjectName("PrimaryBtn")
        self.btn_run_test.setFixedHeight(45)
        self.btn_run_test.clicked.connect(self.run_verification)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_run_test)
        
        self.test_progress = QProgressBar()
        self.test_progress.setFixedHeight(12)
        self.test_progress.setRange(0, 100) # 百分比进度
        self.test_progress.setTextVisible(False)
        self.test_progress.setValue(0)
        btn_layout.addWidget(self.test_progress)
        
        layout.addLayout(btn_layout)
        return page

    # --- 逻辑控制 ---

    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择数据集")
        if d:
            self.path_input.setText(d)
            self.update_preview(d)

    def browse_test_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择测试目录")
        if d: self.test_path_input.setText(d)

    def update_preview(self, base_dir):
        for i in reversed(range(self.preview_layout.count())): 
            self.preview_layout.itemAt(i).widget().setParent(None)
        
        # 改进：如果不存在 train 目录，尝试直接从根目录读取类别
        p = Path(base_dir) / "train"
        if not p.exists():
            # 同时也检查是否已经有我们处理过的目录
            processed_p = Path(base_dir) / "processed_dataset" / "train"
            if processed_p.exists():
                p = processed_p
            else:
                p = Path(base_dir)
            
        exclude_dirs = ["train", "val", "test", "cache", "model", "build", "modules", "qt", "test", "lib", ".git", ".github", "processed_dataset"]
        cats = [d for d in p.iterdir() if d.is_dir() and d.name not in exclude_dirs]
        if not cats: return
        
        self.info_label.setText(f"类别: {', '.join([c.name for c in cats])}")
        all_imgs = []
        for c in cats:
            for f in c.glob("*"):
                if f.suffix.lower() in [".jpg", ".png", ".jpeg"]: all_imgs.append((f, c.name))
        
        if not all_imgs: return
        samples = random.sample(all_imgs, min(len(all_imgs), 8))
        for img_p, name in samples:
            w = QWidget(); v = QVBoxLayout(w)
            l = QLabel(); pix = QPixmap(str(img_p))
            l.setPixmap(pix.scaled(150, 150, Qt.KeepAspectRatio))
            v.addWidget(l); v.addWidget(QLabel(name))
            self.preview_layout.addWidget(w)

    def go_to_config(self):
        if not os.path.exists(self.path_input.text()): return
        try:
            r1 = float(self.split_train.text())
            r2 = float(self.split_val.text())
            r3 = float(self.split_test.text())
            if abs(r1 + r2 + r3 - 1.0) > 1e-4:
                QMessageBox.warning(self, "参数错误", "划分比例总和必须为 1.0")
                return
            self.params["split_ratio"] = (r1, r2, r3)
        except ValueError:
            QMessageBox.warning(self, "参数错误", "请输入有效的数字比例")
            return

        self.params["base_dir"] = self.path_input.text()
        self.nav_list.setCurrentRow(1)

    def go_to_train(self):
        try:
            self.params.update({
                "model_type": self.model_combo.currentText(),
                "alpha": float(self.alpha_input.currentText()),
                "img_size": (int(self.size_input.text()), int(self.size_input.text())),
                "batch_size": int(self.batch_input.text()),
                "learning_rate": float(self.lr_input.text()),
                "learning_rate2": float(self.lr2_input.text()),
                "patience": int(self.patience_input.text()),
                "patience2": int(self.patience2_input.text()),
                "dropout_rate": float(self.dropout_input.text()),
                "epochs_stage1": int(self.epoch1_input.text()),
                "epochs_stage2": int(self.epoch2_input.text()),
                "run_stage2": self.stage2_cb.isChecked()
            })
            self.nav_list.setCurrentRow(2)
        except: pass

    def append_log(self, text):
        self.log_output.append(text)
        self.log_output.moveCursor(QTextCursor.End)

    def start_train_thread(self):
        self.btn_run.setEnabled(False)
        threading.Thread(target=self.run_full_pipeline, daemon=True).start()

    def run_full_pipeline(self):
        try:
            p = self.params
            self.signals.log_signal.emit("="*50)
            self.signals.log_signal.emit("开始初始化训练环境...")
            
            # 打印详细设备信息
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.signals.log_signal.emit(f"训练设备: GPU (检测到 {len(gpus)} 个)")
                try:
                    # 使用 subprocess 调用 nvidia-smi 获取更详细的信息
                    import subprocess
                    cmd = "nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits"
                    res = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')
                    for i, line in enumerate(res):
                        name, mem, cap, driver = line.split(', ')
                        self.signals.log_signal.emit(f" -> GPU {i}: {name}")
                        self.signals.log_signal.emit(f"    显存总量: {mem} MB")
                        self.signals.log_signal.emit(f"    硬件算力: {cap}")
                        self.signals.log_signal.emit(f"    驱动版本: {driver}")
                    
                except Exception:
                    self.signals.log_signal.emit(f"设备详情: {[gpu.name for gpu in gpus]}")
                    self.signals.log_signal.emit("(无法获取显存详情，请确保安装了 NVIDIA 驱动)")
            else:
                self.signals.log_signal.emit("训练设备: CPU (未检测到兼容的 NVIDIA GPU)")

            # 打印系统信息
            import platform
            self.signals.log_signal.emit(f"TensorFlow 版本: {tf.__version__}")
            self.signals.log_signal.emit(f"Python 版本: {platform.python_version()}")
            self.signals.log_signal.emit("-" * 30)

            self.signals.log_signal.emit("加载数据中...")
            self.signals.log_signal.emit("正在预热数据集图像至缓存，请耐心等待...")
            self.train_ds, self.val_ds, self.validation_raw, self.class_names = model_utils.prepare_datasets(
                p["base_dir"], 
                img_size=p["img_size"], 
                batch_size=p["batch_size"],
                split_ratio=p["split_ratio"]
            )
            self.model = model_utils.build_model(len(self.class_names), model_type=p["model_type"], alpha=p["alpha"], img_size=p["img_size"], dropout_rate=p["dropout_rate"])
            self.signals.log_signal.emit("数据预热完毕，缓存构建成功！")
            
            # 准备保存目录
            export_dir = os.path.join("model", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(export_dir, exist_ok=True)
            self.export_dir_path = export_dir

            # 计算总 Epoch 进度预期
            total_epochs = p["epochs_stage1"] + (p["epochs_stage2"] if p["run_stage2"] else 0)

            # Phase 1
            self.signals.log_signal.emit("Phase 1: 开始训练...")
            self.signals.progress_signal.emit(0) # 初始化进度条
            self.model.compile(optimizer=tf.keras.optimizers.Adam(p["learning_rate"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            cb1 = LogCallback(self.signals, [], self.export_dir_path, total_epochs=total_epochs, epoch_offset=0)
            es1 = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=p["patience"], restore_best_weights=True)
            h1 = self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=p["epochs_stage1"], 
                                callbacks=[cb1, es1])
            
            # Phase 2
            if p["run_stage2"]:
                self.signals.log_signal.emit("Phase 2: 微调训练...")
                self.model.trainable = True
                self.model.compile(optimizer=tf.keras.optimizers.Adam(p["learning_rate2"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                
                # 如果第一阶段触发了 Early stopping, h1.epoch可能很短，为了进度条连续，我们记录偏移量
                actual_epoch1_done = len(h1.epoch)
                
                cb2 = LogCallback(self.signals, [h1], self.export_dir_path, total_epochs=total_epochs, epoch_offset=actual_epoch1_done)
                es2 = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=p["patience2"], restore_best_weights=True)
                h2 = self.model.fit(self.train_ds, validation_data=self.val_ds, epochs=p["epochs_stage2"], 
                                    callbacks=[cb2, es2])
            
            # 训练结束，进度条拉满并保存最终混淆矩阵
            self.signals.progress_signal.emit(100)
            cm_path = os.path.join(self.export_dir_path, "confusion_matrix.png")
            plot_utils.evaluate_cm(self.model, self.val_ds, self.class_names, save_path=cm_path)

            # 清理训练时生成的缓存目录文件避免堆积
            import shutil
            if os.path.exists("cache"):
                try:
                    shutil.rmtree("cache")
                except Exception as e:
                    self.signals.log_signal.emit(f"清除缓存失败: {e}")

            self.signals.finished_signal.emit(True, "TRAIN_SUCCESS")
        except Exception as e:
            self.signals.log_signal.emit(f"错误: {str(e)}")
            self.signals.finished_signal.emit(False, f"TRAIN_ERROR: {str(e)}")

    def start_export_task(self):
        self.btn_export.setEnabled(False)
        self.export_info.setText("正在导出...") # 缩短文字
        self.export_progress.setValue(0)
        threading.Thread(target=self.run_export_logic, daemon=True).start()

    def run_export_logic(self):
        try:
            out_path = export_utils.export_tflite(
                self.model, 
                self.validation_raw, 
                self.class_names, 
                self.export_dir_path,
                progress_cb=lambda pct: self.signals.export_progress_signal.emit(pct)
            )
            self.params["model_path"] = out_path
            self.signals.export_progress_signal.emit(100) # 完成
            self.signals.log_signal.emit(f"模型已成功导出至: {out_path}")
            self.signals.finished_signal.emit(True, "EXPORT_DONE")
        except Exception as e:
            self.signals.log_signal.emit(f"导出失败: {str(e)}")

    def on_train_finished(self, success, msg):
        if msg.startswith("TRAIN_"):
            self.btn_run.setEnabled(True)
            if success:
                self.export_info.setText("训练已完成，请导出")
                self.btn_export.setEnabled(True)
                self.btn_open_folder.setEnabled(True)
                
                # 展示可视化图表：去掉固定缩放，改用动态缩放以填满标签
                c = os.path.join(self.export_dir_path, "training_curves.png")
                m = os.path.join(self.export_dir_path, "confusion_matrix.png")
                if os.path.exists(c): 
                    self.curve_display.setPixmap(QPixmap(c))
                if os.path.exists(m): 
                    self.cm_display.setPixmap(QPixmap(m))
                    self.btn_enlarge_cm.setEnabled(True)
                
                # 设置默认测试路径
                t = os.path.join(self.params["base_dir"], "test")
                if os.path.exists(t): self.test_path_input.setText(t)
                
                QMessageBox.information(self, "训练完成", "模型训练已完成！\n请前往“结果与导出”页面查看混淆矩阵并导出模型。")
            else:
                QMessageBox.critical(self, "训练失败", f"训练过程中出现错误:\n{msg}")

        elif msg == "EXPORT_DONE":
            self.btn_export.setEnabled(True)
            if success:
                self.export_progress.setValue(100)
                self.export_info.setText("导出完成！")
                QMessageBox.information(self, "导出成功", "TFLite 模型与相关文件导出成功！\n您可以点击“打开文件夹”查看详细结果。")
            else:
                self.export_progress.setValue(0)
                self.export_info.setText("导出失败...")
                QMessageBox.critical(self, "导出失败", "导出模型时遇到错误。")
                
        elif msg.startswith("TEST_"):
            self.btn_run_test.setEnabled(True)
            if success:
                self.test_progress.setValue(100)
            else:
                self.test_progress.setValue(0)

    def run_verification(self):
        if not self.params.get("model_path"):
            self.test_log.append("错误: 请先在步骤 6 中导出模型 (.tflite)")
            return
        d = self.test_path_input.text() or os.path.join(self.params["base_dir"], "test")
        if not os.path.exists(d):
            self.test_log.append(f"错误: 目录不存在 - {d}")
            return
        
        self.test_log.append(f"--- 开始测试集评估 ---")
        self.test_log.append(f"测试目录: {d}")
        self.test_log.append(f"使用模型: {self.params['model_path']}")
        
        self.btn_run_test.setEnabled(False)
        self.test_progress.setValue(0)
        
        def test_worker():
            try:
                from modules import model_test_utils as model_test
                # 捕获打印内容
                from io import StringIO
                import sys as sys_orig
                
                old_stdout = sys_orig.stdout
                sys_orig.stdout = mystdout = StringIO()
                
                # 调用 model_test 的 main，注入进展回调
                # GUI 版本的 main 返回的是 5 个元素的元组，由于提取了 chart_path，我们需要进行解包取首个
                chart_path, *_ = model_test.main(
                    model_path=self.params["model_path"], 
                    test_dir=d,
                    progress_cb=lambda pct: self.signals.test_progress_signal.emit(pct)
                )
                
                sys_orig.stdout = old_stdout
                self.signals.test_log_signal.emit(mystdout.getvalue())
                if chart_path and os.path.exists(chart_path):
                    self.signals.test_img_signal.emit(chart_path)
                self.signals.test_log_signal.emit("--- 测试完成 ---")
                self.signals.finished_signal.emit(True, "TEST_DONE")
            except Exception as e:
                import sys as sys_orig
                if hasattr(sys_orig, "stdout_orig"):
                    sys_orig.stdout = old_stdout
                self.signals.test_log_signal.emit(f"评估失败: {str(e)}")
                self.signals.finished_signal.emit(False, "TEST_ERROR")
        
        threading.Thread(target=test_worker, daemon=True).start()

    def show_large_cm(self):
        m = os.path.join(self.export_dir_path, "confusion_matrix.png")
        if os.path.exists(m):
            from PyQt5.QtWidgets import QDialog
            dlg = QDialog(self)
            dlg.setWindowTitle("混淆矩阵 - 放大视图")
            dlg.resize(800, 650)
            l = QVBoxLayout(dlg)
            lbl = ScalableLabel()
            lbl.setPixmap(QPixmap(m))
            l.addWidget(lbl)
            dlg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv); w = TrainWindow(); w.show(); sys.exit(app.exec_())