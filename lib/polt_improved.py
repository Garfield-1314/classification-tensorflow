import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_combined_curves_improved(history_list):
    """
    改进的合并曲线绘制函数，动态调整 x 轴范围，并将图像保存到 train 文件夹中
    :param history_list: 包含多个 History 对象的列表
    """
    # 初始化列表，用于存储所有阶段的合并数据
    all_train_loss = []
    all_val_loss = []
    all_train_accuracy = []
    all_val_accuracy = []
    all_epochs = []

    # 当前的全局 epoch 计数器
    global_epoch = 0

    # 遍历每个阶段的历史记录，提取数据并分配全局 epoch 编号
    for history in history_list:
        epochs = range(global_epoch + 1, global_epoch + 1 + len(history.history['loss']))
        all_train_loss.extend(history.history['loss'])
        all_val_loss.extend(history.history['val_loss'])
        all_train_accuracy.extend(history.history['accuracy'])
        all_val_accuracy.extend(history.history['val_accuracy'])
        all_epochs.extend(epochs)
        global_epoch += len(history.history['loss'])

    # 创建 train 文件夹（如果不存在）
    save_dir = 'train'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取当前日期并格式化为字符串
    current_date = datetime.now().strftime('%Y-%m-%d')

    # 设置保存文件的路径和名称
    save_filename = f'{current_date}_combined_curves.png'
    save_path = os.path.join(save_dir, save_filename)

    # 绘制曲线
    plt.figure(figsize=(14, 6))

    # 绘制 loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(all_epochs, all_train_loss, label='Train Loss', color='blue')
    plt.plot(all_epochs, all_val_loss, label='Validation Loss', color='orange', linestyle='--')
    plt.title('Combined Loss Curves')
    plt.xlabel('Global Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(all_epochs, all_train_accuracy, label='Train Accuracy', color='green')
    plt.plot(all_epochs, all_val_accuracy, label='Validation Accuracy', color='red', linestyle='--')
    plt.title('Combined Accuracy Curves')
    plt.xlabel('Global Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 调整布局
    plt.tight_layout()

    # 保存图像到指定路径
    plt.savefig(save_path)

    # 显示图像
    plt.show()

    # 打印保存路径
    print(f"图像已保存到: {save_path}")