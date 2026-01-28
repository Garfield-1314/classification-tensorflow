import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 从原始数据集中提取一个批次用于预览
def plot_images(dataset, title, augmentation=None, rows=3, cols=3):
    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=16)
    for images, labels in dataset.take(1):  # 取第一个批次
        for i in range(rows*cols):
            ax = plt.subplot(rows, cols, i+1)
            image = images[i].numpy()
            
            # 如果传入增强层，动态应用增强（模拟训练时的随机性）
            if augmentation is not None:
                image = augmentation(tf.expand_dims(image, axis=0))[0]
            
            # 反归一化（假设预处理已归一化到[0,1]）
            if image.dtype == tf.float32:
                image = np.clip(image * 255, 0, 255).astype("uint8")
            
            plt.imshow(image)
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()

# # 预览原始图像（未增强）
# plot_images(train_dataset_raw, "Raw Images")

# # 预览增强后的图像（注意：每次运行会生成不同的随机增强）
# plot_images(train_dataset_raw, "Augmented Images", augmentation=data_augmentation)