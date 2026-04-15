# Jupyter Notebook - 代码

# 导入必要的库
import sys
sys.path.insert(0, '.')
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm
import datetime
import tensorflow_model_optimization as tfmot
from lib import AU
# 设定日志级别
tf.get_logger().setLevel('ERROR')

# 🔹 超参数
IMG_SIZE = (32, 32)
AUTOTUNE = tf.data.AUTOTUNE
IMG_SHAPE = IMG_SIZE + (3,)

# 创建model目录（如果不存在）
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# 检查缓存目录是否存在并删除
folder = 'cache'
if os.path.exists(folder) and os.path.isdir(folder):
    try:
        shutil.rmtree(folder)
        print(f"成功删除目录: {folder}")
    except Exception as e:
        print(f"删除失败，错误信息: {e}")
else:
    print(f"目录 '{folder}' 不存在")

BATCH_SIZE = 32

# 🔹 数据集路径
cache_dir = os.path.join('cache')
os.makedirs(cache_dir, exist_ok=True)

# 加载 MNIST 数据集并调整为模型需要的格式
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# 将单通道灰度图转换为三通道，并缩放到 224x224
def preprocess_mnist(image, label):
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


train_dataset_raw = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).batch(BATCH_SIZE)
validation_dataset_raw = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)

class_names = [str(i) for i in range(10)]
print("Class Names:", class_names)

# 获取当前日期作为文件名 (格式: YYYYMMDD)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
labels_path = os.path.join(model_dir, f'labels_{timestamp}.txt')
with open(labels_path, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print(f"标签文件已生成: {labels_path}")

# 使用磁盘缓存，减小内存压力
train_cache_path = os.path.join(cache_dir, 'train_cache1')
val_cache_path = os.path.join(cache_dir, 'val_cache1')

train_dataset = (
    train_dataset_raw.map(preprocess_mnist,
                          num_parallel_calls=AUTOTUNE).cache()  # 内存缓存，简单起见
    .shuffle(1000, reshuffle_each_iteration=True).prefetch(AUTOTUNE))

validation_dataset = (validation_dataset_raw.map(
    preprocess_mnist, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE))

# 关键：训练前先完整遍历一次，确保cache文件写完整，避免partial cache warning
print("开始预热缓存（stage1）...")
for _ in train_dataset:
    pass
for _ in validation_dataset:
    pass
print("缓存预热完成（stage1）")

# 🔹 构建模型

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    pooling='avg',
    alpha=0.35,
    # include_preprocessing=False,
    weights='imagenet')

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255), base_model,
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])
model.build((None, 32, 32, 3))
model.summary()

# 编译模型

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.000015,
    decay_steps=len(train_dataset),
    decay_rate=0.99,
    staircase=True)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=1,  # 仅测试流程，跑一个 epoch
    callbacks=[early_stopping])

model.save(f'./{model_dir}/stage1_model.h5')


# 🔹 直接导出为TFLite格式 (无需保存H5)
def representative_dataset():
    # 单独构建校准数据管道
    calibration_dataset = (validation_dataset_raw.map(
        preprocess_mnist,
        num_parallel_calls=AUTOTUNE).take(500).cache().prefetch(AUTOTUNE))

    for images, _ in tqdm(calibration_dataset, desc="Calibration"):
        yield [tf.cast(images, tf.float32)]  # 输入需为浮点型


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # 输入为uint8 (0-255)
converter.inference_output_type = tf.uint8  # 输出为uint8类别索引

tflite_model = converter.convert()

# 保存带时间戳的TFLite模型
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
output_path = f'./model/model_{timestamp}.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite模型已保存至: {output_path}")

target_dir = model_dir
# 直接匹配当前目录下的 .h5 文件
for file in os.listdir(target_dir):
    if file.endswith(".h5"):
        file_path = os.path.join(target_dir, file)
        try:
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        except Exception as e:
            print(f"删除 {file_path} 时出错: {e}")

y_pred_probs = model.predict(validation_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([labels.numpy() for _, labels in validation_dataset])

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm,
            annot=True,
            cmap="Blues",
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 检查缓存目录是否存在并删除
folder = 'cache'
if os.path.exists(folder) and os.path.isdir(folder):
    try:
        shutil.rmtree(folder)
        print(f"成功删除目录: {folder}")
    except Exception as e:
        print(f"删除失败，错误信息: {e}")
else:
    print(f"目录 '{folder}' 不存在")
