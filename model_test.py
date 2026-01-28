import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
# 加载量化后的TFLite模型
model_path = 'your_model.tflite'  # 替换为你的模型路径
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
if 'GPU' in tf.config.list_physical_devices():
    delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
    interpreter.modify_graph_with_delegate(delegate)

# 获取输入输出详细信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 检查模型输入类型和尺寸
expected_height = input_details[0]['shape'][1]
expected_width = input_details[0]['shape'][2]
input_dtype = input_details[0]['dtype']
print(f"模型输入尺寸: {expected_height}x{expected_width}, 数据类型: {input_dtype}")

# 数据集路径
test_dir = os.path.join('yourdataset', 'test')  # 确保test目录存在

# 超参数设置
BATCH_SIZE = 1  # 适当调大batch size提升推理速度
IMG_SIZE = (expected_height, expected_width)  # 使用模型期望的尺寸
print(IMG_SIZE)
# 加载完整测试集（无需split）
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False  # 无需打乱顺序
)
class_names = test_dataset.class_names
print(class_names)
num_classes = len(class_names)
def preprocess_image(image):
    """根据模型需求预处理图像"""
    # 量化模型通常需要uint8输入，若训练时已归一化则无需额外处理
    if input_dtype == np.uint8:
        return tf.cast(image, tf.uint8)
    else:
        # 若模型需要float输入，进行归一化（示例为除以255）
        return tf.cast(image, tf.float32) / 255.0

def predict_batch(images):
    """批量推理提升效率"""
    # 预处理整个batch
    processed_images = []
    for img in images:
        processed_images.append(preprocess_image(img))
    input_data = np.array(processed_images, dtype=input_dtype)
    
    # 推理
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# 初始化统计变量
correct_predictions = np.zeros(num_classes)
total_predictions = np.zeros(num_classes)
# 遍历测试集
for images, labels in tqdm(test_dataset, desc="测试进度"):
    batch_preds = predict_batch(images)
    predicted_labels = np.argmax(batch_preds, axis=1)
    
    # 更新统计
    for true_label, pred_label in zip(labels.numpy(), predicted_labels):
        total_predictions[true_label] += 1
        if true_label == pred_label:
            correct_predictions[true_label] += 1

# 计算准确率（处理除零情况）
class_accuracies = np.zeros_like(correct_predictions, dtype=np.float32)
for i in range(num_classes):
    if total_predictions[i] > 0:
        class_accuracies[i] = correct_predictions[i] / total_predictions[i]
    else:
        class_accuracies[i] = 0.0
class_accuracies_percentage = class_accuracies * 100 

# 绘制柱状图
plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, class_accuracies_percentage, color='skyblue')

# 添加数值标签
for bar, acc in zip(bars, class_accuracies_percentage):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1f}%',
             ha='center', va='bottom')

plt.xlabel('class')
plt.ylabel('acc (%)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 获取模型名称（去掉路径和扩展名）
model_name = os.path.splitext(os.path.basename(model_path))[0]
save_dir = 'test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 定义保存路径
save_path = os.path.join("./test", f"{model_name}_accuracy_bar_chart.png")

# 保存柱状图到 test 文件夹
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

# 打印保存路径
print(f"柱状图已保存到: {save_path}")



print("\n各类别准确率：")
for i in range(num_classes):
    class_name = class_names[i].ljust(15)  # 对齐类名
    acc = class_accuracies_percentage[i]
    samples = int(total_predictions[i])
    print(f"  {class_name}: {acc:.2f}%  ({correct_predictions[i]}/{samples})")

# 计算整体准确率
total_correct = np.sum(correct_predictions)
total_samples = np.sum(total_predictions)
overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0

# 打印整体准确率（带醒目格式）
print("\n\033[1;36m" + "-" * 50)
print(f" 整体测试准确率: {overall_accuracy:.2f}%  ({total_correct}/{total_samples})")
print("-" * 50 + "\033[0m")