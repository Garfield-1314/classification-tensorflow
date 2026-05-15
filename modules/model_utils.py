import os
import shutil
import random
import tensorflow as tf
from lib import AU

def split_dataset(base_dir, split_ratio=(0.75, 0.2, 0.05)):
    """
    如果 base_dir 下没有 train/val 文件夹，则自动按照比例划分。
    改进：不再修改原始文件夹，而是在 base_dir 下创建一个 processed_dataset 文件夹。
    返回实际使用的数据集路径。
    """
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    # 1. 检查是否已经存在 train/val 结构
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        return base_dir

    # 2. 检查是否已经存在处理过的目录
    processed_dir = os.path.join(base_dir, "processed_dataset")
    if os.path.exists(processed_dir):
        return processed_dir

    # 3. 开始划分逻辑
    # 获取所有类别（排除一些可能的干扰项）
    exclude_dirs = ["train", "val", "test", "cache", "model", "processed_dataset", "build", "modules", "qt", "lib", ".git"]
    categories = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) and d not in exclude_dirs]

    if not categories:
        return base_dir # 如果找不到类别，退回到原始目录交由 tf 处理（可能会报错，但保证逻辑不崩）

    print(f"检测到数据集未划分，正在处理至 {processed_dir} ...")
    os.makedirs(processed_dir, exist_ok=True)
    
    for cat in categories:
        cat_path = os.path.join(base_dir, cat)
        images = [f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))]
        if not images:
            continue
            
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        for i, img in enumerate(images):
            src = os.path.join(cat_path, img)
            if i < n_train:
                tag = "train"
            elif i < n_train + n_val:
                tag = "val"
            else:
                tag = "test"
            
            dst_dir = os.path.join(processed_dir, tag, cat)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, img)) # 使用 copy2 保留元数据，且不破坏原文件

    print(f"数据集备份并划分完成，存储于: {processed_dir}")
    return processed_dir

def prepare_datasets(base_dir, img_size=(160, 160), batch_size=16, split_ratio=(0.75, 0.2, 0.05)):
    """从界面参数准备数据集"""
    # 自动划分数据集，获取处理后的路径
    working_dir = split_dataset(base_dir, split_ratio=split_ratio)
    
    train_dir = os.path.join(working_dir, "train")
    valid_dir = os.path.join(working_dir, "val")
    
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, batch_size=batch_size, image_size=img_size
    )
    validation_raw = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir, batch_size=batch_size, image_size=img_size
    )
    
    class_names = train_raw.class_names
    autotune = tf.data.AUTOTUNE
    
    # 建立缓存并打乱数据
    os.makedirs("cache", exist_ok=True)
    
    train_ds = (
        train_raw.map(AU.preprocess_image, num_parallel_calls=autotune)
        .cache("cache/train_cache")
        .shuffle(1000, reshuffle_each_iteration=True)
        .prefetch(autotune)
    )
    
    val_ds = (
        validation_raw.map(AU.preprocess_image, num_parallel_calls=autotune)
        .cache("cache/val_cache")
        .prefetch(autotune)
    )
    
    # 数据集预热（Warm up）将数据推入缓存以加速后续多轮迭代
    print("正在预热数据集至底层以加速后续训练...")
    for _ in train_ds: pass
    for _ in val_ds: pass
    print("预热完毕！")
    
    # 暴露 validation_raw 以供 TFLite 导出校准时使用（避免读 cache 中断告警）
    return train_ds, val_ds, validation_raw, class_names

def build_model(num_classes, model_type="MobileNetV2", alpha=0.35, img_size=(160, 160), dropout_rate=0.8):
    """根据 GUI 配置构建模型"""
    img_shape = img_size + (3,)
    if model_type == "MobileNetV1":
        base_model = tf.keras.applications.MobileNet(
            input_shape=img_shape, include_top=False, pooling="avg", alpha=alpha, weights="imagenet"
        )
    else:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=img_shape, include_top=False, pooling="avg", alpha=alpha, weights="imagenet"
        )

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255),
        base_model,
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.build((None, *img_size, 3))
    return model
