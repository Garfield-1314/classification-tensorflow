import os
import tensorflow as tf
from lib import AU

def prepare_datasets(base_dir, img_size=(160, 160), batch_size=16):
    """从界面参数准备数据集"""
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "val")
    
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
