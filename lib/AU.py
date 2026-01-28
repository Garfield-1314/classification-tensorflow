import tensorflow as tf

# 🔹 数据增强
data_augmentation = tf.keras.Sequential([
    # tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(factor=(-0.1,0.1),fill_mode="nearest"),
    tf.keras.layers.RandomZoom(0.10,fill_mode="nearest"),
    tf.keras.layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
    tf.keras.layers.RandomBrightness(0.10),
    tf.keras.layers.RandomContrast(0.10)
])


# 预处理函数（添加增强）
def preprocess_image(image, label):
    return image, label

# 预处理函数（添加增强）
def preprocess_image_aug(image, label):
    image = data_augmentation(image)
    return image, label