import os
import tensorflow as tf
from lib import AU

def export_tflite(model, validation_raw, class_names, out_dir, progress_cb=None):
    """导出逻辑解耦"""
    def representative_dataset():
        autotune = tf.data.AUTOTUNE
        calibration_ds = (validation_raw.map(AU.preprocess_image, num_parallel_calls=autotune)
                          .take(500).cache().prefetch(autotune))
        
        # 为了获得进度，我们知道这里 take(500) 设定了最大步数
        # 如果 raw validation_raw 不足 500，长度就是其自带长度
        total_steps = len(validation_raw)
        total_steps = min(total_steps, 500) if total_steps > 0 else 500
        
        if progress_cb: progress_cb(0)
        for i, (images, _) in enumerate(calibration_ds):
            if progress_cb:
                # 留出10% 给 convert 本身执行
                pct = int((i / total_steps) * 90)
                progress_cb(min(pct, 90))
            yield [tf.cast(images, tf.float32)]
        if progress_cb: progress_cb(90)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    out_path = os.path.join(out_dir, "model.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    return out_path
