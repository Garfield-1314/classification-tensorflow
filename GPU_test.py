import tensorflow as tf

def check_gpu_availability():
    print("="*50)
    print(f"TensorFlow 版本: {tf.__version__}")
    
    gpu_available = tf.config.list_physical_devices('GPU')
    print("\nGPU 可用状态:", "可用 ✅" if gpu_available else "不可用 ❌")
    
    if gpu_available:
        print("\nGPU 设备信息:")
        for i, gpu in enumerate(gpu_available):
            details = tf.config.experimental.get_device_details(gpu)
            print(f"[GPU {i}]")
            print(f"  设备名称: {gpu.name}")
            print(f"  设备类型: {gpu.device_type}")
            print(f"  计算架构: {details.get('compute_capability', '未知')}")
            
            # 兼容性处理：检查是否存在memory_limit属性
            if hasattr(gpu, 'memory_limit'):
                print(f"  显存大小: {gpu.memory_limit // 1024 // 1024} MB")
            else:
                # 旧版本获取显存的方法
                from tensorflow.python.client import device_lib
                local_devices = device_lib.list_local_devices()
                gpu_info = [x for x in local_devices if x.device_type == 'GPU'][i]
                print(f"  显存大小: {int(gpu_info.memory_limit // 1024 // 1024)} MB (兼容模式)")
            
            print(f"  TF设备类型: {details.get('device_name', '未知')}")
    else:
        print("\n提示：请检查CUDA/cuDNN是否安装正确，并确认安装了GPU版TensorFlow")

    print("\n所有可见设备:")
    devices = tf.config.list_physical_devices()
    for device in devices:
        print(f"- {device.name} ({device.device_type})")      # 打印设备

if __name__ == "__main__":
    check_gpu_availability()
    print("\n" + "="*50)
