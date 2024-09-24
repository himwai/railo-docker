import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np

def run_matrix_mult_test(device, size, num_iterations):
    with tf.device(device):
        total_time = 0
        for _ in range(num_iterations):
            start = time.time()
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            c = tf.matmul(a, b)
            # 強制執行計算
            _ = c.numpy()
            end = time.time()
            total_time += (end - start)
    avg_time = total_time / num_iterations
    return avg_time

def run_conv_test(device, size, num_iterations):
    with tf.device(device):
        total_time = 0
        for _ in range(num_iterations):
            start = time.time()
            input = tf.random.normal([1, size, size, 3])
            filter = tf.random.normal([3, 3, 3, 64])
            result = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
            # 強制執行計算
            _ = result.numpy()
            end = time.time()
            total_time += (end - start)
    avg_time = total_time / num_iterations
    return avg_time

def test_gpu():
    print("TensorFlow 版本:", tf.__version__)
    print("GPU 可用:", tf.config.list_physical_devices('GPU'))

    # 列出可用的 GPU 設備
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print("GPU:", gpu.name, tf.config.experimental.get_device_details(gpu))

    # 矩陣乘法測試
    sizes = [1000, 2000, 5000]
    num_iterations = 10

    for size in sizes:
        print(f"\n執行 {size}x{size} 矩陣乘法測試 ({num_iterations} 次迭代):")
        gpu_time = run_matrix_mult_test('/GPU:0', size, num_iterations)
        cpu_time = run_matrix_mult_test('/CPU:0', size, num_iterations)
        print(f"GPU 平均計算時間: {gpu_time:.4f} 秒")
        print(f"CPU 平均計算時間: {cpu_time:.4f} 秒")
        print(f"GPU 加速比: {cpu_time / gpu_time:.2f}x")

    # 卷積測試
    conv_sizes = [224, 512, 1024]
    num_iterations = 10

    for size in conv_sizes:
        print(f"\n執行 {size}x{size} 卷積測試 ({num_iterations} 次迭代):")
        gpu_time = run_conv_test('/GPU:0', size, num_iterations)
        cpu_time = run_conv_test('/CPU:0', size, num_iterations)
        print(f"GPU 平均計算時間: {gpu_time:.4f} 秒")
        print(f"CPU 平均計算時間: {cpu_time:.4f} 秒")
        print(f"GPU 加速比: {cpu_time / gpu_time:.2f}x")

    # 簡單的 GPU 操作和繪圖
    with tf.device('/GPU:0'):
        x = tf.linspace(-2, 2, 1000)
        y = tf.sin(x)

    plt.figure(figsize=(12, 6))
    plt.plot(x.numpy(), y.numpy())
    plt.title('Sin 函數圖 (1000 點)')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.savefig('sin_plot.png')
    print("\n已生成 Sin 函數圖並保存為 sin_plot.png")

if __name__ == "__main__":
    test_gpu()