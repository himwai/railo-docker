import tensorflow as tf
import time

# Check if TensorFlow is using the GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

if physical_devices:
    print("GPU is available")
else:
    print("GPU is not available")

# Check GPU memory usage
memory_info = tf.config.experimental.get_memory_info('GPU:0')
print("GPU memory info:", memory_info)


# Perform a simple tensor operation to ensure GPU is being used
# a = tf.constant([1, 2, 3, 4])
# b = tf.constant([5, 6, 7, 8])
# c = tf.add(a, b)

# print("Result of tensor addition:", c.numpy())

# Perform a simple GPU computation
try:
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result:")
    print(c.numpy())

    # Performance test
    print("Running performance test...")
    start = time.time()
    for _ in range(1000):
        tf.tidy(lambda: tf.matmul(tf.random.normal([100, 100]), tf.random.normal([100, 100])))
    print(f"Time taken for 1000 matrix multiplications: {time.time() - start} seconds")

except Exception as e:
    print("Error performing GPU operation:", e)