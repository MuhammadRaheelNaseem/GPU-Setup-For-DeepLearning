import tensorflow as tf

# Check if TensorFlow is using the GPU
if tf.test.is_gpu_available():
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is using CPU.")
