#!/bin/bash

# Installing TensorFlow with GPU support
echo "Installing TensorFlow with GPU support..."
pip install tensorflow-gpu

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"

echo "TensorFlow installation complete!"
