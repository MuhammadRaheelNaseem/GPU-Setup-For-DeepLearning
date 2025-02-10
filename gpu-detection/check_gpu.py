import torch
import tensorflow as tf
import subprocess

# Function to check GPU status using nvidia-smi
def check_gpu_with_nvidia_smi():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, universal_newlines=True)
        if "NVIDIA-SMI" in output:
            print("NVIDIA GPU detected!")
            print(output)
        else:
            print("No GPU detected via nvidia-smi.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running nvidia-smi: {e.output}")

# Check PyTorch GPU status
def check_pytorch_gpu():
    if torch.cuda.is_available():
        print("PyTorch: CUDA is available! GPU detected:", torch.cuda.get_device_name(0))
    else:
        print("PyTorch: CUDA is not available. GPU not detected.")

# Check TensorFlow GPU status
def check_tensorflow_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("TensorFlow: GPU detected:", physical_devices)
    else:
        print("TensorFlow: GPU not detected.")

if __name__ == "__main__":
    print("Checking for GPU detection:")
    check_gpu_with_nvidia_smi()
    check_pytorch_gpu()
    check_tensorflow_gpu()
