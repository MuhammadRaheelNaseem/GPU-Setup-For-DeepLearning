### **Repository Structure**

```
GPU-Setup-For-DeepLearning/
│
├── README.md
├── setup-scripts/
│   ├── install-cuda.sh
│   ├── install-pytorch.sh
│   ├── install-tensorflow.sh
│
├── gpu-detection/
│   ├── check_gpu.py
│
├── troubleshooting/
│   ├── troubleshooting-guide.md
│
└── examples/
    ├── pytorch_gpu_test.py
    ├── tensorflow_gpu_test.py
```

### **1. `README.md`**

```markdown
# GPU Setup for Deep Learning

This repository provides comprehensive instructions, scripts, and troubleshooting tips for setting up and forcing GPU usage in deep learning frameworks such as **TensorFlow** and **PyTorch**. It will guide you through ensuring that your system's NVIDIA GPU is properly installed and utilized for deep learning tasks.

## Repository Structure

- **setup-scripts/**: Contains scripts for installing **CUDA**, **PyTorch**, and **TensorFlow** with GPU support.
- **gpu-detection/**: A Python script (`check_gpu.py`) to check if your system detects the GPU and supports CUDA.
- **troubleshooting/**: A guide to resolve common issues like GPU not being detected.
- **examples/**: Basic example scripts to force the use of GPU in TensorFlow and PyTorch.

---

## Installation Instructions

Follow the instructions below to ensure that you have the correct setup.

### 1. Clone this Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/muhammadraheelnaseem/GPU-Setup-For-DeepLearning.git
cd GPU-Setup-For-DeepLearning
```

### 2. Install CUDA and Drivers
Run the appropriate setup script to install **CUDA**:

```bash
bash setup-scripts/install-cuda.sh
```

This script will install **CUDA Toolkit** and the **NVIDIA Drivers** necessary for TensorFlow and PyTorch to use the GPU.

### 3. Install PyTorch with GPU support
To install **PyTorch** with GPU support:

```bash
bash setup-scripts/install-pytorch.sh
```

### 4. Install TensorFlow with GPU support
To install **TensorFlow** with GPU support:

```bash
bash setup-scripts/install-tensorflow.sh
```

### 5. Verify GPU Usage
Run the script to check if your GPU is being detected by both **TensorFlow** and **PyTorch**:

```bash
python gpu-detection/check_gpu.py
```

---

## Troubleshooting

If your GPU is not detected, visit the **`troubleshooting/`** directory for detailed troubleshooting steps:

- **Windows Users**: Ensure your CUDA path is correctly set in the environment variables.
- **Linux Users**: Ensure that the NVIDIA driver and CUDA are installed and configured correctly.

You can also consult the [official TensorFlow GPU installation guide](https://www.tensorflow.org/install/gpu) or [PyTorch GPU installation guide](https://pytorch.org/get-started/locally/) for further help.

---

## Example Scripts

To force the use of GPU in your deep learning models, check out the examples:

- [PyTorch GPU Example](examples/pytorch_gpu_test.py)
- [TensorFlow GPU Example](examples/tensorflow_gpu_test.py)

```

---

### **2. `setup-scripts/install-cuda.sh`**

```bash
#!/bin/bash

# Installing NVIDIA Drivers
echo "Installing NVIDIA Drivers..."
sudo apt update
sudo apt install -y nvidia-driver-460  # Modify based on GPU

# Installing CUDA Toolkit
echo "Installing CUDA Toolkit..."
sudo apt install -y nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version
nvidia-smi

echo "CUDA installation complete!"
```

This script installs the **NVIDIA drivers** and **CUDA toolkit** for the system. It verifies the installation by running `nvcc --version` and `nvidia-smi`.

---

### **3. `setup-scripts/install-pytorch.sh`**

```bash
#!/bin/bash

# Installing PyTorch with GPU support
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

echo "PyTorch installation complete!"
```

This script installs **PyTorch** with GPU support. It also verifies whether PyTorch detects CUDA and the GPU after installation.

---

### **4. `setup-scripts/install-tensorflow.sh`**

```bash
#!/bin/bash

# Installing TensorFlow with GPU support
echo "Installing TensorFlow with GPU support..."
pip install tensorflow-gpu

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"

echo "TensorFlow installation complete!"
```

This script installs **TensorFlow** with GPU support and verifies if TensorFlow detects the GPU after installation.

---

### **5. `gpu-detection/check_gpu.py`**

```python
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
```

This Python script checks if **NVIDIA GPU** is available using `nvidia-smi`, and then verifies if **PyTorch** and **TensorFlow** can detect the GPU.

---

### **6. `troubleshooting/troubleshooting-guide.md`**

```markdown
# Troubleshooting Guide for GPU Detection Issues

## 1. Common Issues

### Issue: GPU not detected by PyTorch/TensorFlow
- Ensure that your **NVIDIA drivers** and **CUDA** are installed properly.
- Check if your system recognizes the GPU with the command:
  ```bash
  nvidia-smi
  ```

### Issue: TensorFlow or PyTorch fails to use GPU
- Set the correct environment variables, such as `CUDA_VISIBLE_DEVICES`.
- If using Windows, ensure that your **CUDA installation path** is included in the **system's PATH**.

## 2. Steps for Reinstallation

### Reinstall NVIDIA Drivers:
- Download the latest driver from [NVIDIA's official website](https://www.nvidia.com/Download/index.aspx).
- Follow the instructions to uninstall and reinstall the drivers.

### Reinstall CUDA:
- Visit the [CUDA installation page](https://developer.nvidia.com/cuda-toolkit).
- Follow the installation steps based on your distribution (Ubuntu, CentOS, etc.).

### Reinstall PyTorch and TensorFlow:
If you're still facing issues, try reinstalling **PyTorch** and **TensorFlow** with the correct versions compatible with your CUDA installation.

## 3. Helpful Links

- [TensorFlow GPU Installation](https://www.tensorflow.org/install/gpu)
- [PyTorch GPU Installation](https://pytorch.org/get-started/locally/)
```

This markdown file contains common troubleshooting issues related to GPU detection and solutions. It also provides links to official installation guides for TensorFlow and PyTorch.

---

### **7. `examples/pytorch_gpu_test.py`**

```python
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")
```

This script checks if **PyTorch** is utilizing the GPU. If available, it prints the GPU's name; otherwise, it will fall back to the CPU.

---

### **8. `examples/tensorflow_gpu_test.py`**

```python
import tensorflow as tf

# Check if TensorFlow is using the GPU
if tf.test.is_gpu_available():
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is using CPU.")
```

This script checks if **TensorFlow** is using the GPU and prints the status accordingly.
