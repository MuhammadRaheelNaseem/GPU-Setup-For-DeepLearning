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
