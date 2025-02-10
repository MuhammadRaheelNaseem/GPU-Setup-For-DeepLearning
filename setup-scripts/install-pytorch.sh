#!/bin/bash

# Installing PyTorch with GPU support
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

echo "PyTorch installation complete!"
