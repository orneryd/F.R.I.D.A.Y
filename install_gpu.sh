#!/bin/bash

echo "========================================================================"
echo "F.R.I.D.A.Y - GPU Installation (CUDA 12.1)"
echo "========================================================================"
echo ""

echo "Checking CUDA Toolkit..."
if ! command -v nvcc &> /dev/null; then
    echo ""
    echo "[ERROR] CUDA Toolkit not found!"
    echo "Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

nvcc --version

echo ""
echo "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "[ERROR] NVIDIA GPU or driver not found!"
    echo "Please install NVIDIA GPU driver"
    exit 1
fi

nvidia-smi

echo ""
echo "========================================================================"
echo "Installing PyTorch with CUDA 12.1 support..."
echo "========================================================================"
echo ""

echo "Uninstalling old PyTorch versions..."
pip3 uninstall -y torch torchvision torchaudio

echo ""
echo "Installing PyTorch with CUDA 12.1..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] PyTorch installation failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Installing other dependencies..."
echo "========================================================================"
echo ""

pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Dependencies installation failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Verifying GPU installation..."
echo "========================================================================"
echo ""

python3 cli.py gpu-info

echo ""
echo "========================================================================"
echo "Installation complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Check GPU info: python3 cli.py gpu-info --test"
echo "  2. Start training: python3 cli.py reddit dataset.json"
echo ""
