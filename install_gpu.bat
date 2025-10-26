@echo off
echo ========================================
echo F.R.I.D.A.Y GPU Installation (CUDA 12.1)
echo ========================================
echo.

echo Step 1: Uninstalling old PyTorch...
pip uninstall -y torch torchvision torchaudio
echo.

echo Step 2: Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

echo Step 3: Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo.

echo Step 4: Running GPU info...
python cli.py gpu-info
echo.

echo ========================================
echo Installation complete!
echo ========================================
pause
