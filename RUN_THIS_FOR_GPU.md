# 🚀 GPU-Setup für dein System

## Du hast CUDA 12.9 installiert! ✅

Führe jetzt diesen Befehl aus:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python cli.py gpu-info
```

## Oder nutze das automatische Script:

```bash
install_gpu.bat
```

Das war's! 🎉
