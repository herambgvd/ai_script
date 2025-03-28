#!/bin/bash

set -e

echo "🐍 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing base packages from static_requirements.txt..."
pip install -r static_requirements.txt

echo "🧠 Detecting CUDA support for PyTorch..."
PYTORCH_CMD=""

if command -v nvidia-smi &> /dev/null; then
    echo "🚀 CUDA GPU detected. Installing PyTorch with GPU support..."
    PYTORCH_CMD="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
else
    echo "⚙️ No CUDA GPU detected. Installing CPU version of PyTorch..."
    PYTORCH_CMD="pip install torch torchvision torchaudio"
fi

eval $PYTORCH_CMD

echo "🧹 Uninstalling OpenCV-related packages (if any)..."
pip uninstall -y opencv-python opencv-python-headless || true

echo "🧊 Freezing final environment to prod_requirements.txt..."
pip freeze > prod_requirements.txt

echo "✅ Installation complete. Requirements saved to prod_requirements.txt"
