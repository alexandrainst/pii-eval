#!/usr/bin/env bash
set -e

uv venv --allow-existing

# Check for NVIDIA GPU first
if command -v nvidia-smi &> /dev/null || lspci | grep -qi "NVIDIA"; then
    echo "🟢 NVIDIA GPU detected — installing PyTorch with CUDA support"
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Check for Intel GPU next
elif lspci | grep -qi "Intel Corporation.*Graphics"; then
    echo "🟢 Intel GPU detected — installing PyTorch with XPU support"
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
# Default: CPU-only
else
    echo "⚪️ No GPU found — installing CPU-only PyTorch"
    uv pip install torch torchvision torchaudio
fi
