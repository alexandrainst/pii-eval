#!/usr/bin/env bash
set -e

uv venv --allow-existing

# Check for Intel GPU device (via /dev/dri or lspci)
if lspci | grep -qi "Intel Corporation.*Graphics"; then
    echo "🟢 Intel GPU detected — installing PyTorch with XPU support"
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
else
    echo "⚪️ No Intel GPU found — installing regular PyTorch"
    uv pip install torch torchvision torchaudio
fi
