#!/bin/bash
# LawFirmAI Production Startup Script

echo "Starting LawFirmAI Production Application..."

# ȯ�� ���� ����
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="7860"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

# �α� ���丮 ����
mkdir -p logs

# ���δ��� �� ����
echo "Launching production interface..."
python gradio/app_final_production.py

echo "LawFirmAI Production Application started successfully!"
echo "Access the application at: http://localhost:7860"
