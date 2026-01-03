#!/bin/bash

# 1. Crea environment isolato
echo "Creating vLLM environment..."
python3 -m venv vllm-env
source vllm-env/bin/activate

# 2. Installa vLLM (aggiornato per supporto FP8)
echo "Installing vLLM & Utilities..."
pip install --upgrade pip
pip install vllm "huggingface_hub[cli]" accelerated

echo "Setup Finished!"