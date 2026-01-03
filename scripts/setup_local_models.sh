#!/bin/bash

# 1. Cerca l'interprete Python corretto (vLLM vuole < 3.13)
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
else
    echo "❌ Errore: Python 3.10 o 3.11 non trovato. Installalo con 'sudo apt install python3.11 python3.11-venv'"
    exit 1
fi

echo "Using python: $($PYTHON_CMD --version)"

# 2. Crea environment isolato
echo "Creating vLLM environment..."
rm -rf vllm-env
$PYTHON_CMD -m venv vllm-env
source vllm-env/bin/activate

# 3. Installa vLLM (Fix: accelerate invece di accelerated)
echo "Installing vLLM & Utilities..."
pip install --upgrade pip
pip install vllm accelerate "huggingface_hub[cli]"

echo "✅ Setup Finished!"