#!/bin/bash

# ==============================================================================
# GREEN CODE REFACTORING - LOCAL LLM SERVER LAUNCHER
# Infrastructure: vLLM on NVIDIA RTX 4090 (24GB)
# Strategy: Dual Model Serving (Split Memory)
# ==============================================================================

# Definiamo i nomi dei modelli (HuggingFace IDs)
# 1. Qwen2.5-Coder-7B-Instruct: Il "Manovale" (Coding puro, Zero-Shot)
MODEL_CODER="Qwen/Qwen2.5-Coder-7B-Instruct"

# 2. DeepSeek-R1-Distill-Qwen-7B: Il "Pensatore" (Reasoning, CoT, LDB)
MODEL_REASONER="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Configurazioni Porte
PORT_CODER=8000
PORT_REASONER=8001

# Configurazione GPU
# Usiamo 0.45 per lasciare un 10% di margine per il sistema operativo e overhead
GPU_UTILIZATION=0.45
MAX_MODEL_LEN=8192  # Riduciamo leggermente il contesto per sicurezza memoria (puoi alzare a 16k/32k se regge)

echo "================================================================="
echo "🚀 STARTING LOCAL LLM INFRASTRUCTURE (vLLM)"
echo "GPU Memory Budget per model: ~10.8 GB (45%)"
echo "================================================================="

# Funzione per uccidere i processi alla chiusura dello script
trap 'kill $(jobs -p)' EXIT

# 1. Avvio Server Qwen (Coder) sulla porta 8000
echo "Starting Coder Model: $MODEL_CODER on port $PORT_CODER..."
vllm serve $MODEL_CODER \
    --port $PORT_CODER \
    --gpu-memory-utilization $GPU_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-dtype fp8 \
    --dtype half \
    --trust-remote-code &

# Attendiamo qualche secondo per dare precedenza all'allocazione del primo
sleep 10

# 2. Avvio Server DeepSeek (Reasoner) sulla porta 8001
echo "Starting Reasoner Model: $MODEL_REASONER on port $PORT_REASONER..."
vllm serve $MODEL_REASONER \
    --port $PORT_REASONER \
    --gpu-memory-utilization $GPU_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --kv-cache-dtype fp8 \
    --dtype half \
    --trust-remote-code &

echo "================================================================="
echo "⏳ Waiting for servers to be ready..."
echo "   - Qwen Coder:     http://localhost:$PORT_CODER/v1"
echo "   - DeepSeek R1:    http://localhost:$PORT_REASONER/v1"
echo "================================================================="
echo "Press Ctrl+C to stop both servers."

# Mantiene lo script attivo
wait