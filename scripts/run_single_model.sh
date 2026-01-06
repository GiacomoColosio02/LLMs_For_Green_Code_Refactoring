#!/bin/bash

# ==============================================================================
# SINGLE MODEL LAUNCHER (AWQ 4-BIT QUANTIZED - ULTRA CONTEXT)
# Usage: bash scripts/run_single_model.sh [qwen|deepseek]
# ==============================================================================

MODEL_ALIAS=$1
PORT=8000
GPU_UTIL=0.95  # Usiamo il 95% della GPU (4-bit lascia spazio per il contesto)
MAX_LEN=60000  # 🚀 CONTESTO ENORME: 60k Token (possibile grazie a AWQ)

if [ "$MODEL_ALIAS" == "qwen" ]; then
    # Versione Ufficiale Quantizzata di Qwen 2.5 Coder
    MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    echo "🔵 SELECTED: Qwen 2.5 Coder AWQ (The Executor - High Context)"

elif [ "$MODEL_ALIAS" == "deepseek" ]; then
    # Versione Quantizzata Community (CasperHansen è lo standard per R1 AWQ)
    MODEL_NAME="casperhansen/deepseek-r1-distill-qwen-7b-awq"
    echo "🟣 SELECTED: DeepSeek R1 AWQ (The Reasoner - High Context)"

else
    echo "❌ Error: Specify model alias."
    echo "Usage: bash scripts/run_single_model.sh qwen"
    echo "       bash scripts/run_single_model.sh deepseek"
    exit 1
fi

echo "================================================================="
echo "🚀 STARTING SINGLE vLLM INSTANCE (AWQ 4-bit)"
echo "   Model:   $MODEL_NAME"
echo "   Context: $MAX_LEN tokens"
echo "   Port:    $PORT"
echo "================================================================="

# Uccidiamo eventuali vecchi vLLM per liberare la porta e la VRAM
echo "🧹 Cleaning up old processes..."
pkill -f vllm
sleep 2 # Aspetta che la GPU si liberi

# Lanciamo vLLM con il flag --quantization awq
vllm serve $MODEL_NAME \
    --port $PORT \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len $MAX_LEN \
    --quantization awq \
    --dtype half \
    --trust-remote-code