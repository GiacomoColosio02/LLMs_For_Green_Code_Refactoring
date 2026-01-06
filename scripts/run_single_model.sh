#!/bin/bash

# ==============================================================================
# SINGLE MODEL LAUNCHER (FP16 FULL PRECISION)
# Usage: bash scripts/run_single_model.sh [qwen|deepseek]
# ==============================================================================

MODEL_ALIAS=$1
PORT=8000
GPU_UTIL=0.90  # Usiamo quasi tutta la GPU per un solo modello
MAX_LEN=16384  # Possiamo permetterci un contesto più ampio ora!

if [ "$MODEL_ALIAS" == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
    echo "🔵 SELECTED: Qwen 2.5 Coder (The Executor)"

elif [ "$MODEL_ALIAS" == "deepseek" ]; then
    MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    echo "🟣 SELECTED: DeepSeek R1 (The Reasoner)"

else
    echo "❌ Error: Specify model alias."
    echo "Usage: bash scripts/run_single_model.sh qwen"
    echo "       bash scripts/run_single_model.sh deepseek"
    exit 1
fi

echo "================================================================="
echo "🚀 STARTING SINGLE vLLM INSTANCE (Full Precision FP16)"
echo "   Model: $MODEL_NAME"
echo "   Port:  $PORT"
echo "================================================================="

# Uccidiamo eventuali vecchi vLLM per liberare la porta
pkill -f vllm

vllm serve $MODEL_NAME \
    --port $PORT \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len $MAX_LEN \
    --dtype half \
    --trust-remote-code