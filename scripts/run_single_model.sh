#!/bin/bash

# ==============================================================================
# SINGLE MODEL LAUNCHER (AWQ 4-BIT)
# Fix: Respects Qwen native limit (32k) to avoid RoPE scaling errors.
# ==============================================================================

MODEL_ALIAS=$1
PORT=8000
GPU_UTIL=0.95
MAX_LEN=32768  # 🟢 LIMITE SICURO: 32k Tokens (~120k caratteri). Più che sufficiente.

if [ "$MODEL_ALIAS" == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    echo "🔵 SELECTED: Qwen 2.5 Coder AWQ (32k Context)"

elif [ "$MODEL_ALIAS" == "deepseek" ]; then
    MODEL_NAME="casperhansen/deepseek-r1-distill-qwen-7b-awq"
    echo "🟣 SELECTED: DeepSeek R1 AWQ"

else
    echo "❌ Error: Specify model alias."
    echo "Usage: bash scripts/run_single_model.sh qwen"
    exit 1
fi

echo "================================================================="
echo "🚀 STARTING SINGLE vLLM INSTANCE"
echo "   Model:   $MODEL_NAME"
echo "   Context: $MAX_LEN tokens"
echo "================================================================="

pkill -f vllm
sleep 2

# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 serve solo se vuoi forzare oltre i 32k (rischio crash)
# Qui usiamo il limite nativo per la massima stabilità.
vllm serve $MODEL_NAME \
    --port $PORT \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len $MAX_LEN \
    --quantization awq \
    --dtype half \
    --trust-remote-code