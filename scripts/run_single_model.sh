#!/bin/bash

# ==============================================================================
# SINGLE MODEL LAUNCHER (AWQ 4-BIT)
# ==============================================================================

MODEL_ALIAS=$1
PORT=8000
GPU_UTIL=0.90
MAX_LEN=32768

if [ "$MODEL_ALIAS" == "qwen" ]; then
    MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
    echo "�� SELECTED: Qwen 2.5 Coder 7B AWQ"

elif [ "$MODEL_ALIAS" == "qwen32" ]; then
    MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"
    GPU_UTIL=0.95
    MAX_LEN=16384  # Ridotto per stare in 24GB
    echo "🔵 SELECTED: Qwen 2.5 Coder 32B AWQ (più intelligente!)"

elif [ "$MODEL_ALIAS" == "deepseek" ]; then
    MODEL_NAME="casperhansen/deepseek-r1-distill-qwen-7b-awq"
    echo "🟣 SELECTED: DeepSeek R1 Distill 7B AWQ"

elif [ "$MODEL_ALIAS" == "deepseek14" ]; then
    MODEL_NAME="casperhansen/deepseek-r1-distill-qwen-14b-awq"
    GPU_UTIL=0.95
    MAX_LEN=16384
    echo "🟣 SELECTED: DeepSeek R1 Distill 14B AWQ (reasoning!)"

else
    echo "❌ Error: Specify model alias."
    echo ""
    echo "Usage: bash scripts/run_single_model.sh <alias>"
    echo ""
    echo "Available models:"
    echo "  qwen       - Qwen 2.5 Coder 7B  (fast, ~5GB)"
    echo "  qwen32     - Qwen 2.5 Coder 32B (smart, ~18GB)"
    echo "  deepseek   - DeepSeek R1 7B    (reasoning, ~5GB)"
    echo "  deepseek14 - DeepSeek R1 14B   (best reasoning, ~10GB)"
    exit 1
fi

echo "================================================================="
echo "🚀 STARTING SINGLE vLLM INSTANCE"
echo "   Model:   $MODEL_NAME"
echo "   Context: $MAX_LEN tokens"
echo "   VRAM %:  $GPU_UTIL"
echo "================================================================="

# Pulizia processi orfani
pkill -f vllm
sleep 3

# Avvio
vllm serve $MODEL_NAME \
    --port $PORT \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len $MAX_LEN \
    --quantization awq \
    --dtype half \
    --trust-remote-code
