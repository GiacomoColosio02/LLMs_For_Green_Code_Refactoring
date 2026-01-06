#!/bin/bash

# ==============================================================================
# SINGLE MODEL LAUNCHER (AWQ 4-BIT)
# Fix: Reduced GPU_UTIL to 0.90 to prevent OOM during warmup.
# ==============================================================================

MODEL_ALIAS=$1
PORT=8000
GPU_UTIL=0.90  # 🟢 MODIFICATO: 90% (Lascia ~2.5GB liberi per le attivazioni)
MAX_LEN=32768  # Rimaniamo a 32k, ora ci sta comodo.

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
echo "   VRAM %:  $GPU_UTIL"
echo "================================================================="

# Pulizia processi orfani
pkill -f vllm
sleep 3

# Avvio con parametri di memoria sicuri
vllm serve $MODEL_NAME \
    --port $PORT \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len $MAX_LEN \
    --quantization awq \
    --dtype half \
    --trust-remote-code