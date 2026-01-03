#!/bin/bash

# Attiva environment
source vllm-env/bin/activate

# Configurazione Porte
PORT_QWEN=8000
PORT_DEEPSEEK=8001

# Calcolo memoria: Diamo il 45% della GPU a testa (45% + 45% = 90%, lascia 10% al sistema)
GPU_UTIL=0.45

echo "========================================================"
echo "🚀 STARTING LOCAL GREEN AI SERVER (RTX 4090 OPTIMIZED)"
echo "========================================================"

# 1. Start Qwen2.5-Coder (Port 8000)
echo "Starting Qwen2.5-Coder-7B..."
nohup vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
    --port $PORT_QWEN \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 8192 \
    --api-key "EMPTY" > qwen.log 2>&1 &

# Nota: --kv-cache-dtype fp8 comprime la memoria del contesto, ottimo per repo grandi!

# 2. Start DeepSeek-R1 (Port 8001)
echo "Starting DeepSeek-R1-Distill-Qwen-7B..."
nohup vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --port $PORT_DEEPSEEK \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization $GPU_UTIL \
    --max-model-len 8192 \
    --api-key "EMPTY" > deepseek.log 2>&1 &

echo "⏳ Waiting 30s for models to load..."
sleep 30

echo "✅ SERVERS UP!"
echo "   - Qwen:     http://localhost:$PORT_QWEN/v1"
echo "   - DeepSeek: http://localhost:$PORT_DEEPSEEK/v1"
echo "   (Check qwen.log and deepseek.log for errors)"