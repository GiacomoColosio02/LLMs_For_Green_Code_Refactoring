#!/bin/bash

echo "=========================================="
echo "🔍 AUDIT REPOSITORY LOCALE"
echo "=========================================="
echo ""

echo "📁 STEP 1: Struttura Directory"
echo "=========================================="
tree -L 3 -I '__pycache__|*.pyc|venv|.git' 2>/dev/null || find . -type d -not -path '*/\.*' -not -path '*/venv/*' -not -path '*/__pycache__/*' | head -30
echo ""

echo "=========================================="
echo "📂 STEP 2: File Measurement"
echo "=========================================="
echo "File in src/measurement/:"
ls -lh src/measurement/*.py 2>/dev/null || echo "❌ Directory src/measurement/ non trovata"
echo ""

echo "=========================================="
echo "⚙️ STEP 3: File Configurazione"
echo "=========================================="
echo "File in configs/:"
ls -lh configs/ 2>/dev/null || echo "❌ Directory configs/ non trovata"
echo ""

echo "=========================================="
echo "📦 STEP 4: Dipendenze Installate"
echo "=========================================="
pip list 2>/dev/null | grep -E "psutil|codecarbon|pyyaml|PyYAML|datasets|pandas" || echo "⚠️ Alcune dipendenze potrebbero mancare"
echo ""

echo "=========================================="
echo "🔄 STEP 5: Git Status"
echo "=========================================="
git status --short
echo ""
git log --oneline -5
echo ""

echo "=========================================="
echo "✅ AUDIT COMPLETATO"
echo "=========================================="
echo ""
echo "Prossimo step: python test_local_metrics.py"

