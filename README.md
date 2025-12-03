# LLMs For Green Code Refactoring

Benchmarking Large Language Models for Green Code Refactoring using SWE-Perf extended with GSMM-aligned sustainability metrics.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [GSMM Metrics Implementation](#gsmm-metrics-implementation)
- [Measurement Infrastructure](#measurement-infrastructure)
- [Quick Start Guide](#quick-start-guide)
- [LLM Models](#llm-models-to-be-defined)
- [Prompting Strategies](#prompting-strategies-to-be-defined)
- [Repository Structure](#repository-structure)
- [Implementation Status](#implementation-status)
- [Next Steps](#next-steps)
- [Thesis Information](#thesis-information)

---

## Project Overview

This thesis project evaluates the ability of **state-of-the-art Large Language Models** to perform **green code refactoring and optimization**. The project extends the **SWE-Perf benchmark** with comprehensive energy and carbon measurements aligned with the **Green Software Measurement Model (GSMM)**.

### Project Goals

1. ✅ **Extend SWE-Perf** with 13 GSMM-aligned metrics (6 GREEN + 7 EFFICIENCY)
2. ⏳ **Benchmark LLMs** for green code optimization (planned)
3. ⏳ **Compare** LLM-generated vs human expert optimizations

### Current Status

- ✅ **Phase 1: GSMM Measurement System** - **COMPLETED**
  - 13 metrics implemented and validated
  - Measurement infrastructure deployed on gaissa.essi.upc.edu
  - Baseline measurements ready (140 instances)
  
- ⏳ **Phase 2: LLM Evaluation** - **UPCOMING**
  - Model selection (2 proprietary + 2 open-source)
  - Prompting strategies design (2 single-turn + 2 multi-turn)
  - Code generation and measurement pipeline

---

## GSMM Metrics Implementation

### ✅ COMPLETED: 13 Metrics (6 GREEN + 7 EFFICIENCY)

Our measurement system captures **13 sustainability metrics** aligned with the Green Software Measurement Model:

#### 🟢 GREEN Metrics (Energy & Carbon)

| Metric | Unit | Description | Tool |
|--------|------|-------------|------|
| **`gpu_energy_joules`** | J | GPU energy consumption | pynvml (power × duration) |
| **`cpu_energy_joules`** | J | CPU energy consumption | EnergiBridge (Intel RAPL) |
| **`total_energy_joules`** | J | Total energy (GPU + CPU) | Calculated |
| **`power_watts`** | W | Mean power draw | Calculated (E / duration) |
| **`carbon_grams`** | gCO₂e | Carbon emissions | E × grid_intensity |
| **`energy_efficiency`** | - | Energy efficiency ratio | Calculated |

**Carbon Grid Intensities:**
- Spain (ESP): 250 gCO₂e/kWh
- USA: 417 gCO₂e/kWh  
- Germany (DEU): 311 gCO₂e/kWh
- France (FRA): 52 gCO₂e/kWh
- UK (GBR): 233 gCO₂e/kWh

#### ⚡ EFFICIENCY Metrics (Resource Usage)

| Metric | Unit | Description | Tool |
|--------|------|-------------|------|
| **`duration_seconds`** | s | Test execution time | time.time() |
| **`cpu_usage_mean_percent`** | % | Average CPU utilization | psutil (system-wide) |
| **`cpu_usage_peak_percent`** | % | Peak CPU utilization | max(cpu_samples) |
| **`ram_usage_peak_mb`** | MB | Peak RAM usage | psutil.virtual_memory() |
| **`gpu_usage_mean_percent`** | % | Average GPU utilization | pynvml (compute) |
| **`gpu_usage_peak_percent`** | % | Peak GPU utilization | max(gpu_samples) |
| **`gpu_memory_peak_mb`** | MB | Peak GPU memory usage | pynvml.nvmlDeviceGetMemoryInfo() |

**Sampling Configuration:**
- CPU/RAM sampling rate: 100ms (10 Hz)
- GPU sampling rate: 100ms (10 Hz)
- Baseline measurement: 5 seconds
- Measurement overhead: <1%

### Example Measurement Output
```json
{
  "instance_id": "astropy__astropy-16065",
  "base_commit": "48a792f9",
  "head_commit": "7eac388c",
  "measurements": {
    "BASE": {
      "test_1": {
        "green_metrics": {
          "gpu_energy_joules": 31.34,
          "cpu_energy_joules": 50.25,
          "total_energy_joules": 81.59,
          "power_watts": 45.26,
          "carbon_grams": 0.00567,
          "energy_efficiency": 0.0123
        },
        "efficiency_metrics": {
          "duration_seconds": 1.80,
          "cpu_usage_mean_percent": 8.02,
          "cpu_usage_peak_percent": 61.2,
          "ram_usage_peak_mb": 4896.95,
          "gpu_usage_mean_percent": 0.0,
          "gpu_usage_peak_percent": 0.0,
          "gpu_memory_peak_mb": 6384.88
        },
        "repetitions": [
          {"run": 1, "duration": 1.80, "energy": 81.59},
          {"run": 2, "duration": 1.10, "energy": 47.12},
          {"run": 3, "duration": 1.10, "energy": 48.76}
        ]
      }
    }
  }
}
```

---

## Measurement Infrastructure

### Hardware Configuration

**Server:** gaissa.essi.upc.edu (UPC Barcelona ESSI Department)

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 9 7950X (16 cores, 32 threads) |
| **GPU** | NVIDIA GeForce RTX 4090 (24GB VRAM) |
| **RAM** | 32GB DDR5 |
| **OS** | Ubuntu 24.04 LTS |
| **Python** | 3.12 |

### Software Stack

| Tool | Purpose | Installation |
|------|---------|--------------|
| **EnergiBridge** | CPU energy (Intel RAPL) | `~/LLMs_For_Green_Code_Refactoring/energibridge` |
| **pynvml** | GPU monitoring | `pip install pynvml` |
| **psutil** | System resources | `pip install psutil` |
| **pytest** | Test execution | `pip install pytest` |

### Implementation Files
```
src/measurement/
├── energy_monitor_gsmm.py      # Main orchestrator (EnergyMonitorGSMM)
│   ├── SystemResourceTracker   # CPU/RAM monitoring thread
│   ├── GPUMonitorThread        # GPU monitoring thread
│   └── measure_test_energy()   # Main measurement method
│
├── cpu_energy_monitor.py       # EnergiBridge wrapper (sudo)
├── gpu_monitor.py              # pynvml wrapper (NVIDIA)
├── wattmeter_monitor.py        # NETIO PowerBOX (future, optional)
└── base_monitor.py             # Abstract base class

scripts/
├── measure_instance.py         # Measure single instance
└── measure_all_instances.py    # Batch measurement (140 instances)

configs/
└── measurement_config.yaml     # Measurement parameters
```

### Measurement Process

For each SWE-Perf instance (BASE + HEAD commits):
```
1. Clone repository
2. Checkout commit (BASE or HEAD)
3. Install dependencies
4. Measure baseline (5s idle)
5. For each test (3 repetitions):
   a. Start GPU monitoring thread
   b. Start CPU/RAM monitoring thread
   c. Execute test with EnergiBridge
   d. Stop monitoring threads
   e. Calculate metrics
6. Save results to JSON
```

**Key Features:**
- ✅ Baseline subtraction (software-induced changes only)
- ✅ 3 repetitions per test (statistical reliability)
- ✅ Thread-safe monitoring (no race conditions)
- ✅ Automatic GPU detection (disabled if no NVIDIA GPU)
- ✅ Error handling with cleanup (try-finally blocks)

---

## Quick Start Guide

### 1. Installation
```bash
# Clone repository
git clone https://github.com/GiacomoColosio02/LLMs_For_Green_Code_Refactoring.git
cd LLMs_For_Green_Code_Refactoring

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure EnergiBridge
```bash
# EnergiBridge should be installed at:
~/LLMs_For_Green_Code_Refactoring/energibridge

# Verify it works (requires sudo for RAPL access)
cd ~/LLMs_For_Green_Code_Refactoring/energibridge
sudo ./energibridge --help
```

### 3. Test Measurement System
```bash
# Measure single instance (test system)
python3 scripts/measure_instance.py \
    --instance astropy__astropy-16065 \
    --dataset data/original/swe_perf_original_20251124.json \
    --country ESP \
    --output data/raw/measurements

# Expected output: JSON with 13 metrics
cat data/raw/measurements/astropy__astropy-16065/measurements.json
```

### 4. Measure All 140 Instances (Baseline)

#### Option A: Small Test (10 instances, ~25 minutes)
```bash
python3 scripts/measure_all_instances.py \
    --dataset data/original/swe_perf_original_20251124.json \
    --country ESP \
    --output data/raw/measurements \
    --limit 10
```

#### Option B: Full Measurement (140 instances, ~5-6 hours)
```bash
# Launch in background with nohup
nohup python3 scripts/measure_all_instances.py \
    --dataset data/original/swe_perf_original_20251124.json \
    --country ESP \
    --output data/raw/measurements \
    > measurement_full.log 2>&1 &

# Save process ID for monitoring
echo $! > measurement.pid

# Monitor progress
tail -f measurement_full.log

# Count completed instances
find data/raw/measurements -name "measurements.json" | wc -l
```

#### Resume Interrupted Measurement
```bash
# Count already completed instances
COMPLETED=$(find data/raw/measurements -name "measurements.json" | wc -l)

# Resume from where you left off
python3 scripts/measure_all_instances.py \
    --dataset data/original/swe_perf_original_20251124.json \
    --country ESP \
    --output data/raw/measurements \
    --start-from $COMPLETED
```

### 5. Verify Measurement Results
```bash
# Check measurement summary
cat data/raw/measurements/measurement_summary.json

# Verify all 13 metrics present in a measurement
python3 -c "
import json
with open('data/raw/measurements/astropy__astropy-16065/measurements.json') as f:
    data = json.load(f)
    test_metrics = data['measurements']['BASE']['test_1']
    
    green = test_metrics['green_metrics']
    efficiency = test_metrics['efficiency_metrics']
    
    print('🟢 GREEN METRICS:')
    for k, v in green.items():
        print(f'  {k}: {v}')
    
    print('\n⚡ EFFICIENCY METRICS:')
    for k, v in efficiency.items():
        print(f'  {k}: {v}')
"
```

---

## LLM Models [TO BE DEFINED]

> **Status:** Model selection in progress  
> **Target:** 4 models total (2 proprietary + 2 open-source)

### Selection Criteria

- SWE-bench performance (code generation capability)
- API availability and cost
- Context window size (for repository-level tasks)
- Licensing (open-source requirement)

### Tentative Candidates

**Proprietary:**
- OpenAI GPT-4/5 series
- Anthropic Claude 3/4 series

**Open-Source:**
- Alibaba Qwen2.5-Coder
- Meta Llama 3/4 series
- Google CodeGemma

**Final selection will be documented in Phase 2.**

---

## Prompting Strategies [TO BE DEFINED]

> **Status:** Strategy design in progress  
> **Target:** 4 strategies total (2 single-turn + 2 multi-turn)

### Design Principles

1. **Green-oriented:** All prompts emphasize energy and carbon reduction
2. **Fair comparison:** Same information across all strategies
3. **Reproducible:** Minimal variability in prompt structure
4. **Inspired by:** Recent LLM-for-code research (SWE-bench, CodeGen studies)

### Tentative Framework

**Single-Turn Strategies (2):**
- Direct optimization request
- Role-based expert prompt

**Multi-Turn Strategies (2):**
- Self-collaboration (analysis → optimization → validation)
- Iterative refinement with feedback

**Final strategies will be documented in Phase 2.**

---

## Repository Structure
```
LLMs_For_Green_Code_Refactoring/
├── data/
│   ├── original/              # SWE-Perf original dataset
│   │   └── swe_perf_original_20251124.json  # 140 instances
│   ├── raw/                   # Raw measurement data (JSON per instance)
│   └── processed/             # Extended dataset with GSMM metrics (future)
│
├── src/
│   ├── measurement/           # ✅ Metric collection (COMPLETED)
│   │   ├── energy_monitor_gsmm.py      # Main orchestrator
│   │   ├── cpu_energy_monitor.py       # EnergiBridge wrapper
│   │   ├── gpu_monitor.py              # pynvml wrapper
│   │   ├── wattmeter_monitor.py        # NETIO PowerBOX (optional)
│   │   └── base_monitor.py             # Abstract base
│   │
│   ├── llm_clients/           # ⏳ LLM API clients (TO BE IMPLEMENTED)
│   ├── prompt_templates/      # ⏳ Prompt generation (TO BE IMPLEMENTED)
│   ├── code_extraction/       # ⏳ Parse SWE-Perf instances (TO BE IMPLEMENTED)
│   ├── patch_application/     # ⏳ Apply LLM patches (TO BE IMPLEMENTED)
│   ├── data_processing/       # ⏳ Data processing pipelines
│   ├── analysis/              # ⏳ Statistical analysis tools
│   └── utils/                 # Shared utilities
│
├── scripts/
│   ├── measure_instance.py           # ✅ Measure single instance
│   └── measure_all_instances.py      # ✅ Batch measurement (140 instances)
│
├── configs/
│   ├── measurement_config.yaml       # ✅ Measurement parameters
│   └── llm_api_keys.yaml             # ⏳ API keys (future)
│
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
├── results/                   # Analysis results and figures
└── energibridge/              # ✅ EnergiBridge tool (sudo required)
```

---

## Implementation Status

### ✅ Phase 1: GSMM Measurement System (COMPLETED)

- [x] **13 GSMM-aligned metrics implemented**
  - 6 GREEN metrics (energy, power, carbon)
  - 7 EFFICIENCY metrics (CPU, RAM, GPU usage)
- [x] **Measurement infrastructure deployed**
  - GPU monitoring with pynvml (NVIDIA RTX 4090)
  - CPU energy with EnergiBridge (Intel RAPL via sudo)
  - System resources with psutil (CPU%, RAM)
  - Thread-safe monitoring (SystemResourceTracker, GPUMonitorThread)
- [x] **Scripts operational**
  - `measure_instance.py` (single instance)
  - `measure_all_instances.py` (batch 140 instances)
- [x] **Validation complete**
  - Tested on 2 instances (astropy__astropy-16065, astropy__astropy-16058)
  - All 13 metrics present in output JSON
  - ~146 seconds per instance (5.7 hours for 140)

**Key Commits:**
- `6aa41af`: Initial GPU monitoring
- `eb984bd`: SystemResourceTracker for CPU/RAM
- `791f2f8`: GPUMonitorThread for GPU metrics
- `307e951`: Fix import paths in measure_all_instances.py

### ⏳ Phase 2: LLM Evaluation Pipeline (UPCOMING)

- [ ] **LLM client infrastructure**
  - Model selection (2 proprietary + 2 open-source)
  - API client implementations
  - Retry logic and rate limiting
- [ ] **Prompt template system**
  - 2 single-turn strategies
  - 2 multi-turn strategies
  - Green-oriented prompts (energy/carbon focus)
- [ ] **Code extraction & patch application**
  - Parse SWE-Perf problem statements
  - Extract LLM-generated patches
  - Apply patches to repositories
- [ ] **Integration & measurement**
  - Generate LLM optimizations (4 models × 2 strategies = 8 versions per instance)
  - Measure LLM versions with GSMM metrics
  - Total: 140 instances × (2 baseline + 8 LLM) = 1,400 measurements

### ⏳ Phase 3: Analysis & Thesis (FUTURE)

- [ ] Statistical analysis (LLM vs human performance)
- [ ] Visualization (plots, tables, comparative analysis)
- [ ] Thesis writing
- [ ] Defense preparation

---

## Next Steps

### Immediate (Phase 2 - LLM Evaluation)

1. **Finalize LLM model selection**
   - Identify 2 proprietary models (GPT-4/5, Claude 3/4)
   - Identify 2 open-source models (Qwen2.5-Coder, Llama/CodeGemma)
   - Obtain API keys and test access

2. **Design prompting strategies**
   - Define 2 single-turn strategies (direct, role-based)
   - Define 2 multi-turn strategies (self-collaboration, iterative)
   - Implement prompt template system with green focus

3. **Build code generation pipeline**
   - Parse SWE-Perf problem statements
   - Generate prompts for each (model, strategy) combination
   - Call LLM APIs and extract patches
   - Apply patches to repositories

4. **Measure LLM-generated code**
   - Run measurement system on LLM versions
   - Collect 13 GSMM metrics per version
   - Store results in structured format

### Medium-Term (Phase 3 - Analysis)

5. **Statistical analysis**
   - Compare LLM vs human performance
   - Analyze trade-offs (energy vs speed, carbon vs memory)
   - Identify best-performing models and strategies

6. **Visualization and reporting**
   - Generate plots and tables
   - Write results section
   - Prepare presentation materials

---

## Thesis Information

**Title:** Benchmarking Large Language Models for Green Code Refactoring

**Student:** Giacomo Colosio  
**Email:** giacomo.colosio@estudiantat.upc.edu  
**GitHub:** [@GiacomoColosio02](https://github.com/GiacomoColosio02)

**Institution:** Universitat Politècnica de Catalunya (UPC Barcelona)  
**Department:** ESSI (Barcelona School of Informatics)  
**Program:** Master's in Computer Science  

**Supervisors:**
- Prof. Silverio Martínez-Fernández (UPC Barcelona)
- Dr. Vincenzo De Martino (UPC Barcelona)

**Research Focus:** LLMs × Software Engineering × Sustainability (Green Software)

**Academic Year:** 2024-2025

---

## References

1. **SWE-Perf:** Performance Optimization Benchmark for Software Engineering
2. **Green Software Measurement Model (GSMM):** Green Software Foundation
3. **Intel RAPL:** Running Average Power Limit for CPU energy measurement
4. **EnergiBridge:** Open-source tool for RAPL access
5. **pynvml:** Python bindings for NVIDIA Management Library

---

## Acknowledgments

- **SWE-Perf creators** - For the foundation benchmark
- **Green Software Foundation** - For GSMM metrics framework
- **UPC Barcelona ESSI** - For computational infrastructure (gaissa server)
- **Supervisors** - Prof. Silverio Martínez-Fernández, Dr. Vincenzo De Martino

---

**Last Updated:** December 2025  
**Current Phase:** Phase 1 Complete ✅ | Phase 2 Starting ⏳