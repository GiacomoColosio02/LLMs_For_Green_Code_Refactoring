# LLMs For Green Code Refactoring

Benchmarking Large Language Models for Green Code Refactoring using SWE-Perf extended with GSMM-aligned sustainability metrics.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [GSMM Metrics Implementation](#gsmm-metrics-implementation)
- [Measurement Infrastructure](#measurement-infrastructure)
- [Quick Start Guide](#quick-start-guide)
- [Batch Measurement Tutorial](#batch-measurement-tutorial)
- [LLM Models](#llm-models-to-be-defined)
- [Prompting Strategies](#prompting-strategies-to-be-defined)
- [Repository Structure](#repository-structure)
- [Implementation Status](#implementation-status)
- [Thesis Information](#thesis-information)

---

## Project Overview

This thesis project evaluates the ability of **state-of-the-art Large Language Models** to perform **green code refactoring and optimization**. The project extends the **SWE-Perf benchmark** with comprehensive energy and carbon measurements aligned with the **Green Software Measurement Model (GSMM)**.

### Project Goals

1. ✅ **Extend SWE-Perf** with 17 GSMM-aligned metrics (10 GREEN + 7 EFFICIENCY)
2. ⏳ **Benchmark LLMs** for green code optimization (planned)
3. ⏳ **Compare** LLM-generated vs human expert optimizations

### Current Status

- ✅ **Phase 1: GSMM Measurement System** - **COMPLETED**
  - 17 metrics implemented and validated (13 base + 4 wattmeter)
  - 100% energy coverage with wattmeter integration
  - Measurement infrastructure deployed on gaissa.essi.upc.edu
  - Ready for batch measurement (140 instances)
  
- ⏳ **Phase 2: LLM Evaluation** - **UPCOMING**
  - Model selection (2 proprietary + 2 open-source)
  - Prompting strategies design (2 single-turn + 2 multi-turn)
  - Code generation and measurement pipeline

---

## GSMM Metrics Implementation

### ✅ COMPLETED: 17 Metrics (10 GREEN + 7 EFFICIENCY)

Our measurement system captures **17 sustainability metrics** with **100% energy coverage**:

#### 🟢 GREEN Metrics (Energy & Carbon)

**Core Metrics (GPU + CPU, ~85% coverage):**

| Metric | Unit | Description | Tool |
|--------|------|-------------|------|
| **`gpu_energy_joules`** | J | GPU energy consumption | pynvml (power × duration) |
| **`cpu_energy_joules`** | J | CPU energy consumption | EnergiBridge (Intel RAPL) |
| **`total_energy_joules`** | J | Total energy (GPU + CPU) | Calculated |
| **`power_watts`** | W | Mean power draw | Calculated (E / duration) |
| **`carbon_grams`** | gCO₂e | Carbon emissions (partial) | E × grid_intensity |
| **`energy_efficiency`** | - | Energy efficiency ratio | Calculated |

**System Metrics (Wattmeter, 100% coverage):**

| Metric | Unit | Description | Tool |
|--------|------|-------------|------|
| **`system_energy_joules`** 🔌 | J | Complete system energy | NETIO PowerBOX 4KF |
| **`system_power_mean_watts`** 🔌 | W | Mean system power | Wattmeter sampling |
| **`system_power_peak_watts`** �� | W | Peak system power | max(power_samples) |
| **`carbon_grams_system`** 🔌 | gCO₂e | Carbon emissions (complete) | system_E × grid_intensity |

**Energy Coverage Analysis:**
- **GPU + CPU**: ~85% (missing RAM, Storage, PSU, Motherboard)
- **System (Wattmeter)**: 100% (wall power measurement)
- **Example**: GPU+CPU = 71.36 J, System = 84.00 J → 15% missing components

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
- Wattmeter sampling rate: 100ms (10 Hz)
- Baseline measurement: 5 seconds
- Measurement overhead: <1%

### Example Measurement Output (with Wattmeter)
```json
{
  "instance_id": "astropy__astropy-16065",
  "base_commit": "48a792f9",
  "head_commit": "7eac388c",
  "base_measurements": {
    "tests": [
      {
        "test_name": "test_distribution[False-True-log]",
        "measurements": [
          {
            "gpu_energy_joules": 26.23,
            "cpu_energy_joules": 45.13,
            "total_energy_joules": 71.36,
            "power_watts": 46.03,
            "carbon_grams": 0.00496,
            "energy_efficiency": 0.01401,
            "system_energy_joules": 84.00,
            "system_power_mean_watts": 93.33,
            "system_power_peak_watts": 96.00,
            "carbon_grams_system": 0.00583,
            "duration_seconds": 1.55,
            "cpu_usage_mean_percent": 9.36,
            "cpu_usage_peak_percent": 61.1,
            "ram_usage_peak_mb": 3237.10,
            "gpu_usage_mean_percent": 0.0,
            "gpu_usage_peak_percent": 0.0,
            "gpu_memory_peak_mb": 574.19,
            "repetition": 1
          }
        ]
      }
    ]
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
| **Python** | 3.12.3 |
| **Wattmeter** | NETIO PowerBOX 4KF (IP: 10.4.60.25) |

### Software Stack

| Tool | Purpose | Metrics Provided |
|------|---------|------------------|
| **EnergiBridge** | CPU energy (Intel RAPL) | cpu_energy_joules, cpu_power_watts |
| **pynvml** | GPU monitoring | gpu_energy_joules, gpu_usage_*, gpu_memory_* |
| **psutil** | System resources | cpu_usage_*, ram_usage_* |
| **NETIO PowerBOX** | System-level power (100%) | system_energy_joules, system_power_* |
| **pytest** | Test execution | - |

### Implementation Files
```
src/measurement/
├── energy_monitor_gsmm.py      # Main orchestrator (EnergyMonitorGSMM)
│   ├── SystemResourceTracker   # CPU/RAM monitoring thread
│   ├── GPUMonitorThread        # GPU monitoring thread
│   ├── WattmeterMonitorThread  # Wattmeter monitoring thread
│   └── measure_test_energy()   # Main measurement method
│
├── cpu_energy_monitor.py       # EnergiBridge wrapper (sudo)
├── gpu_monitor.py              # pynvml wrapper (NVIDIA)
├── wattmeter_monitor.py        # NETIO PowerBOX wrapper
└── collector.py                # Metrics collection orchestration

scripts/
├── measure_instance.py         # Measure single instance
├── measure_all_instances.py    # Batch measurement (140 instances)
└── verify_measurements.py      # Verify JSON validity

configs/
└── measurement_config.yaml     # Measurement parameters + wattmeter config
```

### Measurement Process

For each SWE-Perf instance (BASE + HEAD commits):
```
1. Clone repository
2. Checkout commit (BASE or HEAD)
3. Install dependencies (with Python 3.12 workarounds)
4. Measure baseline (5s idle)
5. For each test (3 repetitions):
   a. Start wattmeter monitoring thread (100% coverage)
   b. Start GPU monitoring thread
   c. Start CPU/RAM monitoring thread
   d. Execute test with EnergiBridge
   e. Stop all monitoring threads (reverse order)
   f. Calculate 17 metrics
6. Save results to JSON (only if measurements successful)
```

**Key Features:**
- ✅ **100% energy coverage** with wattmeter
- ✅ Baseline subtraction (software-induced changes only)
- ✅ 3 repetitions per test (statistical reliability)
- ✅ Thread-safe monitoring (no race conditions)
- ✅ Automatic GPU detection (disabled if no NVIDIA GPU)
- ✅ Python 3.12 compatibility workarounds (setuptools, urllib3)
- ✅ Error handling: No JSON saved if dependencies fail

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

### 2. Configure System (Server Only)
```bash
# Verify EnergiBridge (requires sudo for RAPL access)
cd ~/LLMs_For_Green_Code_Refactoring/energibridge
sudo ./energibridge --help

# Install ninja (for matplotlib compatibility)
sudo apt update
sudo apt install -y ninja-build

# Verify wattmeter access (UPC network only)
python3 -c "
import requests
r = requests.get('http://10.4.60.25/netio.json', timeout=5)
print(f'✅ Wattmeter accessible: {r.status_code}')
"
```

### 3. Test Single Instance
```bash
# Measure single instance (with wattmeter)
python3 scripts/measure_instance.py \
    --instance astropy__astropy-16065 \
    --dataset data/original/swe_perf_original_20251124.json \
    --country ESP \
    --output data/raw/measurements

# Verify output
cat data/raw/measurements/astropy__astropy-16065/measurements.json | python3 -m json.tool | grep -E "system_energy|system_power"
```

Expected output shows wattmeter metrics:
```json
"system_energy_joules": 84.00,
"system_power_mean_watts": 93.33,
"system_power_peak_watts": 96.00,
"carbon_grams_system": 0.00583
```

---

## Batch Measurement Tutorial

### 📊 Extending SWE-Perf Dataset with GSMM Metrics

This section shows how to measure **all 140 SWE-Perf instances** and create the extended dataset with sustainability metrics.

#### Expected Results

- **Total instances**: 140
- **Success rate**: ~80-85% (110-120 instances)
- **Metrics per instance**: 17 (10 GREEN + 7 EFFICIENCY)
- **Time required**: 6-8 hours
- **Storage**: ~50-100 MB

#### Step-by-Step Guide

**Step 1: Verify Setup**
```bash
# On server gaissa.essi.upc.edu
cd ~/LLMs_For_Green_Code_Refactoring

# Check wattmeter connection
python3 -c "
from src.measurement.wattmeter_monitor import WattmeterMonitor
w = WattmeterMonitor(ip='10.4.60.25', output_id=1)
print(f'Current power: {w.get_current_power():.2f} W')
"

# Expected: ✅ Wattmeter connected: 10.4.60.25
#           Current power: ~80-100 W
```

**Step 2: Clean Output Directory (Optional)**
```bash
# Remove previous measurements if starting fresh
rm -rf data/raw/measurements/*

# Verify clean state
ls data/raw/measurements/ || echo "✅ Directory clean"
```

**Step 3: Launch Batch Measurement**
```bash
# Launch full batch (140 instances) in background
nohup python3 scripts/measure_all_instances.py \
    --dataset data/original/swe_perf_original_20251124.json \
    --country ESP \
    --output data/raw/measurements \
    > measurement_full_140.log 2>&1 &

# Save process ID
echo $! > measurement.pid

echo "✅ Batch measurement started!"
echo "📋 Process ID: $(cat measurement.pid)"
echo "📊 Estimated time: 6-8 hours"
echo "📈 Expected success: 110-120 instances (80-85%)"
```

**Alternative: Test Run (10 instances, ~25 minutes)**
```bash
# Test with first 10 instances
python3 scripts/measure_all_instances.py \
    --dataset data/original/swe_perf_original_20251124.json \
    --country ESP \
    --output data/raw/measurements \
    --limit 10

# Good for verifying system works before full run
```

**Step 4: Monitor Progress**
```bash
# Watch live log
tail -f measurement_full_140.log

# Count completed instances
COMPLETED=$(find data/raw/measurements -name "measurements.json" | wc -l)
echo "Progress: $COMPLETED/140 instances"

# Check process status
ps -p $(cat measurement.pid) -o pid,etime,cmd

# View last 20 lines of log
tail -20 measurement_full_140.log
```

**Step 5: Resume if Interrupted**
```bash
# If process stops, resume from where it left off
COMPLETED=$(find data/raw/measurements -name "measurements.json" | wc -l)
echo "Resuming from instance $COMPLETED"

nohup python3 scripts/measure_all_instances.py \
    --dataset data/original/swe_perf_original_20251124.json \
    --country ESP \
    --output data/raw/measurements \
    --start-from $COMPLETED \
    > measurement_resume.log 2>&1 &

echo $! > measurement.pid
```

**Step 6: Verify Results**
```bash
# After completion, verify measurements
python3 scripts/verify_measurements.py \
    --output-dir data/raw/measurements \
    --verbose

# Expected output:
# ✅ Valid (17 metrics): 110-120 instances
# ❌ Invalid/Incomplete: 20-30 instances
```

**Step 7: View Summary**
```bash
# Check final summary
cat data/raw/measurements/measurement_summary.json | python3 -m json.tool

# Example output:
# {
#   "total_measured": 140,
#   "successes": 115,
#   "failures": 25,
#   "success_rate": 82.1
# }
```

#### Understanding Failures

**Common failure reasons (~15-20% of instances):**
1. **Incompatible dependencies** (old packages with Python 3.12)
2. **Missing test files** (tests removed/renamed in newer commits)
3. **Build failures** (missing system libraries)
4. **Test execution errors** (environment-specific issues)

**These failures are expected** and don't affect the validity of successful measurements. The ~115 valid instances provide sufficient data for statistical analysis.

#### Output Structure
```
data/raw/measurements/
├── astropy__astropy-16065/
│   └── measurements.json          # 17 metrics × 2 commits × N tests
├── astropy__astropy-16058/
│   └── measurements.json
├── ...
├── measurement_log.txt            # Per-instance status log
└── measurement_summary.json       # Final statistics
```

#### Next Steps After Measurement
```bash
# 1. Verify all measurements valid
python3 scripts/verify_measurements.py --output-dir data/raw/measurements

# 2. Save invalid instances list
python3 scripts/verify_measurements.py \
    --output-dir data/raw/measurements \
    --save-invalid invalid_instances.txt

# 3. Create processed dataset (future)
# python3 scripts/create_extended_dataset.py

# 4. Statistical analysis (future)
# jupyter notebook notebooks/analyze_measurements.ipynb
```

---

## LLM Models [TO BE DEFINED]

> **Status:** Model selection in progress  
> **Target:** 4 models total (2 proprietary + 2 open-source)

---

## Prompting Strategies [TO BE DEFINED]

> **Status:** Strategy design in progress  
> **Target:** 4 strategies total (2 single-turn + 2 multi-turn)

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
│   │   ├── energy_monitor_gsmm.py      # Main orchestrator + wattmeter
│   │   ├── cpu_energy_monitor.py       # EnergiBridge wrapper
│   │   ├── gpu_monitor.py              # pynvml wrapper
│   │   ├── wattmeter_monitor.py        # NETIO PowerBOX wrapper
│   │   └── collector.py                # Metrics orchestration
│   │
│   ├── llm_clients/           # ⏳ LLM API clients (TO BE IMPLEMENTED)
│   ├── prompt_templates/      # ⏳ Prompt generation (TO BE IMPLEMENTED)
│   └── ...
│
├── scripts/
│   ├── measure_instance.py           # ✅ Measure single instance
│   ├── measure_all_instances.py      # ✅ Batch measurement (140 instances)
│   └── verify_measurements.py        # ✅ Validate measurement JSONs
│
├── configs/
│   └── measurement_config.yaml       # ✅ Measurement + wattmeter config
│
└── energibridge/              # ✅ EnergiBridge tool (sudo required)
```

---

## Implementation Status

### ✅ Phase 1: GSMM Measurement System (COMPLETED)

- [x] **17 GSMM-aligned metrics implemented**
  - 6 GREEN metrics (GPU+CPU, ~85% coverage)
  - 4 GREEN metrics (wattmeter, 100% coverage)
  - 7 EFFICIENCY metrics (CPU, RAM, GPU usage)
- [x] **Wattmeter integration** (NETIO PowerBOX 4KF)
  - IP: 10.4.60.25 (UPC network)
  - 100% system energy coverage
  - Real-time power monitoring
- [x] **Measurement infrastructure deployed**
  - GPU monitoring with pynvml
  - CPU energy with EnergiBridge
  - System resources with psutil
  - Wattmeter with requests API
- [x] **Python 3.12 compatibility**
  - setuptools/wheel workarounds
  - ninja installation
  - urllib3 dependencies
- [x] **Scripts operational and tested**
  - Single instance measurement validated
  - Batch measurement ready
  - Verification script implemented

**Key Commits:**
- `97214c7`: Fix wattmeter BaseMonitor dependency
- `7fa7ab5`: Complete wattmeter integration
- `b998717`: Add dependency workarounds

### ⏳ Phase 2: LLM Evaluation Pipeline (UPCOMING)

- [ ] Model selection and API setup
- [ ] Prompt template system
- [ ] Code generation pipeline
- [ ] Integration with measurement system

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

## Acknowledgments

- **SWE-Perf creators** - For the foundation benchmark
- **Green Software Foundation** - For GSMM metrics framework
- **UPC Barcelona ESSI** - For computational infrastructure and wattmeter access
- **Alexandra (UPC IT)** - For wattmeter network configuration
- **Supervisors** - Prof. Silverio Martínez-Fernández, Dr. Vincenzo De Martino

---

**Last Updated:** December 4, 2025  
**Current Phase:** Phase 1 Complete ✅ (with Wattmeter) | Phase 2 Starting ⏳
