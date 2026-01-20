# SWE-Perf-Green: Benchmarking Energy Efficiency of LLM-Generated Code Optimizations

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

**Replication package for the paper:**

> **Faster but Fatter: Benchmarking Energy Efficiency of LLM-Generated Code Optimizations**
> 
> Giacomo Colosio
> 
> Universitat Politècnica de Catalunya, Barcelona, Spain

---

## Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in automated code refactoring. While existing benchmarks like SWE-Perf evaluate LLMs on code *performance* optimization, they neglect a critical dimension: **energy consumption**. We introduce **SWE-Perf-Green**, extending SWE-Perf with comprehensive energy measurement capabilities aligned with the Green Software Measurement Model (GSMM).

**Key Findings:**
- 🔴 LLM-generated optimizations consume **1.3–6.8× more energy** than human solutions despite achieving speedup
- 🟡 Context granularity (Oracle vs Realistic) has **no significant impact** on energy efficiency (p>0.6)
- 🟢 Simpler prompting strategies produce greener code: **Zero-Shot achieves 15% better energy efficiency** than Self-Collaboration (p<0.001)
- 🔵 Human-written optimizations remain **significantly more energy-efficient** than all LLM configurations (p<0.001)

---

## Benchmark Overview

| Component | Description |
|-----------|-------------|
| **Instances** | 125 real-world optimization tasks from GitHub PRs |
| **Repositories** | 12 popular Python projects (scikit-learn, pandas, matplotlib, etc.) |
| **Tests** | 874 executable performance tests |
| **Metrics** | 17 GSMM-aligned metrics (energy, power, carbon, efficiency) |
| **Models Evaluated** | Qwen2.5-Coder-7B, DeepSeek-R1-Distill-7B |
| **Configurations** | 16 (2 models × 4 prompting strategies × 2 context settings) |
| **Total Patches** | 2,050 generated, 618 valid for energy analysis |

---

## Repository Structure
```
SWE-Perf-Green/
├── data/
│   ├── benchmark/                    # Curated benchmark instances (125)
│   ├── measurements/                 # Energy measurement results
│   └── patches/                      # Generated patches (all configurations)
│
├── src/
│   ├── measurement/                  # Energy measurement infrastructure
│   │   ├── energy_monitor_gsmm.py   # Main orchestrator (17 metrics)
│   │   ├── cpu_energy_monitor.py    # EnergiBridge/RAPL wrapper
│   │   ├── gpu_monitor.py           # NVML wrapper
│   │   └── wattmeter_monitor.py     # NETIO PowerBOX integration
│   │
│   ├── generation/                   # Patch generation pipeline
│   │   ├── prompts/                 # 4 prompting strategy templates
│   │   └── llm_client.py            # vLLM inference client
│   │
│   └── analysis/                     # Statistical analysis scripts
│
├── scripts/
│   ├── measure_instance.py          # Measure single instance
│   ├── measure_batch.py             # Batch measurement (all instances)
│   ├── generate_patches.py          # Generate patches with LLMs
│   └── run_analysis.py              # Reproduce paper statistics
│
├── configs/
│   ├── measurement_config.yaml      # Hardware & measurement parameters
│   └── generation_config.yaml       # LLM generation parameters
│
├── results/                          # Paper results & figures
│   ├── tables/                      # LaTeX tables
│   └── figures/                     # Generated figures
│
└── docker/                           # Reproducible environments
    └── Dockerfile                   # Container for measurements
```

---

## Quick Start

### 1. Installation
```bash
git clone https://github.com/GiacomoColosio02/LLMs_For_Green_Code_Refactoring.git
cd LLMs_For_Green_Code_Refactoring

# Create environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce Paper Results
```bash
# Run statistical analysis (no hardware required)
python scripts/run_analysis.py --results data/measurements/

# Output: Tables I-XI and Figures 4-5 from the paper
```

### 3. Run Energy Measurements (Requires Hardware)
```bash
# Single instance
python scripts/measure_instance.py \
    --instance scikit-learn__scikit-learn-12345 \
    --output data/measurements/

# Full benchmark (requires ~8 hours)
python scripts/measure_batch.py \
    --config configs/measurement_config.yaml \
    --output data/measurements/
```

---

## GSMM Metrics

We collect **17 metrics** aligned with the Green Software Measurement Model:

### Energy Metrics (10)
| Metric | Unit | Source |
|--------|------|--------|
| `cpu_energy` | J | Intel RAPL via EnergiBridge |
| `gpu_energy` | J | NVIDIA NVML |
| `system_energy` | J | NETIO PowerBOX 4KF wattmeter |
| `total_energy` | J | Aggregated |
| `power_mean` | W | Calculated |
| `power_peak` | W | max(samples) |
| `carbon_emissions` | gCO₂eq | Energy × grid intensity |
| `energy_per_test` | J/test | Calculated |
| `energy_ratio` | - | E_LLM / E_human |
| `edp` | J·s | Energy × Duration |

### Efficiency Metrics (7)
| Metric | Unit | Source |
|--------|------|--------|
| `duration` | s | Timer |
| `cpu_usage_mean` | % | psutil |
| `cpu_usage_peak` | % | psutil |
| `ram_usage_peak` | MB | psutil |
| `gpu_usage_mean` | % | NVML |
| `gpu_usage_peak` | % | NVML |
| `gpu_memory_peak` | MB | NVML |

---

## Hardware Configuration

Experiments conducted on dedicated measurement server:

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD EPYC 7742 (64 cores, 2.25 GHz) |
| **GPU** | NVIDIA GeForce RTX 4090 (24GB VRAM) |
| **RAM** | 512 GB DDR4-3200 |
| **Storage** | NVMe SSD (Samsung 980 PRO) |
| **Power Monitor** | NETIO PowerBOX 4KF (0.5% accuracy) |
| **OS** | Ubuntu 22.04 LTS (kernel 5.15) |

---

## Prompting Strategies

| Strategy | Structure | Description |
|----------|-----------|-------------|
| **Zero-Shot (ZS)** | Single-turn | Direct optimization request |
| **Chain-of-Thought (CoT)** | Two-phase | Analysis → Generation |
| **Self-Collaboration (SC)** | Three-turn | Analyst → Optimizer → Reviewer |
| **LLM Debugger (LDB)** | Iterative | Refinement with execution feedback |

Prompt templates available in `src/generation/prompts/`.

---

## Citation
```bibtex
@inproceedings{colosio2025swepergreen,
  author    = {Colosio, Giacomo},
  title     = {Faster but Fatter: Benchmarking Energy Efficiency of LLM-Generated Code Optimizations},
  booktitle = {Proceedings of the [Conference Name]},
  year      = {2025},
  publisher = {ACM/IEEE},
  address   = {[Location]},
  doi       = {[DOI]}
}
```

---

## Acknowledgments

- **Supervisors:** Prof. Silverio Martínez-Fernández, Dr. Vincenzo De Martino, Prof. Fabio Palomba
- **Institution:** Universitat Politècnica de Catalunya (UPC Barcelona)
- **Baseline Benchmark:** [SWE-Perf](https://github.com/anthropics/swe-perf)
- **Metrics Framework:** [Green Software Measurement Model (GSMM)](https://doi.org/10.1016/j.future.2024.01.001)

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

**Giacomo Colosio**  
📧 giacomo.colosio@estudiantat.upc.edu  
🏛️ Universitat Politècnica de Catalunya, Barcelona, Spain
