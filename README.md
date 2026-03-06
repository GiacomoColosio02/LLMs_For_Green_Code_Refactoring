# SWE-PERF-GREEN: Are Humans Greener than Language Models to Optimize Code?

**Replication package for the ICSME 2026 submission.**

> A Benchmarking Study of Energy Efficiency in LLM-Generated Code Optimizations

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

---

## Overview

This repository contains the replication package for the paper *"Are Humans Greener than Language Models to Optimize Code? A Benchmarking Study"*.

We introduce **SWE-PERF-GREEN**, extending the SWE-Perf benchmark with comprehensive energy measurement capabilities aligned with the **Green Software Measurement Model (GSMM)**. We evaluate two open-source 7B models — **Qwen2.5-Coder-7B** and **DeepSeek-R1-Distill-7B** — across 125 real-world Python optimization instances using 17 GSMM-aligned metrics.

### Key Findings

- LLM-generated optimizations consume **1.3–6.8× more energy** than human solutions despite achieving comparable speedup
- Context granularity (Oracle vs Realistic) has **no significant impact** on energy efficiency
- **Zero-Shot prompting achieves ~15% better energy efficiency** than Self-Collaboration
- Human-written optimizations remain **significantly more energy-efficient** than all LLM configurations

---

## Benchmark Summary

| Component | Description |
|-----------|-------------|
| **Instances** | 125 real-world optimization tasks from GitHub PRs |
| **Repositories** | 12 popular Python projects |
| **Metrics** | 17 GSMM-aligned (energy, power, carbon, efficiency) |
| **Models** | Qwen2.5-Coder-7B, DeepSeek-R1-Distill-7B |
| **Configurations** | 16 (2 models × 4 strategies × 2 context settings) |
| **Prompting Strategies** | Zero-Shot, Chain-of-Thought, Self-Collaboration, LDB |

---

## Repository Structure
```
├── data/
│   ├── raw/                    # Original immutable data
│   ├── processed/              # Final datasets for analysis
│   │   └── green/              # Green datasets (CSV + JSON per config)
│   └── external/               # Third-party data (grid intensities)
│
├── src/
│   ├── measurement/            # Energy measurement infrastructure
│   ├── prompt_templates/       # 4 prompting strategy templates
│   ├── llm_clients/            # vLLM inference clients
│   ├── patch_engine/           # Patch parsing and application
│   ├── data_processing/        # Data processing utilities
│   ├── analysis/               # Statistical analysis scripts
│   └── utils/                  # Configuration utilities
│
├── scripts/                    # Executable scripts
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_test_results_analysis.ipynb
│   ├── 03_swe_perf_green_analysis.ipynb
│   └── 04_results.ipynb
│
├── configs/                    # Configuration files
├── results/                    # Raw experiment results per model
├── reports/
│   ├── figures/                # Paper figures
│   └── tables/                 # LaTeX tables
│
├── docs/                       # Documentation
└── logs/                       # Experiment logs
```

---

## Reproduce Paper Results (No Hardware Required)
```bash
# Clone and setup
git clone <ANONYMOUS_REPO_URL>
cd LLMs_For_Green_Code_Refactoring
pip install -r requirements.txt

# Open analysis notebooks
jupyter notebook notebooks/03_swe_perf_green_analysis.ipynb
```

The pre-computed results are available in `data/processed/green/` and `results/` and can be analyzed without access to the measurement hardware.

---

## Run Energy Measurements (Requires Hardware)

To reproduce the full measurement pipeline, the following hardware is required:

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 4090 (24GB VRAM) |
| **Power Monitor** | NETIO PowerBOX 4KF |
| **CPU Energy** | Intel RAPL via EnergiBridge |
```bash
# Single instance measurement
python scripts/measure_instance.py \
    --instance scikit-learn__scikit-learn-12345 \
    --output data/raw/measurements/

# Batch measurement (all configurations)
python scripts/run_batch_experiments.py \
    --config configs/measurement_config.yaml
```

---

## GSMM Metrics (17 Total)

### Energy Metrics (10)

| Metric | Unit | Source |
|--------|------|--------|
| `cpu_energy` | J | Intel RAPL via EnergiBridge |
| `gpu_energy` | J | NVIDIA NVML |
| `system_energy` | J | NETIO PowerBOX 4KF |
| `total_energy` | J | Aggregated |
| `power_mean` | W | Calculated |
| `power_peak` | W | max(samples) |
| `carbon_emissions` | gCO₂eq | Energy × grid intensity |
| `energy_per_test` | J/test | Calculated |
| `energy_ratio` | — | E_LLM / E_human |
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

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
