# LLMs For Green Code Refactoring

Benchmarking Large Language Models for Green Code Refactoring using SWE-Perf extended with GSMM metrics.

## 🎯 Project Overview

### Metrics

**Efficiency Metrics (Resource-Oriented)**
- `cpu_usage_mean` (%) - Average CPU utilization
- `cpu_usage_peak` (%) - Peak CPU utilization  
- `ram_usage_peak` (MB) - Peak memory consumption
- `duration` (s) - Execution time

**Greenness Metrics (Energy-Oriented)**
- `energy_consumption` (J) - Total energy consumed
- `mean_power_draw` (W) - Average power draw
- `energy_efficiency` (tests/J) - Useful work per energy
- `carbon_emissions` (gCO₂e) - Carbon footprint

## Repository Structure
```
LLMs_For_Green_Code_Refactoring/
├── data/
│   ├── original/          # SWE-Perf original dataset
│   ├── raw/               # Raw measurement data
│   └── processed/         # Extended dataset with metrics
├── src/
│   ├── measurement/       # Metric collection
│   ├── data_processing/   # Data processing
│   ├── analysis/          # Statistical analysis
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
├── configs/               # Configuration files
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
└── results/               # Analysis results
```

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt
# Measure single instance
python scripts/measure_instance.py --instance astropy__astropy-16065
```

## Author

**Giacomo Colosio**  
Master's Student @ UPC Barcelona  
Supervisors: Prof. Silverio Martínez-Fernández, Dr. Vincenzo De Martino

## Status

- [ ] Dataset collection
- [ ] Measurement pipeline
- [ ] LLM evaluation
- [ ] Analysis & results