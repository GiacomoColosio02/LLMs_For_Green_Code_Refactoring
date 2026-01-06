# Phase 1: Green Software Dataset Creation

## Overview

This phase extends the SWE-Perf benchmark with comprehensive energy measurement capabilities to assess the environmental impact of code optimizations. The goal is to create a dataset that enables evaluation of whether performance-optimized code is also more energy-efficient ("green").

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SWE-PERF ORIGINAL                            │
│                      140 instances                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼ Test validation & filtering
┌─────────────────────────────────────────────────────────────────┐
│                    REDUCED DATASET                              │
│                 131 instances (valid tests)                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼ Energy measurement (k repetitions)
                          │   • EnergiBridge (CPU)
                          │   • NVML (GPU)  
                          │   • Wattmeter (System)
┌─────────────────────────────────────────────────────────────────┐
│                   RAW MEASUREMENTS                              │
│           131 folders with measurements.json                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼ Statistical aggregation (mean, std, min, max)
                          │   + Failed test cleanup
┌─────────────────────────────────────────────────────────────────┐
│                    GREEN DATASET                                │
│       131 instances │ 813 tests │ 52 metrics per test           │
│                                                                 │
│     For each test: base (pre-opt) vs head (post-opt)            │
│     13 metrics × 4 aggregations = 52 values                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Original Dataset → Reduced Dataset

### Input
- `swe_perf_original_20251124.json` (140 instances)

### Process

The reduction process ensures we only work with instances that have valid, executable tests:

1. **Test Validation**: For each instance, execute all tests in `efficiency_test` to verify which ones actually pass
2. **Instance Removal**: Remove instances where no tests pass successfully
3. **Test Filtering**: Within each instance, remove tests that fail from the `efficiency_test` list
4. **Field Cleanup**: Remove unnecessary fields:
   - `problem_statement_oracle`
   - `problem_statement_realistic`
   - `duration_changes`

### Output
- `swe_perf_reduced.json` (131 instances with valid tests)

### Statistics
| Metric | Count |
|--------|-------|
| Original instances | 140 |
| Reduced instances | 131 |
| Removed instances | 9 |
| Retention rate | 93.6% |

---

## Step 2: Reduced Dataset → Raw Measurements

### Input
- `swe_perf_reduced.json` (131 instances)

### Measurement Process

For each instance, the `measure_instance.py` script performs:

#### 2.1 Repository Setup
```
1. Clone repository from GitHub
2. Checkout base_commit (code BEFORE optimization)
3. Setup environment (venv with correct dependencies)
```

#### 2.2 Environment Configuration

Different repositories require specific configurations:

| Repository | Python Version | Special Requirements |
|------------|---------------|---------------------|
| astropy | 3.10 | Build isolation, pytest-astropy |
| matplotlib | 3.11 | Build isolation, pyparsing<3.2 |
| sphinx | 3.9 | Build isolation (flit) |
| scikit-learn | 3.9 | Conda for older versions, joblib fix |
| xarray | 3.10 | pandas<2.1 |
| Others | 3.9 | Standard setup |

#### 2.3 Baseline Measurement
- 5 seconds of idle measurement for system calibration
- Captures baseline CPU usage, RAM, power consumption

#### 2.4 Test Execution & Measurement

For each test in `efficiency_test`:

```
┌─────────────────────────────────────────────────────────┐
│                    TEST EXECUTION                        │
├─────────────────────────────────────────────────────────┤
│  1. Start all monitors simultaneously:                   │
│     • EnergiBridge (CPU energy via RAPL)                │
│     • NVML (GPU energy, temperature, utilization)       │
│     • Wattmeter (System-level power)                    │
│     • Resource Monitor (CPU%, RAM usage)                │
│                                                         │
│  2. Execute: pytest {test_name} -v                      │
│                                                         │
│  3. Stop monitors and collect metrics                   │
│                                                         │
│  4. Repeat k times for statistical stability            │
└─────────────────────────────────────────────────────────┘
```

#### 2.5 Head Commit Measurement
- Checkout `head_commit` (code AFTER optimization)
- Repeat the entire measurement process

### Metrics Collected

| Category | Metrics | Source |
|----------|---------|--------|
| **GREEN** | cpu_energy_joules | EnergiBridge (RAPL) |
| | gpu_energy_joules | NVML |
| | total_energy_joules | Wattmeter |
| | power_watts | Wattmeter |
| | carbon_grams | Calculated (grid intensity) |
| | energy_efficiency | ops/joule |
| **EFFICIENCY** | duration_seconds | Timer |
| | cpu_usage_mean_percent | Resource Monitor |
| | cpu_usage_peak_percent | Resource Monitor |
| | ram_usage_mean_mb | Resource Monitor |
| | ram_usage_peak_mb | Resource Monitor |
| | gpu_temperature_mean_celsius | NVML |
| | gpu_temperature_peak_celsius | NVML |

### Output
- `data/raw/measurements/{instance_id}/measurements.json`

---

## Step 3: Raw Measurements → Green Dataset

### Input
- `swe_perf_reduced.json`
- `data/raw/measurements/*/measurements.json`

### Aggregation Process

The `measure_and_create_green_dataset.py` script performs:

#### 3.1 Validation
For each test, verify:
- `return_code == 0` (test passed) in both base and head
- Valid measurements exist for all metrics

#### 3.2 Statistical Aggregation

From k repetitions, calculate 4 statistical measures:

| Aggregation | Description |
|-------------|-------------|
| `mean` | Average across k repetitions |
| `std` | Standard deviation |
| `min` | Minimum value |
| `max` | Maximum value |

#### 3.3 Final Structure

```json
{
  "test_name": {
    "base": {
      "cpu_energy_joules_mean": 45.2,
      "cpu_energy_joules_std": 2.1,
      "cpu_energy_joules_min": 43.0,
      "cpu_energy_joules_max": 47.5,
      "gpu_energy_joules_mean": 12.3,
      // ... 48 more metrics
    },
    "head": {
      "cpu_energy_joules_mean": 38.1,
      "cpu_energy_joules_std": 1.8,
      // ... comparison metrics
    }
  }
}
```

#### 3.4 Cleanup
- Remove tests without valid measurements in both base and head
- Align `efficiency_test` list with `green_metrics`
- Auto-update reduced dataset to reflect valid tests

### Output
- `swe_perf_green_k1.json` (k=1 repetitions)
- `swe_perf_green_k3.json` (k=3 repetitions)

---

## Final Dataset Structure

### Metadata (Root Level)

| Field | Description |
|-------|-------------|
| `name` | Dataset name: "SWE-Perf Green Extended" |
| `green_metrics` | List of 6 green metrics |
| `efficiency_metrics` | List of 7 efficiency metrics |
| `aggregations` | ["mean", "std", "min", "max"] |
| `instance_count` | 131 instances |
| `creation_date` | Timestamp |

### Instance Structure

| Field | Description |
|-------|-------------|
| `repo` | Repository (e.g., "matplotlib/matplotlib") |
| `instance_id` | Unique identifier |
| `base_commit` | Commit hash BEFORE optimization |
| `head_commit` | Commit hash AFTER optimization |
| `version` | Repository version |
| `efficiency_test` | List of valid tests |
| `patch` | Code diff of the optimization |
| `green_metrics` | Measurements per test (base vs head) |
| `_green_metadata` | Validation info |

### Metrics Summary

**52 metrics per test** (13 base metrics × 4 aggregations):

#### GREEN Metrics (6 × 4 = 24)
- `cpu_energy_joules_{mean,std,min,max}`
- `gpu_energy_joules_{mean,std,min,max}`
- `total_energy_joules_{mean,std,min,max}`
- `power_watts_{mean,std,min,max}`
- `carbon_grams_{mean,std,min,max}`
- `energy_efficiency_{mean,std,min,max}`

#### EFFICIENCY Metrics (7 × 4 = 28)
- `duration_seconds_{mean,std,min,max}`
- `cpu_usage_mean_percent_{mean,std,min,max}`
- `cpu_usage_peak_percent_{mean,std,min,max}`
- `ram_usage_mean_mb_{mean,std,min,max}`
- `ram_usage_peak_mb_{mean,std,min,max}`
- `gpu_temperature_mean_celsius_{mean,std,min,max}`
- `gpu_temperature_peak_celsius_{mean,std,min,max}`

---

## Hardware Infrastructure

| Component | Tool | Description |
|-----------|------|-------------|
| **CPU** | EnergiBridge | RAPL interface for CPU energy |
| **GPU** | NVIDIA RTX 4090 + NVML | GPU energy, temperature, utilization |
| **System** | NETIO PowerBOX 4KF | Wall power measurement (100% coverage) |
| **Server** | GAISSA Server (UPC) | Dedicated measurement environment |

### Measurement Coverage

```
┌────────────────────────────────────────────────────────────┐
│                    SYSTEM BOUNDARY                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              NETIO Wattmeter (100%)                  │  │
│  │  ┌─────────────────┐    ┌─────────────────────────┐  │  │
│  │  │   CPU (RAPL)    │    │   GPU (NVML)            │  │  │
│  │  │   EnergiBridge  │    │   Energy + Temperature  │  │  │
│  │  └─────────────────┘    └─────────────────────────┘  │  │
│  │                                                      │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │   Resource Monitor (CPU%, RAM)                  │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## Final Statistics

| Metric | Value |
|--------|-------|
| Total instances | 131 |
| Total tests | 813 |
| Metrics per test | 52 |
| Repositories covered | 9 |
| Success rate | 100% |

### Repositories in Dataset

| Repository | Instances | Tests |
|------------|-----------|-------|
| pydata/xarray | 53 | ~350 |
| scikit-learn/scikit-learn | 32 | ~180 |
| sympy/sympy | 19 | ~100 |
| astropy/astropy | 12 | ~25 |
| mwaskom/seaborn | 4 | ~10 |
| sphinx-doc/sphinx | 4 | ~15 |
| pylint-dev/pylint | 3 | ~8 |
| matplotlib/matplotlib | 2 | ~15 |
| psf/requests | 2 | ~5 |

---

## Carbon Intensity Calculation

Carbon emissions are calculated using grid intensity data:

```
carbon_grams = total_energy_joules × grid_intensity / 3600
```

Where:
- `grid_intensity` = 250 gCO2e/kWh (Spain average)
- Division by 3600 converts from Joules to kWh

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `measure_instance.py` | Measure a single instance |
| `measure_and_create_green_dataset.py` | Full pipeline: measure all + create dataset |
| `reduced_dataset.py` | Create reduced dataset from original |
| `download_sweperf.py` | Download SWE-Perf dataset |

### Usage

```bash
# Measure single instance
python scripts/measure_instance.py --instance astropy__astropy-16065 --country ESP

# Run full pipeline
python scripts/measure_and_create_green_dataset.py --country ESP

# Create reduced dataset from scratch
python scripts/reduced_dataset.py
```

---

## Quality Assurance

### Test Validation
- Each test must pass (`return_code == 0`) in both base and head commits
- Tests failing in either commit are excluded

### Measurement Stability
- Multiple repetitions (k=1 or k=3) for statistical reliability
- Baseline measurement for system calibration
- Aggregated statistics (mean, std, min, max) capture variability

### Auto-Cleanup
- Failed instances automatically removed from reduced dataset
- Failed tests automatically removed from instance test lists
- Dataset remains synchronized across all files
