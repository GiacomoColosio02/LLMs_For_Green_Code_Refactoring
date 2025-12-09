# GSMM Metrics Implementation for Green Code Refactoring
## Detailed Technical Specification

**Student:** Giacomo Colosio  
**Supervisors:** Prof. Silverio Martínez-Fernández, Dr. Vincenzo De Martino  
**Institution:** UPC Barcelona - ESSI Department  
**Date:** December 2024

---

## 1. Project Overview and Goals

This thesis project investigates the capability of **Large Language Models (LLMs)** to perform **green code refactoring** - i.e., optimizing software to reduce energy consumption and carbon emissions while maintaining functional correctness. The project is structured in three distinct phases:

### 1.1 Project Goals

#### ✅ **Phase 1: SWE-Perf Extension with GSMM Metrics** (COMPLETED)

The first phase extends the **SWE-Perf benchmark** with 17 sustainability metrics aligned with the **Green Software Measurement Model (GSMM)** proposed by the Green Software Foundation. This phase establishes a comprehensive measurement infrastructure capable of capturing both energy consumption (GREEN metrics) and resource efficiency (EFFICIENCY metrics) for software test execution.

**Deliverables:**
- 10 GREEN metrics for energy and carbon quantification
- 7 EFFICIENCY metrics for resource utilization analysis
- Measurement infrastructure deployed on high-performance server (gaissa.essi.upc.edu)
- Baseline measurements for 140 SWE-Perf instances

**Key Achievement:** 100% system energy coverage through wattmeter integration, overcoming the typical limitation of component-level measurements (GPU+CPU) that capture only ~85% of total system energy.

#### ⏳ **Phase 2: LLM Benchmarking** (PLANNED)

The second phase will evaluate four state-of-the-art LLMs across different prompting strategies:

**Models (tentative):**
- 2 proprietary models (e.g., GPT-4/5, Claude 3/4)
- 2 open-source models (e.g., Qwen2.5-Coder, Llama/CodeGemma)

**Prompting Strategies:**
- 2 single-turn approaches (direct optimization, role-based expert)
- 2 multi-turn approaches (self-collaboration, iterative refinement)

Each LLM will generate optimized versions of the 140 baseline instances, which will then be measured using the same 17 GSMM metrics established in Phase 1.

#### ⏳ **Phase 3: Comparative Analysis** (PLANNED)

The final phase will perform statistical analysis comparing:
- LLM-generated optimizations vs. human expert optimizations (HEAD commits)
- Performance across different models and prompting strategies
- Trade-offs between energy efficiency, execution time, and resource usage

**Research Questions:**
1. Can LLMs effectively optimize code for energy and carbon reduction?
2. Which LLM architectures and prompting strategies are most effective?
3. How do LLM optimizations compare to human expert optimizations?

---

## 2. GSMM Metrics: Detailed Implementation

The measurement system implements **17 metrics** divided into two categories aligned with the Green Software Measurement Model:

- **10 GREEN Metrics:** Energy consumption and carbon emissions
- **7 EFFICIENCY Metrics:** Resource utilization and execution performance

### 2.1 Measurement Architecture

The implementation uses a **multi-threaded monitoring architecture** to capture metrics from different system components simultaneously:
```
┌─────────────────────────────────────────────────────┐
│         EnergyMonitorGSMM (Orchestrator)            │
└─────────────────────────────────────────────────────┘
           │
           ├─→ WattmeterMonitorThread  (System-level power)
           ├─→ GPUMonitorThread        (GPU metrics)
           ├─→ SystemResourceTracker   (CPU, RAM metrics)
           └─→ CPUEnergyMonitor        (CPU energy via EnergiBridge)
```

**Key Design Principles:**
- **Baseline subtraction:** 5-second idle baseline measured before each test to isolate software-induced energy consumption
- **Thread-safe sampling:** Independent monitoring threads at 100ms intervals (10 Hz)
- **Statistical reliability:** 3 repetitions per test with mean aggregation
- **Non-intrusive measurement:** Overhead <1% of total execution time

---

## 3. GREEN Metrics (Energy & Carbon)

### 3.1 Core Energy Metrics (Component-Level, ~85% Coverage)

These metrics capture energy consumption from the two primary computational components: GPU and CPU. Together they represent approximately 85% of total system energy, excluding secondary components like RAM, storage, PSU efficiency losses, and motherboard.

---

#### 3.1.1 GPU Energy (`gpu_energy_joules`)

**Description:**  
Energy consumed by the GPU during test execution, measured in Joules (J). This metric is particularly important for computation-intensive workloads that leverage GPU acceleration.

**Implementation:**  
Using NVIDIA's pynvml (Python bindings for NVIDIA Management Library), we sample GPU power at 100ms intervals during test execution:
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Sample power at 100ms intervals
power_samples = []
while test_running:
    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
    power_watts = power_mw / 1000.0
    power_samples.append(power_watts)
    time.sleep(0.1)  # 100ms sampling

# Calculate energy
mean_power = sum(power_samples) / len(power_samples)
gpu_energy_joules = mean_power * duration_seconds
```

**Formula:**
$$E_{GPU} = \overline{P_{GPU}} \times \Delta t$$

Where:
- $E_{GPU}$ = GPU energy in Joules [J]
- $\overline{P_{GPU}}$ = Mean GPU power draw in Watts [W]
- $\Delta t$ = Test duration in seconds [s]

**Hardware:** NVIDIA GeForce RTX 4090 (24GB VRAM)

---

#### 3.1.2 CPU Energy (`cpu_energy_joules`)

**Description:**  
Energy consumed by the CPU during test execution, measured in Joules (J). This is the most critical metric for CPU-bound workloads and represents the base energy cost of all computation.

**Implementation:**  
Using **EnergiBridge**, an open-source tool that reads Intel RAPL (Running Average Power Limit) registers. RAPL provides hardware-level energy counters for CPU packages and DRAM:
```python
import subprocess

# EnergiBridge measures CPU energy for command execution
result = subprocess.run(
    ['sudo', 'energibridge', '-o', 'temp.csv', '--', 'pytest', 'test.py'],
    capture_output=True,
    check=True
)

# Parse EnergiBridge CSV output
df = pd.read_csv('temp.csv')
cpu_energy_joules = df['package-0'].sum() + df.get('package-1', 0).sum()
```

**Formula:**
$$E_{CPU} = \sum_{i=0}^{N} E_{package_i}$$

Where:
- $E_{CPU}$ = Total CPU energy in Joules [J]
- $E_{package_i}$ = Energy from CPU package $i$ (RAPL counter)
- $N$ = Number of CPU packages (typically 1-2)

**Hardware:** AMD Ryzen 9 7950X (16 cores, 32 threads)  
**Note:** EnergiBridge requires sudo access for RAPL register reads.

---

#### 3.1.3 Total Energy (`total_energy_joules`)

**Description:**  
Combined energy consumption from GPU and CPU components. This represents the **minimum measurable energy** for the computational workload.

**Implementation:**  
Simple aggregation of GPU and CPU measurements:
```python
total_energy_joules = gpu_energy_joules + cpu_energy_joules
```

**Formula:**
$$E_{total} = E_{GPU} + E_{CPU}$$

**Coverage Analysis:**  
While this metric captures the major computational components, it underestimates total system energy by ~15% because it excludes:
- RAM energy consumption
- Storage (SSD/HDD) I/O energy
- PSU (Power Supply Unit) efficiency losses (~10-15%)
- Motherboard, fans, and peripheral components

**Typical Values:** 85% of system-level energy measured by wattmeter

---

#### 3.1.4 Mean Power (`power_watts`)

**Description:**  
Average power consumption during test execution, measured in Watts (W). This metric provides an intensity measure independent of execution time.

**Implementation:**
```python
power_watts = total_energy_joules / duration_seconds
```

**Formula:**
$$P = \frac{E_{total}}{\Delta t}$$

Where:
- $P$ = Mean power in Watts [W]
- $E_{total}$ = Total energy in Joules [J]
- $\Delta t$ = Duration in seconds [s]

**Interpretation:** Higher power indicates more energy-intensive computation per unit time.

---

#### 3.1.5 Carbon Emissions (`carbon_grams`)

**Description:**  
Carbon dioxide equivalent (CO₂e) emissions from electricity consumption, measured in grams. This metric translates energy consumption into environmental impact using grid carbon intensity.

**Implementation:**
```python
# Convert energy to kWh
energy_kwh = total_energy_joules / 3_600_000

# Apply grid carbon intensity
carbon_grams = energy_kwh * grid_intensity_gCO2e_per_kWh
```

**Formula:**
$$C = \frac{E_{total}}{3,600,000} \times I_{grid}$$

Where:
- $C$ = Carbon emissions in grams CO₂e [gCO₂e]
- $E_{total}$ = Total energy in Joules [J]
- $I_{grid}$ = Grid carbon intensity in grams CO₂e per kWh [gCO₂e/kWh]

**Grid Intensities (2024 data):**
- **Spain (ESP):** 250 gCO₂e/kWh (server location)
- USA: 417 gCO₂e/kWh
- Germany: 311 gCO₂e/kWh
- France: 52 gCO₂e/kWh (high nuclear)
- UK: 233 gCO₂e/kWh

**Note:** This metric uses component-level energy (GPU+CPU), thus underestimating true carbon emissions by ~15%.

---

#### 3.1.6 Energy Efficiency (`energy_efficiency`)

**Description:**  
Ratio of useful computation to energy consumed. Lower values indicate higher efficiency (less energy per unit of work). This is a **dimensionless metric** for comparing implementations.

**Implementation:**
```python
# Normalized metric (0-1 range typically)
energy_efficiency = total_energy_joules / baseline_energy_joules
```

**Formula:**
$$\eta = \frac{E_{implementation}}{E_{baseline}}$$

Where:
- $\eta$ = Energy efficiency ratio (dimensionless)
- $E_{implementation}$ = Energy of current implementation [J]
- $E_{baseline}$ = Energy of baseline implementation [J]

**Interpretation:**
- $\eta < 1.0$ → More efficient than baseline
- $\eta = 1.0$ → Same efficiency as baseline
- $\eta > 1.0$ → Less efficient than baseline

---

### 3.2 System Energy Metrics (Wattmeter, 100% Coverage)

These metrics capture **complete system-level energy** by measuring wall power consumption through a hardware wattmeter. This approach provides 100% coverage of all system components and is the **ground truth** for total energy consumption.

---

#### 3.2.1 System Energy (`system_energy_joules`)

**Description:**  
Complete system energy consumption measured at the wall socket, including all components: CPU, GPU, RAM, storage, PSU losses, motherboard, fans, and peripherals. This is the **most accurate energy metric**.

**Implementation:**  
Using a **NETIO PowerBOX 4KF** hardware wattmeter connected to the server's power outlet:
```python
import requests
import threading
import time

class WattmeterMonitorThread:
    def __init__(self, ip='10.4.60.25', output_id=1):
        self.ip = ip
        self.output_id = output_id
        self.power_samples = []
        
    def sample_power(self):
        """Sample power at 100ms intervals"""
        while self.running:
            response = requests.get(
                f'http://{self.ip}/netio.json',
                timeout=5
            )
            data = response.json()
            power_watts = data['Outputs'][self.output_id - 1]['Load']
            self.power_samples.append(power_watts)
            time.sleep(0.1)  # 100ms sampling
    
    def calculate_energy(self):
        mean_power = sum(self.power_samples) / len(self.power_samples)
        duration = len(self.power_samples) * 0.1
        return mean_power * duration
```

**Formula:**
$$E_{system} = \overline{P_{wall}} \times \Delta t$$

Where:
- $E_{system}$ = System energy in Joules [J]
- $\overline{P_{wall}}$ = Mean wall power in Watts [W]
- $\Delta t$ = Duration in seconds [s]

**Hardware:** NETIO PowerBOX 4KF (IP: 10.4.60.25, UPC network)  
**Coverage:** 100% (all system components)

---

#### 3.2.2 System Mean Power (`system_power_mean_watts`)

**Description:**  
Average power consumption measured at the wall socket during test execution. This is the **true average power draw** of the complete system.

**Implementation:**
```python
system_power_mean_watts = sum(power_samples) / len(power_samples)
```

**Formula:**
$$\overline{P_{system}} = \frac{1}{N} \sum_{i=1}^{N} P_i$$

Where:
- $\overline{P_{system}}$ = Mean system power [W]
- $P_i$ = Power sample $i$ [W]
- $N$ = Number of samples

**Typical Values:** 80-120W for test execution on our hardware

---

#### 3.2.3 System Peak Power (`system_power_peak_watts`)

**Description:**  
Maximum instantaneous power consumption during test execution. This metric captures power spikes that may not be visible in the mean value and is important for power provisioning and thermal management.

**Implementation:**
```python
system_power_peak_watts = max(power_samples)
```

**Formula:**
$$P_{peak} = \max_{i \in [1,N]} P_i$$

**Use Cases:**
- Data center power provisioning
- Thermal design validation
- Power surge analysis

---

#### 3.2.4 System Carbon Emissions (`carbon_grams_system`)

**Description:**  
Carbon dioxide equivalent emissions calculated from **complete system energy** (100% coverage). This is the **most accurate carbon metric** as it includes all components.

**Implementation:**
```python
energy_kwh = system_energy_joules / 3_600_000
carbon_grams_system = energy_kwh * grid_intensity
```

**Formula:**
$$C_{system} = \frac{E_{system}}{3,600,000} \times I_{grid}$$

**Comparison with Component Carbon:**
- Component-level carbon (`carbon_grams`): ~85% of true emissions
- System-level carbon (`carbon_grams_system`): 100% of true emissions
- Difference: ~15% underestimation in component-level measurement

**Example:**
```
Component: 0.00496 gCO₂e (GPU+CPU only)
System:    0.00583 gCO₂e (complete system)
Missing:   0.00087 gCO₂e (17.5% underestimation)
```

---

## 4. EFFICIENCY Metrics (Resource Utilization)

These metrics quantify how efficiently the software uses system resources during execution. They are essential for understanding performance bottlenecks and resource constraints.

---

### 4.1 Execution Performance

#### 4.1.1 Duration (`duration_seconds`)

**Description:**  
Total execution time of the test in seconds. This is the fundamental performance metric and is used to normalize energy consumption into power.

**Implementation:**
```python
import time

start_time = time.time()
# Execute test
run_pytest(test_command)
end_time = time.time()

duration_seconds = end_time - start_time
```

**Formula:**
$$\Delta t = t_{end} - t_{start}$$

**Importance:** Faster execution generally reduces energy consumption, but the relationship is not always linear due to power-time trade-offs.

---

### 4.2 CPU Utilization Metrics

#### 4.2.1 CPU Mean Usage (`cpu_usage_mean_percent`)

**Description:**  
Average CPU utilization across all cores during test execution, expressed as a percentage (0-100%). This indicates how effectively the software uses available CPU resources.

**Implementation:**
```python
import psutil
import threading

cpu_samples = []
def monitor_cpu():
    while running:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_samples.append(cpu_percent)

cpu_usage_mean_percent = sum(cpu_samples) / len(cpu_samples)
```

**Formula:**
$$\overline{CPU} = \frac{1}{N} \sum_{i=1}^{N} CPU_i$$

Where:
- $\overline{CPU}$ = Mean CPU usage [%]
- $CPU_i$ = CPU percentage sample $i$
- $N$ = Number of samples

**Interpretation:**
- Low (<10%): I/O-bound or idle workload
- Medium (10-50%): Mixed workload
- High (>50%): CPU-bound computation

---

#### 4.2.2 CPU Peak Usage (`cpu_usage_peak_percent`)

**Description:**  
Maximum CPU utilization observed during test execution. This captures brief periods of intense computation that may not be visible in the mean.

**Implementation:**
```python
cpu_usage_peak_percent = max(cpu_samples)
```

**Formula:**
$$CPU_{peak} = \max_{i \in [1,N]} CPU_i$$

**Use Case:** Identify computation spikes and parallelization effectiveness

---

### 4.3 Memory Metrics

#### 4.3.1 RAM Peak Usage (`ram_usage_peak_mb`)

**Description:**  
Maximum RAM consumption during test execution, measured in megabytes (MB). This metric is critical for understanding memory requirements and potential swap usage.

**Implementation:**
```python
import psutil

ram_samples = []
def monitor_ram():
    while running:
        mem = psutil.virtual_memory()
        ram_mb = mem.used / (1024 * 1024)
        ram_samples.append(ram_mb)
        time.sleep(0.1)

ram_usage_peak_mb = max(ram_samples)
```

**Formula:**
$$RAM_{peak} = \max_{i \in [1,N]} \left( \frac{RAM_{used,i}}{1024^2} \right)$$

**System Configuration:** 32GB DDR5 total available  
**Warning Threshold:** >28GB (approaching system limits)

---

### 4.4 GPU Utilization Metrics

#### 4.4.1 GPU Mean Usage (`gpu_usage_mean_percent`)

**Description:**  
Average GPU compute utilization during test execution, expressed as a percentage (0-100%). This measures how effectively the software leverages GPU acceleration.

**Implementation:**
```python
import pynvml

gpu_samples = []
def monitor_gpu():
    while running:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_samples.append(util.gpu)
        time.sleep(0.1)

gpu_usage_mean_percent = sum(gpu_samples) / len(gpu_samples)
```

**Formula:**
$$\overline{GPU} = \frac{1}{N} \sum_{i=1}^{N} GPU_i$$

**Interpretation:**
- 0%: No GPU usage (CPU-only workload)
- Low (1-20%): Minimal GPU acceleration
- High (>50%): GPU-accelerated computation

---

#### 4.4.2 GPU Peak Usage (`gpu_usage_peak_percent`)

**Description:**  
Maximum GPU compute utilization during test execution.

**Implementation:**
```python
gpu_usage_peak_percent = max(gpu_samples)
```

**Formula:**
$$GPU_{peak} = \max_{i \in [1,N]} GPU_i$$

---

#### 4.4.3 GPU Memory Peak (`gpu_memory_peak_mb`)

**Description:**  
Maximum GPU memory (VRAM) usage during test execution, measured in megabytes. This is critical for GPU-accelerated workloads as it determines whether computations fit in VRAM or require expensive host-device transfers.

**Implementation:**
```python
import pynvml

gpu_mem_samples = []
def monitor_gpu_memory():
    while running:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_mb = mem_info.used / (1024 * 1024)
        gpu_mem_samples.append(mem_mb)
        time.sleep(0.1)

gpu_memory_peak_mb = max(gpu_mem_samples)
```

**Formula:**
$$VRAM_{peak} = \max_{i \in [1,N]} \left( \frac{VRAM_{used,i}}{1024^2} \right)$$

**Hardware:** RTX 4090 with 24GB VRAM  
**Warning Threshold:** >20GB (approaching VRAM limits)

---

## 5. Measurement Process Summary

### 5.1 Per-Instance Measurement Flow

For each SWE-Perf instance, the following process is executed for both BASE (pre-optimization) and HEAD (post-optimization) commits:
```
1. Repository Setup
   ├── Clone repository from GitHub
   ├── Checkout specific commit (BASE or HEAD)
   └── Install dependencies in virtual environment

2. Baseline Measurement (5 seconds)
   ├── Start all monitoring threads
   ├── Wait 5 seconds (idle system)
   ├── Stop all monitoring threads
   └── Store baseline metrics

3. Test Execution (3 repetitions)
   For each repetition:
   ├── Start wattmeter monitoring (first)
   ├── Start GPU monitoring thread
   ├── Start CPU/RAM monitoring thread
   ├── Execute test with EnergiBridge
   ├── Stop CPU/RAM monitoring (first)
   ├── Stop GPU monitoring
   ├── Stop wattmeter monitoring (last)
   ├── Subtract baseline from measurements
   └── Calculate all 17 metrics

4. Result Aggregation
   ├── Average across 3 repetitions
   ├── Calculate mean, std, min, max
   └── Save to JSON file
```

### 5.2 Baseline Subtraction

All energy and resource metrics undergo **baseline subtraction** to isolate software-induced consumption:

$$M_{net} = M_{measured} - M_{baseline}$$

Where:
- $M_{net}$ = Net metric value (software-induced only)
- $M_{measured}$ = Raw measured value during test
- $M_{baseline}$ = Idle system value (5-second average)

This ensures measurements reflect **only the energy/resources consumed by the test execution**, excluding background system processes and idle power.

### 5.3 Statistical Reliability

Each test is executed **3 times** to account for measurement variability. The final metrics reported are:

- **Mean:** Primary value used for analysis
- **Standard deviation:** Measure of variability
- **Min/Max:** Range of observed values

Repetitions with >20% deviation from the mean are flagged for manual review.

---

## 6. Output Format

Each measured instance produces a JSON file with the following structure:
```json
{
  "instance_id": "astropy__astropy-16065",
  "base_commit": "48a792f9",
  "head_commit": "7eac388c",
  "measurement_timestamp": "2024-12-09T10:30:45",
  
  "base_measurements": {
    "tests": [
      {
        "test_name": "test_distribution[False-True-log]",
        "measurements": [
          {
            "repetition": 1,
            
            "green_metrics": {
              "gpu_energy_joules": 26.23,
              "cpu_energy_joules": 45.13,
              "total_energy_joules": 71.36,
              "power_watts": 46.03,
              "carbon_grams": 0.00496,
              "energy_efficiency": 0.01401,
              "system_energy_joules": 84.00,
              "system_power_mean_watts": 93.33,
              "system_power_peak_watts": 96.00,
              "carbon_grams_system": 0.00583
            },
            
            "efficiency_metrics": {
              "duration_seconds": 1.55,
              "cpu_usage_mean_percent": 9.36,
              "cpu_usage_peak_percent": 61.1,
              "ram_usage_peak_mb": 3237.10,
              "gpu_usage_mean_percent": 0.0,
              "gpu_usage_peak_percent": 0.0,
              "gpu_memory_peak_mb": 574.19
            }
          }
        ]
      }
    ]
  },
  
  "head_measurements": {
    "tests": [ /* same structure */ ]
  }
}
```

---

## 7. Validation and Quality Assurance

### 7.1 Measurement Validation

A dedicated verification script ensures all measurements are valid:
```bash
python3 scripts/verify_measurements.py \
    --output-dir data/raw/measurements \
    --verbose
```

**Validation Checks:**
- All 17 metrics present in each measurement
- No negative values for energy/duration
- Reasonable ranges (e.g., duration >0.01s, energy >0J)
- System energy >= component energy (sanity check)
- Baseline properly subtracted

### 7.2 Known Limitations

**Expected Failure Rate:** 15-30% of instances may fail due to:
1. **Dependency incompatibilities:** Old packages incompatible with Python 3.12
2. **Missing tests:** Tests removed/renamed in newer commits
3. **Build failures:** Missing system libraries
4. **Network issues:** Wattmeter connection timeouts

These failures are **expected and acceptable** as they reflect the realistic challenges of measuring legacy codebases. The ~70-85% success rate provides sufficient data for statistical analysis.

### 7.3 Energy Coverage Comparison

**Component-Level (GPU + CPU):**
- Coverage: ~85% of total system energy
- Missing: RAM, storage, PSU losses, motherboard
- Use case: Understanding computational workload distribution

**System-Level (Wattmeter):**
- Coverage: 100% of total system energy
- Includes: All components + overhead
- Use case: Ground truth for total energy and carbon

**Example Comparison:**
```
GPU Energy:       26.23 J  (31.2%)
CPU Energy:       45.13 J  (53.8%)
Component Total:  71.36 J  (85.0%)
─────────────────────────────────
System Energy:    84.00 J  (100%)
Missing Components: 12.64 J  (15.0%)
```

The 15% gap represents energy consumed by secondary components that would otherwise be unaccounted for without wattmeter measurement.

---

## 8. Next Steps: Phase 2 Implementation

With the GSMM measurement infrastructure complete, the next phase will:

1. **Select and configure 4 LLM models** (2 proprietary + 2 open-source)
2. **Design 4 prompting strategies** emphasizing green optimization
3. **Generate LLM-optimized code** for all 140 instances
4. **Measure LLM versions** using the same 17 metrics
5. **Compare LLM vs. human** optimizations statistically

**Expected Dataset:**
- 140 instances × (1 BASE + 1 HEAD + 8 LLM versions) = 1,400 measurements
- 1,400 × 17 metrics = 23,800 data points for analysis

---

## 9. References

1. **Green Software Measurement Model (GSMM)** - Green Software Foundation
2. **SWE-Perf Benchmark** - Software Engineering Performance Benchmark
3. **Intel RAPL** - Running Average Power Limit Documentation
4. **NVIDIA NVML** - NVIDIA Management Library API
5. **EnergiBridge** - Open-source RAPL Wrapper
6. **NETIO PowerBOX 4KF** - Hardware Wattmeter Documentation

---

**Document Version:** 1.0  
**Last Updated:** December 9, 2024  
**Status:** Phase 1 Complete | Phase 2 In Planning
