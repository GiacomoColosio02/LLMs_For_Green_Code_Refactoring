# LLMs For Green Code Refactoring

Benchmarking Large Language Models for Green Code Refactoring using SWE-Perf extended with GSMM-aligned sustainability metrics.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
  - [Starting Point: SWE-Perf Dataset](#starting-point-swe-perf-dataset)
  - [Step 1: Adding New Metrics](#step-1-adding-new-metrics)
  - [Step 2: Modifying LLM Usage](#step-2-modifying-llm-usage)
- [Repository Structure](#repository-structure)
- [Implementation Status](#implementation-status)
- [Quick Start](#quick-start)
- [LLM Models](#llm-models)
- [Measurement Infrastructure](#measurement-infrastructure)
- [Thesis Information](#thesis-information)

---

## Project Overview

This thesis project evaluates the ability of **state-of-the-art Large Language Models** to perform **green code refactoring and optimization**. The project extends the SWE-Perf benchmark with comprehensive energy and carbon measurements, evaluating **6 LLMs** across **140 instances** with **26 versions each** (3,640 total measurements).

### Starting Point: SWE-Perf Dataset

The **SWE-Perf dataset** provides real-world performance optimization instances from GitHub pull requests. Each instance contains:

| Column | Type | Description |
|--------|------|-------------|
| `repo` | string | GitHub repository identifier (owner/name) |
| `instance_id` | string | Formatted ID: `repo_owner__repo_name-PR-number` |
| `patch` | string | Gold patch that resolved the performance issue |
| `test_patch` | string | Test file patch from the solution PR |
| `base_commit` | string | Commit hash before optimization |
| `head_commit` | string | Commit hash after optimization (human solution) |
| `created_at` | string | PR creation date |
| `version` | string | Installation version for evaluation |
| `duration_changes` | List[dict] | Runtime measurements (base vs head) |
| `efficiency_test` | List[string] | Performance-related unit tests |
| `patch_functions` | string | Functions modified in code patch |
| `problem_statement_oracle` | Dict[list] | Complete problem description (Oracle mode) |
| `test_functions` | string | Functions modified in unit tests |
| `problem_statement_realistic` | Dict[list] | Limited problem description (Realistic mode) |
| `human_performance` | float | Human performance baseline metrics |

---

## Step 1: Adding New Metrics

We extend SWE-Perf with **8 new sustainability metrics** aligned with the **Green Software Measurement Model (GSMM)**:

### 1. Efficiency Metrics (Resource-Oriented)

| Metric | Unit | Tool | Rationale |
|--------|------|------|-----------|
| **`duration_seconds`** | s | `time.time()` | Already in SWE-Perf, ensures backward compatibility |
| **`cpu_usage_mean_percent`** | % | `psutil.cpu_percent()` | CPU is primary energy consumer, direct correlation |
| **`cpu_usage_peak_percent`** | % | `max(cpu_samples)` | Identifies computational bursts, hardware sizing |
| **`ram_usage_peak_mb`** | MB | `psutil.memory_info().rss` | Hardware obsolescence prevention, memory leak detection |

**Sampling Configuration:**
- Sampling rate: 0.1s (10 Hz)
- Trade-off: <1% measurement overhead vs. sufficient granularity

### 2. Greenness Metrics (Energy-Oriented)

| Metric | Unit | Tool | Calculation | Rationale |
|--------|------|------|-------------|-----------|
| **`energy_consumption_joules`** | J | CodeCarbon (Intel RAPL) | ∫ P(t) dt | Primary energy impact metric |
| **`mean_power_draw_watts`** | W | Derived | E / duration | Computational intensity, CPU load indicator |
| **`carbon_emissions_grams`** | gCO₂e | CodeCarbon | E × grid_intensity | Final environmental impact, SDG alignment |
| **`energy_efficiency_tests_per_joule`** | tests/J | Calculated | useful_work / E | Normalized comparison, GSMM efficiency factor |

**Energy Configuration:**
- Grid intensity: 250 gCO₂e/kWh (Spain, configurable by location)
- Baseline subtraction: All metrics measure software-induced changes (test_measurement - baseline_measurement)

---

## Step 2: Modifying LLM Usage

### 2.1 What We Want to Evaluate

We generate **24 optimized versions per instance** across a **3-dimensional design space**:

**Dimensions:**
1. **Problem Statement Type** (2): Oracle vs Realistic
2. **Prompt Strategy** (2): Zero-Shot vs Self-Collaboration
3. **LLM Model** (6): GPT-5, Claude Opus 4.5, Gemini 3 Pro, Qwen2.5-Coder-32B, Llama 4 Maverick, Gemma 3 27B

**Evaluation Matrix:**

| Problem Type | Prompt Strategy | Models | Versions |
|--------------|----------------|--------|----------|
| **ORACLE** | Zero-Shot | 6 models | 6 |
| **ORACLE** | Self-Collaboration | 6 models | 6 |
| **REALISTIC** | Zero-Shot | 6 models | 6 |
| **REALISTIC** | Self-Collaboration | 6 models | 6 |
| | | **Total:** | **24** |

**Per Instance:** 24 LLM-generated + 2 baseline (BASE + HEAD) = **26 versions**  
**Total Project:** 140 instances × 26 versions = **3,640 measurements**

---

### 2.2 Oracle vs Realistic Problem Statements

The key difference lies in **what information the LLM receives**:

| Aspect | Oracle (File-Level) | Realistic (Repo-Level) |
|--------|---------------------|------------------------|
| **Dataset Column Used** | `problem_statement_oracle` | `problem_statement_realistic` |
| **Target Functions** | `patch_functions` (exact functions modified by humans) | `test_functions` (functions executed during tests) |
| **Code Context** | Only files containing target functions | Entire repository code |
| **Description** | Complete, detailed performance issue | Limited, partial description |
| **LLM Task** | Optimize given functions (guided) | Find AND optimize bottlenecks (autonomous) |
| **Realism** | Idealized scenario (low difficulty) | Real-world scenario (high difficulty) |

**Oracle Mode:**
- LLM knows exactly which functions to optimize
- Has privileged information (functions from `patch_functions`)
- Simulates perfect developer intuition
- Measures pure code generation capability

**Realistic Mode:**
- LLM must navigate entire repository
- Only knows which functions are tested (`test_functions`)
- Must identify optimization targets autonomously
- Simulates real developer workflow
- Measures end-to-end autonomous capability

---

### 2.3 New Prompting Strategies

To evaluate LLMs' ability to understand different collaboration patterns, we implement **two prompting strategies**, both adapted for **green software engineering**.

#### Strategy 1: Zero-Shot (Single-Turn)

**Approach:** Direct, single-turn optimization request

**Structure:**
```
System Prompt: Expert software engineer specializing in performance + green optimization
Problem Statement: [Oracle or Realistic context]
Task: Optimize code to reduce time, memory, energy, and carbon
Output: Unified diff patch
```

**Green Optimization Focus:**
- Explicit mention of energy consumption reduction
- Carbon emissions minimization as goal
- Energy efficiency metrics as success criteria

---

#### Strategy 2: Self-Collaboration (Multi-Turn)

**Approach:** Structured conversation with specialized roles

**Three-Turn System:**

| Turn | Role | Task | Output |
|------|------|------|--------|
| **1** | Performance Analyst | Identify bottlenecks, analyze complexity | Analysis report |
| **2** | Code Optimizer | Generate optimized code based on analysis | Code patch |
| **3** | Quality Validator | Validate correctness and performance | Final approved patch |

**Green Adaptation:**
Each role receives green-specific instructions:
- **Analyst**: Identify energy hotspots, resource waste
- **Optimizer**: Prioritize energy-efficient algorithms, minimize allocations
- **Validator**: Verify energy improvements, check carbon impact

**Rationale:**
- Self-collaboration simulates team code review process
- Tests LLM's ability to reason through optimization systematically
- Provides intermediate artifacts (analysis) for interpretability

---

**Note on Prompts:**
The focus of this project is **not** on prompt engineering perfection, but on establishing a **fair, reproducible baseline** for comparing LLM capabilities. The prompts are designed to be:
- Clear and unambiguous
- Consistent across models
- Green-oriented (energy/carbon aware)
- Modifiable for future research

---

## Repository Structure
```
LLMs_For_Green_Code_Refactoring/
├── data/
│   ├── original/              # SWE-Perf original dataset
│   ├── raw/                   # Raw measurement data
│   └── processed/             # Extended dataset with GSMM metrics
│
├── src/
│   ├── llm_clients/           # ✅ LLM API client implementations
│   │   ├── base_client.py           # Abstract base with retry logic
│   │   ├── openai_client.py         # GPT-5 client
│   │   ├── anthropic_client.py      # Claude Opus 4.5 client
│   │   ├── google_client.py         # Gemini 3 Pro + Gemma 3 27B
│   │   ├── alibaba_client.py        # Qwen2.5-Coder-32B client
│   │   ├── meta_client.py           # Llama 4 Maverick client
│   │   └── client_manager.py        # Centralized orchestration
│   │
│   ├── prompt_templates/      # ✅ Prompt generation system
│   │   ├── base_template.py         # Abstract template + PromptContext
│   │   ├── zero_shot_template.py    # Single-turn prompts
│   │   ├── self_collaboration_template.py  # Multi-turn with roles
│   │   └── template_manager.py      # Template factory
│   │
│   ├── measurement/           # Metric collection infrastructure
│   ├── data_processing/       # Data processing pipelines
│   ├── analysis/              # Statistical analysis tools
│   └── utils/                 # Shared utilities
│
├── scripts/                   # Executable scripts
│   ├── test_llm_setup.py            # ✅ Test config (no API calls)
│   ├── test_free_llm_clients.py     # ✅ Test free models only
│   ├── test_llm_clients.py          # ✅ Test all 6 models
│   └── test_prompt_templates.py     # ✅ Test template generation
│
├── configs/
│   └── llm_api_keys.yaml      # API keys (gitignored for security)
│
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
└── results/                   # Analysis results and figures
```

---

## Implementation Status

### ✅ Completed Components

#### 1️⃣ LLM API Client Infrastructure
- [x] **6 LLM clients implemented**
  - OpenAI (GPT-5)
  - Anthropic (Claude Opus 4.5)
  - Google (Gemini 3 Pro + Gemma 3 27B)
  - Alibaba (Qwen2.5-Coder-32B)
  - Meta (Llama 4 Maverick)
- [x] **Unified interface** via `BaseLLMClient` abstract class
- [x] **Centralized management** with `LLMClientManager` (caching, lazy loading)
- [x] **Retry logic** with exponential backoff (max 3 attempts)
- [x] **Token counting** with provider-specific tokenizers
- [x] **3 testing scripts** (setup-only, free-tier, full)

#### 2️⃣ Prompt Template System
- [x] **Base infrastructure** with `PromptContext` data class
- [x] **Zero-Shot templates**
  - `ZeroShotOracleTemplate` (file-level optimization)
  - `ZeroShotRealisticTemplate` (repo-level optimization)
- [x] **Self-Collaboration templates**
  - 3-turn system: Analyst → Optimizer → Validator
  - `SelfCollaborationOracleTemplate`
  - `SelfCollaborationRealisticTemplate`
- [x] **Template Manager** for centralized generation
- [x] **Code extraction utilities** for parsing LLM responses (unified diff)
- [x] **Green-oriented prompts** (energy/carbon focus in all templates)

### ⏳ In Progress / Upcoming

- [ ] **Code extraction system** (parse SWE-Perf instances)
- [ ] **LLM generation pipeline** (orchestrate 24 patches per instance)
- [ ] **Patch application system** (apply LLM patches to repositories)
- [ ] **Integration** (connect all components end-to-end)
- [ ] **Batch measurement** (3,640 versions with GSMM metrics)
- [ ] **Statistical analysis** (compare LLM vs human performance)
- [ ] **Results visualization** (plots, tables, comparative analysis)

---

## Quick Start

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

---

### 2. LLM Setup

#### Get Free API Keys (Recommended for Testing)

1. **Google** (Gemini 3 Pro + Gemma 3 27B):  
   - Visit: https://makersuite.google.com/app/apikey
   - Free tier: 15 requests/min, 1500 requests/day
   - No credit card required

2. **Alibaba** (Qwen2.5-Coder-32B):  
   - Visit: https://dashscope.console.aliyun.com/
   - Free trial with initial credits (~1M tokens)

#### Create API Keys Configuration
```bash
# Create config file
nano configs/llm_api_keys.yaml
```

Add your keys:
```yaml
google:
  api_key: "YOUR_GOOGLE_API_KEY"

alibaba:
  api_key: "YOUR_ALIBABA_API_KEY"

# Optional paid models (for full evaluation)
openai:
  api_key: null  # Get from: https://platform.openai.com/api-keys
anthropic:
  api_key: null  # Get from: https://console.anthropic.com/settings/keys
meta:
  together_api_key: null  # Get from: https://www.together.ai/

timeout_seconds: 120
max_retries: 3
```

---

### 3. Test LLM Clients
```bash
# Test configuration only (no API calls, 100% free)
python scripts/test_llm_setup.py

# Test free models only (Google + Alibaba)
python scripts/test_free_llm_clients.py

# Test all 6 models (requires paid API keys)
python scripts/test_llm_clients.py
```

---

### 4. Test Prompt Templates
```bash
# Verify all 4 template combinations work
python scripts/test_prompt_templates.py
```

Expected output: 4 template combinations (2 strategies × 2 problem types) successfully generated.

---

### 5. Measure Baseline Performance (Future)
```bash
# Measure single instance
python scripts/measure_instance.py --instance astropy__astropy-16065

# Measure all 140 instances
python scripts/measure_all_instances.py
```

---

## LLM Models

| Model | Provider | Free Tier | SWE-bench Score | Context | Specialization |
|-------|----------|-----------|-----------------|---------|----------------|
| **GPT-5** | OpenAI | ❌ ($5 min) | 74.9% | 128K | Coding, reasoning, multimodal |
| **Claude Opus 4.5** | Anthropic | ❌ ($5 min) | **80.9%** ⭐ | 200K | Coding (SOTA), agentic tasks |
| **Gemini 3 Pro** | Google | ✅ **FREE** | - | 1M | Multimodal, long context |
| **Qwen2.5-Coder-32B** | Alibaba | ✅ **FREE trial** | 73.7% | 128K | Code generation, repair |
| **Llama 4 Maverick** | Meta | ⚠️ ($0.19/1M) | - | 1M | Efficient MoE (128 experts) |
| **Gemma 3 27B** | Google | ✅ **FREE** | - | 128K | Lightweight, efficient |

### Cost Estimation

**For 140 instances × 24 versions = 3,360 LLM calls:**

| Scenario | Models | Estimated Cost |
|----------|--------|----------------|
| **Free Tier Only** | Gemini 3 Pro, Gemma 3 27B, Qwen2.5-Coder | **$0** |
| **Partial Evaluation** | + GPT-5 | ~$100-200 |
| **Full Evaluation** | All 6 models | **~$300-570** |

**Note:** Costs are estimates based on average prompt/response sizes. Actual costs may vary.

---

## Measurement Infrastructure

**Deployment:** gaissa.essi.upc.edu (UPC Barcelona ESSI Department)

### Tools & Libraries

| Tool | Purpose | Metrics |
|------|---------|---------|
| **Intel RAPL** | CPU energy measurement | `energy_consumption_joules` |
| **psutil** | System resource monitoring | `cpu_usage_*`, `ram_usage_peak_mb` |
| **CodeCarbon** | Carbon emissions tracking | `carbon_emissions_grams` |
| **time** | Execution timing | `duration_seconds` |
| **pytest** | Test execution | Pass/fail status |

### Measurement Process

For each version (BASE, HEAD, or LLM-generated):

1. **Baseline measurement** (idle system, 30s)
2. **Test execution** with monitoring (0.1s sampling rate)
3. **Metric calculation** (test - baseline)
4. **Data storage** (raw + processed)

### Version Types Per Instance

| Version Type | Count | Description |
|--------------|-------|-------------|
| `BASE_COMMIT` | 1 | Original unoptimized code |
| `HEAD_COMMIT` | 1 | Human expert optimization (gold standard) |
| `LLM_ORACLE_ZEROSHOT` | 6 | One per model, Oracle mode, single-turn |
| `LLM_ORACLE_SELFCOLLAB` | 6 | One per model, Oracle mode, multi-turn |
| `LLM_REALISTIC_ZEROSHOT` | 6 | One per model, Realistic mode, single-turn |
| `LLM_REALISTIC_SELFCOLLAB` | 6 | One per model, Realistic mode, multi-turn |
| **Total** | **26** | Per instance |

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

## Project Timeline

- [x] **Phase 1:** Systematic literature review (Completed)
- [x] **Phase 2:** Measurement infrastructure deployment (Completed)
- [x] **Phase 3:** Baseline measurements (BASE + HEAD commits) (Completed)
- [x] **Phase 4:** LLM client architecture implementation (Completed)
- [x] **Phase 5:** Prompt template system design (Completed)
- [ ] **Phase 6:** Code extraction & LLM generation pipeline (In Progress)
- [ ] **Phase 7:** Batch measurement (3,640 versions)
- [ ] **Phase 8:** Statistical analysis & visualization
- [ ] **Phase 9:** Thesis writing
- [ ] **Phase 10:** Defense preparation

---

## Contributing

This is an academic thesis project. For questions, collaboration inquiries, or suggestions:

- **Email:** giacomo.colosio@estudiantat.upc.edu
- **GitHub Issues:** [Create an issue](https://github.com/GiacomoColosio02/LLMs_For_Green_Code_Refactoring/issues)

---

## License

**Academic Research Project** - UPC Barcelona 2024-2025

This project is part of a Master's thesis. The code and methodology are open for educational and research purposes. Please cite appropriately if you use or build upon this work.

---

## Acknowledgments

- **SWE-Perf benchmark creators** - For providing the foundation dataset
- **Green Software Foundation** - For GSMM sustainability metrics framework
- **UPC Barcelona ESSI Department** - For infrastructure and computational resources
- **LLM Providers** - OpenAI, Anthropic, Google, Alibaba, Meta - For API access
- **Supervisors** - Prof. Silverio Martínez-Fernández, Dr. Vincenzo De Martino - For guidance

---

## References

1. SWE-Perf: Performance Optimization Benchmark
2. Green Software Measurement Model (GSMM) - Green Software Foundation
3. Intel RAPL: Running Average Power Limit
4. CodeCarbon: Carbon Emissions Tracking Library

---

**Last Updated:** November 2025