# LLMs For Green Code Refactoring

Benchmarking Large Language Models for Green Code Refactoring using SWE-Perf extended with GSMM-aligned metrics.

## Project Overview

This thesis project evaluates how well **6 state-of-the-art LLMs** can generate energy-efficient code optimizations. We extend the SWE-Perf benchmark with comprehensive energy and carbon measurements across **3,640 optimization instances** (140 base instances × 26 versions).

### Design Space (3D)

1. **Problem Statement Type**
   - `ORACLE`: Complete performance issue description
   - `REALISTIC`: Limited/partial description

2. **Prompt Strategy**
   - `Zero-Shot`: Single-turn direct request
   - `Self-Collaboration`: Multi-turn with specialized roles

3. **LLM Models** (6 total)
   - GPT-5 (OpenAI)
   - Claude Opus 4.5 (Anthropic)
   - Gemini 3 Pro (Google)
   - Qwen2.5-Coder-32B (Alibaba)
   - Llama 4 Maverick (Meta)
   - Gemma 3 27B (Google)

**Total**: 2 × 2 × 6 = 24 LLM-generated versions per instance

---

## Metrics (GSMM-Aligned)

### Efficiency Metrics (Resource-Oriented)
- `cpu_usage_mean` (%) - Average CPU utilization
- `cpu_usage_peak` (%) - Peak CPU utilization  
- `ram_usage_peak` (MB) - Peak memory consumption
- `duration` (s) - Execution time

### Greenness Metrics (Energy-Oriented)
- `energy_consumption` (J) - Total energy consumed
- `mean_power_draw` (W) - Average power draw
- `energy_efficiency` (tests/J) - Useful work per energy unit
- `carbon_emissions` (gCO₂e) - Carbon footprint

---

## Repository Structure
```
LLMs_For_Green_Code_Refactoring/
├── data/
│   ├── original/          # SWE-Perf original dataset
│   ├── raw/               # Raw measurement data
│   └── processed/         # Extended dataset with metrics
├── src/
│   ├── llm_clients/       # 🆕 LLM API client implementations
│   │   ├── base_client.py
│   │   ├── openai_client.py
│   │   ├── anthropic_client.py
│   │   ├── google_client.py
│   │   ├── alibaba_client.py
│   │   ├── meta_client.py
│   │   └── client_manager.py
│   ├── measurement/       # Metric collection
│   ├── data_processing/   # Data processing
│   ├── analysis/          # Statistical analysis
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
│   ├── test_llm_setup.py           # 🆕 Test setup (no API calls)
│   ├── test_free_llm_clients.py    # 🆕 Test free models only
│   └── test_llm_clients.py         # 🆕 Test all 6 models
├── configs/
│   └── llm_api_keys.yaml  # 🆕 API keys (gitignored)
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
└── results/               # Analysis results
```

---

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/LLMs_For_Green_Code_Refactoring.git
cd LLMs_For_Green_Code_Refactoring

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. LLM Setup (Free Tier)

Get free API keys:
- **Google** (Gemini 3 Pro + Gemma 3 27B): https://makersuite.google.com/app/apikey
- **Alibaba** (Qwen2.5-Coder): https://dashscope.console.aliyun.com/

Create `configs/llm_api_keys.yaml`:
```yaml
google:
  api_key: "YOUR_GOOGLE_API_KEY"

alibaba:
  api_key: "YOUR_ALIBABA_API_KEY"

# Optional paid models
openai:
  api_key: null
anthropic:
  api_key: null
meta:
  together_api_key: null
```

### 3. Test LLM Clients
```bash
# Test setup (no API calls, completely free)
python scripts/test_llm_setup.py

# Test free models only (Google + Alibaba)
python scripts/test_free_llm_clients.py

# Test all 6 models (requires paid API keys)
python scripts/test_llm_clients.py
```

### 4. Measure Baseline Performance
```bash
# Measure single instance
python scripts/measure_instance.py --instance astropy__astropy-16065

# Measure all 140 instances
python scripts/measure_all_instances.py
```

---

## LLM Models

| Model | Provider | Free Tier | SWE-bench Score | Specialization |
|-------|----------|-----------|-----------------|----------------|
| **GPT-5** | OpenAI | ❌ ($5 min) | 74.9% | Coding, reasoning |
| **Claude Opus 4.5** | Anthropic | ❌ ($5 min) | 80.9% ⭐ | Coding (SOTA) |
| **Gemini 3 Pro** | Google | ✅ FREE | - | Multimodal, 1M context |
| **Qwen2.5-Coder-32B** | Alibaba | ✅ FREE trial | 73.7% | Code generation/repair |
| **Llama 4 Maverick** | Meta | ⚠️ ($0.19/1M) | - | Efficient MoE, multimodal |
| **Gemma 3 27B** | Google | ✅ FREE | - | Lightweight, 128K context |

**Estimated Cost** (140 instances × 24 versions):
- Free tier models: $0 (Google + Alibaba)
- Full evaluation: ~$300-570 (all 6 models)

---

## Measurement Infrastructure

**Server**: gaissa.essi.upc.edu (UPC Barcelona)

**Tools**:
- Intel RAPL (energy consumption)
- psutil (CPU/RAM monitoring)
- CodeCarbon (carbon emissions)
- pytest (test execution)

**Per-Instance Versions**:
- `BASE_COMMIT`: Original unoptimized code
- `HEAD_COMMIT`: Human-optimized code
- 24 × `LLM_GENERATED`: AI-optimized code (2 statements × 2 strategies × 6 models)

---

## Thesis Information

**Title**: Benchmarking Large Language Models for Green Code Refactoring

**Student**: Giacomo Colosio  
**Institution**: UPC Barcelona (Universitat Politècnica de Catalunya)  
**Supervisors**: Prof. Silverio Martínez-Fernández, Dr. Vincenzo De Martino

**Program**: Master's in Computer Science  
**Focus**: LLMs × Software Engineering × Sustainability

---

## Project Status

- [x] Systematic literature review
- [x] Measurement infrastructure deployment
- [x] Baseline measurements (BASE_COMMIT + HEAD_COMMIT)
- [x] LLM client architecture implementation
- [ ] Prompt template design (Zero-Shot + Self-Collaboration)
- [ ] LLM optimization generation (3,360 versions)
- [ ] Complete measurements with energy/carbon metrics
- [ ] Statistical analysis and comparison
- [ ] Thesis writing

---

## Contributing

This is a thesis project. For questions or collaboration inquiries, please contact:
- **Email**: giacomo.colosio@estudiantat.upc.edu
- **GitHub**: [Your GitHub username]

---

## License

Academic research project - UPC Barcelona 2024-2025

---

## Acknowledgments

- SWE-Perf benchmark creators
- Green Software Foundation (GSMM metrics)
- UPC Barcelona ESSI department
- OpenAI, Anthropic, Google, Alibaba, Meta (LLM providers)