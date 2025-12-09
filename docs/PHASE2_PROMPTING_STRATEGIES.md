# Phase 2: LLM Benchmarking for Green Code Refactoring
## Prompting Strategies and Evaluation Framework

**Student:** Giacomo Colosio  
**Supervisors:** Prof. Silverio Martínez-Fernández, Dr. Vincenzo De Martino  
**Institution:** UPC Barcelona - ESSI Department  
**Date:** December 2024

---

## 1. Project Context: Three-Phase Structure

This document details **Phase 2** of the thesis project, which builds upon the GSMM measurement infrastructure established in Phase 1.

### 1.1 Project Overview

| Phase | Status | Description | Deliverables |
|-------|--------|-------------|--------------|
| **Phase 1** | ✅ Complete | GSMM Metrics Implementation | 17 metrics, 140 baseline measurements |
| **Phase 2** | 🔄 In Design | LLM Benchmarking | 4 models × 4 strategies × 2 settings = 32 configurations |
| **Phase 3** | ⏳ Planned | Comparative Analysis | Statistical analysis, thesis writing |

### 1.2 Phase 2 Goals

Phase 2 evaluates the capability of **state-of-the-art Large Language Models** to perform **green code refactoring** - generating optimized code that reduces energy consumption and carbon emissions while maintaining functional correctness.

**Research Questions:**
1. Can LLMs effectively optimize code for energy and performance?
2. Which prompting strategies are most effective for green optimization?
3. How do LLM optimizations compare to human expert optimizations?
4. What is the trade-off between single-turn simplicity and multi-turn sophistication?

---

## 2. Evaluation Framework Design

### 2.1 Model Selection Strategy

We will evaluate **4 LLM models** representing different architectural approaches and accessibility models:

**Tentative Selection:**

| Category | Model | Rationale | Access |
|----------|-------|-----------|--------|
| **Proprietary (Closed)** | GPT-4o/GPT-4.5 | Industry-leading code generation, SWE-bench SOTA | OpenAI API |
| | Claude Sonnet 4 | Strong reasoning, long context | Anthropic API |
| **Open-Source** | Qwen2.5-Coder-32B | SOTA open-source coding model | HuggingFace/vLLM |
| | Llama 3.1-70B/CodeGemma | Meta/Google backing, broad adoption | HuggingFace/vLLM |

**Selection Criteria:**
- **Code generation capability**: Performance on SWE-bench, HumanEval, MBPP
- **Context window**: Sufficient for repository-level reasoning (≥32K tokens)
- **API availability**: Stable, documented API for reproducibility
- **Cost/Accessibility**: Balance between proprietary and open-source

**Final selection will be determined after preliminary testing in early Phase 2.**

### 2.2 Problem Settings: Oracle vs. Realistic

Following recent code generation literature (SWE-bench, Agentless), we evaluate models under two distinct information settings that simulate different levels of task difficulty:

#### 2.2.1 Oracle Setting (File-Level)

**Definition:** The model is provided with the **exact files and functions** that require optimization (ground truth from human expert patches).

**Input:**
- Specific files containing target functions
- Function-level specifications: `{'file.py': ['func1', 'func2']}`
- Complete file content

**Task:** Generate optimization patch for provided code.

**Evaluation Focus:** Pure code generation and optimization capability.

**Example (astropy__astropy-16065):**
```
Files: ['astropy/units/decorators.py']
Functions: ['as_decorator', 'wrapper']
```

#### 2.2.2 Realistic Setting (Repository-Level)

**Definition:** The model receives the **entire repository** (or significant portion) and must autonomously identify where to optimize.

**Input:**
- Full repository structure
- List of **measured functions** (functions executed by performance tests)
- Repository-level context

**Task:** 
1. Analyze repository structure
2. Identify optimization opportunities
3. Generate patch targeting measured functions

**Evaluation Focus:** Repository understanding, function localization, planning, and optimization.

**Example (astropy__astropy-16065):**
```
Measured functions: ['Quantity.to', 'Unit.compose']
Repository: 247 Python files, 156K lines
Model must: Find relevant files → Identify bottlenecks → Generate patch
```

**Comparison:**

| Aspect | Oracle | Realistic |
|--------|--------|-----------|
| Information | Ground truth files/functions | Measured functions only |
| Search space | ~2-5 files | ~50-300 files |
| Task complexity | Code generation | Analysis + localization + generation |
| Real-world analogy | Expert-guided optimization | Autonomous optimization |
| Inspiration | SWE-bench "oracle" prompts | Agentless, OpenHands workflows |

---

## 3. Prompting Strategies

We design **4 prompting strategies** with increasing sophistication, inspired by recent advances in LLM reasoning and code generation.

### 3.1 Strategy Matrix

| Strategy | Inspiration | Turns | Complexity | Oracle | Realistic |
|----------|-------------|-------|------------|--------|-----------|
| **Zero-Shot** | In-context learning | 1 | Low | ✅ | ✅ |
| **Chain-of-Thought (CoT)** | Wei et al. (2022) | 1 | Medium | ✅ | ✅ |
| **Self-Collaboration** | AgentCoder (2024) | 4 | High | ✅ | ✅ |
| **LDB (Iterative)** | Zhong et al. (2024) | 2-5 | Very High | ✅ | ✅ |

**Total configurations:** 4 strategies × 2 settings = **8 prompt templates**

---

## 4. Detailed Strategy Specifications

### 4.1 Strategy 1: Zero-Shot

**Inspiration:** Pure in-context learning capability of foundation models.

**Design:** Direct instruction with GSMM-aligned guidelines, no intermediate reasoning, no examples.

**Hypothesis:** Strong foundation models should optimize code from instruction alone.

#### 4.1.1 Zero-Shot Oracle Prompt

```
You are an expert in energy-efficient software engineering. Your task is to
generate a patch that optimizes the code for computational performance and
reduced energy consumption.
<problem_statement>
Please enhance the computational efficiency, execution speed, and energy
efficiency across the entire repository. The optimization efforts may target
one or more objective functions: {TARGET_FUNCTIONS_DICT}
Conditions:

Acceleration of at least one function is sufficient for success
Optimization may be direct (target function) or indirect (subroutines)
Prioritize maximal efficiency gains where feasible
All existing unit tests must remain unaltered
</problem_statement>

<code_files>
{CONTENT_OF_PROVIDED_FILES}
</code_files>
<green_optimization_guidelines>
Consider optimizations that positively impact GSMM metrics:

CPU/GPU Energy: Reduce loops, use efficient data structures,
avoid redundant calculations
Memory: Minimize peak RAM usage, prefer generators over lists
Execution Time: Improve algorithmic complexity, add caching
System Power: Smoother CPU usage profiles reduce power spikes
</green_optimization_guidelines>

<output_format>
Generate optimization as SEARCH/REPLACE patch:
### path/to/file.py
<<<<<<< SEARCH
# Exact code to replace (maintain indentation)
=======
# New optimized code
>>>>>>> REPLACE
```
Provide ONLY the patch. No explanations or analysis.
</output_format>
```

**Placeholder Population:**
- `{TARGET_FUNCTIONS_DICT}`: `{'astropy/units/decorators.py': ['as_decorator', 'wrapper']}`
- `{CONTENT_OF_PROVIDED_FILES}`: Complete file content with line numbers

#### 4.1.2 Zero-Shot Realistic Prompt
```
You are an autonomous AI software engineer agent. Your goal is to improve 
performance and energy efficiency of a code repository.

<problem_statement>
Analyze the provided repository and optimize it to enhance computational  efficiency, execution speed, and reduce energy consumption. Performance 
evaluations will be based on the following functions, which are measured  by unit tests: {REALISTIC_TARGET_FUNCTIONS_LIST}

Conditions:
1. Acceleration of at least one measured function is sufficient
2. You can modify ANY part of the repository to achieve optimization
3. Prioritize maximal efficiency gains
4. Do not modify existing unit tests
</problem_statement>

<repository_context>
You have access to the full repository structure and code. You must:
1. Understand the codebase
2. Locate sections impacting target functions
3. Generate optimization patch
</repository_context>

<green_optimization_guidelines>
[Same as Oracle]
</green_optimization_guidelines>

<output_format>
[Same as Oracle]
</output_format>
```

**Placeholder Population:**
- `{REALISTIC_TARGET_FUNCTIONS_LIST}`: `['Quantity.to', 'Unit.compose', 'Quantity.__mul__']`
- Repository content: Top-level structure + 10 largest Python files (~20K tokens)

**Key Difference:** Model must infer which files to modify from function names alone.

---

### 4.2 Strategy 2: Chain-of-Thought (CoT)

**Inspiration:** Wei et al. (2022) - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

**Design:** Explicit instruction to "think step-by-step" before generating the patch, forcing the model to articulate its reasoning process.

**Hypothesis:** Explicit reasoning improves optimization quality by encouraging systematic analysis of bottlenecks and impact prediction.

#### 4.2.1 CoT Oracle Prompt
```
You are an expert in green software optimization. Before generating the
final patch, you MUST first analyze the code step-by-step.
<problem_statement>
[Same as Zero-Shot Oracle]
</problem_statement>
<code_files>
{CONTENT_OF_PROVIDED_FILES}
</code_files>
<thinking_process_instructions>
Perform a step-by-step (Chain-of-Thought) analysis:

Identify Targets: Locate the objective functions {TARGET_FUNCTIONS}
in the provided code.
Bottleneck Analysis: For each function, analyze:

Time/space complexity (Big-O notation)
Loop structures and nesting levels
Data structure operations (list ops, dict lookups, etc.)
Redundant computations or allocations


Green Impact Hypothesis: For each proposed change, predict impact
on GSMM metrics:

Example: "Replacing O(n²) nested loop with dict lookup will reduce
CPU energy by ~40% and duration proportionally"


Solution Synthesis: Based on analysis, formulate final optimization
strategy.
</thinking_process_instructions>

<output_format>
Your response MUST have two sections:
ANALYSIS:
[Your detailed step-by-step reasoning here]
PATCH:

```

### path/to/file.py
<<<<<<< SEARCH
...
=======
...
>>>>>>> REPLACE
```
</output_format>
```

**Expected Output Structure:**
```
ANALYSIS:
Step 1 - Identify Targets:
- Function 'as_decorator' found at astropy/units/decorators.py:42
- Function 'wrapper' found at astropy/units/decorators.py:67

Step 2 - Bottleneck Analysis:
- 'as_decorator' performs isinstance() check in loop (line 45-48)
  → O(n) per invocation, called frequently
- 'wrapper' rebuilds Quantity objects unnecessarily (line 72)
  → Allocates new objects even when unit unchanged

Step 3 - Green Impact Hypothesis:
- Cache isinstance() results → Reduce CPU cycles by ~30%
- Reuse Quantity objects when possible → Reduce memory allocations
- Expected: 25-30% reduction in CPU energy, 15% reduction in duration

Step 4 - Solution Synthesis:
Implement: (1) isinstance() caching via lru_cache, 
           (2) early return in wrapper for same-unit case

PATCH:
[actual code patch]
```

#### 4.2.2 CoT Realistic Prompt
```
[Same structure as CoT Oracle, but with modified thinking process]

<thinking_process_instructions>
Perform step-by-step analysis:

1. **Repository Exploration Strategy**: Plan how to locate code related 
   to target functions {REALISTIC_TARGET_FUNCTIONS_LIST}:
   - Search strategy (grep for function names, analyze imports)
   - Call graph inference (which files likely contain these functions?)
   - Dependency analysis

2. **Bottleneck Analysis**: [Same as Oracle once files identified]

3. **Green Impact Hypothesis**: [Same as Oracle]

4. **Solution Synthesis**: [Same as Oracle]
</thinking_process_instructions>
```

**Key Addition:** Step 1 now includes repository navigation planning.

---

### 4.3 Strategy 3: Self-Collaboration

**Inspiration:** 
- Du et al. (2024) - "AgentCoder: Multi-Agent-based Code Generation"
- Self-Collaboration paradigm - simulating expert team within single LLM

**Design:** Single model simulates conversation between three specialized roles across 4 turns, mimicking real software team dynamics.

**Hypothesis:** Role specialization and multi-turn refinement produce higher-quality optimizations than single-pass generation.

#### 4.3.1 Self-Collaboration System Prompt (Oracle & Realistic)
```
You will simulate a collaboration between three AI experts to optimize code. 
Follow this conversation flow strictly:

<roles>
1. **Sustainability Analyst**: Analyzes code to identify inefficiencies 
   with focus on energy consumption (GSMM metrics), CPU/GPU usage, memory, 
   and algorithm complexity. Proposes high-level optimization goals.

2. **Senior Refactoring Engineer**: Takes Analyst's goals and writes 
   actual code changes. Focuses on performance, readability, and correctness.

3. **Critical Reviewer**: Reviews new code for bugs, missed optimization 
   opportunities, and adherence to Analyst's goals. Suggests concrete improvements.
</roles>

<conversation_flow>
TURN 1 (Sustainability Analyst):
- Input: Original code and problem statement
- Output: List of 3-5 specific, actionable optimization goals targeting 
  GSMM metrics

Example output format:
"OPTIMIZATION GOALS:
1. Reduce CPU energy in function X by eliminating O(n²) loop
2. Minimize memory allocations in function Y by reusing objects
3. ..."

TURN 2 (Senior Refactoring Engineer):
- Input: Original code, problem statement, Analyst's goals
- Output: Complete SEARCH/REPLACE patch implementing optimizations

TURN 3 (Critical Reviewer):
- Input: Original code, Analyst's goals, Engineer's patch
- Output: Critique noting potential issues and concrete suggestions

Example output format:
"REVIEW:
✅ Good: Loop elimination in function X
❌ Issue: Edge case not handled on line 45
💡 Suggestion: Add bounds check to prevent IndexError"

TURN 4 (Senior Refactoring Engineer - Final):
- Input: Reviewer's critique
- Output: Final, revised SEARCH/REPLACE patch addressing all critiques
</conversation_flow>

<problem_statement>
[INSERT ORACLE OR REALISTIC PROBLEM STATEMENT HERE]
</problem_statement>

<initial_code_context>
[INSERT CODE FILES (Oracle) OR REPOSITORY DESCRIPTION (Realistic)]
</initial_code_context>

Now, begin the simulation as the Sustainability Analyst for TURN 1.
```

**Implementation Workflow:**
1. Send system prompt → Model generates Turn 1 (Analyst goals)
2. Append Turn 1 to conversation → Model generates Turn 2 (Engineer patch)
3. Append Turn 2 to conversation → Model generates Turn 3 (Reviewer critique)
4. Append Turn 3 to conversation → Model generates Turn 4 (Final patch)

**Example Turn Outputs:**
```
TURN 1 (Analyst):
OPTIMIZATION GOALS:
1. Eliminate redundant unit compatibility checks in Quantity.__mul__ 
   (occurs in tight loop, ~40% of CPU time)
2. Cache unit conversion factors in Unit.to method 
   (recomputed unnecessarily, ~25% of calls repeat)
3. Reduce memory allocations in decorators.wrapper 
   (creates new Quantity even when unnecessary)
Expected Impact: 50-60% reduction in CPU energy, 30-40% duration reduction

TURN 2 (Engineer):
[Generates initial patch implementing above goals]

TURN 3 (Reviewer):
REVIEW:
✅ Good: Unit conversion caching implementation is sound
❌ Issue: Cache may grow unbounded, consider LRU policy
❌ Issue: Quantity reuse optimization breaks when quantity has .info attribute
💡 Suggestion: Add cache size limit (e.g., @lru_cache(maxsize=128))
💡 Suggestion: Check for .info before reusing Quantity object

TURN 4 (Engineer - Final):
[Generates revised patch addressing all reviewer points]
```

**Advantages:**
- Systematic analysis before implementation (Analyst)
- Quality control through review (Reviewer)
- Iterative refinement without manual intervention

**Challenges:**
- Requires 4 API calls per instance
- Model must maintain consistency across turns
- Higher cost (4× tokens vs. Zero-Shot)

---

### 4.4 Strategy 4: LDB (Large Language Model Debugger)

**Inspiration:** Zhong et al. (2024) - "LDB: A Large Language Model Debugger via Verifying Runtime Execution Step-by-step"

**Design:** Iterative optimization loop using **runtime execution feedback** (profiling data, test results) to guide successive refinements.

**Hypothesis:** Concrete execution evidence (profiler traces, actual measurements) enables more targeted optimization than static analysis alone.

#### 4.4.1 LDB Workflow
```
┌─────────────────────────────────────────────────────┐
│  ITERATION 1                                        │
├─────────────────────────────────────────────────────┤
│  1. Model generates initial patch (Zero-Shot)       │
│  2. Apply patch → Run tests → Measure GSMM metrics  │
│  3. Collect feedback:                               │
│     - cProfile output (CPU bottlenecks)             │
│     - memory_profiler output (RAM usage)            │
│     - GSMM metrics (energy, duration)               │
│     - Test pass/fail status                         │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  ITERATION 2 (if needed)                            │
├─────────────────────────────────────────────────────┤
│  1. Model receives feedback from Iteration 1        │
│  2. Diagnose issues (why energy not reduced?)       │
│  3. Generate revised patch                          │
│  4. Apply → Test → Measure                          │
└─────────────────────────────────────────────────────┘
                          ↓
                    (repeat 2-5 times)
```

#### 4.4.2 LDB Debugging Prompt (Oracle & Realistic)
```
You are a Code Debugger and Optimizer specialized in performance. You will
use execution feedback to fix and optimize code.
<previous_attempt>
The following code patch was previously generated but showed suboptimal
performance or failed tests.
Previous Patch:
{PREVIOUS_PATCH}
</previous_attempt>
<execution_feedback>
Runtime analysis indicates:
cProfile Output (CPU Time):
{CPROFILE_DATA}
Memory Profiler:
{MEMORY_PROFILER_DATA}
GSMM Metrics (Before → After):

CPU Energy: {BEFORE_CPU_ENERGY}J → {AFTER_CPU_ENERGY}J ({CHANGE}%)
Duration: {BEFORE_DURATION}s → {AFTER_DURATION}s ({CHANGE}%)
Peak RAM: {BEFORE_RAM}MB → {AFTER_RAM}MB ({CHANGE}%)

Test Results:
{TEST_PASS_FAIL_INFO}
Key Observations:
{AUTOMATED_ANALYSIS}
Examples of observations:

"Function foo() is still the bottleneck (95% CPU time in cProfile)"
"Test test_bar fails with IndexError on edge case X"
"Memory usage increased by 20% due to cache not being released"
"Energy reduction achieved (30%) but duration barely changed (5%),
suggesting power increased"
</execution_feedback>

<code_context>
{CURRENT_CODE_AFTER_PATCH}
</code_context>
<problem_statement>
[Original Oracle or Realistic problem statement]
</problem_statement>
<task>
1. **Diagnose**: Analyze the feedback to identify why previous patch 
   failed or was inefficient. Be specific about root causes.

Correct & Optimize: Propose a new patch that addresses the
performance/energy issues while passing all tests.
Justify: Briefly explain how the new patch resolves issues
highlighted in feedback.
</task>


<output_format>
DIAGNOSIS:
[Specific root cause analysis based on profiler data]
JUSTIFICATION:
[How new patch addresses each identified issue]
NEW_PATCH:

### path/to/file.py
<<<<<<< SEARCH
...
=======
...
>>>>>>> REPLACE
```
</output_format>
```

**Example Execution Feedback (Iteration 1):**
```
<execution_feedback>
**cProfile Output (Top 5 functions by cumtime):**
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    0.450    0.005    2.341    0.023 decorators.py:42(as_decorator)
     1000    0.892    0.001    1.234    0.001 decorators.py:67(wrapper)
      500    0.234    0.000    0.567    0.001 core.py:123(__mul__)

**GSMM Metrics (Before → After):**
- CPU Energy: 45.2J → 42.8J (-5.3%)
- Duration: 1.85s → 1.78s (-3.8%)
- Peak RAM: 245MB → 289MB (+17.9%)

**Test Results:**
✅ All 12 tests passed

**Key Observations:**
- Optimization reduced CPU energy by only 5%, below 20% target
- Memory usage INCREASED significantly (45MB)
- cProfile shows 'as_decorator' still taking 50% of total time
- Root cause: Caching introduced but cache size unbounded, causing 
  memory bloat. Function still called frequently.
</execution_feedback>
```

**Model Response (Iteration 2):**
```
DIAGNOSIS:
The previous patch introduced @lru_cache on 'as_decorator' but failed to 
limit cache size, causing 45MB memory increase. Additionally, the function 
is called with many unique argument combinations, reducing cache hit rate. 
The 5% energy reduction suggests the cache overhead (memory operations) 
partially negates computational savings.

JUSTIFICATION:
New patch: (1) Adds maxsize=128 to lru_cache to cap memory, (2) Refactors 
'as_decorator' to compute expensive isinstance() check OUTSIDE the cached 
portion, reducing cache misses. This should maintain cache benefits while 
eliminating memory bloat.

NEW_PATCH:
[revised code with bounded cache and refactored logic]
```

**Termination Criteria:**
- **Success**: GSMM metrics improve by ≥20% AND all tests pass
- **Max iterations**: 5 (prevent infinite loops)
- **Degradation**: If metrics worsen for 2 consecutive iterations, return best patch

**Advantages:**
- Data-driven: Actual runtime evidence guides optimization
- Iterative refinement based on concrete failures
- Can recover from incorrect initial assumptions

**Challenges:**
- Requires execution infrastructure (sandbox, profilers)
- Multiple API calls (2-5× vs. Zero-Shot)
- Significant computational cost (run tests 2-5 times per instance)

---

## 5. Evaluation Metrics

Each generated patch will be evaluated using the **17 GSMM metrics** from Phase 1, comparing LLM-optimized code against human expert optimizations (HEAD commits).

### 5.1 Primary Success Metrics

**Energy Efficiency:**
- **ΔE_cpu**: Percentage change in CPU energy
- **ΔE_system**: Percentage change in system energy (wattmeter)
- **ΔC_system**: Percentage change in carbon emissions

**Performance:**
- **Δt**: Percentage change in execution duration
- **ΔP_mean**: Change in mean power draw

**Resource Efficiency:**
- **ΔRAM_peak**: Change in peak memory usage
- **ΔCPU_usage**: Change in CPU utilization

### 5.2 Secondary Metrics

**Functional Correctness:**
- Test pass rate (must be 100%)
- Semantic equivalence to original code

**Optimization Quality:**
- Number of instances improved (ΔE ≥ 20%)
- Magnitude of improvement (mean ΔE across successful instances)
- Consistency (std dev of ΔE)

**Strategy Efficiency:**
- Token usage per instance
- API call count
- Wall-clock time to generate patch

### 5.3 Comparative Analysis

For each (model, strategy, setting) combination:
```
Success Rate = (# instances with ΔE_system ≥ 20%) / (# total instances)

Mean Improvement = mean(ΔE_system) for successful instances

vs. Human Performance:
  ΔE_LLM vs. ΔE_human = (E_base - E_LLM) vs. (E_base - E_head)
```

**Expected Dataset Size:**
- 140 instances × 4 models × 4 strategies × 2 settings = **4,480 measurements**
- Each measurement: 17 GSMM metrics
- Total data points: 4,480 × 17 = **76,160 metrics**

---

## 6. Implementation Plan

### 6.1 Pipeline Architecture
```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2 PIPELINE                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each (instance, model, strategy, setting):            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 1. PROMPT GENERATION                                │  │
│  │    - Load instance from SWE-Perf                    │  │
│  │    - Populate prompt template                       │  │
│  │    - Oracle: Extract target files/functions        │  │
│  │    - Realistic: Compile repository context         │  │
│  └─────────────────────────────────────────────────────┘  │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 2. LLM INFERENCE                                    │  │
│  │    - Call model API (OpenAI, Anthropic, HF)        │  │
│  │    - Handle multi-turn (Self-Collab, LDB)          │  │
│  │    - Parse SEARCH/REPLACE output                   │  │
│  └─────────────────────────────────────────────────────┘  │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 3. PATCH APPLICATION                                │  │
│  │    - Validate patch format                          │  │
│  │    - Apply to BASE commit                           │  │
│  │    - Handle conflicts/errors                        │  │
│  └─────────────────────────────────────────────────────┘  │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 4. MEASUREMENT (using Phase 1 infrastructure)      │  │
│  │    - Run tests with GSMM monitoring                 │  │
│  │    - Collect 17 metrics                             │  │
│  │    - For LDB: Generate feedback, iterate           │  │
│  └─────────────────────────────────────────────────────┘  │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 5. STORAGE                                          │  │
│  │    - Save: patch, metrics, logs, intermediate data │  │
│  │    - JSON format for analysis                       │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Expected Timeline

| Task | Duration | Dependencies |
|------|----------|--------------|
| API setup & model selection | 1 week | - |
| Prompt template implementation | 1 week | Model selection |
| Patch parsing & application | 1 week | Prompt templates |
| Integration with Phase 1 | 1 week | All above |
| Pilot run (10 instances) | 3 days | Full pipeline |
| Full run (140 instances) | 2-3 weeks | Pilot validation |
| Data processing & analysis | 2 weeks | Full run complete |

**Total estimated duration: 8-10 weeks**

### 6.3 Cost Estimation

**API Costs (Proprietary Models):**
- Average prompt size: ~8K tokens (Oracle), ~20K tokens (Realistic)
- Average output size: ~2K tokens
- Calls per instance: 1 (Zero-Shot/CoT), 4 (Self-Collab), 2-5 (LDB)

**Rough estimate (GPT-4/Claude):**
- Oracle: $0.20-0.50 per instance
- Realistic: $0.50-1.50 per instance
- Total (2 proprietary × 140 instances × 8 configs): ~$3,000-6,000

**Computational Costs (Open-Source Models):**
- Inference on local GPU (RTX 4090) or rented compute
- Estimated: $500-1,000 for full run

**Total Phase 2 budget: ~$4,000-7,000**

---

## 7. Expected Outcomes & Deliverables

### 7.1 Quantitative Results

**For each (model, strategy, setting) combination:**
- Success rate (% instances with ≥20% energy reduction)
- Mean/median/std energy improvement
- Performance metrics (duration, power, memory)
- Comparison to human expert optimizations

**Hypotheses to test:**
1. **H1**: CoT outperforms Zero-Shot (explicit reasoning helps)
2. **H2**: Self-Collaboration outperforms CoT (multi-agent refinement helps)
3. **H3**: LDB achieves highest success rate (feedback-driven optimization)
4. **H4**: Oracle setting has higher success rates than Realistic (reduced search space)
5. **H5**: Proprietary models outperform open-source (more parameters, better training)

### 7.2 Qualitative Analysis

- **Failure modes**: Why do certain strategies/models fail?
- **Optimization patterns**: What code patterns do LLMs successfully optimize?
- **Green awareness**: Do models correctly reason about GSMM metrics?
- **Trade-offs**: Energy vs. memory vs. duration

### 7.3 Thesis Deliverables

1. **Extended SWE-Perf dataset** with 17 GSMM metrics + LLM patches
2. **Comprehensive evaluation** across 32 configurations
3. **Statistical analysis** of strategy effectiveness
4. **Insights** for green software engineering with LLMs
5. **Open-source artifacts**: Prompts, code, data

---

## 8. References

**Prompting Strategies:**
1. Wei et al. (2022) - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
2. Du et al. (2024) - "AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation"
3. Zhong et al. (2024) - "LDB: A Large Language Model Debugger via Verifying Runtime Execution Step-by-step"

**Benchmarks:**
4. Jimenez et al. (2024) - "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"
5. Xia et al. (2024) - "Agentless: Demystifying LLM-based Software Engineering Agents"

**Green Software:**
6. Green Software Foundation - "Green Software Measurement Model (GSMM)"
7. Verdecchia et al. (2023) - "A Systematic Review of Green AI"

---

**Document Version:** 1.0  
**Last Updated:** December 9, 2024  
**Status:** Phase 1 Complete ✅ | Phase 2 Design Complete 📋 | Implementation Pending ⏳