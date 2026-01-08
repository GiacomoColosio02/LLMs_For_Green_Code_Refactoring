"""
Chain-of-Thought (CoT) Prompt Templates for Green Code Optimization.

Based on:
- Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- Kojima et al. (2022): "Large Language Models are Zero-Shot Reasoners"

Key Innovation: Structured reasoning path tailored to Green Software Engineering,
forcing the model to analyze before generating code (System 2 thinking).

The prompt enforces two strict sections:
- SECTION 1: ANALYSIS (Identification → Diagnosis → Hypothesis)
- SECTION 2: PATCH (SEARCH/REPLACE blocks only)
"""

import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .base_template import BasePromptTemplate, PromptContext, ProblemStatementType


# =============================================================================
# COT RESPONSE PARSER
# =============================================================================

@dataclass
class CoTResponse:
    """Parsed Chain-of-Thought response."""
    raw_response: str
    analysis_section: str
    patch_section: str
    identification: str
    diagnosis: str
    hypothesis: str
    has_valid_structure: bool
    

def parse_cot_response(response: str) -> CoTResponse:
    """
    Parse a CoT response into its structured components.
    
    Extracts:
    - SECTION 1: ANALYSIS (with Identification, Diagnosis, Hypothesis)
    - SECTION 2: PATCH (the actual code changes)
    
    Args:
        response: Raw LLM response
        
    Returns:
        CoTResponse with parsed sections
    """
    # Remove <think> tags if present (for reasoning models like DeepSeek R1)
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Try to find section markers
    analysis_section = ""
    patch_section = ""
    
    # Pattern 1: Explicit SECTION markers
    section1_match = re.search(
        r'SECTION\s*1\s*:?\s*ANALYSIS(.*?)(?=SECTION\s*2|$)', 
        clean_response, 
        re.IGNORECASE | re.DOTALL
    )
    section2_match = re.search(
        r'SECTION\s*2\s*:?\s*PATCH(.*?)$', 
        clean_response, 
        re.IGNORECASE | re.DOTALL
    )
    
    if section1_match:
        analysis_section = section1_match.group(1).strip()
    if section2_match:
        patch_section = section2_match.group(1).strip()
    
    # Fallback: If no explicit markers, try to find SEARCH/REPLACE blocks
    if not patch_section and "<<<<<<< SEARCH" in clean_response:
        # Find where the first SEARCH block starts and take everything from there
        search_start = clean_response.find("<<<<<<< SEARCH")
        # Look backwards for file path marker
        pre_search = clean_response[:search_start]
        last_newline = pre_search.rfind('\n###')
        if last_newline == -1:
            last_newline = pre_search.rfind('\n')
        
        patch_section = clean_response[max(0, last_newline):].strip()
        analysis_section = clean_response[:max(0, last_newline)].strip()
    
    # Extract sub-components from analysis
    identification = ""
    diagnosis = ""
    hypothesis = ""
    
    # Try to extract Identification
    id_match = re.search(
        r'(?:1\.\s*)?(?:IDENTIFICATION|Identification|Where)[:\s]*(.*?)(?=(?:2\.|DIAGNOSIS|Diagnosis|Why)|$)',
        analysis_section,
        re.IGNORECASE | re.DOTALL
    )
    if id_match:
        identification = id_match.group(1).strip()
    
    # Try to extract Diagnosis
    diag_match = re.search(
        r'(?:2\.\s*)?(?:DIAGNOSIS|Diagnosis|Why)[:\s]*(.*?)(?=(?:3\.|HYPOTHESIS|Hypothesis|What if)|$)',
        analysis_section,
        re.IGNORECASE | re.DOTALL
    )
    if diag_match:
        diagnosis = diag_match.group(1).strip()
    
    # Try to extract Hypothesis
    hyp_match = re.search(
        r'(?:3\.\s*)?(?:HYPOTHESIS|Hypothesis|What if)[:\s]*(.*?)$',
        analysis_section,
        re.IGNORECASE | re.DOTALL
    )
    if hyp_match:
        hypothesis = hyp_match.group(1).strip()
    
    # Determine if structure is valid
    has_valid_structure = bool(patch_section and "<<<<<<< SEARCH" in patch_section)
    
    return CoTResponse(
        raw_response=response,
        analysis_section=analysis_section,
        patch_section=patch_section,
        identification=identification,
        diagnosis=diagnosis,
        hypothesis=hypothesis,
        has_valid_structure=has_valid_structure
    )


def extract_patch_from_cot(response: str) -> str:
    """
    Extract only the patch section from a CoT response.
    
    This is the key function for integration with the patch applicator.
    It discards all reasoning text and returns only the SEARCH/REPLACE blocks.
    
    Args:
        response: Raw LLM response
        
    Returns:
        String containing only the patch content (ready for PatchEngine)
    """
    parsed = parse_cot_response(response)
    
    if parsed.patch_section:
        return parsed.patch_section
    
    # Ultimate fallback: return everything from first SEARCH block
    if "<<<<<<< SEARCH" in response:
        # Clean think tags
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        idx = clean.find("<<<<<<< SEARCH")
        # Include potential file marker before SEARCH
        pre_idx = clean.rfind('\n', 0, idx)
        if pre_idx > idx - 200:  # File marker should be close
            return clean[pre_idx:].strip()
        return clean[idx:].strip()
    
    return response


# =============================================================================
# COT TEMPLATE CLASS
# =============================================================================

class ChainOfThoughtTemplate(BasePromptTemplate):
    """
    Chain-of-Thought template for Green Code Optimization.
    
    Forces structured reasoning before code generation:
    1. SECTION 1: ANALYSIS
       - Identification (Where is the problem?)
       - Diagnosis (Why is it inefficient?)
       - Hypothesis (What will the fix achieve?)
    2. SECTION 2: PATCH
       - SEARCH/REPLACE blocks only
    
    Supports both ORACLE and REALISTIC modes.
    """
    
    def __init__(self):
        super().__init__()
        self.template_name = "ChainOfThought"
    
    def generate_prompt(self, context: PromptContext) -> str:
        """Generate CoT prompt based on context type."""
        if context.problem_statement_type == ProblemStatementType.ORACLE:
            return self._generate_oracle_prompt(context)
        else:
            return self._generate_realistic_prompt(context)
    
    def _get_cot_instructions_oracle(self) -> str:
        """Get CoT instructions for ORACLE mode."""
        return '''
## RESPONSE FORMAT

Your response MUST follow this EXACT structure with two sections:

### SECTION 1: ANALYSIS

Start this section by writing: "Let's think step by step."

Then address these three points:

1. **IDENTIFICATION (The "Where")**
   Analyze the target function(s) provided. What is their current algorithmic complexity?
   What data structures are being used?

2. **DIAGNOSIS (The "Why")**
   Explain the inefficiency in physical/computational terms. For example:
   - "This nested loop creates O(N²) complexity, causing excessive CPU cycles"
   - "Repeated string concatenation causes memory allocation spikes"
   - "This function recalculates the same values multiple times"

3. **HYPOTHESIS (The "What if")**
   Predict the impact of your proposed fix. For example:
   - "Using a dictionary for lookups will reduce complexity from O(N) to O(1)"
   - "Caching the result will eliminate redundant computations"
   - "Using list comprehension will reduce memory overhead"

### SECTION 2: PATCH

Only AFTER completing the analysis, provide your code changes using SEARCH/REPLACE format:

```
### path/to/file.py
<<<<<<< SEARCH
original code (minimum unique context)
=======
optimized code
>>>>>>> REPLACE
```

CRITICAL RULES FOR PATCHES:
- Use SMALL, UNIQUE SEARCH blocks - just enough lines to locate the code
- Do NOT copy entire functions - only the specific lines being changed
- Preserve all existing functionality - tests MUST still pass
- Do NOT add new external dependencies
'''

    def _get_cot_instructions_realistic(self) -> str:
        """Get CoT instructions for REALISTIC mode."""
        return '''
## RESPONSE FORMAT

Your response MUST follow this EXACT structure with two sections:

### SECTION 1: ANALYSIS

Start this section by writing: "Let's think step by step."

Then address these three points:

1. **IDENTIFICATION (The "Where")**
   From the retrieved files, identify the HOTSPOT - the specific function or code block
   that is most likely causing the performance issue. Filter out noise files that are
   not relevant. Explain why you selected this location.

2. **DIAGNOSIS (The "Why")**
   Explain the inefficiency in physical/computational terms. For example:
   - "This nested loop creates O(N²) complexity, causing excessive CPU cycles"
   - "Repeated string concatenation causes memory allocation spikes"
   - "This function recalculates the same values multiple times"

3. **HYPOTHESIS (The "What if")**
   Predict the impact of your proposed fix. For example:
   - "Using a dictionary for lookups will reduce complexity from O(N) to O(1)"
   - "Caching the result will eliminate redundant computations"
   - "Using vectorized operations will leverage CPU SIMD instructions"

### SECTION 2: PATCH

Only AFTER completing the analysis, provide your code changes using SEARCH/REPLACE format:

```
### path/to/file.py
<<<<<<< SEARCH
original code (minimum unique context)
=======
optimized code
>>>>>>> REPLACE
```

CRITICAL RULES FOR PATCHES:
- Use SMALL, UNIQUE SEARCH blocks - just enough lines to locate the code
- Do NOT copy entire functions - only the specific lines being changed
- Preserve all existing functionality - tests MUST still pass
- Do NOT add new external dependencies
- Do NOT modify test files
'''

    def _generate_oracle_prompt(self, context: PromptContext) -> str:
        """Generate ORACLE mode CoT prompt."""
        
        # Build code context
        code_section = self._format_code_files(context.code_files)
        
        # Build target functions info
        target_info = ""
        if context.target_functions:
            if isinstance(context.target_functions, dict):
                target_info = f"**Target Functions to Optimize:**\n```\n{context.target_functions}\n```"
            elif isinstance(context.target_functions, list):
                target_info = f"**Target Files:** {', '.join(context.target_functions)}"
        
        prompt = f'''You are an expert Green Software Engineer specializing in energy-efficient code optimization.

## TASK

Optimize the provided code for **energy efficiency and execution speed** while maintaining 100% functional correctness.

{self._get_green_context()}

## CONTEXT

**Repository:** `{context.repo_name}`

{target_info}

**Problem Description:**
{context.problem_description}

## TARGET CODE

{code_section}

{self._get_cot_instructions_oracle()}
'''
        return prompt.strip()
    
    def _generate_realistic_prompt(self, context: PromptContext) -> str:
        """Generate REALISTIC mode CoT prompt."""
        
        # Build code context
        code_section = self._format_code_files(context.code_files)
        
        # Build repo map if available
        repo_map_section = ""
        if context.repo_map:
            repo_map_section = f'''## REPOSITORY STRUCTURE

```
{context.repo_map}
```
'''
        
        prompt = f'''You are an expert Green Software Engineer specializing in energy-efficient code optimization.

## TASK

A performance regression has been detected in this codebase. Your task is to:
1. Analyze the retrieved code to find the performance bottleneck
2. Apply targeted optimizations to improve energy efficiency and execution speed

{self._get_green_context()}

## CONTEXT

**Repository:** `{context.repo_name}`

**Performance Issue:**
{context.problem_description}

{repo_map_section}

## RETRIEVED CODE CONTEXT

The following files were retrieved based on the failing tests.
**WARNING:** Some files may be NOISE - not all files are relevant to the performance issue.
You must identify which file(s) contain the actual bottleneck.

{code_section}

{self._get_cot_instructions_realistic()}
'''
        return prompt.strip()
    
    def _format_code_files(self, code_files: Dict[str, str]) -> str:
        """Format code files for prompt."""
        if not code_files:
            return "*No code files provided*"
        
        sections = []
        for filepath, content in code_files.items():
            sections.append(f"[start of {filepath}]\n{content}\n[end of {filepath}]")
        
        return "\n\n".join(sections)
    
    def _get_green_context(self) -> str:
        """Get green software optimization context."""
        return '''## GREEN SOFTWARE OPTIMIZATION GOALS

Focus on optimizations that reduce:
- **CPU cycles** (algorithmic efficiency, reduced complexity)
- **Memory allocations** (reuse objects, avoid copies)
- **I/O operations** (batch operations, caching)
- **Energy consumption** (fewer instructions = less power)

Remember: The most efficient code is code that doesn't run. Eliminate redundant computations.'''


# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

class CoTOracleTemplate(ChainOfThoughtTemplate):
    """CoT template that always uses ORACLE mode."""
    
    def generate_prompt(self, context: PromptContext) -> str:
        # Force ORACLE mode
        context.problem_statement_type = ProblemStatementType.ORACLE
        return self._generate_oracle_prompt(context)


class CoTRealisticTemplate(ChainOfThoughtTemplate):
    """CoT template that always uses REALISTIC mode."""
    
    def generate_prompt(self, context: PromptContext) -> str:
        # Force REALISTIC mode
        context.problem_statement_type = ProblemStatementType.REALISTIC
        return self._generate_realistic_prompt(context)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ChainOfThoughtTemplate',
    'CoTOracleTemplate', 
    'CoTRealisticTemplate',
    'CoTResponse',
    'parse_cot_response',
    'extract_patch_from_cot'
]