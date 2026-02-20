"""
Chain-of-Thought (CoT) Prompt Templates for Green Code Optimization.

Version: 3.0 - Green Software Oriented
"""

import re
from typing import Optional, Dict, List
from dataclasses import dataclass

from .base_template import BasePromptTemplate, PromptContext, ProblemStatementType, PromptStrategy


@dataclass
class CoTResponse:
    """Parsed Chain-of-Thought response."""
    raw_response: str
    analysis_section: str
    patch_section: str
    has_valid_structure: bool


def parse_cot_response(response: str) -> CoTResponse:
    """Parse a CoT response into its structured components."""
    # Remove <think> tags if present (for reasoning models like DeepSeek R1)
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    analysis_section = ""
    patch_section = ""
    
    # Try multiple patterns for section markers
    section1_match = re.search(
        r'(?:SECTION\s*1|##\s*SECTION\s*1|###\s*SECTION\s*1)\s*:?\s*(?:ANALYSIS)?(.*?)(?=SECTION\s*2|##\s*SECTION\s*2|###\s*SECTION\s*2|<<<<<<< SEARCH|$)', 
        clean_response, re.IGNORECASE | re.DOTALL
    )
    section2_match = re.search(
        r'(?:SECTION\s*2|##\s*SECTION\s*2|###\s*SECTION\s*2)\s*:?\s*(?:PATCH)?(.*?)$', 
        clean_response, re.IGNORECASE | re.DOTALL
    )
    
    if section1_match:
        analysis_section = section1_match.group(1).strip()
    if section2_match:
        patch_section = section2_match.group(1).strip()
    
    # Fallback: if no section markers but SEARCH/REPLACE exists
    if not patch_section and "<<<<<<< SEARCH" in clean_response:
        search_start = clean_response.find("<<<<<<< SEARCH")
        
        # Look for file path marker before SEARCH
        file_path_match = re.search(r'(###\s+[\w/._-]+\.py\s*\n)', clean_response[:search_start])
        if file_path_match:
            patch_start = file_path_match.start()
        else:
            pre_search = clean_response[:search_start]
            last_newline = pre_search.rfind('\n')
            patch_start = max(0, last_newline)
        
        patch_section = clean_response[patch_start:].strip()
        analysis_section = clean_response[:patch_start].strip()
    
    has_valid_structure = bool(patch_section and "<<<<<<< SEARCH" in patch_section)
    
    return CoTResponse(
        raw_response=response,
        analysis_section=analysis_section,
        patch_section=patch_section,
        has_valid_structure=has_valid_structure
    )


def extract_patch_from_cot(response: str) -> str:
    """Extract only the patch section from a CoT response."""
    # First, remove <think>...</think> blocks (DeepSeek R1)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    parsed = parse_cot_response(response)
    
    patch = parsed.patch_section if parsed.patch_section else response
    
    # Remove markdown code block wrappers (```python ... ```)
    patch = re.sub(r'^```(?:python|diff|text)?\s*\n?', '', patch, flags=re.MULTILINE)
    patch = re.sub(r'\n?```\s*$', '', patch, flags=re.MULTILINE)
    
    # Also try to extract from within code blocks if SEARCH is inside
    if "<<<<<<< SEARCH" not in patch and "<<<<<<< SEARCH" in response:
        code_block_match = re.search(r'```(?:python|diff|text)?\s*\n(.*?<<<<<<< SEARCH.*?)```', 
                                      response, re.DOTALL)
        if code_block_match:
            patch = code_block_match.group(1).strip()
    
    # Final cleanup: ensure we have the file path marker
    if "<<<<<<< SEARCH" in patch and "### " not in patch:
        file_match = re.search(r'###\s+([\w/._-]+\.py)', response)
        if file_match:
            patch = f"### {file_match.group(1)}\n{patch}"
    
    return patch.strip()


class ChainOfThoughtTemplate(BasePromptTemplate):
    """Chain-of-Thought template for Green Code Optimization."""
    
    def __init__(self):
        super().__init__(PromptStrategy.COT)
        self.template_name = "ChainOfThought"
    
    def generate_prompt(self, context: PromptContext) -> str:
        if context.problem_statement_type == ProblemStatementType.ORACLE:
            return self._generate_oracle_prompt(context)
        else:
            return self._generate_realistic_prompt(context)
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract code/patch from LLM response - required by base class."""
        return extract_patch_from_cot(response)
    
    def _get_cot_instructions(self) -> str:
        """CoT instructions - green-oriented analysis, strict patch format. DeepSeek R1 compatible."""
        return '''## RESPONSE FORMAT

**IMPORTANT: Do NOT use <think> tags or internal reasoning blocks.**
**Write your analysis directly in plain text, then provide the patch.**

**PART 1 - ENERGY ANALYSIS (3-5 lines only):**
Start with "ANALYSIS:" then briefly explain:
- What code pattern is energy-intensive? (e.g., redundant computations, inefficient algorithm, excessive memory allocation)
- Why does it consume unnecessary energy? (e.g., O(n²) complexity causes excessive CPU cycles)
- What green optimization will you apply to reduce energy consumption?

**PART 2 - PATCH:**
Start with "PATCH:" then provide code changes using this EXACT format (no ```python``` blocks!):

### path/to/file.py
<<<<<<< SEARCH
exact original code to find
=======
your energy-efficient replacement
>>>>>>> REPLACE

---

## EXAMPLE (follow this format exactly)

ANALYSIS:
The `find_duplicates` function in `utils.py` uses O(n²) nested loops, causing excessive CPU cycles and energy waste. Using a set gives O(1) lookups, reducing time complexity to O(n) and significantly lowering CPU energy consumption.

PATCH:
### myproject/utils.py
<<<<<<< SEARCH
def find_duplicates(items):
    duplicates = []
    for i, item in enumerate(items):
        for j, other in enumerate(items):
            if i != j and item == other and item not in duplicates:
                duplicates.append(item)
    return duplicates
=======
def find_duplicates(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)
>>>>>>> REPLACE

---

## CRITICAL RULES
1. **Do NOT use <think> tags** - write analysis directly as plain text
2. Keep analysis SHORT (3-5 lines max) - start with "ANALYSIS:"
3. Focus the analysis on **energy impact**: why the code wastes energy and how your change reduces it
4. Start patch section with "PATCH:" 
5. File path line (### path/to/file.py) MUST come immediately before <<<<<<< SEARCH
6. SEARCH block must match original code EXACTLY (copy-paste from TARGET CODE)
7. Do NOT wrap patch in ```python``` code blocks
8. Do NOT add external dependencies
9. Do NOT modify test files
'''

    def _generate_oracle_prompt(self, context: PromptContext) -> str:
        code_section = self._format_code_files(context.code_files)
        
        prompt = f'''You are an expert Green Software Engineer focused on reducing code energy consumption and environmental impact.

**IMPORTANT: Respond directly without using <think> tags or hidden reasoning blocks.**

The optimized code will be measured with energy profiling tools that track CPU energy, GPU energy, total system power, and carbon emissions. Your changes should minimize these metrics.

## TASK
Analyze the provided code to identify energy-intensive patterns, then provide an optimized patch that reduces energy consumption while maintaining correctness.
First provide a brief ANALYSIS of the energy inefficiency (3-5 lines), then provide the PATCH.

## CONTEXT
**Repository:** `{context.repo_name}`
**Problem:** {context.problem_description}

{self._get_green_software_context()}

## TARGET CODE
{code_section}

{self._get_cot_instructions()}
'''
        return prompt.strip()
    
    def _generate_realistic_prompt(self, context: PromptContext) -> str:
        code_section = self._format_code_files(context.code_files)
        repo_map_section = f"## REPO STRUCTURE\n```\n{context.repo_map}\n```" if context.repo_map else ""
        
        prompt = f'''You are an expert Green Software Engineer focused on reducing code energy consumption and environmental impact.

**IMPORTANT: Respond directly without using <think> tags or hidden reasoning blocks.**

The optimized code will be measured with energy profiling tools that track CPU energy, GPU energy, total system power, and carbon emissions. Your changes should minimize these metrics.

## TASK
Find and fix the energy-intensive code pattern in this codebase.
First provide a brief ANALYSIS (3-5 lines) identifying which code wastes the most energy and why, then provide the PATCH.

## CONTEXT
**Repository:** `{context.repo_name}`
**Problem:** {context.problem_description}

{repo_map_section}

## RETRIEVED CODE (some files may be noise - identify the energy-intensive bottleneck)
{code_section}

{self._get_cot_instructions()}
'''
        return prompt.strip()
    
    def _format_code_files(self, code_files: Dict[str, str]) -> str:
        if not code_files:
            return "*No code files provided*"
        sections = []
        for filepath, content in sorted(code_files.items()):
            sections.append(f"[start of {filepath}]\n{content}\n[end of {filepath}]")
        return "\n\n".join(sections)


class CoTOracleTemplate(ChainOfThoughtTemplate):
    def generate_prompt(self, context: PromptContext) -> str:
        context.problem_statement_type = ProblemStatementType.ORACLE
        return self._generate_oracle_prompt(context)


class CoTRealisticTemplate(ChainOfThoughtTemplate):
    def generate_prompt(self, context: PromptContext) -> str:
        context.problem_statement_type = ProblemStatementType.REALISTIC
        return self._generate_realistic_prompt(context)


# Backward compatibility alias
CoTTemplate = ChainOfThoughtTemplate

__all__ = [
    'ChainOfThoughtTemplate', 'CoTTemplate', 'CoTOracleTemplate', 'CoTRealisticTemplate',
    'CoTResponse', 'parse_cot_response', 'extract_patch_from_cot'
]