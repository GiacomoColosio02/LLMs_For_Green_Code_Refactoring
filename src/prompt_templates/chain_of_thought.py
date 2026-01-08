"""
Chain-of-Thought (CoT) Prompt Templates for Green Code Optimization.
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
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    analysis_section = ""
    patch_section = ""
    
    section1_match = re.search(
        r'SECTION\s*1\s*:?\s*ANALYSIS(.*?)(?=SECTION\s*2|$)', 
        clean_response, re.IGNORECASE | re.DOTALL
    )
    section2_match = re.search(
        r'SECTION\s*2\s*:?\s*PATCH(.*?)$', 
        clean_response, re.IGNORECASE | re.DOTALL
    )
    
    if section1_match:
        analysis_section = section1_match.group(1).strip()
    if section2_match:
        patch_section = section2_match.group(1).strip()
    
    if not patch_section and "<<<<<<< SEARCH" in clean_response:
        search_start = clean_response.find("<<<<<<< SEARCH")
        pre_search = clean_response[:search_start]
        last_newline = pre_search.rfind('\n')
        patch_section = clean_response[max(0, last_newline):].strip()
        analysis_section = clean_response[:max(0, last_newline)].strip()
    
    has_valid_structure = bool(patch_section and "<<<<<<< SEARCH" in patch_section)
    
    return CoTResponse(
        raw_response=response,
        analysis_section=analysis_section,
        patch_section=patch_section,
        has_valid_structure=has_valid_structure
    )


def extract_patch_from_cot(response: str) -> str:
    """Extract only the patch section from a CoT response."""
    parsed = parse_cot_response(response)
    
    if parsed.patch_section:
        return parsed.patch_section
    
    if "<<<<<<< SEARCH" in response:
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        idx = clean.find("<<<<<<< SEARCH")
        pre_idx = clean.rfind('\n', 0, idx)
        if pre_idx > idx - 200:
            return clean[pre_idx:].strip()
        return clean[idx:].strip()
    
    return response


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
        return '''
## RESPONSE FORMAT

Your response MUST follow this EXACT structure:

### SECTION 1: ANALYSIS

Start by writing: "Let's think step by step."

Then address:
1. **IDENTIFICATION**: What function/code has the inefficiency?
2. **DIAGNOSIS**: Why is it inefficient? (O(N^2), memory spikes, redundant computation, etc.)
3. **HYPOTHESIS**: What will your fix achieve? (O(1) lookup, caching, etc.)

### SECTION 2: PATCH

Only AFTER the analysis, provide code changes:

### path/to/file.py
<<<<<<< SEARCH
original code (minimum unique context)
=======
optimized code
>>>>>>> REPLACE

CRITICAL RULES:
- Use SMALL SEARCH blocks - just enough to locate uniquely
- Preserve functionality - tests MUST still pass
- Do NOT add new dependencies
'''

    def _generate_oracle_prompt(self, context: PromptContext) -> str:
        code_section = self._format_code_files(context.code_files)
        
        prompt = f'''You are an expert Green Software Engineer.

## TASK
Optimize the provided code for energy efficiency and execution speed.

## CONTEXT
**Repository:** `{context.repo_name}`
**Problem:** {context.problem_description}

## TARGET CODE
{code_section}

{self._get_cot_instructions()}
'''
        return prompt.strip()
    
    def _generate_realistic_prompt(self, context: PromptContext) -> str:
        code_section = self._format_code_files(context.code_files)
        repo_map_section = f"## REPO STRUCTURE\n```\n{context.repo_map}\n```" if context.repo_map else ""
        
        prompt = f'''You are an expert Green Software Engineer.

## TASK
Find and fix the performance bottleneck in this codebase.

## CONTEXT
**Repository:** `{context.repo_name}`
**Problem:** {context.problem_description}

{repo_map_section}

## RETRIEVED CODE (some may be noise - identify the hotspot!)
{code_section}

{self._get_cot_instructions()}
'''
        return prompt.strip()
    
    def _format_code_files(self, code_files: Dict[str, str]) -> str:
        if not code_files:
            return "*No code files provided*"
        sections = []
        for filepath, content in code_files.items():
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


__all__ = [
    'ChainOfThoughtTemplate', 'CoTOracleTemplate', 'CoTRealisticTemplate',
    'CoTResponse', 'parse_cot_response', 'extract_patch_from_cot'
]
