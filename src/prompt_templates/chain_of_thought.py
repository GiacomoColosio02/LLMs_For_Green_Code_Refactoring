"""
Chain-of-Thought (CoT) Prompt Templates for Green Code Optimization.

Version: 3.0 - Simplified to match ZS success rate
"""

import re
from typing import Optional, Dict, List
from dataclasses import dataclass

from .base_template import BasePromptTemplate, PromptContext, ProblemStatementType, PromptStrategy


def extract_patch_from_cot(response: str) -> str:
    """Extract only the patch section from a CoT response."""
    # Remove <think> tags if present (for reasoning models)
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove markdown code block wrappers
    clean_response = re.sub(r'```(?:python|diff|text)?\s*\n?', '', clean_response)
    clean_response = re.sub(r'\n?```', '', clean_response)
    
    # Strategy 1: Find SECTION 2 marker and extract everything after
    section2_match = re.search(
        r'(?:SECTION\s*2|##\s*SECTION\s*2|###\s*SECTION\s*2)\s*:?\s*(?:PATCH)?(.*)$',
        clean_response, re.IGNORECASE | re.DOTALL
    )
    if section2_match:
        patch_content = section2_match.group(1).strip()
        if '<<<<<<< SEARCH' in patch_content:
            return patch_content
    
    # Strategy 2: Find first file path marker (### path/to/file.py) followed by SEARCH
    file_patch_match = re.search(
        r'(###\s+[\w/._-]+\.py\s*\n\s*<<<<<<< SEARCH.*)',
        clean_response, re.DOTALL
    )
    if file_patch_match:
        return file_patch_match.group(1).strip()
    
    # Strategy 3: Find first <<<<<<< SEARCH and include file path before it
    search_match = re.search(r'<<<<<<< SEARCH', clean_response)
    if search_match:
        pre_search = clean_response[:search_match.start()]
        
        # Look for file path marker before SEARCH
        file_path_match = re.search(r'(###\s+[\w/._-]+\.py\s*\n?)\s*$', pre_search)
        if file_path_match:
            start_pos = file_path_match.start()
        else:
            # Try to find any file path in the line before
            lines = pre_search.rstrip().split('\n')
            for i in range(len(lines) - 1, -1, -1):
                if re.match(r'###\s+[\w/._-]+\.py', lines[i]):
                    start_pos = pre_search.rfind(lines[i])
                    break
            else:
                start_pos = search_match.start()
        
        return clean_response[start_pos:].strip()
    
    # Fallback: return everything (let PatchEngine handle it)
    return clean_response.strip()


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
        """Extract code/patch from LLM response."""
        return extract_patch_from_cot(response)
    
    def _get_cot_instructions(self) -> str:
        """Simplified CoT instructions - analysis is brief, patch format matches ZS exactly."""
        return '''## RESPONSE FORMAT

Your response has TWO parts:

**PART 1 - BRIEF ANALYSIS (2-3 lines max):**
Write "ANALYSIS:" then in 2-3 lines explain: what is slow and why.

**PART 2 - PATCH (main output):**
Write "PATCH:" then provide code changes using EXACT format below.

---

## PATCH FORMAT (CRITICAL - MUST MATCH EXACTLY)

### path/to/file.py
<<<<<<< SEARCH
[exact original code to find]
=======
[your optimized replacement]
>>>>>>> REPLACE

---

## COMPLETE EXAMPLE

ANALYSIS:
The `find_duplicates` function uses O(n²) nested loops. Using a set gives O(n).

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

## RULES
- Keep ANALYSIS to 2-3 lines MAX
- File path line MUST come immediately before <<<<<<< SEARCH
- SEARCH block must match original code EXACTLY
- Do NOT wrap in ```python``` code blocks
- Do NOT add external dependencies
- Do NOT modify test files
'''

    def _generate_oracle_prompt(self, context: PromptContext) -> str:
        code_section = self._format_code_files(context.code_files)
        
        prompt = f'''You are an expert Green Software Engineer.

## TASK
Optimize the provided code for energy efficiency and execution speed.
First briefly analyze, then provide the patch.

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
First briefly analyze to identify the bottleneck, then provide the patch.

## CONTEXT
**Repository:** `{context.repo_name}`
**Problem:** {context.problem_description}

{repo_map_section}

## RETRIEVED CODE (some files may be noise - identify the real bottleneck)
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


__all__ = [
    'ChainOfThoughtTemplate', 'CoTOracleTemplate', 'CoTRealisticTemplate',
    'extract_patch_from_cot'
]