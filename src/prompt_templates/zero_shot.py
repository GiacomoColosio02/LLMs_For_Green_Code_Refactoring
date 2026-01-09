"""
Zero-Shot Prompting Strategy (Unified).
Handles both ORACLE and REALISTIC settings via PromptContext.

ORACLE: LLM receives exact target files (gold context) - pure optimization task.
REALISTIC: LLM receives retrieved files + repo map - must identify bottleneck first.

Version: 2.0 - Enhanced with explicit output format examples
"""
from typing import Union, List, Dict
from .base_template import BasePromptTemplate, PromptStrategy, PromptContext, ProblemStatementType


class ZeroShotTemplate(BasePromptTemplate):
    """
    Unified Zero-Shot template for green code optimization.
    
    Automatically adapts prompt based on context.problem_statement_type:
    - ORACLE: Direct optimization with gold files
    - REALISTIC: Analysis + optimization with retrieved files
    """
    
    def __init__(self):
        super().__init__(PromptStrategy.ZERO_SHOT)
    
    def generate_prompt(self, context: PromptContext) -> str:
        """
        Generate prompt based on context type.
        
        Args:
            context: PromptContext with problem_statement_type set
            
        Returns:
            Complete prompt string
        """
        if context.problem_statement_type == ProblemStatementType.REALISTIC:
            return self._generate_realistic_prompt(context)
        else:
            return self._generate_oracle_prompt(context)
    
    # =========================================================================
    # ORACLE PROMPT (Gold Context - Direct Optimization)
    # =========================================================================
    
    def _generate_oracle_prompt(self, context: PromptContext) -> str:
        """
        Generate ORACLE prompt: LLM knows exactly which files to modify.
        Focus on pure optimization without retrieval noise.
        """
        sections = []
        
        # 1. Role & Task
        sections.append(self._get_oracle_header())
        
        # 2. Critical Rules
        sections.append(self._get_optimization_rules())
        
        # 3. Problem Description
        if context.repo_name:
            sections.append(f"### Repository: `{context.repo_name}`\n")
        
        sections.append("### Optimization Goal:")
        sections.append(context.problem_description)
        sections.append("")
        
        # 4. Target Code (Gold Files)
        sections.append("### Target Code Files:")
        sections.append("The following files contain the code that needs optimization.\n")
        sections.append(context.get_formatted_code())
        sections.append("")
        
        # 5. Output Format
        sections.append(self._get_search_replace_format_instruction())
        
        return "\n".join(sections)
    
    def _get_oracle_header(self) -> str:
        """Header for ORACLE setting."""
        return (
            "You are an expert Green Software Engineer specializing in energy-efficient Python code.\n"
            "Your task is to optimize the provided code for **energy efficiency and execution speed** "
            "while maintaining 100% functional correctness.\n"
        )
    
    # =========================================================================
    # REALISTIC PROMPT (Retrieved Context - Analysis + Optimization)
    # =========================================================================
    
    def _generate_realistic_prompt(self, context: PromptContext) -> str:
        """
        Generate REALISTIC prompt: LLM must analyze retrieved files to find bottleneck.
        Includes repo map and warning about noise in retrieved files.
        """
        sections = []
        
        # 1. Role & Task
        sections.append(self._get_realistic_header())
        
        # 2. Critical Rules (with noise warning)
        sections.append(self._get_realistic_rules())
        
        # 3. Repository Structure (if available)
        if context.repo_map:
            sections.append("### Repository Structure:")
            sections.append("```")
            sections.append(context.repo_map)
            sections.append("```")
            sections.append("")
        
        # 4. Problem Description / Failing Tests
        sections.append("### Performance Issue:")
        sections.append(context.problem_description)
        sections.append("")
        
        # 5. Retrieved Code Context
        sections.append("### Retrieved Code Context:")
        sections.append(
            "The following files were retrieved based on the test code. "
            "**Note:** Some files may be irrelevant - analyze carefully.\n"
        )
        sections.append(context.get_formatted_code())
        sections.append("")
        
        # 6. Output Format
        sections.append(self._get_search_replace_format_instruction())
        
        return "\n".join(sections)
    
    def _get_realistic_header(self) -> str:
        """Header for REALISTIC setting."""
        return (
            "You are an expert Green Software Engineer specializing in energy-efficient Python code.\n"
            "You have identified a performance regression in the codebase based on failing tests.\n\n"
            "### TASK:\n"
            "1. Analyze the retrieved code context to identify the performance bottleneck\n"
            "2. Apply targeted optimizations to improve energy efficiency and execution speed\n"
        )
    
    def _get_realistic_rules(self) -> str:
        """Rules for REALISTIC setting (includes noise warning)."""
        return (
            "### CRITICAL RULES:\n"
            "1. **Analyze Retrieved Files Carefully:** The context contains files found by code search. "
            "Some files may be NOISE - focus only on code relevant to the performance issue.\n"
            "2. **Output ONLY the code patch** using SEARCH/REPLACE format (see below).\n"
            "3. **Use SMALL, UNIQUE SEARCH blocks** - include only enough lines to locate the code uniquely. "
            "Do NOT copy entire functions or files.\n"
            "4. **Do NOT import new external libraries** unless already present in the file.\n"
            "5. **Do NOT modify test files** - only optimize the source code.\n"
            "6. Focus on: algorithmic efficiency, reducing redundant computations, memory usage, I/O optimization.\n"
        )
    
    # =========================================================================
    # SHARED COMPONENTS
    # =========================================================================
    
    def _get_optimization_rules(self) -> str:
        """Optimization rules for ORACLE setting."""
        return (
            "### CRITICAL RULES:\n"
            "1. **Output ONLY the code patch** using SEARCH/REPLACE format (see below).\n"
            "2. **Use SMALL, UNIQUE SEARCH blocks** - include only enough context to locate the code uniquely. "
            "Do NOT copy entire functions or files into SEARCH blocks.\n"
            "3. **Do NOT import new external libraries** (numpy, pandas, aiohttp, etc.) "
            "unless they are ALREADY imported in the file.\n"
            "4. **Do NOT change function signatures** - maintain API compatibility.\n"
            "5. **Do NOT modify test files** - only optimize the source code.\n"
            "6. Focus on: algorithmic efficiency, reducing redundant computations, "
            "efficient data structures, memory optimization.\n"
        )
    
    def _get_search_replace_format_instruction(self) -> str:
        """
        SEARCH/REPLACE format instructions.
        Enhanced with multiple examples and common mistakes to avoid.
        """
        return '''### OUTPUT FORMAT (CRITICAL - FOLLOW EXACTLY)

Generate your changes using *SEARCH/REPLACE* blocks with this exact format:

### path/to/file.py
<<<<<<< SEARCH
[exact lines from the original file to find]
=======
[your optimized replacement code]
>>>>>>> REPLACE

---

**CORRECT EXAMPLES:**

**Example 1: List comprehension optimization**

### myproject/utils.py
<<<<<<< SEARCH
def process_items(items):
    result = []
    for item in items:
        result.append(transform(item))
    return result
=======
def process_items(items):
    return [transform(item) for item in items]
>>>>>>> REPLACE

**Example 2: Adding caching**

### myproject/compute.py
<<<<<<< SEARCH
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
=======
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
>>>>>>> REPLACE

**Example 3: Multiple changes in same file**

### myproject/data.py
<<<<<<< SEARCH
    for key in dict.keys():
        process(key)
=======
    for key in dict:
        process(key)
>>>>>>> REPLACE

### myproject/data.py
<<<<<<< SEARCH
    if len(items) > 0:
=======
    if items:
>>>>>>> REPLACE

---

**WRONG - DO NOT DO THIS:**

WRONG (missing file path):
<<<<<<< SEARCH
def foo():
=======
def foo():
>>>>>>> REPLACE

WRONG (wrapped in code blocks):
```python
### file.py
<<<<<<< SEARCH
```

WRONG (explanation instead of code):
To optimize this, you should use a dictionary...

---

**RULES:**
1. Always specify file path on line before <<<<<<< SEARCH
2. SEARCH must match original code EXACTLY (including whitespace)
3. Keep SEARCH blocks SMALL - just enough to locate uniquely
4. Multiple SEARCH/REPLACE blocks allowed for different changes
5. Do NOT wrap in ```python``` code blocks
6. Start your response directly with ### path/to/file.py
'''

    def extract_code_from_response(self, response: str) -> str:
        """
        Extract code from LLM response.
        Actual parsing is done by PatchEngine, this just returns raw response.
        """
        return response


# =============================================================================
# CONVENIENCE ALIASES (backward compatibility)
# =============================================================================

class ZeroShotOracleTemplate(ZeroShotTemplate):
    """Alias for backward compatibility with existing runners."""
    
    def generate_prompt(self, context: PromptContext) -> str:
        # Force ORACLE mode
        context.problem_statement_type = ProblemStatementType.ORACLE
        return super().generate_prompt(context)


class ZeroShotRealisticTemplate(ZeroShotTemplate):
    """Alias for backward compatibility with existing runners."""
    
    def generate_prompt(self, context: PromptContext) -> str:
        # Force REALISTIC mode
        context.problem_statement_type = ProblemStatementType.REALISTIC
        return super().generate_prompt(context)