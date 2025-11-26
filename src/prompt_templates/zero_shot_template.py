"""
Zero-Shot prompt template (single-turn optimization)
"""
from typing import Dict, List
import re

from .base_template import (
    BasePromptTemplate,
    PromptStrategy,
    PromptContext,
    ProblemStatementType
)


class ZeroShotTemplate(BasePromptTemplate):
    """Single-turn direct optimization request"""
    
    def __init__(self):
        super().__init__(PromptStrategy.ZERO_SHOT)
    
    def generate_prompt(self, context: PromptContext) -> str:
        """Generate single comprehensive prompt"""
        
        system_prompt = self._get_system_prompt()
        problem_statement = self._format_problem_statement(context)
        task_instructions = self._get_task_instructions(context)
        output_format = self._format_output_instructions()
        
        # Combine all sections
        full_prompt = f"""
{system_prompt}

{problem_statement}

{task_instructions}

{output_format}

Now, generate the optimized code patch.
"""
        
        return full_prompt.strip()
    
    def _get_system_prompt(self) -> str:
        """System-level instructions"""
        return """
# Code Performance Optimization Task

You are an expert software engineer specializing in performance optimization and green software engineering.

Your goal is to optimize the provided code to:
- Reduce execution time
- Minimize memory usage
- Decrease energy consumption
- Lower carbon emissions

While maintaining:
- Functional correctness (all tests must pass)
- Code readability and maintainability
- Existing API contracts
"""
    
    def _get_task_instructions(self, context: PromptContext) -> str:
        """Task-specific instructions"""
        
        if context.problem_statement_type == ProblemStatementType.ORACLE:
            focus = "Focus on optimizing the provided target functions."
        else:
            focus = "Analyze the entire code context and identify the best optimization opportunities."
        
        return f"""
## Your Task

{focus}

Consider these optimization strategies:
1. **Algorithmic improvements**: Better data structures, algorithms with lower complexity
2. **Memory efficiency**: Reduce allocations, reuse objects, use generators
3. **Computation reduction**: Eliminate redundant calculations, cache results
4. **I/O optimization**: Batch operations, reduce disk/network access
5. **Concurrency**: Use parallelization where appropriate

**Test Command:**
```bash
{context.test_command}
```

Ensure your optimized code passes all existing tests.
"""
    
    def extract_code_from_response(self, response: str) -> str:
        """
        Extract unified diff patch from LLM response
        
        Looks for code blocks marked as 'diff' or between --- and +++
        """
        # Try to find diff code block
        diff_pattern = r'```diff\n(.*?)```'
        matches = re.findall(diff_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Try to find anything between --- and +++ (unified diff format)
        diff_lines = []
        in_diff = False
        for line in response.split('\n'):
            if line.startswith('---'):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
            if line.startswith('+++') and len(diff_lines) > 1:
                # Found start of diff, continue capturing
                continue
        
        if diff_lines:
            return '\n'.join(diff_lines).strip()
        
        # Fallback: return entire response if no clear diff found
        return response.strip()


class ZeroShotOracleTemplate(ZeroShotTemplate):
    """Zero-Shot template specialized for ORACLE mode"""
    
    def generate_prompt(self, context: PromptContext) -> str:
        # Override to add Oracle-specific hints
        base_prompt = super().generate_prompt(context)
        
        oracle_hint = """
**IMPORTANT**: You have been given the exact functions that experts identified as optimization targets. Focus your efforts on these specific functions.
"""
        # Insert hint after system prompt
        return base_prompt.replace(
            "# Your Task",
            f"{oracle_hint}\n# Your Task"
        )


class ZeroShotRealisticTemplate(ZeroShotTemplate):
    """Zero-Shot template specialized for REALISTIC mode"""
    
    def generate_prompt(self, context: PromptContext) -> str:
        # Override to add Realistic-specific hints
        base_prompt = super().generate_prompt(context)
        
        realistic_hint = """
**IMPORTANT**: You have access to the full repository. The listed functions are executed during tests, but the performance bottleneck might be in functions they call or in data preparation steps. Analyze carefully before optimizing.
"""
        # Insert hint after system prompt
        return base_prompt.replace(
            "# Your Task",
            f"{realistic_hint}\n# Your Task"
        )