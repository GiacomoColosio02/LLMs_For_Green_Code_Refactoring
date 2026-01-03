"""
Zero-Shot prompt template specialized for Green Code Refactoring.
Based on SWE-perf 'prompt_efficiency' structure.
"""
from typing import Dict, List
import re
from .base_template import BasePromptTemplate, PromptStrategy, PromptContext, ProblemStatementType

class ZeroShotTemplate(BasePromptTemplate):
    """
    Implementation of the Zero-Shot strategy.
    Reference: Jimenez et al. (2024) - SWE-bench Baseline
    """
    
    def __init__(self):
        super().__init__(PromptStrategy.ZERO_SHOT)
    
    def generate_prompt(self, context: PromptContext) -> str:
        # 1. PREMISE
        premise = self._get_sweperf_header()

        # 2. PROBLEM STATEMENT (Green Adaptation)
        green_guidelines = (
            "\nGREEN OPTIMIZATION GOALS:\n"
            "1. Reduce CPU Energy Consumption (Joules).\n"
            "2. Reduce Wall-clock Execution Time.\n"
            "3. Minimize Memory Spikes (Peak RAM).\n"
            "4. Maintain 100% functional correctness (pass all tests).\n"
        )

        if context.problem_statement_type == ProblemStatementType.ORACLE:
            targets = context.get_target_functions_str()
            problem_body = (
                f"{context.problem_description}\n"
                f"{green_guidelines}\n"
                f"Focus on optimizing these specific targets:\n{targets}"
            )
        else:
            # Realistic: Focus on symptoms (tests)
            problem_body = (
                f"REALISTIC SETTING: The following tests are showing poor energy performance:\n"
                f"{context.test_command}\n\n"
                f"{green_guidelines}\n"
                "Analyze the provided repository context (files retrieved via BM25), "
                "identify the bottleneck causing the high consumption, and optimize it."
            )

        problem_block = f"<problem_statement>\n{problem_body}\n</problem_statement>"

        # 3. CODE CONTEXT
        code_block = f"<code>\n{context.get_formatted_code(add_line_numbers=False)}\n</code>"

        # 4. FINAL ASSEMBLY
        final_prompt = [
            premise,
            problem_block,
            "",
            code_block,
            "",
            self._get_search_replace_format_instruction()
        ]
        
        return "\n".join(final_prompt)

    def extract_code_from_response(self, response: str) -> str:
        """Extracts the python code block containing the patch."""
        if "```python" in response:
            parts = response.split("```python")
            # Return the last block or scan for SEARCH key
            for part in reversed(parts):
                if "<<<<<<< SEARCH" in part:
                    return f"```python{part.split('```')[0]}```"
            return f"```python{parts[-1].split('```')[0]}```"
        return response