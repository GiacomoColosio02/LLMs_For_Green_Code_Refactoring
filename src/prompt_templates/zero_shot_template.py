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
        # 1. SYSTEM PERSONA (Green Adaptation)
        # Defines the role and primary objective before the specific task.
        system_persona = (
            "You are an expert in Green Software Engineering. "
            "Your goal is to refactor code to minimize energy consumption and carbon emissions "
            "while maintaining strict functional correctness.\n"
        )

        # 2. PREMISE (Standard SWE-perf)
        premise = self._get_sweperf_header()

        # 3. PROBLEM STATEMENT & GOALS
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
            # Realistic: Focus on symptoms (tests) and autonomous detection
            problem_body = (
                f"REALISTIC SETTING: The following tests are showing poor energy performance:\n"
                f"{context.test_command}\n\n"
                f"{green_guidelines}\n"
                "Analyze the provided repository context (files retrieved via BM25), "
                "identify the bottleneck causing the high consumption, and optimize it."
            )

        problem_block = f"<problem_statement>\n{problem_body}\n</problem_statement>"

        # 4. CODE CONTEXT
        # Line numbers are False to facilitate copy-paste for SEARCH/REPLACE blocks
        code_block = f"<code>\n{context.get_formatted_code(add_line_numbers=False)}\n</code>"

        # 5. FINAL ASSEMBLY
        final_prompt = [
            system_persona,  # <--- Added: Sets the expert role
            premise,         # <--- Standard instruction
            problem_block,   # <--- The task + Green Goals
            "",
            code_block,      # <--- The code to fix
            "",
            self._get_search_replace_format_instruction() # <--- Mandatory output format
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