"""
Chain-of-Thought prompt template for Green Code Refactoring.
"""
from typing import Dict
from .base_template import BasePromptTemplate, PromptStrategy, PromptContext, ProblemStatementType

class CoTTemplate(BasePromptTemplate):
    """
    Implementation of CoT Strategy.
    Reference: Wei et al. (2022) - Chain-of-Thought Prompting
    """
    
    def __init__(self):
        super().__init__(PromptStrategy.COT)
    
    def generate_prompt(self, context: PromptContext) -> str:
        premise = self._get_sweperf_header()
        
        # Context instructions specific to setting
        if context.problem_statement_type == ProblemStatementType.ORACLE:
            context_instr = f"TARGETS: {context.get_target_functions_str()}"
        else:
            context_instr = (
                f"TARGETS: The performance issue is triggered by these tests:\n{context.test_command}\n"
                "The provided code includes files retrieved via BM25. Some might be irrelevant noise."
            )

        # The core CoT Instruction
        cot_instructions = """
Before writing the patch, you MUST perform a Deep Analysis following these steps.
Output your reasoning in an 'ANALYSIS:' section.

ANALYSIS GUIDELINES:
1. **Context Analysis**: 
   - Identify which provided file contains the computationally intensive logic ("Hotspot").
   - If in Realistic mode, explicitly filter out irrelevant files.

2. **Green Impact Assessment**:
   - Identify the specific inefficiency (e.g., O(N^2) loop, redundant memory allocation).
   - Hypothesize why this consumes excess Energy (CPU/RAM).

3. **Optimization Strategy**:
   - Propose a concrete refactoring plan to reduce Joules/Time.

PATCH:
(Only after the analysis, provide the SEARCH/REPLACE block)
"""

        final_prompt = [
            premise,
            "<problem_context>",
            context.problem_description,
            context_instr,
            "</problem_context>",
            "",
            "<code>",
            context.get_formatted_code(add_line_numbers=False),
            "</code>",
            "",
            cot_instructions,
            self._get_search_replace_format_instruction()
        ]
        
        return "\n".join(final_prompt)

    def extract_code_from_response(self, response: str) -> str:
        # Extract only the part after "PATCH:" to ignore reasoning
        if "PATCH:" in response:
            patch_part = response.split("PATCH:")[-1]
            return patch_part
        return response