"""
Chain-of-Thought prompt template for Green Code Refactoring.
"""
from typing import Dict
from .base_template import BasePromptTemplate, PromptStrategy, PromptContext, ProblemStatementType

class CoTTemplate(BasePromptTemplate):
    """
    Implementation of Chain-of-Thought (CoT) Strategy.
    
    Scientific References:
    - Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    - Kojima et al. (2022): "Large Language Models are Zero-Shot Reasoners" (Source of 'Let's think step by step')
    """
    
    def __init__(self):
        super().__init__(PromptStrategy.COT)
    
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
        
        # 3. PROBLEM CONTEXT & TARGETS
        if context.problem_statement_type == ProblemStatementType.ORACLE:
            context_instr = f"TARGETS: {context.get_target_functions_str()}"
        else:
            context_instr = (
                f"TARGETS: The performance issue is triggered by these tests:\n{context.test_command}\n"
                "The provided code includes files retrieved via BM25. Some might be irrelevant noise."
            )

        # 4. CoT INSTRUCTIONS (The Core Mechanism)
        # We explicitly inject "Let's think step by step" to trigger Zero-Shot CoT.
        cot_instructions = """
INSTRUCTIONS FOR REASONING & OUTPUT:
You must strictly follow this format. Do not output the patch immediately.

SECTION 1: ANALYSIS
Start this section by writing: "Let's think step by step."
Analyze the code following these Green Refactoring pillars:
1. **Identification**: Locate the exact "Hotspot" (loops, recursion, heavy I/O) in the provided files.
   (If in Realistic mode, filter out the noise files).
2. **Diagnosis**: Explain WHY it consumes excess energy (e.g., "O(N^2) complexity on CPU", "Redundant object allocation in RAM").
3. **Hypothesis**: Predict the impact of your fix (e.g., "Changing List to Set will reduce lookup energy by X%").

SECTION 2: PATCH
Only after the analysis is complete, provide the SEARCH/REPLACE block for the fix.
"""

        # 5. CODE CONTEXT
        code_block = f"<code>\n{context.get_formatted_code(add_line_numbers=False)}\n</code>"

        # 6. FINAL ASSEMBLY
        final_prompt = [
            system_persona,
            premise,
            "<problem_context>",
            context.problem_description,
            context_instr,
            "</problem_context>",
            "",
            code_block,
            "",
            cot_instructions,
            self._get_search_replace_format_instruction()
        ]
        
        return "\n".join(final_prompt)

    def extract_code_from_response(self, response: str) -> str:
        """
        Extracts only the patch part, ignoring the CoT analysis.
        Robustly handles cases where the model might be chatty.
        """
        # Strategy 1: Split by explicit section header "SECTION 2: PATCH" or just "PATCH"
        # We look for the last occurrence of PATCH to avoid false positives in the analysis text
        if "PATCH" in response:
            # We assume the patch comes after the analysis
            parts = response.split("PATCH")
            candidate = parts[-1] 
            if "<<<<<<< SEARCH" in candidate:
                return candidate
        
        # Strategy 2: Fallback - Look for the standard SEARCH block directly
        # This catches cases where the model forgets the "SECTION 2" header but provides the code
        if "<<<<<<< SEARCH" in response:
            start_index = response.find("<<<<<<< SEARCH")
            # We try to find the filename line which usually precedes the search block
            # But strictly returning from SEARCH is safer for the parser
            
            # Optional: Try to include the filename line if it's immediately before
            # For now, returning from SEARCH is safe as SWE-perf parser handles it
            return response[start_index:]
            
        return response