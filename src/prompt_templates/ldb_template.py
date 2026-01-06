"""
LDB (Large Language Model Debugger) prompt template for Green Code Refactoring.
Strategy: Iterative refinement based on Runtime Execution Feedback.
"""
from typing import Dict, Any, Optional
import json
from .base_template import BasePromptTemplate, PromptStrategy, PromptContext, ProblemStatementType
from .zero_shot_template import ZeroShotTemplate

class LDBTemplate(BasePromptTemplate):
    """
    Implementation of LDB (LLM Debugger) Strategy.
    
    Scientific Reference:
    - Zhong et al. (2024): "LDB: A Large Language Model Debugger via Verifying Runtime Execution Step-by-step"
    
    Workflow:
    1. Initial Generation (via Zero-Shot)
    2. Execution & Measurement (via measure_instance.py)
    3. Feedback Loop (LDB Prompt with runtime metrics)
    """
    
    def __init__(self):
        super().__init__(PromptStrategy.LDB)
        self._initial_template = ZeroShotTemplate()
    
    def generate_prompt(self, context: PromptContext) -> str:
        """
        Standard interface. For LDB, the first turn is identical to Zero-Shot.
        """
        return self._initial_template.generate_prompt(context)

    def generate_debugging_prompt(
        self, 
        context: PromptContext, 
        previous_patch: str, 
        feedback: str
    ) -> str:
        """
        Generates the Debugging Prompt for subsequent iterations.
        
        Args:
            context: The prompt context
            previous_patch: The code generated in the previous failed/suboptimal attempt
            feedback: Structured string containing runtime metrics and errors
            
        Returns:
            The complete prompt for the LDB iteration
        """
        # 1. SYSTEM PERSONA
        system_persona = (
            "You are a Code Debugger and Green Optimization Expert. "
            "Your task is to use actual runtime execution data to fix and optimize code.\n"
        )
        
        # 2. PREVIOUS ATTEMPT CONTEXT
        # We explicitly show what didn't work well
        previous_attempt_block = (
            "<previous_attempt>\n"
            "The following patch was applied but produced suboptimal results or errors:\n"
            f"{previous_patch}\n"
            "</previous_attempt>\n"
        )
        
        # 3. RUNTIME FEEDBACK (The Core of LDB)
        # This contains the 'Ground Truth' from measure_instance.py
        feedback_block = (
            "<execution_feedback>\n"
            "Runtime analysis from the test server:\n"
            f"{feedback}\n"
            "</execution_feedback>\n"
        )
        
        # 4. DIAGNOSIS INSTRUCTIONS
        task_instructions = (
            "INSTRUCTIONS:\n"
            "1. **Diagnose**: Analyze the feedback. Why did energy/performance not improve? "
            "Is there a regression?\n"
            "2. **Correct & Optimize**: Propose a NEW patch. Do not simply repeat the old one.\n"
            "3. **Justify**: Explain briefly how the new patch addresses the specific feedback metrics.\n\n"
            "Output Format:\n"
            "DIAGNOSIS: [Your analysis]\n"
            "PATCH:\n"
            "[SEARCH/REPLACE block]\n"
        )
        
        # 5. CURRENT CODE CONTEXT
        # We provide the original code again for reference
        code_block = f"<code>\n{context.get_formatted_code(add_line_numbers=False)}\n</code>"

        # 6. ASSEMBLY
        final_prompt = [
            system_persona,
            "<problem_statement>",
            f"{context.problem_description}",
            "</problem_statement>",
            "",
            previous_attempt_block,
            feedback_block,
            task_instructions,
            "",
            code_block,
            "",
            self._get_search_replace_format_instruction()
        ]
        
        return "\n".join(final_prompt)

    def extract_code_from_response(self, response: str) -> str:
        """
        Robust extraction logic compatible with LDB format.
        """
        # Look for PATCH section first
        if "PATCH:" in response:
            parts = response.split("PATCH:")
            candidate = parts[-1]
            if "<<<<<<< SEARCH" in candidate:
                return candidate
        
        # Fallback to standard search block
        if "<<<<<<< SEARCH" in response:
            start_index = response.find("<<<<<<< SEARCH")
            return response[start_index:]
            
        return response

    @staticmethod
    def format_feedback_from_measurement(
        measurement_results: Dict[str, Any]
    ) -> str:
        """
        Helper to convert measure_instance.py output JSON into LDB text format.
        
        Args:
            measurement_results: The dictionary loaded from measurements.json
                                 (output of measure_instance.py)
        
        Returns:
            Formatted string for the prompt
        """
        # Handle cases where measurement might have failed completely
        if "error" in measurement_results:
            return f"CRITICAL ERROR: Measurement failed. Reason: {measurement_results['error']}"

        base = measurement_results.get("base", {})
        head = measurement_results.get("head", {})
        
        # 1. Functional Status
        passed = head.get("successful_tests", 0)
        total = head.get("total_tests", 0)
        status = "SUCCESS" if passed == total and total > 0 else "FAILURE"
        
        # 2. Extract Metrics
        # Default to 0 if missing to avoid crashes
        e_base = base.get("cpu_energy_joules", 0)
        e_head = head.get("cpu_energy_joules", 0)
        
        t_base = base.get("duration_seconds", 0)
        t_head = head.get("duration_seconds", 0)
        
        ram_base = base.get("ram_peak_mb", 0)
        ram_head = head.get("ram_peak_mb", 0)
        
        # Calculate Deltas (%)
        def calc_delta(b, h):
            if b == 0: return 0.0
            return ((h - b) / b) * 100

        delta_e = calc_delta(e_base, e_head)
        delta_t = calc_delta(t_base, t_head)
        delta_ram = calc_delta(ram_base, ram_head)
        
        # 3. Construct Feedback String
        feedback_lines = []
        feedback_lines.append(f"1. Test Status: {status} ({passed}/{total} tests passed)")
        
        # Energy Analysis
        target_met = "[TARGET MET]" if delta_e <= -5.0 else "[TARGET MISSED]" # 5% threshold example
        feedback_lines.append(f"2. CPU Energy: {e_base:.2f}J -> {e_head:.2f}J (Change: {delta_e:+.2f}%) {target_met}")
        
        # Duration Analysis
        feedback_lines.append(f"3. Execution Time: {t_base:.2f}s -> {t_head:.2f}s (Change: {delta_t:+.2f}%)")
        
        # Memory Analysis
        feedback_lines.append(f"4. Peak RAM: {ram_base:.2f}MB -> {ram_head:.2f}MB (Change: {delta_ram:+.2f}%)")
        
        # Additional Observations
        if delta_e > 0:
            feedback_lines.append("   ⚠️ WARNING: Energy consumption INCREASED. The patch is less efficient.")
        elif delta_e > -1.0:
            feedback_lines.append("   ℹ️ NOTE: Energy reduction is negligible (< 1%). Needs more aggressive optimization.")
            
        return "\n".join(feedback_lines)