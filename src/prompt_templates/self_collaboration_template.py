"""
Multi-turn Self-Collaboration prompt strategies.
Ref: Du et al. (2024) - "AgentCoder" & Dong et al. (2023) - "Self-Collaboration"
"""
from typing import List, Dict
from .base_template import BasePromptTemplate, PromptStrategy, PromptContext, ProblemStatementType

class SelfCollaborationTemplate(BasePromptTemplate):
    """
    Implements the Multi-Turn Self-Collaboration strategy.
    Orchestrates a dialogue between:
    1. Sustainability Analyst (Diagnosis)
    2. Senior Refactoring Engineer (Implementation)
    3. Critical Reviewer (Verification)
    4. Senior Refactoring Engineer (Final Polish)
    """
    
    def __init__(self):
        super().__init__(PromptStrategy.SELF_COLLABORATION)
        
    def generate_prompt(self, context: PromptContext) -> str:
        """
        Standard interface method. 
        For Self-Collaboration, this returns the INITIAL System Prompt (Turn 1).
        The orchestrator must call specific methods for subsequent turns.
        """
        return self.get_analyst_prompt(context)

    # ==========================================
    # TURN 1: SUSTAINABILITY ANALYST
    # ==========================================
    def get_analyst_prompt(self, context: PromptContext) -> str:
        system_persona = (
            "You are a Sustainability Analyst specialized in Green Software. "
            "Your role is NOT to write code, but to diagnose inefficiencies.\n"
        )
        
        premise = self._get_sweperf_header()
        
        if context.problem_statement_type == ProblemStatementType.ORACLE:
            targets = f"Focus targets: {context.get_target_functions_str()}"
        else:
            targets = f"Failing tests (symptoms): {context.test_command}"

        task = (
            "TASK:\n"
            "Analyze the provided code and identify specific 'Optimization Goals' to reduce "
            "Energy (CPU/GPU Joules) and Time.\n"
            "Focus on:\n"
            "1. Algorithmic complexity (Big-O).\n"
            "2. Memory hotspots (redundant allocations).\n"
            "3. Power spikes (heavy loops).\n\n"
            "OUTPUT FORMAT:\n"
            "Provide a bulleted list of 3-5 clear, actionable Optimization Goals."
        )

        code = f"<code>\n{context.get_formatted_code()}\n</code>"
        
        return "\n".join([system_persona, premise, targets, task, "", code])

    # ==========================================
    # TURN 2: REFACTORING ENGINEER (Initial)
    # ==========================================
    def get_engineer_prompt(self, context: PromptContext, analyst_output: str) -> str:
        system_persona = (
            "You are a Senior Refactoring Engineer. "
            "Your goal is to implement the optimizations proposed by the Analyst.\n"
        )
        
        instructions = (
            "INSTRUCTIONS:\n"
            "Based on the original code and the Analyst's goals below, write the optimized code.\n"
            "Strictly follow the SEARCH/REPLACE format."
        )
        
        analyst_context = f"<analyst_goals>\n{analyst_output}\n</analyst_goals>"
        code = f"<code>\n{context.get_formatted_code()}\n</code>"
        
        return "\n".join([
            system_persona, 
            instructions, 
            analyst_context, 
            code, 
            "", 
            self._get_search_replace_format_instruction()
        ])

    # ==========================================
    # TURN 3: CRITICAL REVIEWER
    # ==========================================
    def get_reviewer_prompt(self, context: PromptContext, analyst_goals: str, engineer_patch: str) -> str:
        system_persona = (
            "You are a Critical Code Reviewer. "
            "Your job is to find bugs or missed green opportunities in the proposed patch.\n"
        )
        
        task = (
            "TASK:\n"
            "Review the Engineer's patch against the Analyst's goals.\n"
            "Check for:\n"
            "1. Functional correctness (did they break logic?).\n"
            "2. Green impact (did they actually reduce complexity?).\n"
            "3. Edge cases.\n\n"
            "OUTPUT:\n"
            "Provide a brief critique. If good, say 'LGTM'. If issues found, list them."
        )
        
        inputs = (
            f"<analyst_goals>\n{analyst_goals}\n</analyst_goals>\n\n"
            f"<proposed_patch>\n{engineer_patch}\n</proposed_patch>"
        )
        
        original_code = f"<original_code>\n{context.get_formatted_code()}\n</original_code>"

        return "\n".join([system_persona, task, inputs, original_code])

    # ==========================================
    # TURN 4: ENGINEER (Final Polish)
    # ==========================================
    def get_final_engineer_prompt(self, context: PromptContext, engineer_patch: str, reviewer_critique: str) -> str:
        system_persona = "You are a Senior Refactoring Engineer. Finalize the code."
        
        task = (
            "TASK:\n"
            "Refine your previous patch based on the Reviewer's critique.\n"
            "If the critique was positive (LGTM), output the same patch.\n"
            "Otherwise, fix the issues mentioned."
        )
        
        inputs = (
            f"<previous_patch>\n{engineer_patch}\n</previous_patch>\n\n"
            f"<reviewer_critique>\n{reviewer_critique}\n</reviewer_critique>"
        )

        return "\n".join([
            system_persona, 
            task, 
            inputs, 
            "", 
            self._get_search_replace_format_instruction()
        ])

    def extract_code_from_response(self, response: str) -> str:
        """
        Robust extraction using the same logic as Prochemy/ZeroShot.
        """
        if "<<<<<<< SEARCH" in response:
            start_index = response.find("<<<<<<< SEARCH")
            return response[start_index:]
        return response

# Boilerplate subclasses for Manager registration
class SelfCollaborationOracleTemplate(SelfCollaborationTemplate):
    def generate_prompt(self, context): return self.get_analyst_prompt(context)

class SelfCollaborationRealisticTemplate(SelfCollaborationTemplate):
    def generate_prompt(self, context): return self.get_analyst_prompt(context)