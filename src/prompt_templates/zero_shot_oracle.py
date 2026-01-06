"""
Zero-Shot ORACLE Prompt Strategy.
Scenario: The LLM receives the exact files that need modification (Gold Context).
Focus: Pure optimization reasoning and coding, without retrieval noise.
"""
from .base_template import BasePromptTemplate, PromptStrategy, PromptContext

class ZeroShotOracleTemplate(BasePromptTemplate):
    def __init__(self):
        super().__init__(PromptStrategy.ZERO_SHOT)

    def generate_prompt(self, context: PromptContext) -> str:
        prompt = (
            "You are an expert Green Software Engineer optimizing Python code for energy efficiency.\n"
            "Your task is to optimize the provided code while maintaining 100% functional correctness.\n\n"
            
            "### CRITICAL RULES:\n"
            "1. **Output ONLY the code patch** using the SWE-bench format (<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE).\n"
            "2. **Use SMALL, UNIQUE SEARCH blocks**. Do NOT copy the entire file in the SEARCH block. Only include enough lines to locate the code uniquely.\n"
            "3. **DO NOT import new external libraries** (like aiohttp, numpy) unless they are already imported in the file.\n"
            "4. Focus on algorithmic efficiency, reducing redundant calculations, and memory usage.\n"
            "5. Do not include explanations or conversational text.\n\n"
        )
        
        # 1. Sintomo (Costruito sinteticamente)
        prompt += f"### Repository: `{context.repo_name}`\n"
        prompt += f"### Optimization Goal:\n{context.problem_description}\n\n"
        
        # 2. Contesto (Solo Gold Files)
        prompt += "### Target Code Files (Relevant Context):\n"
        prompt += context.get_formatted_code() + "\n\n"
        
        # 3. Istruzioni Formato
        prompt += self._get_search_replace_format_instruction()
        
        return prompt

    def extract_code_from_response(self, response: str) -> str:
        return response # L'estrazione viene fatta dal Runner