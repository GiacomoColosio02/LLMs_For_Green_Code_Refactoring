"""
Zero-Shot REALISTIC Prompt Strategy.
Scenario: The LLM acts as a developer who has identified a slow test (Trigger),
has the repository map, and has performed a "Go to Definition" search (Retrieval).
The context contains both relevant code and noise.
"""
from .base_template import BasePromptTemplate, PromptStrategy, PromptContext

class ZeroShotRealisticTemplate(BasePromptTemplate):
    def __init__(self):
        super().__init__(PromptStrategy.ZERO_SHOT)

    def generate_prompt(self, context: PromptContext) -> str:
        prompt = (
            "You are an expert Green Software Engineer optimizing Python code for energy efficiency.\n"
            "You have identified a performance regression in the test suite.\n\n"
            
            "### TASK:\n"
            "Analyze the provided context, identify the code responsible for the bottleneck, and apply optimizations.\n\n"
            
            "### CRITICAL RULES:\n"
            "1. **Analyze the Retrieved Files:** The context below contains the Top-5 files found by searching for the test code symbols. "
            "**WARNING:** Some files might be irrelevant noise. You must discriminate between critical code and noise.\n"
            "2. **Output ONLY the code patch** using the SWE-bench format (<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE).\n"
            "3. **Do NOT import new external libraries** (like aiohttp, numpy) unless they are already imported in the file.\n"
            "4. Focus on algorithmic efficiency and reducing resource consumption.\n\n"
        )
        
        # 1. Repo Map (Architettura)
        prompt += f"### Repository Structure (`tree` output):\n"
        prompt += f"{context.repo_map}\n\n"
        
        # 2. Sintomo (Issue)
        prompt += f"### Issue Description / Failing Tests:\n"
        prompt += f"{context.problem_description}\n\n"
        
        # 3. Contesto (Anchor + Retrieval)
        prompt += "### Code Context (Test Anchor + Retrieved Files):\n"
        prompt += context.get_formatted_code() + "\n\n"
        
        # 4. Istruzioni Formato
        prompt += self._get_search_replace_format_instruction()
        
        return prompt

    def extract_code_from_response(self, response: str) -> str:
        return response # L'estrazione vera avviene nel Runner