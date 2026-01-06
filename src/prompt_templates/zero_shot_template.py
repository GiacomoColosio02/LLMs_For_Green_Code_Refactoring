"""
Zero-Shot Prompting Strategy.
Directly asks the model to optimize the code without examples.
Includes strict constraints to prevent build failures.
"""
import re
from typing import Dict, List, Union
from .base_template import BasePromptTemplate, PromptStrategy, PromptContext

class ZeroShotTemplate(BasePromptTemplate):
    def __init__(self):
        super().__init__(PromptStrategy.ZERO_SHOT)

    def generate_prompt(self, context: PromptContext) -> str:
        """
        Builds the standard SWE-perf prompt with extra safety constraints.
        """
        # 1. Header & Strict Rules
        prompt = (
            "You are an expert Green Software Engineer optimizing Python code for energy efficiency.\n"
            "Your task is to improve the efficiency of the provided code while maintaining 100% functional correctness.\n\n"
            "### CRITICAL RULES:\n"
            "1. **Output ONLY the code patch** using the SWE-bench format (<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE).\n"
            "2. **DO NOT introduce new external dependencies** (e.g., DO NOT import aiohttp, numpy, pandas) unless they are ALREADY used in the file.\n"
            "3. Use ONLY standard Python libraries (collections, itertools, etc.) or existing project modules.\n"
            "4. Focus on algorithmic improvements, reducing redundant calculations, and efficient data structures.\n"
            "5. Do not change function signatures.\n\n"
        )
        
        # 2. Context
        prompt += f"Repository: `{context.repo_name}`\n"
        prompt += f"Issue to solve:\n{context.problem_description}\n\n"
        
        # 3. Code
        prompt += "### Target Code Files:\n"
        prompt += context.get_formatted_code() + "\n\n"
        
        # 4. Instructions
        prompt += self._get_search_replace_format_instruction()
        
        return prompt

    def extract_code_from_response(self, response: str) -> str:
        """
        Logic moved to ExperimentRunner's Hunter Parser, 
        but kept here for interface compatibility.
        """
        return response