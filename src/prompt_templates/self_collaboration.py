"""
Self-Collaboration Prompting Strategy for Green Code Optimization.

Based on: "Unleashing the Emergent Cognitive Synergy in Large Language Models"
https://arxiv.org/abs/2310.01234

This strategy simulates multiple expert roles collaborating:
1. ANALYST: Identifies performance bottlenecks (brief, 3-5 lines)
2. OPTIMIZER: Proposes concrete optimization approach (brief, 3-5 lines)
3. IMPLEMENTER: Produces the final SEARCH/REPLACE patch

Version: 3.0 - Simplified roles, strict patch format matching ZS success
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

from .base_template import (
    BasePromptTemplate, 
    PromptContext, 
    ProblemStatementType, 
    PromptStrategy
)


@dataclass
class SelfCollabResponse:
    """Parsed Self-Collaboration response."""
    raw_responses: List[str]
    analyst_output: str
    optimizer_output: str
    implementer_output: str
    final_patch: str
    has_valid_structure: bool


def extract_patch_from_sc(response: str) -> str:
    """Extract SEARCH/REPLACE patch from Self-Collaboration response."""
    # Remove markdown code blocks
    clean = re.sub(r'```(?:python|diff|text)?\s*\n?', '', response)
    clean = re.sub(r'\n?```', '', clean)
    
    # Strategy 1: Find file path + SEARCH pattern together
    match = re.search(
        r'(###\s+[\w/._-]+\.py\s*\n\s*<<<<<<< SEARCH.*?>>>>>>> REPLACE)',
        clean, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    
    # Strategy 2: Find SEARCH and look backwards for file path
    search_idx = clean.find('<<<<<<< SEARCH')
    if search_idx != -1:
        pre_search = clean[:search_idx]
        
        # Find last file path marker before SEARCH
        file_matches = list(re.finditer(r'###\s+[\w/._-]+\.py', pre_search))
        if file_matches:
            start_pos = file_matches[-1].start()
            return clean[start_pos:].strip()
        
        # No file path found, return from SEARCH
        return clean[search_idx:].strip()
    
    return clean.strip()


class SelfCollaborationTemplate(BasePromptTemplate):
    """
    Self-Collaboration template with 3 streamlined turns.
    
    Turn 1 (Analyst): Brief bottleneck identification (3-5 lines)
    Turn 2 (Optimizer): Brief optimization strategy (3-5 lines)
    Turn 3 (Implementer): Generate SEARCH/REPLACE patch
    """
    
    def __init__(self):
        super().__init__(PromptStrategy.SELF_COLLABORATION)
        self.template_name = "SelfCollaboration"
    
    @property
    def is_multi_turn(self) -> bool:
        return True
    
    @property
    def num_turns(self) -> int:
        return 3
    
    def generate_prompt(self, context: PromptContext) -> Dict[str, any]:
        """Generate configuration for multi-turn collaboration."""
        code_section = self._format_code_files(context.code_files)
        is_oracle = context.problem_statement_type == ProblemStatementType.ORACLE
        
        return {
            "system_prompt": self._get_system_prompt(),
            "initial_context": self._get_initial_context(context, code_section, is_oracle),
            "turns": [
                {"role": "ANALYST", "description": "Identify bottleneck"},
                {"role": "OPTIMIZER", "description": "Propose fix"},
                {"role": "IMPLEMENTER", "description": "Generate patch"}
            ],
            "code_context": code_section,
            "problem_description": context.problem_description
        }
    
    def generate_turn_prompt(
        self, 
        turn_index: int, 
        context: PromptContext,
        previous_responses: List[str]
    ) -> str:
        """Generate prompt for a specific turn."""
        code_section = self._format_code_files(context.code_files)
        is_oracle = context.problem_statement_type == ProblemStatementType.ORACLE
        
        if turn_index == 0:
            return self._build_analyst_turn(context, code_section, is_oracle)
        elif turn_index == 1:
            analyst_output = previous_responses[0] if previous_responses else ""
            return self._build_optimizer_turn(context, code_section, analyst_output)
        elif turn_index == 2:
            analyst_output = previous_responses[0] if len(previous_responses) > 0 else ""
            optimizer_output = previous_responses[1] if len(previous_responses) > 1 else ""
            return self._build_implementer_turn(context, code_section, analyst_output, optimizer_output)
        else:
            raise ValueError(f"Invalid turn index: {turn_index}")
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract patch from final response."""
        return extract_patch_from_sc(response)
    
    def parse_collaboration_responses(self, responses: List[str]) -> SelfCollabResponse:
        """Parse all responses from collaboration session."""
        analyst = responses[0] if len(responses) > 0 else ""
        optimizer = responses[1] if len(responses) > 1 else ""
        implementer = responses[2] if len(responses) > 2 else ""
        
        final_patch = extract_patch_from_sc(implementer)
        has_valid = bool(final_patch and "<<<<<<< SEARCH" in final_patch and "### " in final_patch)
        
        return SelfCollabResponse(
            raw_responses=responses,
            analyst_output=analyst,
            optimizer_output=optimizer,
            implementer_output=implementer,
            final_patch=final_patch,
            has_valid_structure=has_valid
        )
    
    # =========================================================================
    # SYSTEM PROMPT
    # =========================================================================
    
    def _get_system_prompt(self) -> str:
        return "You are an expert software engineer participating in a code optimization session."
    
    # =========================================================================
    # INITIAL CONTEXT
    # =========================================================================
    
    def _get_initial_context(self, context: PromptContext, code_section: str, is_oracle: bool) -> str:
        repo_info = f"**Repository:** `{context.repo_name}`\n" if context.repo_name else ""
        
        return f"""{repo_info}**Problem:** {context.problem_description}

## CODE
{code_section}
"""
    
    # =========================================================================
    # TURN 1: ANALYST (Brief)
    # =========================================================================
    
    def _build_analyst_turn(self, context: PromptContext, code_section: str, is_oracle: bool) -> str:
        repo_info = f"**Repository:** `{context.repo_name}`\n" if context.repo_name else ""
        
        noise_warning = "" if is_oracle else "\n**Note:** Some retrieved files may be noise - identify the real bottleneck.\n"
        
        return f"""You are the **ANALYST** on a green software optimization team.

## CONTEXT
{repo_info}**Problem:** {context.problem_description}
{noise_warning}
## CODE
{code_section}

## YOUR TASK
In **3-5 lines**, identify:
1. Which function/file is the bottleneck?
2. Why is it slow? (O(n²), redundant ops, memory, etc.)

Be specific - name the exact function and file path.
Do NOT provide code - just analysis.

Your analysis:"""

    # =========================================================================
    # TURN 2: OPTIMIZER (Brief)
    # =========================================================================
    
    def _build_optimizer_turn(self, context: PromptContext, code_section: str, analyst_output: str) -> str:
        return f"""You are the **OPTIMIZER** on a green software optimization team.

## ANALYST'S FINDINGS
{analyst_output}

## YOUR TASK
In **3-5 lines**, propose:
1. What specific change to make?
2. Why will it improve performance?

Be concrete - describe the transformation (e.g., "replace nested loop with set lookup").
Do NOT write code yet - the Implementer will do that.

Your optimization strategy:"""

    # =========================================================================
    # TURN 3: IMPLEMENTER (Produces Patch)
    # =========================================================================
    
    def _build_implementer_turn(
        self, 
        context: PromptContext, 
        code_section: str,
        analyst_output: str,
        optimizer_output: str
    ) -> str:
        return f"""You are the **IMPLEMENTER** on a green software optimization team.

## ANALYSIS
{analyst_output}

## OPTIMIZATION STRATEGY
{optimizer_output}

## ORIGINAL CODE (copy EXACTLY from here for SEARCH blocks)
{code_section}

## YOUR TASK
Generate the optimization patch using this EXACT format:

### path/to/file.py
<<<<<<< SEARCH
[exact code from ORIGINAL CODE above - must match perfectly]
=======
[your optimized version]
>>>>>>> REPLACE

## CRITICAL RULES
1. First line must be: ### path/to/file.py
2. SEARCH block must EXACTLY match original code (copy-paste from above)
3. Keep SEARCH blocks SMALL (5-15 lines)
4. Do NOT wrap in ```python``` blocks
5. Do NOT add explanations - ONLY the patch
6. Do NOT add external dependencies

Generate your patch now (start with ### path/to/file.py):"""

    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _format_code_files(self, code_files: Dict[str, str]) -> str:
        if not code_files:
            return "*No code files provided*"
        
        sections = []
        for filepath, content in sorted(code_files.items()):
            sections.append(f"[start of {filepath}]\n{content}\n[end of {filepath}]")
        return "\n\n".join(sections)


# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

class SelfCollabOracleTemplate(SelfCollaborationTemplate):
    """Self-Collaboration for ORACLE context."""
    
    def generate_turn_prompt(
        self, 
        turn_index: int, 
        context: PromptContext,
        previous_responses: List[str]
    ) -> str:
        context.problem_statement_type = ProblemStatementType.ORACLE
        return super().generate_turn_prompt(turn_index, context, previous_responses)


class SelfCollabRealisticTemplate(SelfCollaborationTemplate):
    """Self-Collaboration for REALISTIC context."""
    
    def generate_turn_prompt(
        self, 
        turn_index: int, 
        context: PromptContext,
        previous_responses: List[str]
    ) -> str:
        context.problem_statement_type = ProblemStatementType.REALISTIC
        return super().generate_turn_prompt(turn_index, context, previous_responses)


__all__ = [
    'SelfCollaborationTemplate',
    'SelfCollabOracleTemplate', 
    'SelfCollabRealisticTemplate',
    'SelfCollabResponse',
    'extract_patch_from_sc'
]