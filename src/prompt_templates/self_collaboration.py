"""
Self-Collaboration Prompting Strategy for Green Code Optimization.

Based on: "Unleashing the Emergent Cognitive Synergy in Large Language Models"
https://arxiv.org/abs/2310.01234

This strategy simulates multiple expert roles collaborating:
1. ANALYST: Identifies performance bottlenecks and inefficiencies
2. OPTIMIZER: Proposes concrete code optimizations
3. REVIEWER: Validates and refines the final patch

Each role builds on the previous role's output, creating a collaborative
refinement process within a single LLM.

Version: 2.1 - Shortened Turn 1-2, stricter Turn 3 patch format
"""

import re
from typing import Dict, List, Optional, Tuple
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
    raw_responses: List[str]          # All role responses
    analyst_output: str               # Bottleneck analysis
    optimizer_output: str             # Proposed optimizations
    reviewer_output: str              # Final validated patch
    final_patch: str                  # Extracted patch
    has_valid_structure: bool


class SelfCollaborationTemplate(BasePromptTemplate):
    """
    Self-Collaboration template implementing multi-expert collaboration.
    
    Returns a list of message sequences for multi-turn conversation.
    The runner should call the LLM 3 times, passing previous responses.
    """
    
    def __init__(self):
        super().__init__(PromptStrategy.SELF_COLLABORATION)
        self.template_name = "SelfCollaboration"
    
    @property
    def is_multi_turn(self) -> bool:
        """Indicates this strategy requires multiple LLM calls."""
        return True
    
    @property
    def num_turns(self) -> int:
        """Number of turns/roles in the collaboration."""
        return 3
    
    def generate_prompt(self, context: PromptContext) -> Dict[str, any]:
        """
        Generate the initial prompt and role definitions.
        
        Returns a dict with:
        - 'system_prompt': Base system prompt
        - 'turns': List of turn configs with role prompts
        - 'code_context': Formatted code for reference
        """
        code_section = self._format_code_files(context.code_files)
        
        # Determine if oracle or realistic
        is_oracle = context.problem_statement_type == ProblemStatementType.ORACLE
        
        return {
            "system_prompt": self._get_system_prompt(),
            "initial_context": self._get_initial_context(context, code_section, is_oracle),
            "turns": [
                {
                    "role": "ANALYST",
                    "prompt": self._get_analyst_prompt(is_oracle),
                    "description": "Identify performance bottlenecks"
                },
                {
                    "role": "OPTIMIZER",
                    "prompt": self._get_optimizer_prompt(),
                    "description": "Propose optimizations"
                },
                {
                    "role": "REVIEWER",
                    "prompt": self._get_reviewer_prompt(),
                    "description": "Validate and produce final patch"
                }
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
        """
        Generate prompt for a specific turn, incorporating previous responses.
        
        Args:
            turn_index: 0=Analyst, 1=Optimizer, 2=Reviewer
            context: Original context
            previous_responses: List of responses from previous turns
            
        Returns:
            Complete prompt for this turn
        """
        code_section = self._format_code_files(context.code_files)
        is_oracle = context.problem_statement_type == ProblemStatementType.ORACLE
        
        if turn_index == 0:
            # ANALYST - First turn, no previous responses
            return self._build_analyst_turn(context, code_section, is_oracle)
        
        elif turn_index == 1:
            # OPTIMIZER - Has analyst response
            analyst_output = previous_responses[0] if previous_responses else ""
            return self._build_optimizer_turn(context, code_section, analyst_output)
        
        elif turn_index == 2:
            # REVIEWER - Has analyst and optimizer responses
            analyst_output = previous_responses[0] if len(previous_responses) > 0 else ""
            optimizer_output = previous_responses[1] if len(previous_responses) > 1 else ""
            return self._build_reviewer_turn(context, code_section, analyst_output, optimizer_output)
        
        else:
            raise ValueError(f"Invalid turn index: {turn_index}")
    
    # =========================================================================
    # SYSTEM PROMPT
    # =========================================================================
    
    def _get_system_prompt(self) -> str:
        return """You are participating in a collaborative code optimization session.
Multiple expert roles will work together to optimize code for energy efficiency.

You will be assigned a specific role (Analyst, Optimizer, or Reviewer).
Focus ONLY on your assigned role's responsibilities.
Build upon the work of previous roles when applicable.

Goal: Produce an energy-efficient code patch that maintains correctness."""
    
    # =========================================================================
    # INITIAL CONTEXT
    # =========================================================================
    
    def _get_initial_context(
        self, 
        context: PromptContext, 
        code_section: str,
        is_oracle: bool
    ) -> str:
        """Build the shared context all roles will see."""
        
        repo_info = f"**Repository:** `{context.repo_name}`" if context.repo_name else ""
        
        if is_oracle:
            task_desc = "Optimize the following code for energy efficiency and execution speed."
        else:
            task_desc = "Find and fix performance bottlenecks in the retrieved code."
            if context.repo_map:
                repo_info += f"\n\n**Repository Structure:**\n```\n{context.repo_map}\n```"
        
        return f"""## OPTIMIZATION TASK

{repo_info}

**Problem:** {context.problem_description}

**Objective:** {task_desc}

## CODE CONTEXT

{code_section}
"""
    
    # =========================================================================
    # TURN 1: ANALYST (shortened)
    # =========================================================================
    
    def _get_analyst_prompt(self, is_oracle: bool) -> str:
        if is_oracle:
            return """## YOUR ROLE: ANALYST

Identify the performance bottleneck in the code.

**In 5-7 lines, answer:**
1. Which function/method is the bottleneck? (give exact name and file)
2. Why is it slow? (O(n²), redundant computation, memory issues, etc.)
3. What type of optimization would help?

Be specific and concise. Do NOT provide code - the Optimizer will do that."""
        else:
            return """## YOUR ROLE: ANALYST

Identify the performance bottleneck in the retrieved code (some files may be noise).

**In 5-7 lines, answer:**
1. Which file(s) are relevant to the performance issue?
2. Which function/method is the bottleneck? (exact name)
3. Why is it slow?
4. What optimization would help?

Be specific and concise. Do NOT provide code."""
    
    def _build_analyst_turn(
        self, 
        context: PromptContext, 
        code_section: str,
        is_oracle: bool
    ) -> str:
        initial = self._get_initial_context(context, code_section, is_oracle)
        role_prompt = self._get_analyst_prompt(is_oracle)
        
        return f"""{initial}

{role_prompt}

Your analysis:"""
    
    # =========================================================================
    # TURN 2: OPTIMIZER (shortened)
    # =========================================================================
    
    def _get_optimizer_prompt(self) -> str:
        return """## YOUR ROLE: OPTIMIZER

Based on the Analyst's findings, propose a concrete optimization.

**In 5-7 lines, describe:**
1. What specific change to make (e.g., "replace nested loop with set lookup")
2. Why this improves performance (e.g., "O(n²) → O(n)")
3. Any edge cases to consider

Be concrete about the transformation. Do NOT write the final patch - the Reviewer will do that."""
    
    def _build_optimizer_turn(
        self, 
        context: PromptContext, 
        code_section: str,
        analyst_output: str
    ) -> str:
        return f"""## ANALYST'S FINDINGS

{analyst_output}

---

{self._get_optimizer_prompt()}

Your optimization strategy:"""
    
    # =========================================================================
    # TURN 3: REVIEWER (strict patch format)
    # =========================================================================
    
    def _get_reviewer_prompt(self) -> str:
        return """## YOUR ROLE: REVIEWER - PRODUCE THE FINAL PATCH

Generate the patch using this EXACT format:

### path/to/file.py
<<<<<<< SEARCH
[exact original code - copy from ORIGINAL CODE section above]
=======
[your optimized version]
>>>>>>> REPLACE

**CRITICAL RULES:**
1. First line must be: ### path/to/file.py (the actual file path)
2. SEARCH block must EXACTLY match original code (copy-paste from above)
3. Keep SEARCH blocks small (5-15 lines)
4. Do NOT wrap in ```python``` blocks
5. Do NOT add explanations - ONLY output the patch
6. Do NOT add external dependencies

Generate your patch now (start with ### path/to/file.py):"""
    
    def _build_reviewer_turn(
        self, 
        context: PromptContext, 
        code_section: str,
        analyst_output: str,
        optimizer_output: str
    ) -> str:
        return f"""## OPTIMIZATION STRATEGY

{optimizer_output}

---

## ORIGINAL CODE (copy EXACTLY from here for SEARCH blocks)

{code_section}

---

{self._get_reviewer_prompt()}"""
    
    # =========================================================================
    # RESPONSE PARSING
    # =========================================================================
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract patch from the final (reviewer) response."""
        return self._extract_patch(response)
    
    def parse_collaboration_responses(
        self, 
        responses: List[str]
    ) -> SelfCollabResponse:
        """
        Parse all responses from the collaboration session.
        
        Args:
            responses: List of 3 responses [analyst, optimizer, reviewer]
            
        Returns:
            SelfCollabResponse with parsed components
        """
        analyst = responses[0] if len(responses) > 0 else ""
        optimizer = responses[1] if len(responses) > 1 else ""
        reviewer = responses[2] if len(responses) > 2 else ""
        
        final_patch = self._extract_patch(reviewer)
        has_valid = bool(final_patch and "<<<<<<< SEARCH" in final_patch)
        
        return SelfCollabResponse(
            raw_responses=responses,
            analyst_output=analyst,
            optimizer_output=optimizer,
            reviewer_output=reviewer,
            final_patch=final_patch,
            has_valid_structure=has_valid
        )
    
    def _extract_patch(self, response: str) -> str:
        """Extract SEARCH/REPLACE blocks from response."""
        # Remove markdown code blocks first
        clean = re.sub(r'```(?:python|diff|text)?\s*\n?', '', response)
        clean = re.sub(r'\n?```', '', clean)
        
        # Find file path + SEARCH pattern
        match = re.search(
            r'(###\s+[\w/._-]+\.py\s*\n\s*<<<<<<< SEARCH.*?>>>>>>> REPLACE)',
            clean, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        
        # Fallback: find SEARCH and look backwards for file path
        if "<<<<<<< SEARCH" in clean:
            search_start = clean.find("<<<<<<< SEARCH")
            pre_search = clean[:search_start]
            
            # Find last file path marker before SEARCH
            file_matches = list(re.finditer(r'###\s+[\w/._-]+\.py', pre_search))
            if file_matches:
                start = file_matches[-1].start()
                return clean[start:].strip()
            
            return clean[search_start:].strip()
        
        return response.strip()
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _format_code_files(self, code_files: Dict[str, str]) -> str:
        """Format code files for prompt."""
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
    'SelfCollabResponse'
]