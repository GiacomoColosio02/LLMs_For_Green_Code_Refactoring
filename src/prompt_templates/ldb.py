"""
LDB (LLM Debugger) Prompting Strategy for Green Code Optimization.

Based on: "Teaching Large Language Models to Self-Debug"
https://arxiv.org/abs/2304.05128

This strategy implements iterative refinement with feedback:
1. GENERATE: Initial patch generation
2. VALIDATE: Apply patch and check for errors
3. REFINE: If errors, provide feedback and regenerate
4. Repeat until success or max iterations

The key insight is that LLMs can fix their own mistakes when given
specific error feedback from the validation step.

Version: 2.2 - DeepSeek R1 Compatible - Explicit instructions to avoid <think> tags
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base_template import (
    BasePromptTemplate, 
    PromptContext, 
    ProblemStatementType, 
    PromptStrategy
)


class LDBFeedbackType(Enum):
    """Types of feedback for refinement."""
    PATCH_PARSE_ERROR = "patch_parse_error"      # Couldn't parse SEARCH/REPLACE
    PATCH_APPLY_ERROR = "patch_apply_error"      # SEARCH block not found in file
    TEST_FAILURE = "test_failure"                 # Tests failed after patch
    SYNTAX_ERROR = "syntax_error"                 # Python syntax error
    IMPORT_ERROR = "import_error"                 # Missing import or module
    SUCCESS = "success"                           # Patch applied successfully


@dataclass
class LDBFeedback:
    """Feedback from validation step."""
    feedback_type: LDBFeedbackType
    message: str
    details: Optional[str] = None      # Stack trace, error details
    file_path: Optional[str] = None    # File where error occurred
    line_number: Optional[int] = None  # Line number if applicable


@dataclass
class LDBResponse:
    """Response from an LDB iteration."""
    iteration: int
    raw_response: str
    patch_content: str
    feedback: Optional[LDBFeedback]
    is_final: bool


class LDBTemplate(BasePromptTemplate):
    """
    LDB (LLM Debugger) template implementing iterative refinement.
    
    This is a multi-turn strategy where:
    1. First turn: Generate initial patch
    2. Subsequent turns: Refine based on feedback
    
    The runner should:
    1. Call generate_initial_prompt() for first attempt
    2. Apply patch and validate
    3. If failed, call generate_refinement_prompt() with feedback
    4. Repeat until success or max_iterations
    """
    
    DEFAULT_MAX_ITERATIONS = 3
    
    def __init__(self, max_iterations: int = DEFAULT_MAX_ITERATIONS):
        super().__init__(PromptStrategy.LDB)
        self.template_name = "LDB"
        self.max_iterations = max_iterations
    
    @property
    def is_multi_turn(self) -> bool:
        """Indicates this strategy requires multiple LLM calls."""
        return True
    
    @property
    def is_iterative(self) -> bool:
        """Indicates this strategy uses iterative refinement."""
        return True
    
    def _clean_think_tags(self, response: str) -> str:
        """Remove <think>...</think> blocks from response (DeepSeek R1)."""
        return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    def generate_prompt(self, context: PromptContext) -> Dict[str, any]:
        """
        Generate configuration for LDB strategy.
        
        Returns a dict with:
        - 'system_prompt': Base system prompt
        - 'initial_prompt': First generation prompt
        - 'max_iterations': Maximum refinement attempts
        """
        code_section = self._format_code_files(context.code_files)
        is_oracle = context.problem_statement_type == ProblemStatementType.ORACLE
        
        return {
            "system_prompt": self._get_system_prompt(),
            "initial_prompt": self._build_initial_prompt(context, code_section, is_oracle),
            "max_iterations": self.max_iterations,
            "code_context": code_section,
            "is_oracle": is_oracle
        }
    
    def generate_initial_prompt(self, context: PromptContext) -> str:
        """
        Generate the initial patch generation prompt.
        
        Args:
            context: Prompt context with code and problem description
            
        Returns:
            Complete prompt for initial generation
        """
        code_section = self._format_code_files(context.code_files)
        is_oracle = context.problem_statement_type == ProblemStatementType.ORACLE
        return self._build_initial_prompt(context, code_section, is_oracle)
    
    def generate_refinement_prompt(
        self,
        context: PromptContext,
        previous_patch: str,
        feedback: LDBFeedback,
        iteration: int
    ) -> str:
        """
        Generate a refinement prompt based on feedback.
        
        Args:
            context: Original context
            previous_patch: The patch that failed
            feedback: Feedback from validation
            iteration: Current iteration number (1-indexed)
            
        Returns:
            Prompt for refinement attempt
        """
        code_section = self._format_code_files(context.code_files)
        # Clean previous patch from <think> tags
        clean_patch = self._clean_think_tags(previous_patch)
        return self._build_refinement_prompt(
            context, code_section, clean_patch, feedback, iteration
        )
    
    # =========================================================================
    # SYSTEM PROMPT
    # =========================================================================
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Green Software Engineer with debugging capabilities.
Your goal is to optimize code for energy efficiency while maintaining correctness.
When you receive error feedback, analyze it carefully and fix your patch.

**IMPORTANT: Do NOT use <think> tags or internal reasoning blocks.**
**Respond directly with the patch code only.**"""
    
    # =========================================================================
    # INITIAL PROMPT (matches ZS format for consistency, DeepSeek compatible)
    # =========================================================================
    
    def _build_initial_prompt(
        self,
        context: PromptContext,
        code_section: str,
        is_oracle: bool
    ) -> str:
        """Build the initial generation prompt."""
        
        repo_info = f"**Repository:** `{context.repo_name}`\n" if context.repo_name else ""
        
        if is_oracle:
            task_desc = "Optimize the following code for energy efficiency and execution speed."
        else:
            task_desc = "Find and fix performance bottlenecks in the retrieved code (some files may be noise)."
            if context.repo_map:
                repo_info += f"\n**Repository Structure:**\n```\n{context.repo_map}\n```\n"
        
        return f"""You are an expert Green Software Engineer.

**IMPORTANT: Do NOT use <think> tags or internal reasoning blocks.**
**Start your response directly with ### path/to/file.py**

## TASK
{task_desc}
If your patch has errors, you will receive feedback to fix them.

## CONTEXT
{repo_info}**Problem:** {context.problem_description}

## CODE
{code_section}

## OUTPUT FORMAT

### path/to/file.py
<<<<<<< SEARCH
[exact original code - copy from CODE section above]
=======
[your optimized replacement]
>>>>>>> REPLACE

## RULES
1. **START IMMEDIATELY with ### path/to/file.py** - no introduction or thinking
2. File path line (### path/to/file.py) MUST come immediately before <<<<<<< SEARCH
3. SEARCH block must match original code EXACTLY (copy-paste from CODE section)
4. Keep SEARCH blocks SMALL (5-15 lines)
5. Do NOT wrap in ```python``` code blocks
6. Do NOT add external dependencies
7. Do NOT modify test files
8. **Do NOT use <think> tags** - output ONLY the patch

Generate your optimization patch (start directly with ### path/to/file.py):"""
    
    # =========================================================================
    # REFINEMENT PROMPT (concise error-specific guidance, DeepSeek compatible)
    # =========================================================================
    
    def _build_refinement_prompt(
        self,
        context: PromptContext,
        code_section: str,
        previous_patch: str,
        feedback: LDBFeedback,
        iteration: int
    ) -> str:
        """Build a refinement prompt based on feedback."""
        
        remaining = self.max_iterations - iteration
        if remaining == 1:
            urgency = "⚠️ **LAST ATTEMPT - be precise!**"
        elif remaining <= 0:
            urgency = "❌ No more attempts."
        else:
            urgency = f"({remaining} attempts remaining)"
        
        # Format feedback
        feedback_section = self._format_feedback(feedback)
        
        # Get concise fix hint
        fix_hint = self._get_fix_hint(feedback.feedback_type)
        
        # Truncate previous patch if too long
        patch_preview = previous_patch[:1500]
        if len(previous_patch) > 1500:
            patch_preview += "\n...[truncated]"
        
        return f"""## PATCH FAILED - Attempt {iteration + 1}/{self.max_iterations} {urgency}

**IMPORTANT: Do NOT use <think> tags. Start directly with ### path/to/file.py**

{feedback_section}

{fix_hint}

### Your Previous Patch:
```
{patch_preview}
```

### Original Code (COPY EXACTLY for SEARCH blocks):
{code_section}

Generate your CORRECTED patch (start directly with ### path/to/file.py, no <think> tags):"""
    
    def _format_feedback(self, feedback: LDBFeedback) -> str:
        """Format feedback into a clear message for the LLM."""
        
        sections = [f"**Error:** {feedback.message}"]
        
        if feedback.file_path:
            sections[0] += f" in `{feedback.file_path}`"
        
        if feedback.line_number:
            sections[0] += f" (line {feedback.line_number})"
        
        if feedback.details:
            # Truncate long details
            details = feedback.details[:800]
            if len(feedback.details) > 800:
                details += "\n...[truncated]"
            sections.append(f"```\n{details}\n```")
        
        return "\n".join(sections)
    
    def _get_fix_hint(self, feedback_type: LDBFeedbackType) -> str:
        """Get concise fix instructions based on feedback type."""
        
        hints = {
            LDBFeedbackType.PATCH_PARSE_ERROR: """**Fix:** Patch format is wrong.
- Line before <<<<<<< SEARCH must be: ### path/to/file.py
- Do NOT use ```python``` blocks around the patch
- Do NOT use <think> tags""",

            LDBFeedbackType.PATCH_APPLY_ERROR: """**Fix:** SEARCH block doesn't match original code.
- Copy code EXACTLY from "Original Code" section (including whitespace)
- Use smaller, unique code sections""",

            LDBFeedbackType.SYNTAX_ERROR: """**Fix:** Python syntax error in your optimized code.
- Check parentheses, brackets, colons
- Verify indentation is correct""",

            LDBFeedbackType.IMPORT_ERROR: """**Fix:** Missing import.
- Only use modules already imported in the file
- Or add import in a separate SEARCH/REPLACE block""",

            LDBFeedbackType.TEST_FAILURE: """**Fix:** Tests failing - optimization broke functionality.
- Ensure behavior is preserved exactly
- Check edge cases (empty inputs, None values)"""
        }
        
        return hints.get(feedback_type, "**Fix:** Review the error and correct your patch.")
    
    # =========================================================================
    # RESPONSE PARSING
    # =========================================================================
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract patch from response."""
        # First remove <think> tags (DeepSeek R1)
        clean = self._clean_think_tags(response)
        
        # Remove markdown code blocks if present
        clean = re.sub(r'```(?:python|diff|text)?\s*\n?', '', clean)
        clean = re.sub(r'\n?```', '', clean)
        
        # Find file path + SEARCH pattern together
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
            
            # No file path, try to find in original response
            file_match = re.search(r'(###\s+[\w/._-]+\.py)', response)
            if file_match:
                return f"{file_match.group(1)}\n{clean[search_start:].strip()}"
            
            return clean[search_start:].strip()
        
        return clean.strip()
    
    def create_feedback(
        self,
        feedback_type: LDBFeedbackType,
        message: str,
        details: Optional[str] = None,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None
    ) -> LDBFeedback:
        """Helper to create feedback objects."""
        return LDBFeedback(
            feedback_type=feedback_type,
            message=message,
            details=details,
            file_path=file_path,
            line_number=line_number
        )
    
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

class LDBOracleTemplate(LDBTemplate):
    """LDB for ORACLE context."""
    
    def generate_initial_prompt(self, context: PromptContext) -> str:
        context.problem_statement_type = ProblemStatementType.ORACLE
        return super().generate_initial_prompt(context)


class LDBRealisticTemplate(LDBTemplate):
    """LDB for REALISTIC context."""
    
    def generate_initial_prompt(self, context: PromptContext) -> str:
        context.problem_statement_type = ProblemStatementType.REALISTIC
        return super().generate_initial_prompt(context)


__all__ = [
    'LDBTemplate',
    'LDBOracleTemplate',
    'LDBRealisticTemplate',
    'LDBFeedback',
    'LDBFeedbackType',
    'LDBResponse'
]