"""
LDB (LLM Debugger) Prompting Strategy for Green Code Optimization.

Based on: "Teaching Large Language Models to Self-Debug"
https://arxiv.org/abs/2304.05128

This strategy implements iterative refinement with feedback:
1. GENERATE: Initial patch generation (identical format to ZS)
2. VALIDATE: Apply patch and check for errors
3. REFINE: If errors, provide specific feedback and regenerate
4. Repeat until success or max iterations

Version: 3.0 - Simplified prompts, robust extraction, matches ZS format
"""

import re
from typing import Dict, List, Optional
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
    PATCH_PARSE_ERROR = "patch_parse_error"
    PATCH_APPLY_ERROR = "patch_apply_error"
    TEST_FAILURE = "test_failure"
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    TIMEOUT_ERROR = "timeout_error"
    SUCCESS = "success"


@dataclass
class LDBFeedback:
    """Feedback from validation step."""
    feedback_type: LDBFeedbackType
    message: str
    details: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class LDBIteration:
    """Result from a single LDB iteration."""
    iteration: int
    patch_content: str
    feedback: Optional[LDBFeedback] = None
    success: bool = False


def extract_patch_from_ldb(response: str) -> str:
    """Extract SEARCH/REPLACE patch from LDB response."""
    # Remove markdown code blocks
    clean = re.sub(r'```(?:python|diff|text)?\s*\n?', '', response)
    clean = re.sub(r'\n?```', '', clean)
    
    # Strategy 1: Find file path + SEARCH together
    match = re.search(
        r'(###\s+[\w/._-]+\.py\s*\n\s*<<<<<<< SEARCH.*?>>>>>>> REPLACE)',
        clean, re.DOTALL
    )
    if match:
        return match.group(1).strip()
    
    # Strategy 2: Find all SEARCH/REPLACE blocks with file paths
    all_blocks = []
    pattern = r'(###\s+[\w/._-]+\.py\s*\n\s*<<<<<<< SEARCH.*?>>>>>>> REPLACE)'
    for m in re.finditer(pattern, clean, re.DOTALL):
        all_blocks.append(m.group(1))
    if all_blocks:
        return '\n\n'.join(all_blocks).strip()
    
    # Strategy 3: Find SEARCH and look backwards for file path
    search_idx = clean.find('<<<<<<< SEARCH')
    if search_idx != -1:
        pre_search = clean[:search_idx]
        
        # Find last file path marker before SEARCH
        file_matches = list(re.finditer(r'###\s+[\w/._-]+\.py', pre_search))
        if file_matches:
            start_pos = file_matches[-1].start()
            return clean[start_pos:].strip()
        
        return clean[search_idx:].strip()
    
    return clean.strip()


class LDBTemplate(BasePromptTemplate):
    """
    LDB (LLM Debugger) template with iterative refinement.
    
    Process:
    1. Generate initial patch (same format as Zero-Shot)
    2. If patch fails, provide specific error feedback
    3. LLM regenerates corrected patch
    4. Repeat until success or max_iterations reached
    """
    
    DEFAULT_MAX_ITERATIONS = 3
    
    def __init__(self, max_iterations: int = DEFAULT_MAX_ITERATIONS):
        super().__init__(PromptStrategy.LDB)
        self.template_name = "LDB"
        self.max_iterations = max_iterations
    
    @property
    def is_multi_turn(self) -> bool:
        return True
    
    @property
    def is_iterative(self) -> bool:
        return True
    
    def generate_prompt(self, context: PromptContext) -> Dict[str, any]:
        """Generate configuration for LDB strategy."""
        code_section = self._format_code_files(context.code_files)
        is_oracle = context.problem_statement_type == ProblemStatementType.ORACLE
        
        return {
            "system_prompt": self._get_system_prompt(),
            "initial_prompt": self.generate_initial_prompt(context),
            "max_iterations": self.max_iterations,
            "code_context": code_section,
            "is_oracle": is_oracle
        }
    
    def generate_initial_prompt(self, context: PromptContext) -> str:
        """Generate initial patch generation prompt (identical to ZS)."""
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
        """Generate refinement prompt with error feedback."""
        code_section = self._format_code_files(context.code_files)
        return self._build_refinement_prompt(
            context, code_section, previous_patch, feedback, iteration
        )
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract patch from response."""
        return extract_patch_from_ldb(response)
    
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
    # SYSTEM PROMPT
    # =========================================================================
    
    def _get_system_prompt(self) -> str:
        return "You are an expert Green Software Engineer. Fix errors when given feedback."
    
    # =========================================================================
    # INITIAL PROMPT (matches ZS format exactly)
    # =========================================================================
    
    def _build_initial_prompt(
        self,
        context: PromptContext,
        code_section: str,
        is_oracle: bool
    ) -> str:
        repo_info = f"**Repository:** `{context.repo_name}`\n" if context.repo_name else ""
        
        if is_oracle:
            task = "Optimize the following code for energy efficiency and execution speed."
        else:
            task = "Find and fix the performance bottleneck in the retrieved code."
            if context.repo_map:
                repo_info += f"\n**Repo Structure:**\n```\n{context.repo_map}\n```\n"
        
        return f"""You are an expert Green Software Engineer.

## TASK
{task}
If your patch has errors, you will receive feedback to fix them.

## CONTEXT
{repo_info}**Problem:** {context.problem_description}

## CODE
{code_section}

## OUTPUT FORMAT

### path/to/file.py
<<<<<<< SEARCH
[exact original code to find]
=======
[your optimized replacement]
>>>>>>> REPLACE

## RULES
1. File path MUST be on line before <<<<<<< SEARCH
2. SEARCH must match original code EXACTLY (copy from CODE section above)
3. Keep SEARCH blocks SMALL (5-15 lines)
4. Do NOT wrap in ```python``` code blocks
5. Do NOT add external dependencies
6. Do NOT modify test files

Generate your optimization patch (start with ### path/to/file.py):"""

    # =========================================================================
    # REFINEMENT PROMPT
    # =========================================================================
    
    def _build_refinement_prompt(
        self,
        context: PromptContext,
        code_section: str,
        previous_patch: str,
        feedback: LDBFeedback,
        iteration: int
    ) -> str:
        remaining = self.max_iterations - iteration
        
        if remaining == 1:
            urgency = "⚠️ **LAST ATTEMPT** - be precise!"
        elif remaining <= 0:
            urgency = "❌ No more attempts."
        else:
            urgency = f"({remaining} attempts remaining)"
        
        # Get specific fix instructions
        fix_hint = self._get_fix_hint(feedback.feedback_type)
        
        # Truncate previous patch if too long
        patch_preview = previous_patch[:1500]
        if len(previous_patch) > 1500:
            patch_preview += "\n...[truncated]"
        
        # Truncate error details if too long
        error_details = ""
        if feedback.details:
            details = feedback.details[:800]
            if len(feedback.details) > 800:
                details += "\n...[truncated]"
            error_details = f"\n**Details:**\n```\n{details}\n```"
        
        file_info = f"\n**File:** `{feedback.file_path}`" if feedback.file_path else ""
        line_info = f" (line {feedback.line_number})" if feedback.line_number else ""
        
        return f"""## PATCH FAILED - Iteration {iteration + 1}/{self.max_iterations} {urgency}

**Error:** {feedback.message}{file_info}{line_info}
{error_details}

{fix_hint}

### Your Previous Patch:
```
{patch_preview}
```

### Original Code (COPY EXACTLY for SEARCH blocks):
{code_section}

## Generate CORRECTED patch (start with ### path/to/file.py):"""

    def _get_fix_hint(self, feedback_type: LDBFeedbackType) -> str:
        """Get concise fix instructions based on error type."""
        hints = {
            LDBFeedbackType.PATCH_PARSE_ERROR: """**Fix:** Patch format is wrong.
- Line before <<<<<<< SEARCH must be: ### path/to/file.py
- Do NOT use ```python``` blocks""",

            LDBFeedbackType.PATCH_APPLY_ERROR: """**Fix:** SEARCH block doesn't match original code.
- Copy code EXACTLY from "Original Code" section above
- Check whitespace (spaces vs tabs, trailing spaces)
- Use smaller, unique code sections""",

            LDBFeedbackType.SYNTAX_ERROR: """**Fix:** Python syntax error in your optimized code.
- Check parentheses, brackets, colons
- Verify indentation is correct""",

            LDBFeedbackType.IMPORT_ERROR: """**Fix:** Missing import.
- Only use modules already imported in the file
- Or add import in separate SEARCH/REPLACE block""",

            LDBFeedbackType.TEST_FAILURE: """**Fix:** Tests failing - optimization broke functionality.
- Ensure behavior is preserved
- Check edge cases (empty inputs, None values)
- Verify return types match original""",

            LDBFeedbackType.TIMEOUT_ERROR: """**Fix:** Execution timed out.
- Your optimization may have infinite loop
- Or made performance worse
- Simplify the change"""
        }
        return hints.get(feedback_type, "**Fix:** Review error and correct your patch.")
    
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
    'LDBIteration',
    'extract_patch_from_ldb'
]