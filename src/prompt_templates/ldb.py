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
        return self._build_refinement_prompt(
            context, code_section, previous_patch, feedback, iteration
        )
    
    # =========================================================================
    # SYSTEM PROMPT
    # =========================================================================
    
    def _get_system_prompt(self) -> str:
        return """You are an expert Green Software Engineer with debugging capabilities.

Your goal is to optimize code for energy efficiency while maintaining correctness.

When you receive feedback about errors in your patch:
1. Carefully analyze the error message
2. Identify what went wrong
3. Fix the specific issue
4. Regenerate a corrected patch

Always output patches in SEARCH/REPLACE format.
Learn from your mistakes and improve with each iteration."""
    
    # =========================================================================
    # INITIAL PROMPT
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
            task_section = """## TASK
Optimize the following code for energy efficiency and execution speed.
You will receive feedback if your patch has errors, allowing you to refine it."""
        else:
            task_section = """## TASK
Find and fix performance bottlenecks in the retrieved code.
Note: Some retrieved files may be noise - identify the relevant code first.
You will receive feedback if your patch has errors, allowing you to refine it."""
            
            if context.repo_map:
                repo_info += f"\n**Repository Structure:**\n```\n{context.repo_map}\n```\n"
        
        format_instructions = self._get_format_instructions()
        
        return f"""{task_section}

## CONTEXT
{repo_info}
**Problem:** {context.problem_description}

## CODE
{code_section}

{format_instructions}

Generate your optimization patch now:"""
    
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
        """Build a refinement prompt based on feedback."""
        
        # Format feedback message based on type
        feedback_section = self._format_feedback(feedback)
        
        remaining = self.max_iterations - iteration
        urgency = ""
        if remaining == 1:
            urgency = "\n**⚠️ This is your LAST attempt. Be careful and precise.**"
        elif remaining == 0:
            urgency = "\n**❌ No more attempts remaining.**"
        
        return f"""## REFINEMENT REQUIRED (Attempt {iteration + 1}/{self.max_iterations})
{urgency}

Your previous patch had an error. Please fix it.

### Previous Patch (with error):
```
{previous_patch[:2000]}{'...[truncated]' if len(previous_patch) > 2000 else ''}
```

### Error Feedback:
{feedback_section}

### Original Code (for reference):
{code_section}

{self._get_refinement_instructions(feedback.feedback_type)}

Generate your CORRECTED patch now:"""
    
    def _format_feedback(self, feedback: LDBFeedback) -> str:
        """Format feedback into a clear message for the LLM."""
        
        sections = [f"**Error Type:** {feedback.feedback_type.value}"]
        sections.append(f"**Message:** {feedback.message}")
        
        if feedback.file_path:
            sections.append(f"**File:** {feedback.file_path}")
        
        if feedback.line_number:
            sections.append(f"**Line:** {feedback.line_number}")
        
        if feedback.details:
            # Truncate long details
            details = feedback.details[:1000]
            if len(feedback.details) > 1000:
                details += "\n...[truncated]"
            sections.append(f"**Details:**\n```\n{details}\n```")
        
        return "\n".join(sections)
    
    def _get_refinement_instructions(self, feedback_type: LDBFeedbackType) -> str:
        """Get specific instructions based on feedback type."""
        
        instructions = {
            LDBFeedbackType.PATCH_PARSE_ERROR: """
### How to Fix: PATCH FORMAT ERROR
Your patch was not in the correct format. Ensure you use:

### path/to/file.py
<<<<<<< SEARCH
exact original code
=======
your optimized code
>>>>>>> REPLACE

- Do NOT wrap in ```python``` code blocks
- Include the file path on its own line with ###
- SEARCH must match original code EXACTLY""",

            LDBFeedbackType.PATCH_APPLY_ERROR: """
### How to Fix: SEARCH BLOCK NOT FOUND
The SEARCH block didn't match any code in the file. This usually means:
1. Whitespace mismatch (spaces vs tabs, trailing spaces)
2. The code was slightly different than expected
3. Wrong file path

**Solution:** Copy the EXACT code from the original file into your SEARCH block.
Use smaller, more unique code sections to match.""",

            LDBFeedbackType.SYNTAX_ERROR: """
### How to Fix: SYNTAX ERROR
Your optimized code has a Python syntax error. Common causes:
1. Missing colons, parentheses, or brackets
2. Incorrect indentation
3. Invalid Python syntax

**Solution:** Double-check your REPLACE block for valid Python syntax.""",

            LDBFeedbackType.IMPORT_ERROR: """
### How to Fix: IMPORT ERROR
Your code references a module that isn't imported. 

**Solution:** 
1. Only use modules already imported in the file
2. If adding an import, include it in a separate SEARCH/REPLACE block
3. Prefer standard library modules (functools, itertools, collections)""",

            LDBFeedbackType.TEST_FAILURE: """
### How to Fix: TEST FAILURE
Your patch broke functionality - tests are failing.

**Solution:**
1. Ensure your optimization preserves the original behavior
2. Check edge cases (empty lists, None values, etc.)
3. Make sure return types match the original"""
        }
        
        return instructions.get(feedback_type, """
### How to Fix
Review the error message carefully and correct your patch.
Ensure your SEARCH block matches the original code exactly.""")
    
    # =========================================================================
    # FORMAT INSTRUCTIONS
    # =========================================================================
    
    def _get_format_instructions(self) -> str:
        """Get output format instructions."""
        return """## OUTPUT FORMAT

Generate your patch using SEARCH/REPLACE blocks:

### path/to/file.py
<<<<<<< SEARCH
[exact original code to find]
=======
[your optimized replacement]
>>>>>>> REPLACE

**Rules:**
1. SEARCH must match original code EXACTLY (including whitespace)
2. Keep SEARCH blocks SMALL - just enough to locate uniquely
3. Do NOT wrap in ```python``` code blocks
4. Do NOT add new external dependencies
5. Multiple SEARCH/REPLACE blocks allowed for different changes

**Example:**

### myproject/utils.py
<<<<<<< SEARCH
def slow_function(items):
    result = []
    for item in items:
        result.append(process(item))
    return result
=======
def slow_function(items):
    return [process(item) for item in items]
>>>>>>> REPLACE"""
    
    # =========================================================================
    # RESPONSE PARSING
    # =========================================================================
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract patch from response."""
        # Remove markdown code blocks if present
        patch = re.sub(r'^```(?:python|diff|text)?\s*\n?', '', response, flags=re.MULTILINE)
        patch = re.sub(r'\n?```\s*$', '', patch, flags=re.MULTILINE)
        
        # Try to find SEARCH/REPLACE blocks
        if "<<<<<<< SEARCH" in patch:
            # Find first file path or SEARCH marker
            file_match = re.search(r'(###\s+[\w/._-]+\.py)', patch)
            search_start = patch.find("<<<<<<< SEARCH")
            
            if file_match and file_match.start() < search_start:
                return patch[file_match.start():].strip()
            else:
                # Try to find file path before in original response
                file_match = re.search(r'(###\s+[\w/._-]+\.py)', response)
                if file_match:
                    return f"{file_match.group(1)}\n{patch[search_start:].strip()}"
                return patch[search_start:].strip()
        
        return patch.strip()
    
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