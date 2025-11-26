"""
Base classes for prompt templates
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum


class ProblemStatementType(Enum):
    """Type of problem statement provided to LLM"""
    ORACLE = "oracle"      # File-level: known target functions + relevant files
    REALISTIC = "realistic"  # Repo-level: all functions + entire repository


class PromptStrategy(Enum):
    """Strategy for prompt generation"""
    ZERO_SHOT = "zero_shot"              # Single-turn direct request
    SELF_COLLABORATION = "self_collaboration"  # Multi-turn with roles


@dataclass
class PromptContext:
    """Context information for prompt generation"""
    
    # Instance metadata
    instance_id: str
    problem_statement_type: ProblemStatementType
    
    # Code context
    target_functions: List[Dict[str, Any]]  # Functions to optimize
    code_files: Dict[str, str]  # filename -> content
    
    # Performance context
    problem_description: str  # Description of performance issue
    test_command: str  # Command to run performance tests
    baseline_metrics: Optional[Dict[str, float]] = None  # Optional baseline metrics
    
    # Repository context
    repo_name: str = ""
    base_commit: str = ""
    
    def get_target_functions_str(self) -> str:
        """Format target functions as string"""
        result = []
        for func in self.target_functions:
            func_name = func.get('name', 'unknown')
            file_path = func.get('file', 'unknown')
            result.append(f"- {func_name} in {file_path}")
        return "\n".join(result)
    
    def get_code_context_str(self) -> str:
        """Format code files as string"""
        result = []
        for filepath, content in self.code_files.items():
            result.append(f"### File: {filepath}")
            result.append("```python")
            result.append(content)
            result.append("```")
            result.append("")
        return "\n".join(result)


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates"""
    
    def __init__(self, strategy: PromptStrategy):
        self.strategy = strategy
    
    @abstractmethod
    def generate_prompt(self, context: PromptContext) -> str | List[Dict[str, str]]:
        """
        Generate prompt(s) based on context
        
        Returns:
            - For Zero-Shot: single string prompt
            - For Self-Collaboration: list of {"role": str, "content": str} messages
        """
        pass
    
    @abstractmethod
    def extract_code_from_response(self, response: str) -> str:
        """
        Extract optimized code from LLM response
        
        Args:
            response: Raw LLM output
            
        Returns:
            Extracted code as string (typically a unified diff patch)
        """
        pass
    
    def _format_problem_statement(self, context: PromptContext) -> str:
        """Format problem statement section"""
        if context.problem_statement_type == ProblemStatementType.ORACLE:
            return f"""
## Problem Statement (ORACLE Mode)

You have been provided with the exact functions that need optimization and their containing files.

**Performance Issue:**
{context.problem_description}

**Target Functions to Optimize:**
{context.get_target_functions_str()}

**Relevant Code Files:**
{context.get_code_context_str()}
"""
        else:  # REALISTIC
            return f"""
## Problem Statement (REALISTIC Mode)

You have access to the entire repository. You need to identify and optimize the functions responsible for the performance issue.

**Performance Issue:**
{context.problem_description}

**Functions Executed During Tests (potential optimization targets):**
{context.get_target_functions_str()}

**Repository Code:**
{context.get_code_context_str()}

Note: The functions listed above are executed during performance tests, but the actual bottleneck may be in functions they call. Analyze the code and optimize where appropriate.
"""
    
    def _format_output_instructions(self) -> str:
        """Format output format instructions"""
        return """
## Output Format

Provide your optimization as a unified diff patch that can be applied with `git apply`.

Example format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@ def function_name():
-    old_line
+    new_line
```

Make sure your patch:
1. Maintains functional correctness (passes all tests)
2. Improves performance (reduces execution time, memory usage, or energy consumption)
3. Follows the existing code style
4. Includes only necessary changes
"""