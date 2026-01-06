"""
Base classes for prompt templates
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class ProblemStatementType(Enum):
    """Type of problem statement provided to LLM"""
    ORACLE = "oracle"      # File-level: known target functions + relevant files
    REALISTIC = "realistic"  # Repo-level: all functions + entire repository


class PromptStrategy(Enum):
    """Strategy for prompt generation"""
    ZERO_SHOT = "zero_shot"              # Single-turn direct request
    FEW_SHOT = "few_shot"                # Single-turn with examples
    COT = "cot"                          # Single-turn with Chain-of-Thought
    SELF_COLLABORATION = "self_collaboration"  # Multi-turn with roles
    LDB = "ldb"                          # Iterative Debugging


@dataclass
class PromptContext:
    """Context information for prompt generation"""
    
    # Instance metadata
    instance_id: str = ""
    problem_statement_type: ProblemStatementType = ProblemStatementType.ORACLE
    
    # Code context
    # Flexible type to accept string filenames (from Runner) or detailed dicts
    target_functions: List[Any] = field(default_factory=list) 
    code_files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    
    # Performance context
    problem_description: str = ""
    test_command: str = ""
    baseline_metrics: Optional[Dict[str, float]] = None
    
    # Repository context
    repo_name: str = ""
    base_commit: str = ""
    
    def get_target_functions_str(self) -> str:
        """Format target functions as string, handling both Dict and str inputs"""
        result = []
        for func in self.target_functions:
            if isinstance(func, dict):
                func_name = func.get('name', 'unknown')
                file_path = func.get('file', 'unknown')
                result.append(f"- {func_name} in {file_path}")
            elif isinstance(func, str):
                # Fallback if runner passes just filenames
                result.append(f"- File: {func}")
            else:
                result.append(f"- {str(func)}")
        return "\n".join(result)
    
    def get_formatted_code(self, add_line_numbers: bool = False) -> str:
        """
        Formats code files strictly following SWE-perf conventions.
        Format:
        [start of filename]
        content
        [end of filename]
        """
        all_text = ""
        # Sort files for deterministic output
        for filename, content in sorted(self.code_files.items()):
            all_text += f"[start of {filename}]\n"
            if add_line_numbers:
                lines = [f"{i+1} {line}" for i, line in enumerate(content.splitlines())]
                all_text += "\n".join(lines)
            else:
                all_text += content
            all_text += f"\n[end of {filename}]\n"
        return all_text.strip("\n")


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates"""
    
    def __init__(self, strategy: PromptStrategy):
        self.strategy = strategy
    
    @abstractmethod
    def generate_prompt(self, context: PromptContext) -> Union[str, List[Dict[str, str]]]:
        """
        Generate prompt(s) based on context
        """
        pass

    # Alias for compatibility if Runner calls build_prompt
    def build_prompt(self, context: PromptContext) -> Union[str, List[Dict[str, str]]]:
        return self.generate_prompt(context)
    
    @abstractmethod
    def extract_code_from_response(self, response: str) -> str:
        """
        Extract optimized code from LLM response
        """
        pass

    # Alias for compatibility if Runner calls extract_code
    def extract_code(self, response: str, *args, **kwargs) -> str:
        return self.extract_code_from_response(response)
    
    def _get_sweperf_header(self) -> str:
        """Standard premise from SWE-perf"""
        return (
            "You will be provided with a partial code base and objective functions. "
            "You need to improve the objective function's efficiency and execution speed "
            "by editing the code base."
        )

    def _get_search_replace_format_instruction(self) -> str:
        """
        Exact copy of SWE-perf SEARCH/REPLACE instructions.
        Crucial for parsing compatibility.
        """
        return """
Please improve its efficiency and execution speed by generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE
7. You can't edit the test case, only the code base.
8. Only use standard python libraries, don't suggest installing any packages.

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```"""