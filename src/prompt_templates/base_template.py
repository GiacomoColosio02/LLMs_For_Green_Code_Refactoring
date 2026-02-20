"""
Base classes for prompt templates.
Defines the standard interface and context structure for all prompting strategies.

Version: 3.0 - Green Software Oriented
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum


class ProblemStatementType(Enum):
    """
    Type of problem statement / context provided to LLM.
    
    ORACLE: File-level context - LLM receives exact files that need modification.
            This is the "gold" setting where we know the target.
            
    REALISTIC: Repo-level context - LLM receives retrieved files based on test analysis.
               May contain noise. LLM must identify the bottleneck first.
    """
    ORACLE = "oracle"
    REALISTIC = "realistic"


class PromptStrategy(Enum):
    """
    Strategy for prompt generation.
    
    ZERO_SHOT: Single-turn direct request without examples
    FEW_SHOT: Single-turn with optimization examples
    COT: Chain-of-Thought reasoning before optimization
    SELF_COLLABORATION: Multi-turn with different expert roles
    LDB: Iterative debugging with feedback loop
    """
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    COT = "cot"
    SELF_COLLABORATION = "self_collaboration"
    LDB = "ldb"


@dataclass
class PromptContext:
    """
    Context information for prompt generation.
    
    Contains all information needed to generate a prompt for code optimization:
    - Instance metadata (ID, type)
    - Code context (files, functions)
    - Performance context (description, tests, metrics)
    - Repository context (name, commit)
    """
    
    # Instance metadata
    instance_id: str = ""
    problem_statement_type: ProblemStatementType = ProblemStatementType.ORACLE
    
    # Code context
    target_functions: List[Any] = field(default_factory=list)  # Functions to optimize
    code_files: Dict[str, str] = field(default_factory=dict)   # filename -> content
    
    # REALISTIC-specific context
    repo_map: Optional[str] = None  # Repository structure (tree output)
    
    # Performance context
    problem_description: str = ""                      # What to optimize
    test_command: str = ""                             # How to run tests
    baseline_metrics: Optional[Dict[str, float]] = None  # Pre-optimization metrics
    
    # Repository context
    repo_name: str = ""      # e.g., "mwaskom/seaborn"
    base_commit: str = ""    # Commit hash
    
    def get_target_functions_str(self) -> str:
        """
        Format target functions as readable string.
        
        Handles both dict format (from dataset) and string format (from runner).
        
        Returns:
            Formatted string listing target functions
        """
        if not self.target_functions:
            return "No specific target functions identified."
        
        result = []
        for func in self.target_functions:
            if isinstance(func, dict):
                func_name = func.get('name', 'unknown')
                file_path = func.get('file', 'unknown')
                result.append(f"- {func_name} in {file_path}")
            elif isinstance(func, str):
                result.append(f"- {func}")
            else:
                result.append(f"- {str(func)}")
        return "\n".join(result)
    
    def get_formatted_code(self, add_line_numbers: bool = False) -> str:
        """
        Format code files following SWE-perf conventions.
        
        Format:
            [start of filename]
            content
            [end of filename]
        
        Args:
            add_line_numbers: If True, prefix each line with line number
            
        Returns:
            Formatted code string
        """
        if not self.code_files:
            return "No code files provided."
        
        sections = []
        
        # Sort files for deterministic output
        for filename, content in sorted(self.code_files.items()):
            section = f"[start of {filename}]\n"
            
            if add_line_numbers:
                lines = content.splitlines()
                numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
                section += "\n".join(numbered)
            else:
                section += content
            
            section += f"\n[end of {filename}]"
            sections.append(section)
        
        return "\n\n".join(sections)
    
    def get_total_code_size(self) -> int:
        """Get total character count of all code files."""
        return sum(len(content) for content in self.code_files.values())
    
    def get_file_count(self) -> int:
        """Get number of code files."""
        return len(self.code_files)


class BasePromptTemplate(ABC):
    """
    Abstract base class for prompt templates.
    
    All prompt strategies (zero-shot, few-shot, CoT, etc.) inherit from this.
    Provides common utilities and defines the interface.
    """
    
    def __init__(self, strategy: PromptStrategy):
        """
        Initialize template with strategy type.
        
        Args:
            strategy: The prompting strategy this template implements
        """
        self.strategy = strategy
    
    @abstractmethod
    def generate_prompt(self, context: PromptContext) -> Union[str, List[Dict[str, str]]]:
        """
        Generate prompt(s) based on context.
        
        Args:
            context: PromptContext with all necessary information
            
        Returns:
            Either a single prompt string, or a list of message dicts
            for multi-turn strategies
        """
        pass
    
    @abstractmethod
    def extract_code_from_response(self, response: str) -> str:
        """
        Extract optimized code from LLM response.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Extracted code/patch content
        """
        pass
    
    # Aliases for backward compatibility
    def build_prompt(self, context: PromptContext) -> Union[str, List[Dict[str, str]]]:
        """Alias for generate_prompt (backward compatibility)."""
        return self.generate_prompt(context)
    
    def extract_code(self, response: str, *args, **kwargs) -> str:
        """Alias for extract_code_from_response (backward compatibility)."""
        return self.extract_code_from_response(response)
    
    # =========================================================================
    # SHARED PROMPT COMPONENTS - GREEN SOFTWARE ORIENTED
    # =========================================================================
    
    def _get_sweperf_header(self) -> str:
        """Standard SWE-perf premise with green software focus."""
        return (
            "You will be provided with a partial code base and objective functions. "
            "You need to reduce the code's energy consumption and environmental impact "
            "by editing the code base, while maintaining functional correctness."
        )
    
    def _get_green_software_context(self) -> str:
        """
        Green software optimization context.
        Explains what will be measured and what principles to follow.
        """
        return """### Green Software Engineering Context

Your optimized code will be measured using energy profiling tools that track:
- **CPU energy consumption** (Joules) via hardware energy counters
- **GPU energy consumption** (Joules) via GPU power monitoring
- **Total system energy** (Joules) measured at the power outlet
- **Carbon emissions** (gCO2e) calculated from energy × grid carbon intensity
- **Execution time** (seconds) as a proxy for energy usage

### Green Optimization Principles
To reduce energy consumption, apply these strategies:
1. **Reduce CPU cycles**: Use efficient algorithms (lower time complexity), avoid redundant computations, leverage caching and memoization
2. **Minimize memory allocations**: Reuse objects, use generators instead of lists where possible, avoid unnecessary copies
3. **Optimize data structures**: Choose structures with lower overhead for the access pattern (e.g., sets for lookups, deques for queues)
4. **Reduce I/O operations**: Batch reads/writes, avoid repeated file/network access
5. **Avoid unnecessary work**: Short-circuit evaluations, early returns, skip redundant checks
6. **Use built-in optimizations**: Prefer Python built-ins and standard library functions (implemented in C) over manual Python loops
"""
    
    def _get_search_replace_format_instruction(self) -> str:
        """
        SEARCH/REPLACE format instructions from SWE-perf.
        
        This is the standard format that the PatchEngine expects.
        """
        return """
Please reduce the code's energy consumption by generating *SEARCH/REPLACE* edits.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

**Important:**
- Only edit the source code, NOT the test files
- Only use standard Python libraries or existing project dependencies
- Keep SEARCH blocks small and unique

**Example:**

### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
"""