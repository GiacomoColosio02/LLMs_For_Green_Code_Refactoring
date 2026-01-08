"""
Prompt Templates module for LLM-based code optimization.

Available templates:
- ZeroShotTemplate: Direct optimization request (handles both oracle and realistic)
- ZeroShotOracleTemplate: Alias for oracle mode
- ZeroShotRealisticTemplate: Alias for realistic mode

Usage:
    from src.prompt_templates import ZeroShotTemplate, PromptContext, ProblemStatementType
    
    template = ZeroShotTemplate()
    context = PromptContext(
        problem_statement_type=ProblemStatementType.ORACLE,
        code_files={"path/to/file.py": "content..."},
        problem_description="Optimize energy efficiency",
        repo_name="owner/repo"
    )
    prompt = template.generate_prompt(context)
"""
from .base_template import (
    BasePromptTemplate,
    PromptContext,
    PromptStrategy,
    ProblemStatementType
)
from .zero_shot import (
    ZeroShotTemplate,
    ZeroShotOracleTemplate,
    ZeroShotRealisticTemplate
)

__all__ = [
    # Base classes
    'BasePromptTemplate',
    'PromptContext', 
    'PromptStrategy',
    'ProblemStatementType',
    # Templates
    'ZeroShotTemplate',
    'ZeroShotOracleTemplate',
    'ZeroShotRealisticTemplate',
]