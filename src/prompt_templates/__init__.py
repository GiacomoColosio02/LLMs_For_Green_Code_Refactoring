"""Prompt Templates module."""
from .base_template import (
    BasePromptTemplate, PromptContext, PromptStrategy, ProblemStatementType
)
from .zero_shot import (
    ZeroShotTemplate, ZeroShotOracleTemplate, ZeroShotRealisticTemplate
)
from .chain_of_thought import (
    ChainOfThoughtTemplate, CoTOracleTemplate, CoTRealisticTemplate,
    CoTResponse, parse_cot_response, extract_patch_from_cot
)

__all__ = [
    'BasePromptTemplate', 'PromptContext', 'PromptStrategy', 'ProblemStatementType',
    'ZeroShotTemplate', 'ZeroShotOracleTemplate', 'ZeroShotRealisticTemplate',
    'ChainOfThoughtTemplate', 'CoTOracleTemplate', 'CoTRealisticTemplate',
    'CoTResponse', 'parse_cot_response', 'extract_patch_from_cot',
]
