"""
Prompt Templates for Green Code Refactoring.

This module provides various prompting strategies:
- Zero-Shot: Direct single-turn generation
- Chain-of-Thought (CoT): Reasoning before generation
- Self-Collaboration: Multi-expert collaboration (3 turns)
- LDB: Iterative refinement with feedback

Each strategy supports both ORACLE and REALISTIC contexts.
"""

# Base classes
from .base_template import (
    BasePromptTemplate,
    PromptContext,
    ProblemStatementType,
    PromptStrategy
)

# Zero-Shot templates
from .zero_shot import (
    ZeroShotTemplate,
    ZeroShotOracleTemplate,
    ZeroShotRealisticTemplate
)

# Chain-of-Thought templates
from .chain_of_thought import (
    ChainOfThoughtTemplate,
    CoTOracleTemplate,
    CoTRealisticTemplate,
    CoTResponse,
    parse_cot_response,
    extract_patch_from_cot
)

# Self-Collaboration templates
from .self_collaboration import (
    SelfCollaborationTemplate,
    SelfCollabOracleTemplate,
    SelfCollabRealisticTemplate,
    SelfCollabResponse
)

# LDB templates
from .ldb import (
    LDBTemplate,
    LDBOracleTemplate,
    LDBRealisticTemplate,
    LDBFeedback,
    LDBFeedbackType,
    LDBResponse
)


__all__ = [
    # Base
    'BasePromptTemplate',
    'PromptContext',
    'ProblemStatementType',
    'PromptStrategy',
    
    # Zero-Shot
    'ZeroShotTemplate',
    'ZeroShotOracleTemplate',
    'ZeroShotRealisticTemplate',
    
    # CoT
    'ChainOfThoughtTemplate',
    'CoTOracleTemplate',
    'CoTRealisticTemplate',
    'CoTResponse',
    'parse_cot_response',
    'extract_patch_from_cot',
    
    # Self-Collaboration
    'SelfCollaborationTemplate',
    'SelfCollabOracleTemplate',
    'SelfCollabRealisticTemplate',
    'SelfCollabResponse',
    
    # LDB
    'LDBTemplate',
    'LDBOracleTemplate',
    'LDBRealisticTemplate',
    'LDBFeedback',
    'LDBFeedbackType',
    'LDBResponse',
]


# Strategy mapping for easy lookup
STRATEGY_TEMPLATES = {
    'zero_shot': ZeroShotTemplate,
    'cot': ChainOfThoughtTemplate,
    'self_collab': SelfCollaborationTemplate,
    'ldb': LDBTemplate,
}


def get_template(strategy: str) -> BasePromptTemplate:
    """
    Get template instance by strategy name.
    
    Args:
        strategy: One of 'zero_shot', 'cot', 'self_collab', 'ldb'
        
    Returns:
        Template instance
    """
    if strategy not in STRATEGY_TEMPLATES:
        raise ValueError(f"Unknown strategy: {strategy}. Valid: {list(STRATEGY_TEMPLATES.keys())}")
    return STRATEGY_TEMPLATES[strategy]()