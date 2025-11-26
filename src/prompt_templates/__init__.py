"""
Prompt template system for LLM-based code optimization
"""
from .base_template import (
    BasePromptTemplate,
    PromptStrategy,
    ProblemStatementType,
    PromptContext
)
from .zero_shot_template import (
    ZeroShotTemplate,
    ZeroShotOracleTemplate,
    ZeroShotRealisticTemplate
)
from .self_collaboration_template import (
    SelfCollaborationTemplate,
    SelfCollaborationOracleTemplate,
    SelfCollaborationRealisticTemplate
)
from .template_manager import PromptTemplateManager

__all__ = [
    # Enums
    'PromptStrategy',
    'ProblemStatementType',
    # Data classes
    'PromptContext',
    # Templates
    'BasePromptTemplate',
    'ZeroShotTemplate',
    'ZeroShotOracleTemplate',
    'ZeroShotRealisticTemplate',
    'SelfCollaborationTemplate',
    'SelfCollaborationOracleTemplate',
    'SelfCollaborationRealisticTemplate',
    # Manager
    'PromptTemplateManager',
]