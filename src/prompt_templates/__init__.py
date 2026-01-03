"""
Prompt Templates package initialization.
Exports the strategies used in the benchmark:
- Single Turn: Zero-Shot, Few-Shot
- Multi Turn: Self-Collaboration
"""

from .base_template import (
    BasePromptTemplate,
    PromptContext,
    PromptStrategy,
    ProblemStatementType
)
from .zero_shot_template import ZeroShotTemplate
from .few_shot_template import FewShotTemplate
from .self_collaboration_template import SelfCollaborationTemplate
from .template_manager import TemplateManager

__all__ = [
    'BasePromptTemplate',
    'PromptContext',
    'PromptStrategy',
    'ProblemStatementType',
    'ZeroShotTemplate',
    'FewShotTemplate',
    'SelfCollaborationTemplate',
    'TemplateManager'
]