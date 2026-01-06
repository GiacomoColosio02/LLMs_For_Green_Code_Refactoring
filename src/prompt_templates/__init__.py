"""
Prompt Templates Module.
Exposes the strategies available for the experiment runner.
"""
from .base_template import BasePromptTemplate, PromptStrategy, ProblemStatementType, PromptContext
from .template_manager import PromptTemplateManager

# Esportiamo solo ciò che è stabile
__all__ = [
    "BasePromptTemplate",
    "PromptStrategy", 
    "ProblemStatementType",
    "PromptContext",
    "PromptTemplateManager"
]