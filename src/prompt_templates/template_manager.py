"""
Centralized manager for prompt templates
"""
from typing import Dict, List, Optional
import logging

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

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """
    Factory for creating and managing prompt templates
    """
    
    def __init__(self):
        self._templates: Dict[tuple, BasePromptTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all template combinations"""
        
        # Zero-Shot templates
        self._templates[(PromptStrategy.ZERO_SHOT, ProblemStatementType.ORACLE)] = \
            ZeroShotOracleTemplate()
        
        self._templates[(PromptStrategy.ZERO_SHOT, ProblemStatementType.REALISTIC)] = \
            ZeroShotRealisticTemplate()
        
        # Self-Collaboration templates
        self._templates[(PromptStrategy.SELF_COLLABORATION, ProblemStatementType.ORACLE)] = \
            SelfCollaborationOracleTemplate()
        
        self._templates[(PromptStrategy.SELF_COLLABORATION, ProblemStatementType.REALISTIC)] = \
            SelfCollaborationRealisticTemplate()
        
        logger.info(f"Initialized {len(self._templates)} prompt templates")
    
    def get_template(
        self,
        strategy: PromptStrategy,
        problem_type: ProblemStatementType
    ) -> BasePromptTemplate:
        """
        Get template for given strategy and problem type
        
        Args:
            strategy: Zero-Shot or Self-Collaboration
            problem_type: Oracle or Realistic
            
        Returns:
            Appropriate template instance
        """
        key = (strategy, problem_type)
        
        if key not in self._templates:
            raise ValueError(f"No template found for {strategy.value} + {problem_type.value}")
        
        return self._templates[key]
    
    def generate_prompts(
        self,
        context: PromptContext,
        strategy: PromptStrategy
    ) -> str | List[Dict[str, str]]:
        """
        Generate prompts for given context and strategy
        
        Args:
            context: Prompt context with instance data
            strategy: Which strategy to use
            
        Returns:
            - For Zero-Shot: single string prompt
            - For Self-Collaboration: list of turn prompts
        """
        template = self.get_template(strategy, context.problem_statement_type)
        
        logger.info(
            f"Generating {strategy.value} prompts for {context.problem_statement_type.value} mode"
        )
        
        return template.generate_prompt(context)
    
    def extract_code(
        self,
        response: str,
        strategy: PromptStrategy,
        problem_type: ProblemStatementType,
        turn: int = 1
    ) -> str:
        """
        Extract code patch from LLM response
        
        Args:
            response: Raw LLM output
            strategy: Which strategy was used
            problem_type: Which problem type
            turn: Which turn (for Self-Collaboration)
            
        Returns:
            Extracted code patch
        """
        template = self.get_template(strategy, problem_type)
        
        if strategy == PromptStrategy.SELF_COLLABORATION:
            return template.extract_code_from_response(response, turn)
        else:
            return template.extract_code_from_response(response)
    
    def get_available_combinations(self) -> List[tuple]:
        """Get all available (strategy, problem_type) combinations"""
        return list(self._templates.keys())