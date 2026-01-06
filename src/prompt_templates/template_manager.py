"""
Centralized manager for prompt templates
"""
from typing import Dict, List, Optional, Any
import logging

from .base_template import (
    BasePromptTemplate,
    PromptStrategy,
    ProblemStatementType,
    PromptContext
)
from .zero_shot_template import ZeroShotTemplate
from .cot_template import CoTTemplate
from .self_collaboration_template import SelfCollaborationTemplate

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """
    Factory for creating and managing prompt templates.
    Acts as a facade for both Single-Turn and Multi-Turn strategies.
    """
    
    def __init__(self):
        self._templates: Dict[tuple, BasePromptTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all template combinations"""
        
        # 1. Zero-Shot
        self._templates[(PromptStrategy.ZERO_SHOT, ProblemStatementType.ORACLE)] = \
            ZeroShotTemplate() # Logic handles Oracle internally via context
        self._templates[(PromptStrategy.ZERO_SHOT, ProblemStatementType.REALISTIC)] = \
            ZeroShotTemplate()

        # 2. CoT (Chain of Thought)
        self._templates[(PromptStrategy.COT, ProblemStatementType.ORACLE)] = \
            CoTTemplate()
        self._templates[(PromptStrategy.COT, ProblemStatementType.REALISTIC)] = \
            CoTTemplate()
        
        # 3. Self-Collaboration
        self._templates[(PromptStrategy.SELF_COLLABORATION, ProblemStatementType.ORACLE)] = \
            SelfCollaborationTemplate()
        self._templates[(PromptStrategy.SELF_COLLABORATION, ProblemStatementType.REALISTIC)] = \
            SelfCollaborationTemplate()
        
        logger.info(f"Initialized {len(self._templates)} prompt templates configurations")
    
    def get_template(
        self,
        strategy: PromptStrategy,
        problem_type: ProblemStatementType
    ) -> BasePromptTemplate:
        """
        Get template instance for given strategy and problem type
        """
        key = (strategy, problem_type)
        
        if key not in self._templates:
            # Fallback mechanism if specific combination key is generic
            # (Currently our templates handle oracle/realistic internally, so we reuse instances)
            pass
            
        if key not in self._templates:
             raise ValueError(f"No template found for {strategy.value} + {problem_type.value}")
        
        return self._templates[key]
    
    # =========================================================
    # GENERIC GENERATION (Used for Turn 1 / Single Turn)
    # =========================================================
    def generate_prompts(
        self,
        context: PromptContext,
        strategy: PromptStrategy
    ) -> str:
        """
        Generate the INITIAL prompt.
        - For Zero-Shot/CoT: Returns the full prompt.
        - For Self-Collaboration: Returns Turn 1 (Analyst) prompt.
        """
        template = self.get_template(strategy, context.problem_statement_type)
        logger.info(f"Generating initial prompt for {strategy.value}")
        return template.generate_prompt(context)
    
    # =========================================================
    # MULTI-TURN SPECIFIC METHODS (Self-Collaboration)
    # =========================================================
    def generate_engineer_prompt(
        self, 
        context: PromptContext, 
        strategy: PromptStrategy, 
        analyst_output: str
    ) -> str:
        """Turn 2: Engineer Implementation"""
        template = self.get_template(strategy, context.problem_statement_type)
        if not isinstance(template, SelfCollaborationTemplate):
            raise ValueError("generate_engineer_prompt is only for Self-Collaboration strategy")
        return template.get_engineer_prompt(context, analyst_output)

    def generate_reviewer_prompt(
        self, 
        context: PromptContext, 
        strategy: PromptStrategy, 
        analyst_goals: str, 
        engineer_patch: str
    ) -> str:
        """Turn 3: Critical Review"""
        template = self.get_template(strategy, context.problem_statement_type)
        if not isinstance(template, SelfCollaborationTemplate):
            raise ValueError("generate_reviewer_prompt is only for Self-Collaboration strategy")
        return template.get_reviewer_prompt(context, analyst_goals, engineer_patch)

    def generate_final_engineer_prompt(
        self, 
        context: PromptContext, 
        strategy: PromptStrategy, 
        engineer_patch: str, 
        reviewer_critique: str
    ) -> str:
        """Turn 4: Engineer Final Polish"""
        template = self.get_template(strategy, context.problem_statement_type)
        if not isinstance(template, SelfCollaborationTemplate):
            raise ValueError("generate_final_engineer_prompt is only for Self-Collaboration strategy")
        return template.get_final_engineer_prompt(context, engineer_patch, reviewer_critique)

    # =========================================================
    # EXTRACTION LOGIC
    # =========================================================
    def extract_code(
        self,
        response: str,
        strategy: PromptStrategy,
        problem_type: ProblemStatementType,
    ) -> str:
        """
        Extract code patch from LLM response.
        Compatible with all strategies.
        """
        template = self.get_template(strategy, problem_type)
        # We removed the 'turn' argument because the new SelfCollaborationTemplate 
        # uses a unified extraction logic (looking for SEARCH blocks).
        return template.extract_code_from_response(response)
    
    def get_available_combinations(self) -> List[tuple]:
        """Get all available (strategy, problem_type) combinations"""
        return list(self._templates.keys())