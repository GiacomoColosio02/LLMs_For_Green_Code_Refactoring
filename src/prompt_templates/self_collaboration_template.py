"""
Self-Collaboration prompt template (multi-turn with roles)
"""
from typing import Dict, List
import re

from .base_template import (
    BasePromptTemplate,
    PromptStrategy,
    PromptContext,
    ProblemStatementType
)


class SelfCollaborationTemplate(BasePromptTemplate):
    """Multi-turn optimization with specialized roles"""
    
    def __init__(self):
        super().__init__(PromptStrategy.SELF_COLLABORATION)
        self.roles = [
            "Performance Analyst",
            "Code Optimizer",
            "Quality Validator"
        ]
    
    def generate_prompt(self, context: PromptContext) -> List[Dict[str, str]]:
        """
        Generate multi-turn conversation
        
        Returns:
            List of prompts, one for each turn/role
        """
        
        turns = []
        
        # Turn 1: Performance Analyst
        turns.append({
            "role": "Performance Analyst",
            "prompt": self._generate_analyst_prompt(context)
        })
        
        # Turn 2: Code Optimizer (requires analyst output)
        turns.append({
            "role": "Code Optimizer",
            "prompt": self._generate_optimizer_prompt(context),
            "requires_previous": True
        })
        
        # Turn 3: Quality Validator (requires optimizer output)
        turns.append({
            "role": "Quality Validator",
            "prompt": self._generate_validator_prompt(context),
            "requires_previous": True
        })
        
        return turns
    
    def _generate_analyst_prompt(self, context: PromptContext) -> str:
        """Turn 1: Analyze code and identify bottlenecks"""
        
        problem_statement = self._format_problem_statement(context)
        
        return f"""
# Role: Performance Analyst

You are a performance analyst expert. Your task is to analyze the provided code and identify performance bottlenecks.

{problem_statement}

## Your Analysis Task

Please provide:

1. **Bottleneck Identification**: Which parts of the code are likely causing performance issues?
2. **Complexity Analysis**: What is the time/space complexity of key operations?
3. **Resource Usage**: Where are memory, CPU, or I/O resources being wasted?
4. **Optimization Opportunities**: List specific opportunities for optimization, ordered by expected impact.

## Output Format

Provide your analysis in this structure:
```
### Primary Bottlenecks
- [Location]: [Issue description]

### Complexity Analysis
- [Function/Operation]: O(?) time, O(?) space

### Optimization Opportunities
1. [High Impact] [Description of optimization]
2. [Medium Impact] [Description of optimization]
3. [Low Impact] [Description of optimization]

### Recommendations
[Your prioritized recommendations for the Code Optimizer]
```

Provide only your analysis. Do NOT write optimized code yet.
"""
    
    def _generate_optimizer_prompt(self, context: PromptContext) -> str:
        """Turn 2: Generate optimized code based on analysis"""
        
        return f"""
# Role: Code Optimizer

You are a code optimization expert. Based on the Performance Analyst's analysis, your task is to generate optimized code.

## Previous Analysis

{{ANALYSIS_FROM_TURN_1}}

## Your Optimization Task

Based on the analyst's recommendations:

1. **Implement Optimizations**: Focus on high and medium impact optimizations
2. **Maintain Correctness**: Ensure all functionality is preserved
3. **Code Quality**: Keep code readable and maintainable
4. **Performance Gains**: Aim for measurable improvements in time, memory, or energy

## Test Command
```bash
{context.test_command}
```

{self._format_output_instructions()}

Generate the optimized code patch now.
"""
    
    def _generate_validator_prompt(self, context: PromptContext) -> str:
        """Turn 3: Validate and refine the optimization"""
        
        return f"""
# Role: Quality Validator

You are a quality assurance expert. Your task is to validate the optimized code and suggest refinements.

## Previous Work

**Analysis:**
{{ANALYSIS_FROM_TURN_1}}

**Optimized Code:**
{{CODE_FROM_TURN_2}}

## Your Validation Task

Review the optimized code and check:

1. **Correctness**: Does it maintain all functionality?
2. **Performance**: Will it actually improve performance?
3. **Edge Cases**: Are edge cases handled correctly?
4. **Code Quality**: Is it readable and maintainable?
5. **Test Coverage**: Will it pass all existing tests?

## Output Format

Provide:
```
### Validation Results
- ✓ [Aspect]: [Status/Comments]
- ✗ [Issue]: [Problem found]

### Recommended Refinements
[If any issues found, suggest specific fixes]

### Final Patch
[Either approve the patch from Turn 2, or provide refined version]
```

If the optimization is good, output: "APPROVED: [original patch]"
If refinements needed, output the refined patch in unified diff format.
"""
    
    def extract_code_from_response(self, response: str, turn: int = 3) -> str:
        """
        Extract code from multi-turn response
        
        Args:
            response: Could be from any turn, but typically Turn 3 (validator)
            turn: Which turn this response is from
        
        Returns:
            Extracted patch
        """
        if turn == 1:
            # Analysis turn - no code to extract
            return ""
        
        # For turns 2 and 3, extract diff patch
        # Try to find diff code block
        diff_pattern = r'```diff\n(.*?)```'
        matches = re.findall(diff_pattern, response, re.DOTALL)
        
        if matches:
            return matches[-1].strip()  # Take last match if multiple
        
        # Check for "APPROVED: " marker
        if "APPROVED:" in response:
            # Extract patch after APPROVED marker
            approved_pattern = r'APPROVED:\s*(.*?)(?:\n\n|\Z)'
            match = re.search(approved_pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: find diff lines
        diff_lines = []
        in_diff = False
        for line in response.split('\n'):
            if line.startswith('---'):
                in_diff = True
            if in_diff:
                diff_lines.append(line)
        
        if diff_lines:
            return '\n'.join(diff_lines).strip()
        
        return response.strip()


class SelfCollaborationOracleTemplate(SelfCollaborationTemplate):
    """Self-Collaboration specialized for ORACLE mode"""
    
    def _generate_analyst_prompt(self, context: PromptContext) -> str:
        base_prompt = super()._generate_analyst_prompt(context)
        
        oracle_note = "\n**Note**: Focus your analysis on the provided target functions, as these are known optimization targets.\n"
        
        return base_prompt.replace(
            "## Your Analysis Task",
            f"{oracle_note}\n## Your Analysis Task"
        )


class SelfCollaborationRealisticTemplate(SelfCollaborationTemplate):
    """Self-Collaboration specialized for REALISTIC mode"""
    
    def _generate_analyst_prompt(self, context: PromptContext) -> str:
        base_prompt = super()._generate_analyst_prompt(context)
        
        realistic_note = "\n**Note**: Analyze the entire codebase. The listed functions are entry points from tests, but bottlenecks may be deeper in the call chain.\n"
        
        return base_prompt.replace(
            "## Your Analysis Task",
            f"{realistic_note}\n## Your Analysis Task"
        )