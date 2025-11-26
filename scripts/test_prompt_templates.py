"""
Test script for prompt template system
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_templates import (
    PromptTemplateManager,
    PromptContext,
    PromptStrategy,
    ProblemStatementType
)


def create_dummy_context(problem_type: ProblemStatementType) -> PromptContext:
    """Create dummy context for testing"""
    
    return PromptContext(
        instance_id="test__test-001",
        problem_statement_type=problem_type,
        target_functions=[
            {
                "name": "slow_function",
                "file": "src/module.py",
                "start_line": 10,
                "end_line": 25
            }
        ],
        code_files={
            "src/module.py": """def slow_function(data):
    result = []
    for item in data:
        for i in range(len(data)):
            if item == data[i]:
                result.append(item)
    return result
"""
        },
        problem_description="Function has O(n²) complexity and performs redundant comparisons",
        test_command="pytest tests/test_module.py::test_slow_function -v",
        repo_name="test-repo",
        base_commit="abc123"
    )


def main():
    print("\n" + "="*70)
    print("🧪 TESTING PROMPT TEMPLATE SYSTEM")
    print("="*70)
    
    manager = PromptTemplateManager()
    
    # Test all combinations
    combinations = [
        (PromptStrategy.ZERO_SHOT, ProblemStatementType.ORACLE),
        (PromptStrategy.ZERO_SHOT, ProblemStatementType.REALISTIC),
        (PromptStrategy.SELF_COLLABORATION, ProblemStatementType.ORACLE),
        (PromptStrategy.SELF_COLLABORATION, ProblemStatementType.REALISTIC),
    ]
    
    for strategy, problem_type in combinations:
        print(f"\n{'-'*70}")
        print(f"📝 Testing: {strategy.value} + {problem_type.value}")
        print(f"{'-'*70}\n")
        
        context = create_dummy_context(problem_type)
        prompts = manager.generate_prompts(context, strategy)
        
        if isinstance(prompts, str):
            # Zero-Shot: single prompt
            print(f"Generated prompt ({len(prompts)} chars):")
            print(prompts[:500] + "..." if len(prompts) > 500 else prompts)
        else:
            # Self-Collaboration: multiple turns
            print(f"Generated {len(prompts)} turns:")
            for i, turn in enumerate(prompts, 1):
                role = turn['role']
                prompt = turn['prompt']
                print(f"\n  Turn {i} - {role} ({len(prompt)} chars)")
                print(f"  Preview: {prompt[:200]}...")
    
    print("\n" + "="*70)
    print("✅ All template combinations working!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()