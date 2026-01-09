"""
Unified Experiment Runner for Green Code Refactoring.

Supports all prompting strategies:
- zero_shot: Single-turn direct generation
- cot: Chain-of-Thought reasoning
- self_collab: Multi-turn expert collaboration (3 turns)
- ldb: Iterative refinement with feedback (up to 3 iterations)

Version: 3.0 - Added multi-turn strategy support

Usage:
    python run_experiment.py --instance mwaskom__seaborn-2389 --strategy oracle --prompt-type zero_shot
    python run_experiment.py --instance mwaskom__seaborn-2389 --strategy realistic --prompt-type self_collab
    python run_experiment.py --instance mwaskom__seaborn-2389 --strategy oracle --prompt-type ldb
"""
import sys
import os
import re
import json
import logging
import tempfile
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORTS ---
from src.llm_clients import VLLMClient
from src.prompt_templates import (
    ZeroShotTemplate, 
    ChainOfThoughtTemplate,
    SelfCollaborationTemplate,
    LDBTemplate,
    LDBFeedback,
    LDBFeedbackType,
    PromptContext, 
    ProblemStatementType,
    extract_patch_from_cot
)
from src.patch_engine import PatchEngine, PatchResult
from scripts.measure_instance import SWEPerfMeasurer

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger("ExperimentRunner")

# --- CONSTANTS ---
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "swe_perf_reduced_test.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Context limits per strategy
CONTEXT_LIMITS = {
    "oracle": {
        "max_total_chars": 50000,
        "max_file_chars": 25000,
        "max_files": 5,
    },
    "realistic": {
        "max_total_chars": 35000,
        "max_file_chars": 8000,
        "max_test_chars": 5000,
        "max_retrieved_files": 5,
        "max_repo_map_lines": 100,
    }
}

# Prompt type configurations
PROMPT_CONFIGS = {
    "zero_shot": {"multi_turn": False, "template_class": ZeroShotTemplate},
    "cot": {"multi_turn": False, "template_class": ChainOfThoughtTemplate},
    "self_collab": {"multi_turn": True, "num_turns": 3, "template_class": SelfCollaborationTemplate},
    "ldb": {"multi_turn": True, "iterative": True, "max_iterations": 3, "template_class": LDBTemplate},
}


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Runs green code optimization experiments with various prompting strategies.
    """
    
    def __init__(
        self,
        dataset_path: Path,
        strategy: str = "oracle",
        prompt_type: str = "zero_shot",
        output_dir: Optional[Path] = None
    ):
        self.dataset_path = Path(dataset_path)
        self.strategy = strategy.lower()
        self.prompt_type = prompt_type.lower()
        
        if self.strategy not in ("oracle", "realistic"):
            raise ValueError(f"Strategy must be 'oracle' or 'realistic', got: {strategy}")
        
        if self.prompt_type not in PROMPT_CONFIGS:
            valid = list(PROMPT_CONFIGS.keys())
            raise ValueError(f"Prompt type must be one of {valid}, got: {prompt_type}")
        
        self.prompt_config = PROMPT_CONFIGS[self.prompt_type]
        self.limits = CONTEXT_LIMITS[self.strategy]
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            prefix_map = {"zero_shot": "zs", "cot": "cot", "self_collab": "sc", "ldb": "ldb"}
            prefix = prefix_map.get(self.prompt_type, self.prompt_type)
            self.output_dir = RESULTS_DIR / f"{prefix}_{self.strategy}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Initialize components
        self.llm_client = VLLMClient()
        self.template = self.prompt_config["template_class"]()
        self.measurer = SWEPerfMeasurer(str(dataset_path), country_code="ESP")
        
        logger.info(f"Initialized ExperimentRunner")
        logger.info(f"  Prompt Type: {self.prompt_type.upper()}")
        logger.info(f"  Strategy: {self.strategy.upper()}")
        logger.info(f"  Multi-turn: {self.prompt_config.get('multi_turn', False)}")
        logger.info(f"  Dataset: {self.dataset_path.name} ({len(self.dataset)} instances)")
        logger.info(f"  Model: {self.llm_client.model_name}")
    
    def _load_dataset(self) -> List[Dict]:
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("instances", [])
    
    def _get_instance(self, instance_id: str) -> Dict:
        for item in self.dataset:
            if item.get("instance_id") == instance_id:
                return item
        raise ValueError(f"Instance '{instance_id}' not found")
    
    # =========================================================================
    # CONTEXT BUILDING (same as before)
    # =========================================================================
    
    def _truncate_file(self, content: str, max_chars: int, filename: str = "") -> str:
        if len(content) <= max_chars:
            return content
        
        lines = content.splitlines()
        total_chars = max_chars - 100
        first_portion = int(total_chars * 0.6)
        last_portion = int(total_chars * 0.4)
        
        first_lines = []
        char_count = 0
        for line in lines:
            if char_count + len(line) + 1 > first_portion:
                break
            first_lines.append(line)
            char_count += len(line) + 1
        
        last_lines = []
        char_count = 0
        for line in reversed(lines):
            if char_count + len(line) + 1 > last_portion:
                break
            last_lines.insert(0, line)
            char_count += len(line) + 1
        
        truncation_msg = f"\n# ... [{len(lines) - len(first_lines) - len(last_lines)} lines truncated] ...\n"
        return "\n".join(first_lines) + truncation_msg + "\n".join(last_lines)
    
    def _build_oracle_context(self, repo_path: Path, instance: Dict) -> Tuple[Dict[str, str], List[str]]:
        limits = self.limits
        patch_content = instance.get("patch", "")
        
        target_files = []
        for line in patch_content.splitlines():
            if line.startswith("--- a/"):
                fname = line[6:].strip()
                if (repo_path / fname).exists():
                    target_files.append(fname)
        
        target_files = list(set(target_files))
        
        if not target_files:
            patch_funcs = instance.get("patch_functions", "{}")
            if isinstance(patch_funcs, str):
                patch_funcs = json.loads(patch_funcs)
            target_files = list(patch_funcs.keys())
        
        code_files = {}
        current_chars = 0
        max_total = limits["max_total_chars"]
        max_file = limits["max_file_chars"]
        max_files = limits["max_files"]
        
        for fname in target_files[:max_files]:
            if current_chars >= max_total:
                break
            
            fpath = repo_path / fname
            if not fpath.exists():
                continue
            
            try:
                content = fpath.read_text(errors='ignore')
                
                if len(content) > max_file:
                    content = self._truncate_file(content, max_file, fname)
                
                remaining = max_total - current_chars
                if len(content) > remaining:
                    if remaining > 2000:
                        content = self._truncate_file(content, remaining, fname)
                    else:
                        continue
                
                code_files[fname] = content
                current_chars += len(content)
                
            except Exception as e:
                logger.warning(f"Could not read {fname}: {e}")
        
        logger.info(f"📂 Oracle context: {len(code_files)} files, {current_chars:,} chars")
        return code_files, list(code_files.keys())
    
    def _build_realistic_context(self, repo_path: Path, instance: Dict) -> Tuple[Dict[str, str], List[str], str]:
        limits = self.limits
        max_total = limits["max_total_chars"]
        max_file = limits["max_file_chars"]
        max_test = limits["max_test_chars"]
        max_retrieved = limits["max_retrieved_files"]
        max_repo_lines = limits["max_repo_map_lines"]
        
        test_list = instance.get('efficiency_test', [])
        test_files = list(set(t.split("::")[0] for t in test_list))
        
        repo_map = self._generate_repo_map(repo_path, max_lines=max_repo_lines)
        
        code_files = {}
        current_chars = 0
        
        # Add test files
        for tf in test_files[:3]:
            fpath = repo_path / tf
            if not fpath.exists():
                continue
            
            try:
                content = fpath.read_text(errors='ignore')
                if len(content) > max_test:
                    content = content[:max_test] + "\n# ... [TRUNCATED]"
                
                if current_chars + len(content) <= max_total * 0.4:
                    code_files[tf] = content
                    current_chars += len(content)
            except:
                pass
        
        # Retrieve source files
        query_tokens = set()
        for content in code_files.values():
            query_tokens.update(self._tokenize(content))
        
        stopwords = {'def', 'class', 'self', 'import', 'from', 'in', 'if', 'else', 
                     'return', 'assert', 'test', 'none', 'true', 'false', 'and', 
                     'or', 'pytest', 'for', 'while', 'with', 'as', 'try', 'except'}
        query_tokens -= stopwords
        
        scores = {}
        for root, _, files in os.walk(repo_path):
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                
                rel_path = os.path.relpath(os.path.join(root, fname), repo_path)
                
                if rel_path in test_files or fname == '__init__.py':
                    continue
                if '/tests/' in rel_path or '/test_' in rel_path:
                    continue
                
                try:
                    content = (repo_path / rel_path).read_text(errors='ignore')
                    file_tokens = self._tokenize(content)
                    score = len(query_tokens & file_tokens)
                    if score > 0:
                        scores[rel_path] = (score, len(content))
                except:
                    pass
        
        top_files = sorted(scores.items(), key=lambda x: (x[1][0], -x[1][1]), reverse=True)
        
        files_added = 0
        for fname, (score, _) in top_files:
            if files_added >= max_retrieved or current_chars >= max_total:
                break
            
            try:
                content = (repo_path / fname).read_text(errors='ignore')
                
                if len(content) > max_file:
                    content = self._truncate_file(content, max_file, fname)
                
                remaining = max_total - current_chars
                if len(content) > remaining:
                    if remaining > 2000:
                        content = self._truncate_file(content, remaining, fname)
                    else:
                        continue
                
                code_files[fname] = content
                current_chars += len(content)
                files_added += 1
                
            except:
                pass
        
        logger.info(f"📂 Realistic context: {len(code_files)} files, {current_chars:,} chars")
        return code_files, list(code_files.keys()), repo_map
    
    def _generate_repo_map(self, repo_path: Path, max_lines: int = 100) -> str:
        tree = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
            level = root.replace(str(repo_path), '').count(os.sep)
            if level > 3:
                continue
            indent = '    ' * level
            tree.append(f"{indent}{os.path.basename(root)}/")
            subindent = '    ' * (level + 1)
            py_files = [f for f in files if f.endswith('.py')][:10]
            for f in py_files:
                tree.append(f"{subindent}{f}")
            if len(tree) > max_lines:
                break
        
        if len(tree) > max_lines:
            return "\n".join(tree[:max_lines]) + "\n... (truncated)"
        return "\n".join(tree)
    
    def _tokenize(self, text: str) -> set:
        return set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower()))
    
    # =========================================================================
    # PROMPT CONTEXT
    # =========================================================================
    
    def _build_prompt_context(
        self,
        instance: Dict,
        code_files: Dict[str, str],
        candidates: List[str],
        repo_map: Optional[str] = None
    ) -> PromptContext:
        if self.strategy == "oracle":
            problem_desc = instance.get('problem_statement_oracle',
                                        "Optimize energy efficiency for the provided code.")
            stmt_type = ProblemStatementType.ORACLE
        else:
            problem_desc = instance.get('problem_statement_realistic',
                                        "Optimize energy efficiency based on the failing tests.")
            stmt_type = ProblemStatementType.REALISTIC
        
        test_list = instance.get('efficiency_test', [])
        test_cmd = f"pytest {' '.join(test_list)}" if test_list else ""
        
        return PromptContext(
            instance_id=instance['instance_id'],
            problem_statement_type=stmt_type,
            problem_description=problem_desc,
            code_files=code_files,
            target_functions=candidates,
            test_command=test_cmd,
            repo_name=instance.get('repo', ''),
            base_commit=instance.get('base_commit', ''),
            repo_map=repo_map
        )
    
    def _get_system_prompt(self) -> str:
        return (
            "You are an expert Green Software Engineer specializing in energy-efficient code.\n"
            "Your goal is to optimize the provided code for energy efficiency and execution speed.\n\n"
            "CRITICAL: Output ONLY the code patch using SEARCH/REPLACE format.\n"
            "Do NOT add new external dependencies. Do NOT modify test files."
        )
    
    # =========================================================================
    # SINGLE-TURN EXECUTION
    # =========================================================================
    
    def _run_single_turn(
        self,
        context: PromptContext,
        candidates: List[str],
        repo_path: Path
    ) -> Dict:
        """Execute single-turn strategies (zero_shot, cot)."""
        
        user_prompt = self.template.generate_prompt(context)
        system_prompt = self._get_system_prompt()
        
        total_size = len(system_prompt) + len(user_prompt)
        logger.info(f"   Prompt size: {total_size:,} chars (~{total_size//3:,} tokens)")
        
        response = self.llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=4096
        )
        
        llm_output = response.content
        logger.info(f"   Response: {len(llm_output):,} chars, {response.total_tokens} tokens")
        
        # Extract patch
        if self.prompt_type == "cot":
            patch_content = extract_patch_from_cot(llm_output)
        else:
            patch_content = llm_output
        
        # Apply patch
        patch_engine = PatchEngine(repo_path)
        patch_result = patch_engine.apply_patch(patch_content, candidates)
        
        return {
            "llm_response": llm_output,
            "llm_tokens": response.total_tokens,
            "llm_latency": response.latency_seconds,
            "patch_content": patch_content,
            "patch_result": {
                "success": patch_result.success,
                "changes_applied": patch_result.changes_applied,
                "total_blocks": patch_result.total_blocks,
                "method": patch_result.method_used,
                "modified_files": patch_result.modified_files,
                "error": patch_result.error_message
            },
            "status": "success" if patch_result.success else "patch_failed"
        }
    
    # =========================================================================
    # SELF-COLLABORATION EXECUTION
    # =========================================================================
    
    def _run_self_collaboration(
        self,
        context: PromptContext,
        candidates: List[str],
        repo_path: Path
    ) -> Dict:
        """Execute Self-Collaboration strategy (3 turns)."""
        
        logger.info("🤝 Running Self-Collaboration (3 turns)")
        
        responses = []
        total_tokens = 0
        total_latency = 0.0
        system_prompt = self.template._get_system_prompt()
        
        for turn in range(3):
            role_names = ["ANALYST", "OPTIMIZER", "REVIEWER"]
            logger.info(f"   Turn {turn + 1}/3: {role_names[turn]}")
            
            # Generate turn prompt
            turn_prompt = self.template.generate_turn_prompt(turn, context, responses)
            
            # Call LLM
            response = self.llm_client.generate(
                prompt=turn_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=2048 if turn < 2 else 4096  # More tokens for final turn
            )
            
            responses.append(response.content)
            total_tokens += response.total_tokens
            total_latency += response.latency_seconds
            
            logger.info(f"      Response: {len(response.content):,} chars")
        
        # Parse final response
        parsed = self.template.parse_collaboration_responses(responses)
        patch_content = parsed.final_patch
        
        # Apply patch
        patch_engine = PatchEngine(repo_path)
        patch_result = patch_engine.apply_patch(patch_content, candidates)
        
        return {
            "llm_response": responses[-1],  # Final response
            "all_responses": responses,
            "llm_tokens": total_tokens,
            "llm_latency": total_latency,
            "patch_content": patch_content,
            "turns": {
                "analyst": responses[0] if len(responses) > 0 else "",
                "optimizer": responses[1] if len(responses) > 1 else "",
                "reviewer": responses[2] if len(responses) > 2 else ""
            },
            "patch_result": {
                "success": patch_result.success,
                "changes_applied": patch_result.changes_applied,
                "total_blocks": patch_result.total_blocks,
                "method": patch_result.method_used,
                "modified_files": patch_result.modified_files,
                "error": patch_result.error_message
            },
            "status": "success" if patch_result.success else "patch_failed"
        }
    
    # =========================================================================
    # LDB (ITERATIVE) EXECUTION
    # =========================================================================
    
    def _run_ldb(
        self,
        context: PromptContext,
        candidates: List[str],
        repo_path: Path
    ) -> Dict:
        """Execute LDB strategy (iterative refinement)."""
        
        max_iterations = self.prompt_config.get("max_iterations", 3)
        logger.info(f"🔄 Running LDB (max {max_iterations} iterations)")
        
        iterations = []
        total_tokens = 0
        total_latency = 0.0
        system_prompt = self.template._get_system_prompt()
        
        patch_content = ""
        patch_result = None
        final_status = "error"
        
        for iteration in range(max_iterations):
            logger.info(f"   Iteration {iteration + 1}/{max_iterations}")
            
            # Generate prompt
            if iteration == 0:
                prompt = self.template.generate_initial_prompt(context)
            else:
                # Create feedback from previous failure
                feedback = self._create_feedback_from_result(patch_result, patch_content)
                prompt = self.template.generate_refinement_prompt(
                    context, patch_content, feedback, iteration
                )
            
            # Call LLM
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=4096
            )
            
            total_tokens += response.total_tokens
            total_latency += response.latency_seconds
            
            # Extract patch
            patch_content = self.template.extract_code_from_response(response.content)
            logger.info(f"      Response: {len(response.content):,} chars")
            
            # Try to apply patch
            patch_engine = PatchEngine(repo_path)
            patch_result = patch_engine.apply_patch(patch_content, candidates)
            
            iteration_data = {
                "iteration": iteration + 1,
                "response": response.content,
                "patch_content": patch_content,
                "success": patch_result.success,
                "error": patch_result.error_message
            }
            iterations.append(iteration_data)
            
            if patch_result.success:
                logger.info(f"      ✅ Patch applied successfully!")
                final_status = "success"
                break
            else:
                logger.info(f"      ⚠️ Patch failed: {patch_result.error_message}")
                final_status = "patch_failed"
        
        return {
            "llm_response": iterations[-1]["response"] if iterations else "",
            "iterations": iterations,
            "num_iterations": len(iterations),
            "llm_tokens": total_tokens,
            "llm_latency": total_latency,
            "patch_content": patch_content,
            "patch_result": {
                "success": patch_result.success if patch_result else False,
                "changes_applied": patch_result.changes_applied if patch_result else 0,
                "total_blocks": patch_result.total_blocks if patch_result else 0,
                "method": patch_result.method_used if patch_result else "",
                "modified_files": patch_result.modified_files if patch_result else [],
                "error": patch_result.error_message if patch_result else "No iterations"
            },
            "status": final_status
        }
    
    def _create_feedback_from_result(
        self, 
        patch_result: PatchResult, 
        patch_content: str
    ) -> LDBFeedback:
        """Create LDB feedback from a failed patch result."""
        
        if not patch_result:
            return LDBFeedback(
                feedback_type=LDBFeedbackType.PATCH_PARSE_ERROR,
                message="No patch result available"
            )
        
        error_msg = patch_result.error_message or "Unknown error"
        
        # Determine feedback type based on error
        if "SEARCH block not found" in error_msg or "not found in file" in error_msg:
            return LDBFeedback(
                feedback_type=LDBFeedbackType.PATCH_APPLY_ERROR,
                message="SEARCH block did not match any code in the file",
                details=error_msg
            )
        elif "parse" in error_msg.lower() or "format" in error_msg.lower():
            return LDBFeedback(
                feedback_type=LDBFeedbackType.PATCH_PARSE_ERROR,
                message="Could not parse patch format",
                details=error_msg
            )
        elif "syntax" in error_msg.lower():
            return LDBFeedback(
                feedback_type=LDBFeedbackType.SYNTAX_ERROR,
                message="Python syntax error in optimized code",
                details=error_msg
            )
        elif "import" in error_msg.lower() or "module" in error_msg.lower():
            return LDBFeedback(
                feedback_type=LDBFeedbackType.IMPORT_ERROR,
                message="Import or module error",
                details=error_msg
            )
        else:
            return LDBFeedback(
                feedback_type=LDBFeedbackType.PATCH_APPLY_ERROR,
                message=error_msg,
                details=str(patch_result.__dict__) if patch_result else None
            )
    
    # =========================================================================
    # MAIN RUN METHOD
    # =========================================================================
    
    def run(self, instance_id: str) -> Dict:
        """Run experiment for a single instance."""
        
        logger.info(f"{'='*60}")
        logger.info(f"🚀 STARTING EXPERIMENT: {instance_id}")
        logger.info(f"   Prompt Type: {self.prompt_type.upper()}")
        logger.info(f"   Strategy: {self.strategy.upper()}")
        logger.info(f"{'='*60}")
        
        result = {
            "instance_id": instance_id,
            "prompt_type": self.prompt_type,
            "strategy": self.strategy,
            "model": self.llm_client.model_name,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": None
        }
        
        temp_dir = None
        
        try:
            # 1. Get instance
            instance = self._get_instance(instance_id)
            result["repo"] = instance.get("repo", "")
            
            # 2. Clone repository
            logger.info("📦 Cloning repository...")
            temp_dir = Path(tempfile.mkdtemp(prefix="green_exp_"))
            repo_path = self.measurer.setup_repository(
                instance, temp_dir, instance['base_commit']
            )
            
            # 3. Build context
            logger.info(f"📂 Building {self.strategy} context...")
            
            if self.strategy == "oracle":
                code_files, candidates = self._build_oracle_context(repo_path, instance)
                repo_map = None
            else:
                code_files, candidates, repo_map = self._build_realistic_context(repo_path, instance)
            
            if not code_files:
                raise ValueError("No code files found for context")
            
            result["context_stats"] = {
                "num_files": len(code_files),
                "total_chars": sum(len(c) for c in code_files.values())
            }
            
            # 4. Build prompt context
            context = self._build_prompt_context(instance, code_files, candidates, repo_map)
            
            # 5. Execute based on prompt type
            logger.info(f"🤖 Executing {self.prompt_type} strategy...")
            
            if self.prompt_type in ("zero_shot", "cot"):
                exec_result = self._run_single_turn(context, candidates, repo_path)
            elif self.prompt_type == "self_collab":
                exec_result = self._run_self_collaboration(context, candidates, repo_path)
            elif self.prompt_type == "ldb":
                exec_result = self._run_ldb(context, candidates, repo_path)
            else:
                raise ValueError(f"Unknown prompt type: {self.prompt_type}")
            
            # Merge execution result
            result.update(exec_result)
            
            if result.get("patch_result", {}).get("success"):
                logger.info(f"✅ Patch applied successfully!")
            else:
                logger.warning(f"⚠️ Patch failed")
            
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            result["error"] = str(e)
            result["status"] = "error"
        
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Save result
        self._save_result(instance_id, result)
        
        return result
    
    def _save_result(self, instance_id: str, result: Dict):
        output_file = self.output_dir / f"{instance_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        status_emoji = "✅" if result.get("status") == "success" else "❌"
        logger.info(f"{status_emoji} Saved: {output_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run green code optimization experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero-shot oracle
  python run_experiment.py -i mwaskom__seaborn-2389 -s oracle -p zero_shot
  
  # Chain-of-thought realistic
  python run_experiment.py -i mwaskom__seaborn-2389 -s realistic -p cot
  
  # Self-collaboration oracle
  python run_experiment.py -i mwaskom__seaborn-2389 -s oracle -p self_collab
  
  # LDB (iterative) realistic
  python run_experiment.py -i mwaskom__seaborn-2389 -s realistic -p ldb
        """
    )
    
    parser.add_argument('--instance', '-i', type=str, required=True,
                        help='Instance ID')
    parser.add_argument('--strategy', '-s', type=str, choices=['oracle', 'realistic'],
                        default='oracle', help='Context strategy')
    parser.add_argument('--prompt-type', '-p', type=str,
                        choices=['zero_shot', 'cot', 'self_collab', 'ldb'],
                        default='zero_shot', help='Prompting strategy')
    parser.add_argument('--dataset', '-d', type=str, default=str(DEFAULT_DATASET),
                        help='Dataset path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        dataset_path=Path(args.dataset),
        strategy=args.strategy,
        prompt_type=args.prompt_type,
        output_dir=Path(args.output) if args.output else None
    )
    
    result = runner.run(args.instance)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULT: {result['status'].upper()}")
    print(f"{'='*60}")
    
    if result["status"] == "success":
        pr = result.get("patch_result", {})
        print(f"  Patch: {pr.get('changes_applied', 0)}/{pr.get('total_blocks', 0)} blocks")
        print(f"  Files: {pr.get('modified_files', [])}")
    
    if result.get("num_iterations"):
        print(f"  Iterations: {result['num_iterations']}")
    
    if result.get("error"):
        print(f"  Error: {result['error']}")
    
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())