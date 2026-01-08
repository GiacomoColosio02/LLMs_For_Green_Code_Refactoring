"""
Unified Experiment Runner for Green Code Refactoring.

Runs a single experiment: generates an LLM patch for one instance.
Supports both ORACLE and REALISTIC strategies.

Usage:
    # Oracle mode (knows exact files to modify)
    python run_experiment.py --instance mwaskom__seaborn-2389 --strategy oracle
    
    # Realistic mode (must find files via retrieval)
    python run_experiment.py --instance mwaskom__seaborn-2389 --strategy realistic
    
    # With custom dataset
    python run_experiment.py --instance X --strategy oracle --dataset path/to/data.json
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
from src.prompt_templates import ZeroShotTemplate, PromptContext, ProblemStatementType
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
MAX_CONTEXT_CHARS = 60000  # ~20k tokens, safe for 32k context window


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Runs a single green code optimization experiment.
    
    Flow:
    1. Load instance from dataset
    2. Clone repository at base_commit
    3. Build context (oracle: from patch, realistic: via retrieval)
    4. Generate prompt and call LLM
    5. Apply patch to verify it works
    6. Save results
    """
    
    def __init__(
        self,
        dataset_path: Path,
        strategy: str = "oracle",
        output_dir: Optional[Path] = None
    ):
        """
        Initialize runner.
        
        Args:
            dataset_path: Path to SWE-Perf dataset JSON
            strategy: "oracle" or "realistic"
            output_dir: Directory for results (default: results/{strategy})
        """
        self.dataset_path = Path(dataset_path)
        self.strategy = strategy.lower()
        
        if self.strategy not in ("oracle", "realistic"):
            raise ValueError(f"Strategy must be 'oracle' or 'realistic', got: {strategy}")
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = RESULTS_DIR / f"zs_{self.strategy}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Initialize components
        self.llm_client = VLLMClient()
        self.template = ZeroShotTemplate()
        self.measurer = SWEPerfMeasurer(str(dataset_path), country_code="ESP")
        
        logger.info(f"Initialized ExperimentRunner")
        logger.info(f"  Strategy: {self.strategy.upper()}")
        logger.info(f"  Dataset: {self.dataset_path.name} ({len(self.dataset)} instances)")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Model: {self.llm_client.model_name}")
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        return data.get("instances", [])
    
    def _get_instance(self, instance_id: str) -> Dict:
        """Get instance by ID."""
        for item in self.dataset:
            if item.get("instance_id") == instance_id:
                return item
        raise ValueError(f"Instance '{instance_id}' not found in dataset")
    
    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================
    
    def _build_oracle_context(
        self, 
        repo_path: Path, 
        instance: Dict
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Build context for ORACLE mode.
        Extracts exact target files from the gold patch.
        
        Args:
            repo_path: Path to cloned repository
            instance: Instance data
            
        Returns:
            Tuple of (code_files dict, candidate_files list)
        """
        patch_content = instance.get("patch", "")
        
        # Extract target files from patch
        target_files = []
        for line in patch_content.splitlines():
            if line.startswith("--- a/"):
                fname = line[6:].strip()
                if (repo_path / fname).exists():
                    target_files.append(fname)
        
        target_files = list(set(target_files))
        
        if not target_files:
            logger.warning("No target files found in patch, using patch_functions")
            # Fallback: try patch_functions field
            patch_funcs = instance.get("patch_functions", "{}")
            if isinstance(patch_funcs, str):
                patch_funcs = json.loads(patch_funcs)
            target_files = list(patch_funcs.keys())
        
        # Load files with size limits
        code_files = {}
        current_chars = 0
        
        for fname in target_files:
            if current_chars >= MAX_CONTEXT_CHARS:
                logger.warning(f"Context limit reached, skipping {fname}")
                break
            
            fpath = repo_path / fname
            if not fpath.exists():
                continue
            
            try:
                content = fpath.read_text(errors='ignore')
                
                # Truncate large files
                if len(content) > 30000 and len(target_files) > 1:
                    content = content[:30000] + "\n# ... [TRUNCATED FOR CONTEXT LIMIT] ..."
                
                # Check total limit
                if current_chars + len(content) > MAX_CONTEXT_CHARS:
                    remaining = MAX_CONTEXT_CHARS - current_chars
                    if remaining > 1000:
                        content = content[:remaining] + "\n# ... [TRUNCATED] ..."
                    else:
                        continue
                
                code_files[fname] = content
                current_chars += len(content)
                
            except Exception as e:
                logger.warning(f"Could not read {fname}: {e}")
        
        logger.info(f"📂 Oracle context: {len(code_files)} files, {current_chars:,} chars")
        return code_files, list(code_files.keys())
    
    def _build_realistic_context(
        self,
        repo_path: Path,
        instance: Dict
    ) -> Tuple[Dict[str, str], List[str], str]:
        """
        Build context for REALISTIC mode.
        Uses simulated retrieval based on test file tokens.
        
        Args:
            repo_path: Path to cloned repository
            instance: Instance data
            
        Returns:
            Tuple of (code_files dict, candidate_files list, repo_map string)
        """
        # Get test files
        test_list = instance.get('efficiency_test', [])
        test_files = list(set(t.split("::")[0] for t in test_list))
        
        # Generate repo map
        repo_map = self._generate_repo_map(repo_path)
        
        # Simulated retrieval
        code_files, candidates = self._simulated_retrieval(repo_path, test_files)
        
        logger.info(f"📂 Realistic context: {len(code_files)} files, {len(candidates)} candidates")
        return code_files, candidates, repo_map
    
    def _generate_repo_map(self, repo_path: Path, max_lines: int = 200) -> str:
        """Generate repository structure map."""
        tree = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
            
            level = root.replace(str(repo_path), '').count(os.sep)
            indent = '    ' * level
            tree.append(f"{indent}{os.path.basename(root)}/")
            
            subindent = '    ' * (level + 1)
            for f in files:
                if f.endswith('.py'):
                    tree.append(f"{subindent}{f}")
        
        if len(tree) > max_lines:
            return "\n".join(tree[:max_lines]) + "\n... (truncated)"
        return "\n".join(tree)
    
    def _simulated_retrieval(
        self,
        repo_path: Path,
        test_files: List[str]
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Simulate code retrieval based on test file tokens.
        
        Args:
            repo_path: Repository path
            test_files: List of test file paths
            
        Returns:
            Tuple of (code_files dict, candidate_files list)
        """
        # Extract tokens from test files
        query_tokens = set()
        test_contents = {}
        
        for tf in test_files:
            fpath = repo_path / tf
            if fpath.exists():
                content = fpath.read_text(errors='ignore')
                test_contents[tf] = content
                query_tokens.update(self._tokenize(content))
        
        # Remove stopwords
        stopwords = {
            'def', 'class', 'self', 'import', 'from', 'in', 'if', 'else', 
            'return', 'assert', 'test', 'none', 'true', 'false', 'and', 
            'or', 'pytest', 'for', 'while', 'with', 'as', 'try', 'except'
        }
        query_tokens -= stopwords
        
        # Score all Python files
        scores = {}
        for root, _, files in os.walk(repo_path):
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                
                rel_path = os.path.relpath(os.path.join(root, fname), repo_path)
                
                # Skip test files
                if rel_path in test_files:
                    continue
                
                try:
                    content = (repo_path / rel_path).read_text(errors='ignore')
                    file_tokens = self._tokenize(content)
                    score = len(query_tokens & file_tokens)
                    if score > 0:
                        scores[rel_path] = score
                except:
                    pass
        
        # Build context with limits
        code_files = {}
        current_chars = 0
        
        # First add test files (limited)
        TEST_LIMIT = 20000
        for tf, content in test_contents.items():
            if len(content) > TEST_LIMIT:
                code_files[tf] = content[:TEST_LIMIT] + "\n# ... [TRUNCATED]"
                current_chars += TEST_LIMIT
            else:
                code_files[tf] = content
                current_chars += len(content)
        
        # Then add top retrieved files
        top_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for fname, _ in top_files[:10]:  # Max 10 retrieved files
            if current_chars >= MAX_CONTEXT_CHARS:
                break
            
            try:
                content = (repo_path / fname).read_text(errors='ignore')
                
                if current_chars + len(content) < MAX_CONTEXT_CHARS:
                    code_files[fname] = content
                    current_chars += len(content)
                else:
                    remaining = MAX_CONTEXT_CHARS - current_chars
                    if remaining > 1000:
                        code_files[fname] = content[:remaining] + "\n# ... [TRUNCATED]"
                        current_chars += remaining
            except:
                pass
        
        return code_files, list(code_files.keys())
    
    def _tokenize(self, text: str) -> set:
        """Extract identifier tokens from text."""
        return set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text))
    
    # =========================================================================
    # PROMPT GENERATION
    # =========================================================================
    
    def _build_prompt_context(
        self,
        instance: Dict,
        code_files: Dict[str, str],
        candidates: List[str],
        repo_map: Optional[str] = None
    ) -> PromptContext:
        """Build PromptContext for template."""
        
        # Get problem description based on strategy
        if self.strategy == "oracle":
            problem_desc = instance.get(
                'problem_statement_oracle',
                f"Optimize energy efficiency for the provided code."
            )
            stmt_type = ProblemStatementType.ORACLE
        else:
            problem_desc = instance.get(
                'problem_statement_realistic',
                f"Optimize energy efficiency based on the failing tests."
            )
            stmt_type = ProblemStatementType.REALISTIC
        
        # Build test command
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
        """Get system prompt for LLM."""
        return (
            "You are an expert Green Software Engineer specializing in energy-efficient code.\n"
            "Your goal is to optimize the provided code for energy efficiency and execution speed.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output ONLY the code patch using the SEARCH/REPLACE format shown below.\n"
            "2. Use SMALL, UNIQUE SEARCH blocks - just enough to locate the code.\n"
            "3. Do NOT add new external dependencies.\n"
            "4. Do NOT modify test files.\n"
            "5. Provide the patch immediately, without explanations.\n\n"
            "Format:\n"
            "### path/to/file.py\n"
            "<<<<<<< SEARCH\n"
            "original code\n"
            "=======\n"
            "optimized code\n"
            ">>>>>>> REPLACE"
        )
    
    # =========================================================================
    # MAIN RUN METHOD
    # =========================================================================
    
    def run(self, instance_id: str) -> Dict:
        """
        Run experiment for a single instance.
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Result dictionary with status, response, patch_result, etc.
        """
        logger.info(f"{'='*60}")
        logger.info(f"🚀 STARTING EXPERIMENT: {instance_id}")
        logger.info(f"   Strategy: {self.strategy.upper()}")
        logger.info(f"{'='*60}")
        
        result = {
            "instance_id": instance_id,
            "strategy": self.strategy,
            "model": self.llm_client.model_name,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "llm_response": "",
            "patch_result": None,
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
            
            # 3. Build context based on strategy
            logger.info(f"📂 Building {self.strategy} context...")
            
            if self.strategy == "oracle":
                code_files, candidates = self._build_oracle_context(repo_path, instance)
                repo_map = None
            else:
                code_files, candidates, repo_map = self._build_realistic_context(repo_path, instance)
            
            if not code_files:
                raise ValueError("No code files found for context")
            
            # 4. Generate prompt
            logger.info("📝 Generating prompt...")
            context = self._build_prompt_context(instance, code_files, candidates, repo_map)
            user_prompt = self.template.generate_prompt(context)
            system_prompt = self._get_system_prompt()
            
            # Log prompt size
            total_prompt_size = len(system_prompt) + len(user_prompt)
            logger.info(f"   Prompt size: {total_prompt_size:,} chars (~{total_prompt_size//4:,} tokens)")
            
            # 5. Call LLM
            logger.info("🤖 Querying LLM...")
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=4096
            )
            
            llm_output = response.content
            result["llm_response"] = llm_output
            result["llm_tokens"] = response.total_tokens
            result["llm_latency"] = response.latency_seconds
            
            logger.info(f"   Response: {len(llm_output):,} chars, {response.total_tokens} tokens, {response.latency_seconds:.1f}s")
            
            # 6. Try to apply patch (validation)
            logger.info("🔧 Validating patch...")
            patch_engine = PatchEngine(repo_path)
            patch_result = patch_engine.apply_patch(llm_output, candidates)
            
            result["patch_result"] = {
                "success": patch_result.success,
                "changes_applied": patch_result.changes_applied,
                "total_blocks": patch_result.total_blocks,
                "method": patch_result.method_used,
                "modified_files": patch_result.modified_files
            }
            
            if patch_result.success:
                logger.info(f"   ✅ Patch applied: {patch_result.changes_applied}/{patch_result.total_blocks} blocks")
                result["status"] = "success"
            else:
                logger.warning(f"   ⚠️ Patch failed: {patch_result.error_message}")
                result["status"] = "patch_failed"
            
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            result["error"] = str(e)
            result["status"] = "error"
        
        finally:
            # Cleanup
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        # 7. Save result
        self._save_result(instance_id, result)
        
        return result
    
    def _save_result(self, instance_id: str, result: Dict):
        """Save result to JSON file."""
        output_file = self.output_dir / f"{instance_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        status_emoji = "✅" if result["status"] == "success" else "❌"
        logger.info(f"{status_emoji} Saved: {output_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run green code optimization experiment for a single instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run oracle experiment
  python run_experiment.py --instance mwaskom__seaborn-2389 --strategy oracle
  
  # Run realistic experiment
  python run_experiment.py --instance mwaskom__seaborn-2389 --strategy realistic
  
  # With custom dataset
  python run_experiment.py --instance X --strategy oracle --dataset path/to/data.json
        """
    )
    
    parser.add_argument(
        '--instance', '-i',
        type=str,
        required=True,
        help='Instance ID (e.g., mwaskom__seaborn-2389)'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['oracle', 'realistic'],
        default='oracle',
        help='Experiment strategy (default: oracle)'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=str(DEFAULT_DATASET),
        help='Path to dataset JSON'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (default: results/zs_{strategy})'
    )
    
    args = parser.parse_args()
    
    # Run experiment
    runner = ExperimentRunner(
        dataset_path=Path(args.dataset),
        strategy=args.strategy,
        output_dir=Path(args.output) if args.output else None
    )
    
    result = runner.run(args.instance)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULT: {result['status'].upper()}")
    print(f"{'='*60}")
    
    if result["status"] == "success":
        pr = result.get("patch_result", {})
        print(f"  Patch: {pr.get('changes_applied', 0)}/{pr.get('total_blocks', 0)} blocks applied")
        print(f"  Files: {pr.get('modified_files', [])}")
    elif result.get("error"):
        print(f"  Error: {result['error']}")
    
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())