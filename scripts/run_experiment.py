"""
Unified Experiment Runner for Green Code Refactoring.

Runs a single experiment: generates an LLM patch for one instance.
Supports both ORACLE and REALISTIC strategies.

Version: 2.0 - Improved context management to prevent overflow

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
from src.prompt_templates import (
    ZeroShotTemplate, 
    ChainOfThoughtTemplate,
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

# =============================================================================
# CONTEXT LIMITS - CRITICAL FOR AVOIDING OVERFLOW
# =============================================================================
# For a 32k context model, we need to leave room for:
# - System prompt (~500 tokens)
# - Template instructions (~1000 tokens)  
# - LLM response (~4000 tokens)
# So max input context should be ~26k tokens = ~78k chars
# Being conservative: 20k tokens = 60k chars for ORACLE, less for REALISTIC

CONTEXT_LIMITS = {
    "oracle": {
        "max_total_chars": 50000,       # ~16k tokens - safe for oracle
        "max_file_chars": 25000,        # Single file limit
        "max_files": 5,                 # Max files to include
    },
    "realistic": {
        "max_total_chars": 35000,       # ~12k tokens - more conservative for realistic
        "max_file_chars": 8000,         # Smaller per-file limit (more files expected)
        "max_test_chars": 5000,         # Limit for test files
        "max_retrieved_files": 5,       # Reduce from 10 to 5
        "max_repo_map_lines": 100,      # Limit repo map
    }
}


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
        prompt_type: str = "zero_shot",
        output_dir: Optional[Path] = None
    ):
        """
        Initialize runner.
        
        Args:
            dataset_path: Path to SWE-Perf dataset JSON
            strategy: "oracle" or "realistic"
            prompt_type: "zero_shot" or "cot" (chain-of-thought)
            output_dir: Directory for results (default: results/{prompt_type}_{strategy})
        """
        self.dataset_path = Path(dataset_path)
        self.strategy = strategy.lower()
        self.prompt_type = prompt_type.lower()
        
        if self.strategy not in ("oracle", "realistic"):
            raise ValueError(f"Strategy must be 'oracle' or 'realistic', got: {strategy}")
        
        if self.prompt_type not in ("zero_shot", "cot"):
            raise ValueError(f"Prompt type must be 'zero_shot' or 'cot', got: {prompt_type}")
        
        # Get context limits for this strategy
        self.limits = CONTEXT_LIMITS[self.strategy]
        
        # Set output directory based on prompt type and strategy
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            prefix = "zs" if self.prompt_type == "zero_shot" else "cot"
            self.output_dir = RESULTS_DIR / f"{prefix}_{self.strategy}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Initialize components
        self.llm_client = VLLMClient()
        
        # Select template based on prompt type
        if self.prompt_type == "zero_shot":
            self.template = ZeroShotTemplate()
        else:
            self.template = ChainOfThoughtTemplate()
        
        self.measurer = SWEPerfMeasurer(str(dataset_path), country_code="ESP")
        
        logger.info(f"Initialized ExperimentRunner")
        logger.info(f"  Prompt Type: {self.prompt_type.upper()}")
        logger.info(f"  Strategy: {self.strategy.upper()}")
        logger.info(f"  Dataset: {self.dataset_path.name} ({len(self.dataset)} instances)")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Model: {self.llm_client.model_name}")
        logger.info(f"  Context Limit: {self.limits['max_total_chars']:,} chars")
    
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
    
    def _truncate_file(self, content: str, max_chars: int, filename: str = "") -> str:
        """
        Intelligently truncate a file to max_chars.
        Tries to keep the most relevant parts (imports, class/function definitions).
        """
        if len(content) <= max_chars:
            return content
        
        lines = content.splitlines()
        
        # Strategy: Keep first 40% and last 20%, truncate middle
        total_chars = max_chars - 100  # Reserve for truncation message
        first_portion = int(total_chars * 0.6)
        last_portion = int(total_chars * 0.4)
        
        # Build first part
        first_lines = []
        char_count = 0
        for line in lines:
            if char_count + len(line) + 1 > first_portion:
                break
            first_lines.append(line)
            char_count += len(line) + 1
        
        # Build last part
        last_lines = []
        char_count = 0
        for line in reversed(lines):
            if char_count + len(line) + 1 > last_portion:
                break
            last_lines.insert(0, line)
            char_count += len(line) + 1
        
        truncation_msg = f"\n# ... [{len(lines) - len(first_lines) - len(last_lines)} lines truncated] ...\n"
        
        return "\n".join(first_lines) + truncation_msg + "\n".join(last_lines)
    
    def _build_oracle_context(
        self, 
        repo_path: Path, 
        instance: Dict
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Build context for ORACLE mode.
        Extracts exact target files from the gold patch.
        """
        limits = self.limits
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
            patch_funcs = instance.get("patch_functions", "{}")
            if isinstance(patch_funcs, str):
                patch_funcs = json.loads(patch_funcs)
            target_files = list(patch_funcs.keys())
        
        # Load files with size limits
        code_files = {}
        current_chars = 0
        max_total = limits["max_total_chars"]
        max_file = limits["max_file_chars"]
        max_files = limits["max_files"]
        
        for fname in target_files[:max_files]:
            if current_chars >= max_total:
                logger.warning(f"Context limit reached, skipping remaining files")
                break
            
            fpath = repo_path / fname
            if not fpath.exists():
                continue
            
            try:
                content = fpath.read_text(errors='ignore')
                
                # Apply per-file limit
                if len(content) > max_file:
                    content = self._truncate_file(content, max_file, fname)
                
                # Check total limit
                remaining = max_total - current_chars
                if len(content) > remaining:
                    if remaining > 2000:
                        content = self._truncate_file(content, remaining, fname)
                    else:
                        logger.warning(f"Skipping {fname} - not enough space")
                        continue
                
                code_files[fname] = content
                current_chars += len(content)
                logger.info(f"   Added {fname}: {len(content):,} chars")
                
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
        
        IMPROVED: Better limits to prevent context overflow.
        """
        limits = self.limits
        max_total = limits["max_total_chars"]
        max_file = limits["max_file_chars"]
        max_test = limits["max_test_chars"]
        max_retrieved = limits["max_retrieved_files"]
        max_repo_lines = limits["max_repo_map_lines"]
        
        # Get test files
        test_list = instance.get('efficiency_test', [])
        test_files = list(set(t.split("::")[0] for t in test_list))
        
        # Generate repo map (limited)
        repo_map = self._generate_repo_map(repo_path, max_lines=max_repo_lines)
        
        # Track context size
        code_files = {}
        current_chars = 0
        
        # =====================================================================
        # STEP 1: Add test files (with strict limits)
        # =====================================================================
        test_contents = {}
        
        for tf in test_files[:3]:  # Max 3 test files
            fpath = repo_path / tf
            if not fpath.exists():
                continue
            
            try:
                content = fpath.read_text(errors='ignore')
                
                # Strict limit on test files - we only need them for context
                if len(content) > max_test:
                    # For tests, keep only the relevant test functions
                    content = self._extract_relevant_tests(content, test_list, max_test)
                
                test_contents[tf] = content
                
            except Exception as e:
                logger.warning(f"Could not read test file {tf}: {e}")
        
        # Add test files to context
        for tf, content in test_contents.items():
            if current_chars + len(content) > max_total * 0.4:  # Tests max 40% of context
                logger.warning(f"Test context limit reached, truncating {tf}")
                remaining = int(max_total * 0.4) - current_chars
                if remaining > 1000:
                    content = content[:remaining] + "\n# ... [TRUNCATED]"
                else:
                    continue
            
            code_files[tf] = content
            current_chars += len(content)
            logger.info(f"   Added test {tf}: {len(content):,} chars")
        
        # =====================================================================
        # STEP 2: Retrieve source files using BM25-style scoring
        # =====================================================================
        query_tokens = set()
        for content in test_contents.values():
            query_tokens.update(self._tokenize(content))
        
        # Remove stopwords
        stopwords = {
            'def', 'class', 'self', 'import', 'from', 'in', 'if', 'else', 
            'return', 'assert', 'test', 'none', 'true', 'false', 'and', 
            'or', 'pytest', 'for', 'while', 'with', 'as', 'try', 'except',
            'is', 'not', 'the', 'to', 'of', 'a', 'an'
        }
        query_tokens -= stopwords
        
        # Score all Python files
        scores = {}
        for root, _, files in os.walk(repo_path):
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                
                rel_path = os.path.relpath(os.path.join(root, fname), repo_path)
                
                # Skip test files and __init__.py
                if rel_path in test_files or fname == '__init__.py':
                    continue
                
                # Skip test directories
                if '/tests/' in rel_path or '/test_' in rel_path:
                    continue
                
                try:
                    content = (repo_path / rel_path).read_text(errors='ignore')
                    file_tokens = self._tokenize(content)
                    
                    # BM25-style scoring: penalize very common tokens
                    score = len(query_tokens & file_tokens)
                    
                    # Boost files that match test file names
                    for tf in test_files:
                        test_name = Path(tf).stem.replace('test_', '')
                        if test_name in rel_path:
                            score *= 2
                    
                    if score > 0:
                        scores[rel_path] = (score, len(content))
                except:
                    pass
        
        # =====================================================================
        # STEP 3: Add top retrieved files (with strict limits)
        # =====================================================================
        # Sort by score, then by file size (prefer smaller files)
        top_files = sorted(
            scores.items(), 
            key=lambda x: (x[1][0], -x[1][1]),  # High score, low size
            reverse=True
        )
        
        files_added = 0
        for fname, (score, _) in top_files:
            if files_added >= max_retrieved:
                break
            
            if current_chars >= max_total:
                logger.warning(f"Context limit reached at {current_chars:,} chars")
                break
            
            try:
                content = (repo_path / fname).read_text(errors='ignore')
                
                # Apply per-file limit
                if len(content) > max_file:
                    content = self._truncate_file(content, max_file, fname)
                
                # Check remaining space
                remaining = max_total - current_chars
                if len(content) > remaining:
                    if remaining > 2000:
                        content = self._truncate_file(content, remaining, fname)
                    else:
                        logger.warning(f"Skipping {fname} - not enough space ({remaining} remaining)")
                        continue
                
                code_files[fname] = content
                current_chars += len(content)
                files_added += 1
                logger.info(f"   Added retrieved {fname}: {len(content):,} chars (score: {score})")
                
            except Exception as e:
                logger.warning(f"Could not read {fname}: {e}")
        
        logger.info(f"📂 Realistic context: {len(code_files)} files, {current_chars:,} chars")
        logger.info(f"   Tests: {len(test_contents)}, Retrieved: {files_added}")
        
        return code_files, list(code_files.keys()), repo_map
    
    def _extract_relevant_tests(
        self, 
        content: str, 
        test_list: List[str], 
        max_chars: int
    ) -> str:
        """
        Extract only relevant test functions from a test file.
        """
        # Get test function names from test_list
        test_names = set()
        for t in test_list:
            if "::" in t:
                parts = t.split("::")
                for p in parts[1:]:
                    # Handle parametrized tests like test_foo[param1-param2]
                    name = p.split("[")[0]
                    test_names.add(name)
        
        if not test_names:
            # No specific tests, truncate normally
            return self._truncate_file(content, max_chars)
        
        lines = content.splitlines()
        result_lines = []
        in_relevant_function = False
        current_indent = 0
        
        # Always keep imports and class definitions
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Keep imports
            if stripped.startswith(('import ', 'from ')):
                result_lines.append(line)
                continue
            
            # Keep class definitions
            if stripped.startswith('class '):
                result_lines.append(line)
                continue
            
            # Check if this is a relevant test function
            if stripped.startswith('def '):
                func_name = stripped[4:].split('(')[0]
                if func_name in test_names or any(tn in func_name for tn in test_names):
                    in_relevant_function = True
                    current_indent = len(line) - len(stripped)
                    result_lines.append(line)
                    continue
                else:
                    in_relevant_function = False
            
            # If inside relevant function, keep the line
            if in_relevant_function:
                line_indent = len(line) - len(line.lstrip()) if line.strip() else current_indent + 1
                if line.strip() and line_indent <= current_indent and not stripped.startswith(('@', '#')):
                    in_relevant_function = False
                else:
                    result_lines.append(line)
        
        result = "\n".join(result_lines)
        
        # If still too long, truncate
        if len(result) > max_chars:
            result = result[:max_chars] + "\n# ... [TRUNCATED]"
        
        return result
    
    def _generate_repo_map(self, repo_path: Path, max_lines: int = 100) -> str:
        """Generate repository structure map (limited)."""
        tree = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith(('.', '__'))]
            
            level = root.replace(str(repo_path), '').count(os.sep)
            
            # Limit depth
            if level > 3:
                continue
            
            indent = '    ' * level
            tree.append(f"{indent}{os.path.basename(root)}/")
            
            subindent = '    ' * (level + 1)
            py_files = [f for f in files if f.endswith('.py')][:10]  # Max 10 files per dir
            for f in py_files:
                tree.append(f"{subindent}{f}")
            
            if len(tree) > max_lines:
                break
        
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
        DEPRECATED: Use _build_realistic_context instead.
        """
        return self._build_realistic_context(repo_path, {"efficiency_test": test_files})[:2]
    
    def _tokenize(self, text: str) -> set:
        """Extract identifier tokens from text."""
        return set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower()))
    
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
            "llm_response": "",
            "patch_result": None,
            "error": None,
            "context_stats": {}
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
            
            # Store context stats
            total_chars = sum(len(c) for c in code_files.values())
            result["context_stats"] = {
                "num_files": len(code_files),
                "total_chars": total_chars,
                "files": {f: len(c) for f, c in code_files.items()}
            }
            
            # 4. Generate prompt
            logger.info("📝 Generating prompt...")
            context = self._build_prompt_context(instance, code_files, candidates, repo_map)
            user_prompt = self.template.generate_prompt(context)
            system_prompt = self._get_system_prompt()
            
            # Log prompt size
            total_prompt_size = len(system_prompt) + len(user_prompt)
            estimated_tokens = total_prompt_size // 3  # ~3 chars per token
            logger.info(f"   Prompt size: {total_prompt_size:,} chars (~{estimated_tokens:,} tokens)")
            
            # Check for potential overflow
            if estimated_tokens > 28000:
                logger.warning(f"⚠️ Prompt may be too large! ({estimated_tokens:,} tokens)")
            
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
            
            # 6. Extract patch content (for CoT, strip reasoning section)
            if self.prompt_type == "cot":
                logger.info("🧠 Extracting patch from CoT response...")
                patch_content = extract_patch_from_cot(llm_output)
                logger.info(f"   Extracted patch: {len(patch_content):,} chars")
            else:
                patch_content = llm_output
            
            # 7. Try to apply patch (validation)
            logger.info("🔧 Validating patch...")
            patch_engine = PatchEngine(repo_path)
            patch_result = patch_engine.apply_patch(patch_content, candidates)
            
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
  # Run zero-shot oracle experiment
  python run_experiment.py --instance mwaskom__seaborn-2389 --strategy oracle
  
  # Run zero-shot realistic experiment
  python run_experiment.py --instance mwaskom__seaborn-2389 --strategy realistic
  
  # Run chain-of-thought oracle experiment
  python run_experiment.py --instance mwaskom__seaborn-2389 --strategy oracle --prompt-type cot
  
  # Run CoT realistic experiment  
  python run_experiment.py --instance mwaskom__seaborn-2389 --strategy realistic --prompt-type cot

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
        '--prompt-type', '-p',
        type=str,
        choices=['zero_shot', 'cot'],
        default='zero_shot',
        help='Prompt type: zero_shot or cot (chain-of-thought) (default: zero_shot)'
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
        help='Output directory (default: results/{prompt_type}_{strategy})'
    )
    
    args = parser.parse_args()
    
    # Run experiment
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
        print(f"  Patch: {pr.get('changes_applied', 0)}/{pr.get('total_blocks', 0)} blocks applied")
        print(f"  Files: {pr.get('modified_files', [])}")
    elif result.get("error"):
        print(f"  Error: {result['error']}")
    
    # Print context stats
    ctx = result.get("context_stats", {})
    if ctx:
        print(f"\n  Context: {ctx.get('num_files', 0)} files, {ctx.get('total_chars', 0):,} chars")
    
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())