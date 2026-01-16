"""
LLM Patch Measurement Engine for Green Code Refactoring.

FIXED VERSION: Uses the SAME measurement method as measure_instance.py
which has been proven to work correctly.

Key fix: Uses collector.measure_test_execution() with shell command
instead of the broken MetricsCollector constructor approach.

Supports all prompt types:
- zero_shot: Zero-Shot prompting
- cot: Chain-of-Thought prompting
- self_collab: Self-Collaboration (multi-turn)
- ldb: LDB iterative refinement

Output format:
    data/processed/green/[MODEL]_[PROMPT_TYPE]_[STRATEGY]_k[REPS].json
"""
import sys
import os
import json
import logging
import argparse
import tempfile
import shutil
import time
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORTS ---
from scripts.measure_instance import SWEPerfMeasurer
from src.measurement.collector import MetricsCollector
from src.patch_engine import PatchEngine
from src.utils.config import load_config

# --- LOGGING ---
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "measure_llm_patches.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MeasureLLMPatches")

# --- CONSTANTS ---
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "swe_perf_reduced.json"
ORIGINAL_DATASET = PROJECT_ROOT / "data" / "original" / "swe_perf_original_20251124.json"
GREEN_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "green"
BASE_RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_MODEL_NAME = "Qwen2.5-Coder-7B"

PROMPT_TYPE_TO_DIR_PREFIX = {
    "zero_shot": "zs",
    "cot": "cot",
    "self_collab": "sc",
    "ldb": "ldb"
}

PROMPT_TYPE_TO_CLEAN_NAME = {
    "zero_shot": "ZeroShot",
    "cot": "CoT",
    "self_collab": "SelfCollab",
    "ldb": "LDB"
}

GREEN_METRICS = [
    'cpu_energy_joules', 'gpu_energy_joules', 'total_energy_joules',
    'power_watts', 'carbon_grams', 'energy_efficiency'
]
EFFICIENCY_METRICS = [
    'duration_seconds', 'cpu_usage_mean_percent', 'cpu_usage_peak_percent',
    'ram_usage_mean_mb', 'ram_usage_peak_mb',
    'gpu_temperature_mean_celsius', 'gpu_temperature_peak_celsius'
]
AGGREGATIONS = ['mean', 'std', 'min', 'max']


# =============================================================================
# SYNTAX FIXER
# =============================================================================

class SyntaxFixer:
    """Auto-fixes common syntax errors in Python files."""
    
    def check_syntax(self, file_path: Path) -> Tuple[bool, str]:
        result = subprocess.run(
            ['python', '-m', 'py_compile', str(file_path)],
            capture_output=True, text=True
        )
        return result.returncode == 0, result.stderr or result.stdout
    
    def fix_file(self, file_path: Path) -> bool:
        is_valid, _ = self.check_syntax(file_path)
        if is_valid:
            return True
        
        # Try autopep8
        try:
            subprocess.run(
                ['autopep8', '--in-place', '--aggressive', '--aggressive', str(file_path)],
                capture_output=True, timeout=30
            )
            is_valid, _ = self.check_syntax(file_path)
            if is_valid:
                return True
        except:
            pass
        
        # Try manual fixes
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Normalize tabs to spaces
            lines = [line.replace('\t', '    ') for line in lines]
            
            with open(file_path, 'w') as f:
                f.writelines(lines)
            
            # Try autopep8 again
            subprocess.run(
                ['autopep8', '--in-place', '--aggressive', '--aggressive', str(file_path)],
                capture_output=True, timeout=30
            )
            is_valid, _ = self.check_syntax(file_path)
            return is_valid
        except:
            pass
        
        return False


def fix_syntax_after_patch(repo_path: Path, modified_files: List[str]) -> Tuple[bool, List[str]]:
    """Fix syntax errors in modified files."""
    fixer = SyntaxFixer()
    errors = []
    
    for rel_path in modified_files:
        file_path = repo_path / rel_path
        if not file_path.exists() or not str(file_path).endswith('.py'):
            continue
        
        is_valid, _ = fixer.check_syntax(file_path)
        if not is_valid:
            logger.info(f"    🔧 Fixing syntax in {rel_path}...")
            if not fixer.fix_file(file_path):
                errors.append(rel_path)
                logger.warning(f"    ❌ Could not fix {rel_path}")
            else:
                logger.info(f"    ✅ Fixed {rel_path}")
    
    return len(errors) == 0, errors


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_default_results_dir(prompt_type: str, strategy: str) -> Path:
    prefix = PROMPT_TYPE_TO_DIR_PREFIX.get(prompt_type, prompt_type)
    return BASE_RESULTS_DIR / DEFAULT_MODEL_NAME / f"{prefix}_{strategy}"


def sanitize_model_name(model_name: str) -> str:
    name = model_name.split("/")[-1]
    for suffix in ["-Instruct-AWQ", "-AWQ", "-Instruct", "-instruct", "-awq"]:
        name = name.replace(suffix, "")
    return re.sub(r'[^\w\-.]', '_', name)


def get_output_filename(model_name: str, prompt_type: str, strategy: str, repetitions: int) -> str:
    model_clean = sanitize_model_name(model_name)
    prompt_clean = PROMPT_TYPE_TO_CLEAN_NAME.get(prompt_type, prompt_type)
    return f"{model_clean}_{prompt_clean}_{strategy.capitalize()}_k{repetitions}.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


# =============================================================================
# MAIN MEASURER CLASS
# =============================================================================

class LLMPatchMeasurer:
    """
    Measures LLM-generated patches using the SAME method as measure_instance.py.
    
    Key difference from broken version:
    - Uses collector.measure_test_execution(test_command, repetitions)
    - NOT the MetricsCollector constructor with repo_path/python_path params
    """
    
    def __init__(self, original_dataset_path: Path, reduced_dataset_path: Path, country_code: str = "ESP"):
        self.original_dataset_path = Path(original_dataset_path)
        self.reduced_dataset_path = Path(reduced_dataset_path)
        self.country_code = country_code
        self.config = load_config()
        
        # Load datasets
        self.original_data = self._load_dataset_as_dict(self.original_dataset_path)
        self.reduced_data = self._load_dataset_as_dict(self.reduced_dataset_path)
        
        # Initialize the measurer (for setup_repository and install_dependencies)
        self.measurer = SWEPerfMeasurer(str(original_dataset_path), country_code=country_code)
        
        logger.info(f"LLMPatchMeasurer initialized")
        logger.info(f"  Original: {len(self.original_data)} instances")
        logger.info(f"  Reduced: {len(self.reduced_data)} instances")
    
    def _load_dataset_as_dict(self, path: Path) -> Dict[str, Dict]:
        data = load_json(path)
        instances = data if isinstance(data, list) else data.get("instances", [])
        return {item["instance_id"]: item for item in instances}
    
    def _create_empty_dataset(self, model_name: str, prompt_type: str, strategy: str, repetitions: int) -> Dict:
        prompt_clean = PROMPT_TYPE_TO_CLEAN_NAME.get(prompt_type, prompt_type)
        return {
            'metadata': {
                'name': f'LLM Green Dataset - {model_name} - {prompt_clean} - {strategy}',
                'model': model_name,
                'prompt_type': prompt_clean,
                'strategy': strategy,
                'green_metrics': GREEN_METRICS,
                'efficiency_metrics': EFFICIENCY_METRICS,
                'aggregations': AGGREGATIONS,
                'repetitions': repetitions,
                'creation_date': datetime.now().isoformat(),
                'instance_count': 0,
                'measurement_method': 'measure_test_execution'  # Mark as using correct method
            },
            'instances': []
        }
    
    def _is_instance_in_dataset(self, dataset: Dict, instance_id: str) -> bool:
        return any(inst.get('instance_id') == instance_id for inst in dataset.get('instances', []))
    
    def measure_instance(self, instance_id: str, llm_response: str, repetitions: int = 3) -> Optional[Dict]:
        """
        Measure a single instance with LLM patch applied.
        
        Uses the SAME measurement approach as measure_instance.py:
        1. Create MetricsCollector with instance_id and country_code
        2. Call collector.measure_test_execution(test_command, repetitions)
        """
        logger.info(f"  📏 Measuring {instance_id}")
        
        instance = self.original_data.get(instance_id)
        if not instance:
            logger.error(f"    Instance not found")
            return None
        
        temp_dir = None
        conda_env = None
        
        try:
            # 1. Clone repository at BASE commit
            logger.info(f"    📦 Cloning repository...")
            temp_dir = Path(tempfile.mkdtemp(prefix="llm_meas_"))
            repo_path = self.measurer.setup_repository(instance, temp_dir, instance['base_commit'])
            
            # 2. Apply LLM patch
            logger.info(f"    🔧 Applying LLM patch...")
            patch_engine = PatchEngine(repo_path)
            
            candidates = list(set(re.findall(r'###\s*([\w/._-]+\.py)', llm_response)))
            if not candidates:
                patch_files = re.findall(r'diff --git a/(.*?) b/', instance.get('patch', ''))
                candidates = [f for f in patch_files if f.endswith('.py')]
            
            patch_result = patch_engine.apply_patch(llm_response, candidates)
            
            if not patch_result.success:
                logger.warning(f"    ❌ Failed to apply patch: {patch_result.error_message}")
                return None
            
            logger.info(f"    ✅ Patch applied to: {patch_result.modified_files}")
            
            # 3. Fix syntax errors if any
            logger.info(f"    🔍 Checking syntax...")
            syntax_ok, syntax_errors = fix_syntax_after_patch(repo_path, patch_result.modified_files)
            if not syntax_ok:
                logger.warning(f"    ❌ Syntax errors in: {syntax_errors}")
                return None
            
            # 4. Install dependencies
            logger.info(f"    📦 Installing dependencies...")
            python_path, conda_env = self.measurer.install_dependencies(
                repo_path=repo_path,
                repo=instance['repo'],
                version=instance['version'],
                commit=instance['base_commit']
            )
            
            if python_path is None:
                logger.warning(f"    ❌ Failed to install dependencies")
                return None
            
            # 5. Get tests
            reduced_instance = self.reduced_data.get(instance_id, {})
            tests = reduced_instance.get('efficiency_test', [])
            
            if not tests:
                logger.warning(f"    ❌ No tests found")
                return None
            
            logger.info(f"    🧪 Running {len(tests)} tests x {repetitions} repetitions...")
            
            # 6. FIXED: Use measure_test_execution like measure_instance.py does!
            collector = MetricsCollector(
                instance_id=instance_id,
                country_code=self.country_code
            )
            
            test_results = {}
            valid_tests = 0
            
            for test in tests:
                test_name = test.split("::")[-1] if "::" in test else test
                
                try:
                    # Build test command - SAME as measure_instance.py
                    test_command = f"cd {repo_path} && {python_path} -m pytest '{repo_path}/{test}' -v"
                    
                    # Call measure_test_execution - THIS IS THE KEY FIX!
                    metrics = collector.measure_test_execution(
                        test_command=test_command,
                        repetitions=repetitions
                    )
                    
                    if metrics:
                        test_results[test_name] = {"head": metrics}
                        valid_tests += 1
                        logger.debug(f"      ✅ {test_name}: {metrics.get('total_energy_joules_mean', 'N/A')}J")
                        
                except Exception as e:
                    logger.debug(f"      ❌ {test_name}: {e}")
            
            if valid_tests == 0:
                logger.warning(f"    ❌ No tests passed")
                return None
            
            # 7. Build result
            result = {
                'instance_id': instance_id,
                'repo': instance.get('repo', ''),
                'efficiency_test': tests,
                'green_metrics': test_results,
                '_green_metadata': {
                    'valid_tests': valid_tests,
                    'total_tests': len(tests),
                    'repetitions': repetitions,
                    'patch_files': patch_result.modified_files,
                    'measurement_date': datetime.now().isoformat()
                }
            }
            
            logger.info(f"    ✅ Measured {valid_tests}/{len(tests)} tests")
            return result
            
        except Exception as e:
            logger.error(f"    ❌ Error: {e}", exc_info=True)
            return None
            
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            if conda_env:
                self.measurer.cleanup_conda_env(conda_env)
    
    def measure_strategy(
        self,
        strategy: str,
        prompt_type: str,
        repetitions: int = 3,
        skip_completed: bool = True,
        limit: Optional[int] = None,
        results_dir: Optional[Path] = None,
        model_name: Optional[str] = None
    ) -> Dict:
        """Measure all instances for a strategy."""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📏 MEASURING: {prompt_type.upper()} - {strategy.upper()}")
        logger.info(f"{'='*70}")
        
        # Determine results directory
        if results_dir:
            exp_results_dir = Path(results_dir)
        else:
            exp_results_dir = get_default_results_dir(prompt_type, strategy)
        
        if not exp_results_dir.exists():
            logger.error(f"Results directory not found: {exp_results_dir}")
            return {"error": f"no_results_directory: {exp_results_dir}"}
        
        result_files = list(exp_results_dir.glob("*.json"))
        logger.info(f"Found {len(result_files)} experiment results")
        
        # Detect model
        if model_name:
            detected_model = model_name
        else:
            detected_model = DEFAULT_MODEL_NAME
            for rf in result_files:
                try:
                    data = load_json(rf)
                    if data.get("model"):
                        detected_model = data["model"]
                        break
                except:
                    pass
        
        logger.info(f"Model: {detected_model}")
        
        # Output path
        output_filename = get_output_filename(detected_model, prompt_type, strategy, repetitions)
        output_path = GREEN_OUTPUT_DIR / output_filename
        logger.info(f"Output: {output_path}")
        
        # Load or create dataset
        if output_path.exists():
            output_dataset = load_json(output_path)
            logger.info(f"Loaded existing: {len(output_dataset.get('instances', []))} instances")
        else:
            output_dataset = self._create_empty_dataset(detected_model, prompt_type, strategy, repetitions)
        
        # Filter successful patches
        to_measure = []
        for result_file in result_files:
            try:
                exp_result = load_json(result_file)
                instance_id = exp_result.get("instance_id")
                
                patch_result = exp_result.get("patch_result", {})
                if isinstance(patch_result, dict):
                    is_success = patch_result.get("success", False)
                else:
                    is_success = exp_result.get("status") == "success"
                
                if not is_success:
                    continue
                
                if skip_completed and self._is_instance_in_dataset(output_dataset, instance_id):
                    continue
                
                # Get LLM response
                llm_response = exp_result.get("llm_response", "")
                if not llm_response:
                    llm_response = exp_result.get("patch_content", "")
                if not llm_response:
                    all_responses = exp_result.get("all_responses", [])
                    if all_responses:
                        llm_response = all_responses[-1]
                if not llm_response:
                    iterations = exp_result.get("iterations", [])
                    if iterations:
                        llm_response = iterations[-1].get("patch_content", "")
                
                if not llm_response:
                    continue
                
                to_measure.append({
                    "instance_id": instance_id,
                    "llm_response": llm_response
                })
                
                if limit and len(to_measure) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Error reading {result_file.name}: {e}")
        
        already_done = len(output_dataset.get('instances', []))
        logger.info(f"Already measured: {already_done}")
        logger.info(f"To measure: {len(to_measure)}")
        
        if not to_measure:
            logger.info(f"✅ Nothing to measure!")
            return {"total": len(result_files), "already_done": already_done, 
                    "measured": 0, "success": 0, "failed": 0}
        
        # Measure
        stats = {"total": len(result_files), "already_done": already_done,
                 "measured": 0, "success": 0, "failed": 0}
        
        for idx, item in enumerate(to_measure):
            instance_id = item["instance_id"]
                        # Skip known problematic instances
                        
            if instance_id == "astropy__astropy-13496":
                logger.warning(f"\n[{idx+1}/{len(to_measure)}] ⏭️ Skipping known problematic instance: {instance_id}")
                continue

            logger.info(f"\n[{idx+1}/{len(to_measure)}] {instance_id}")
            
            start_time = time.time()
            result = self.measure_instance(instance_id, item["llm_response"], repetitions)
            elapsed = time.time() - start_time
            
            stats["measured"] += 1
            
            if result:
                stats["success"] += 1
                
                # Update dataset
                output_dataset['instances'] = [
                    i for i in output_dataset['instances']
                    if i.get('instance_id') != instance_id
                ]
                output_dataset['instances'].append(result)
                output_dataset['metadata']['last_updated'] = datetime.now().isoformat()
                output_dataset['metadata']['instance_count'] = len(output_dataset['instances'])
                
                save_json(output_dataset, output_path)
                logger.info(f"  💾 Saved ({elapsed:.1f}s)")
            else:
                stats["failed"] += 1
                logger.warning(f"  ❌ Failed ({elapsed:.1f}s)")
        
        logger.info(f"\n✅ Complete! {stats['success']} success, {stats['failed']} failed")
        return stats
    
    def run(self, strategies: List[str], prompt_type: str = "zero_shot",
            repetitions: int = 3, skip_completed: bool = True, limit: Optional[int] = None,
            results_dir: Optional[Path] = None, model_name: Optional[str] = None) -> Dict:
        """Run measurements for strategies."""
        
        summary = {"prompt_type": prompt_type, "results": {}}
        
        for strategy in strategies:
            stats = self.measure_strategy(
                strategy=strategy,
                prompt_type=prompt_type,
                repetitions=repetitions,
                skip_completed=skip_completed,
                limit=limit,
                results_dir=results_dir,
                model_name=model_name
            )
            summary["results"][strategy] = stats
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Measure LLM patches (FIXED version)")
    parser.add_argument('--strategy', '-s', choices=['oracle', 'realistic', 'both'], default='oracle')
    parser.add_argument('--prompt-type', '-p', choices=['zero_shot', 'cot', 'self_collab', 'ldb'], default='zero_shot')
    parser.add_argument('--repetitions', '-k', type=int, default=3)
    parser.add_argument('--limit', '-l', type=int, default=None)
    parser.add_argument('--no-skip', action='store_true')
    parser.add_argument('--dataset', '-d', type=str, default=str(DEFAULT_DATASET))
    parser.add_argument('--results-dir', '-r', type=str, default=None)
    parser.add_argument('--model-name', '-m', type=str, default=None)
    
    args = parser.parse_args()
    
    strategies = ['oracle', 'realistic'] if args.strategy == 'both' else [args.strategy]
    
    measurer = LLMPatchMeasurer(
        original_dataset_path=ORIGINAL_DATASET,
        reduced_dataset_path=Path(args.dataset)
    )
    
    measurer.run(
        strategies=strategies,
        prompt_type=args.prompt_type,
        repetitions=args.repetitions,
        skip_completed=not args.no_skip,
        limit=args.limit,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()