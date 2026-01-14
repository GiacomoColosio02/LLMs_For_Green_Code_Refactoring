"""
LLM Patch Measurement Engine for Green Code Refactoring.

Measures energy consumption of LLM-generated patches using the same
infrastructure as measure_instance.py (which works correctly).

Supports all prompt types:
- zero_shot: Zero-Shot prompting
- cot: Chain-of-Thought prompting
- self_collab: Self-Collaboration (multi-turn)
- ldb: LDB iterative refinement

MULTI-MODEL SUPPORT:
Results directory and model name can be specified via arguments.

Output format:
    data/processed/green/[MODEL]_[PROMPT_TYPE]_[STRATEGY]_k[REPS].json

Usage:
    python measure_llm_patches.py --strategy oracle --prompt-type cot --repetitions 5
    python measure_llm_patches.py --strategy oracle --prompt-type self_collab -k 3
    python measure_llm_patches.py --dataset data/processed/swe_perf_reduced.json -k 3
    
    # Multi-model support
    python measure_llm_patches.py --strategy oracle --prompt-type zero_shot -k 3 \
        --results-dir results/DeepSeek-Coder-V2/zs_oracle --model-name DeepSeek-Coder-V2
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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- IMPORTS ---
from scripts.measure_instance import SWEPerfMeasurer
from src.measurement.collector import MetricsCollector
from src.patch_engine import PatchEngine

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

# Default model name
DEFAULT_MODEL_NAME = "Qwen2.5-Coder-7B"

# Prompt type mappings
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

# Metrics
GREEN_METRICS = [
    'cpu_energy_joules', 'gpu_energy_joules', 'total_energy_joules',
    'power_watts', 'carbon_grams', 'energy_efficiency'
]
EFFICIENCY_METRICS = [
    'duration_seconds', 'cpu_usage_mean_percent', 'cpu_usage_peak_percent',
    'ram_usage_mean_mb', 'ram_usage_peak_mb',
    'gpu_temperature_mean_celsius', 'gpu_temperature_peak_celsius'
]
ALL_METRICS = GREEN_METRICS + EFFICIENCY_METRICS
AGGREGATIONS = ['mean', 'std', 'min', 'max']


def get_default_results_dir(prompt_type: str, strategy: str) -> Path:
    """Get default results directory for prompt type and strategy."""
    prefix = PROMPT_TYPE_TO_DIR_PREFIX.get(prompt_type, prompt_type)
    return BASE_RESULTS_DIR / f"{prefix}_{strategy}"


def sanitize_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe string."""
    name = model_name.split("/")[-1]
    for suffix in ["-Instruct-AWQ", "-AWQ", "-Instruct", "-instruct", "-awq"]:
        name = name.replace(suffix, "")
    return re.sub(r'[^\w\-.]', '_', name)


def get_output_filename(model_name: str, prompt_type: str, strategy: str, repetitions: int) -> str:
    """Generate output filename."""
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


class LLMPatchMeasurer:
    """
    Measures LLM-generated patches using the same infrastructure as measure_instance.py.
    """
    
    def __init__(self, original_dataset_path: Path, reduced_dataset_path: Path, country_code: str = "ESP"):
        self.original_dataset_path = Path(original_dataset_path)
        self.reduced_dataset_path = Path(reduced_dataset_path)
        self.country_code = country_code
        
        # Load datasets
        self.original_data = self._load_dataset_as_dict(self.original_dataset_path)
        self.reduced_data = self._load_dataset_as_dict(self.reduced_dataset_path)
        
        # Initialize the SAME measurer used by measure_instance.py
        self.measurer = SWEPerfMeasurer(str(original_dataset_path), country_code=country_code)
        
        logger.info(f"LLMPatchMeasurer initialized")
        logger.info(f"  Original dataset: {len(self.original_data)} instances")
        logger.info(f"  Reduced dataset: {len(self.reduced_data)} instances")
    
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
                'instance_count': 0
            },
            'instances': []
        }
    
    def _is_instance_in_dataset(self, dataset: Dict, instance_id: str) -> bool:
        return any(inst.get('instance_id') == instance_id for inst in dataset.get('instances', []))
    
    def measure_instance(self, instance_id: str, llm_response: str, repetitions: int = 5) -> Optional[Dict]:
        """
        Measure a single instance with LLM patch applied.
        Uses the SAME installation logic as measure_instance.py.
        """
        logger.info(f"  📏 Measuring {instance_id}")
        
        # Get instance from original dataset
        instance = self.original_data.get(instance_id)
        if not instance:
            logger.error(f"    Instance not found in original dataset")
            return None
        
        temp_dir = None
        conda_env = None
        
        try:
            # 1. Setup repository at BASE commit (same as measure_instance.py)
            logger.info(f"    📦 Cloning repository...")
            temp_dir = Path(tempfile.mkdtemp(prefix="llm_meas_"))
            repo_path = self.measurer.setup_repository(instance, temp_dir, instance['base_commit'])
            
            # 2. Apply LLM patch BEFORE installing dependencies
            logger.info(f"    🔧 Applying LLM patch...")
            patch_engine = PatchEngine(repo_path)
            
            # Get candidate files from LLM response
            candidates = list(set(re.findall(r'###\s*([\w/._-]+\.py)', llm_response)))
            if not candidates:
                # Fallback: extract from original patch
                patch_files = re.findall(r'diff --git a/(.*?) b/', instance.get('patch', ''))
                candidates = [f for f in patch_files if f.endswith('.py')]
            
            patch_result = patch_engine.apply_patch(llm_response, candidates)
            
            if not patch_result.success:
                logger.warning(f"    ❌ Failed to apply patch: {patch_result.error_message}")
                return None
            
            logger.info(f"    ✅ Patch applied to: {patch_result.modified_files}")
            
            # 3. Install dependencies using measure_instance.py's method
            # This uses REPO_PACKAGE_CONSTRAINTS including matplotlib<3.9
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
            
            # 4. Get tests from reduced dataset
            reduced_instance = self.reduced_data.get(instance_id, {})
            tests = reduced_instance.get('efficiency_test', [])
            
            if not tests:
                logger.warning(f"    ❌ No tests found")
                return None
            
            logger.info(f"    🧪 Running {len(tests)} tests x {repetitions} repetitions...")
            
            # 5. Run measurements using MetricsCollector
            test_results = {}
            valid_tests = 0
            
            for test in tests:
                test_name = test.split("::")[-1] if "::" in test else test
                
                try:
                    collector = MetricsCollector(
                        repo_path=repo_path,
                        python_path=python_path,
                        test_command=test,
                        repetitions=repetitions,
                        country_code=self.country_code,
                        warmup_runs=1
                    )
                    
                    metrics = collector.collect()
                    
                    if metrics:
                        test_results[test_name] = {"head": metrics}
                        valid_tests += 1
                except Exception as e:
                    logger.debug(f"    Test {test_name} failed: {e}")
            
            if valid_tests == 0:
                logger.warning(f"    ❌ No tests passed")
                return None
            
            # 6. Build result in green dataset format
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
            
            logger.info(f"    ✅ Measured {valid_tests}/{len(tests)} tests successfully")
            return result
            
        except Exception as e:
            logger.error(f"    ❌ Error: {e}", exc_info=True)
            return None
            
        finally:
            # Cleanup
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            if conda_env:
                self.measurer.cleanup_conda_env(conda_env)
    
    def measure_strategy(
        self,
        strategy: str,
        prompt_type: str,
        repetitions: int = 5,
        skip_completed: bool = True,
        limit: Optional[int] = None,
        results_dir: Optional[Path] = None,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        Measure all instances for a strategy.
        
        Args:
            strategy: "oracle" or "realistic"
            prompt_type: "zero_shot", "cot", "self_collab", or "ldb"
            repetitions: Number of measurement repetitions
            skip_completed: Skip already measured instances
            limit: Maximum instances to measure
            results_dir: Custom results directory (for multi-model support)
            model_name: Model name override (for multi-model support)
        """
        
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
        logger.info(f"Found {len(result_files)} experiment results in {exp_results_dir}")
        
        # Detect model from results or use provided model_name
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
            logger.info(f"Created new dataset")
        
        # Filter successful patches
        to_measure = []
        for result_file in result_files:
            try:
                exp_result = load_json(result_file)
                instance_id = exp_result.get("instance_id")
                
                # Check for success - handle both old and new formats
                patch_result = exp_result.get("patch_result", {})
                if isinstance(patch_result, dict):
                    is_success = patch_result.get("success", False)
                else:
                    is_success = exp_result.get("status") == "success"
                
                if not is_success:
                    continue
                
                if skip_completed and self._is_instance_in_dataset(output_dataset, instance_id):
                    continue
                
                # Get LLM response - handle different formats
                llm_response = exp_result.get("llm_response", "")
                if not llm_response:
                    # Try patch_content for newer format
                    llm_response = exp_result.get("patch_content", "")
                if not llm_response:
                    # Try all_responses for self_collab
                    all_responses = exp_result.get("all_responses", [])
                    if all_responses:
                        llm_response = all_responses[-1]  # Use reviewer's response
                if not llm_response:
                    # Try iterations for ldb
                    iterations = exp_result.get("iterations", [])
                    if iterations:
                        llm_response = iterations[-1].get("patch_content", "")
                
                if not llm_response:
                    logger.debug(f"No LLM response found in {result_file.name}")
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
                 "measured": 0, "success": 0, "failed": 0, "timing": []}
        
        for idx, item in enumerate(to_measure):
            instance_id = item["instance_id"]
            logger.info(f"\n[{idx+1}/{len(to_measure)}] {instance_id}")
            
            start_time = time.time()
            result = self.measure_instance(instance_id, item["llm_response"], repetitions)
            elapsed = time.time() - start_time
            
            stats["timing"].append(elapsed)
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
        
        logger.info(f"\n✅ Complete! Output: {output_path}")
        logger.info(f"   Instances: {len(output_dataset['instances'])}")
        
        return stats
    
    def run(self, strategies: List[str], prompt_type: str = "zero_shot",
            repetitions: int = 5, skip_completed: bool = True, limit: Optional[int] = None,
            results_dir: Optional[Path] = None, model_name: Optional[str] = None) -> Dict:
        """
        Run measurements for strategies.
        
        Args:
            strategies: List of strategies to measure
            prompt_type: Prompt type
            repetitions: Number of repetitions
            skip_completed: Skip already measured
            limit: Maximum instances
            results_dir: Custom results directory
            model_name: Model name override
        """
        
        start_time = time.time()
        summary = {"prompt_type": prompt_type, "model": model_name or DEFAULT_MODEL_NAME, "results": {}}
        
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
        
        summary["duration_seconds"] = time.time() - start_time
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 SUMMARY - {prompt_type.upper()} - {summary['model']}")
        logger.info(f"{'='*70}")
        for strategy, stats in summary["results"].items():
            if "error" in stats:
                logger.info(f"{strategy}: ERROR - {stats['error']}")
            else:
                logger.info(f"{strategy}: {stats.get('success',0)} success, {stats.get('failed',0)} failed")
        logger.info(f"Duration: {summary['duration_seconds']/60:.1f} minutes")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Measure LLM patches for energy consumption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure Zero-Shot Oracle patches
  python measure_llm_patches.py --strategy oracle --prompt-type zero_shot -k 3
  
  # Measure Self-Collaboration patches
  python measure_llm_patches.py --strategy oracle --prompt-type self_collab -k 3
  
  # Measure LDB patches
  python measure_llm_patches.py --strategy realistic --prompt-type ldb -k 3
  
  # Measure both strategies
  python measure_llm_patches.py --strategy both --prompt-type cot -k 3
  
  # Multi-model support: specify custom results dir and model name
  python measure_llm_patches.py --strategy oracle --prompt-type zero_shot -k 3 \
      --results-dir results/DeepSeek-Coder-V2/zs_oracle --model-name DeepSeek-Coder-V2
        """
    )
    parser.add_argument('--strategy', '-s', 
                        choices=['oracle', 'realistic', 'both'], 
                        default='oracle',
                        help='Strategy to measure')
    parser.add_argument('--prompt-type', '-p', 
                        choices=['zero_shot', 'cot', 'self_collab', 'ldb'], 
                        default='zero_shot',
                        help='Prompt type to measure')
    parser.add_argument('--repetitions', '-k', type=int, default=3,
                        help='Number of measurement repetitions (default: 3)')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of instances to measure')
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-measure already completed instances')
    parser.add_argument('--dataset', '-d', type=str, default=str(DEFAULT_DATASET),
                        help='Path to reduced dataset (default: swe_perf_reduced.json)')
    parser.add_argument('--results-dir', '-r', type=str, default=None,
                        help='Custom results directory (for multi-model support)')
    parser.add_argument('--model-name', '-m', type=str, default=None,
                        help='Model name override (for multi-model support)')
    
    args = parser.parse_args()
    
    strategies = ['oracle', 'realistic'] if args.strategy == 'both' else [args.strategy]
    reduced_dataset = Path(args.dataset)
    
    if not reduced_dataset.exists():
        print(f"❌ Dataset not found: {reduced_dataset}")
        sys.exit(1)
    
    measurer = LLMPatchMeasurer(
        original_dataset_path=ORIGINAL_DATASET,
        reduced_dataset_path=reduced_dataset
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