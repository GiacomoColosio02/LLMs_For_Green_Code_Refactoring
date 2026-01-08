"""
LLM Patch Measurement Engine for Green Code Refactoring.

Measures energy consumption of LLM-generated patches using the same
infrastructure as measure_instance.py (which works correctly).

Output format:
    data/processed/green/[MODEL]_[PROMPT_TYPE]_[STRATEGY]_k[REPS].json

Usage:
    python measure_llm_patches.py --strategy oracle --prompt-type cot --repetitions 5
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
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "swe_perf_reduced_test.json"
ORIGINAL_DATASET = PROJECT_ROOT / "data" / "original" / "swe_perf_original_20251124.json"
GREEN_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "green"
RESULTS_DIR = PROJECT_ROOT / "results"

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


def get_results_dir(prompt_type: str, strategy: str) -> Path:
    """Get results directory for prompt type and strategy."""
    prefix = "zs" if prompt_type.lower() == "zero_shot" else "cot"
    return RESULTS_DIR / f"{prefix}_{strategy}"


def sanitize_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe string."""
    name = model_name.split("/")[-1]
    for suffix in ["-Instruct-AWQ", "-AWQ", "-Instruct", "-instruct", "-awq"]:
        name = name.replace(suffix, "")
    return re.sub(r'[^\w\-.]', '_', name)


def get_output_filename(model_name: str, prompt_type: str, strategy: str, repetitions: int) -> str:
    """Generate output filename."""
    model_clean = sanitize_model_name(model_name)
    prompt_clean = "ZeroShot" if prompt_type.lower() == "zero_shot" else "CoT"
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
        prompt_clean = "ZeroShot" if prompt_type.lower() == "zero_shot" else "CoT"
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
            
            logger.info(f"    ✅ Dependencies installed (python: {python_path})")
            
            # 4. Get tests
            tests = instance.get('efficiency_test', [])
            if isinstance(tests, str):
                tests = [tests]
            
            logger.info(f"    🧪 Measuring {len(tests)} tests (k={repetitions})...")
            
            # 5. Measure each test using MetricsCollector
            test_results = {}
            valid_tests = 0
            
            collector = MetricsCollector(
                instance_id=instance_id,
                country_code=self.country_code
            )
            
            for test in tests:
                test_name = test.split("::")[-1] if "::" in test else test.split("/")[-1]
                
                try:
                    # Build test command (same format as measure_instance.py)
                    test_command = f"cd {repo_path} && {python_path} -m pytest '{test}' -x -v"
                    
                    result = collector.measure_test_execution(
                        test_command=test_command,
                        repetitions=repetitions
                    )
                    
                    if result and result.get('aggregated'):
                        test_results[test_name] = {
                            'head': result['aggregated']  # LLM patch = head
                        }
                        valid_tests += 1
                        logger.info(f"      ✅ {test_name}")
                    else:
                        logger.warning(f"      ❌ {test_name} - no valid results")
                        
                except Exception as e:
                    logger.warning(f"      ❌ {test_name} - {str(e)[:100]}")
            
            # pass  # cleanup not needed  # Not needed
            
            if valid_tests == 0:
                logger.warning(f"    ⚠️ No valid test measurements")
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
        limit: Optional[int] = None
    ) -> Dict:
        """Measure all instances for a strategy."""
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📏 MEASURING: {prompt_type.upper()} - {strategy.upper()}")
        logger.info(f"{'='*70}")
        
        # Find experiment results
        results_dir = get_results_dir(prompt_type, strategy)
        if not results_dir.exists():
            logger.error(f"Results directory not found: {results_dir}")
            return {"error": f"no_results_directory: {results_dir}"}
        
        result_files = list(results_dir.glob("*.json"))
        logger.info(f"Found {len(result_files)} experiment results in {results_dir}")
        
        # Detect model
        model_name = "Unknown"
        for rf in result_files:
            try:
                data = load_json(rf)
                if data.get("model"):
                    model_name = data["model"]
                    break
            except:
                pass
        
        logger.info(f"Model: {model_name}")
        
        # Output path
        output_filename = get_output_filename(model_name, prompt_type, strategy, repetitions)
        output_path = GREEN_OUTPUT_DIR / output_filename
        logger.info(f"Output: {output_path}")
        
        # Load or create dataset
        if output_path.exists():
            output_dataset = load_json(output_path)
            logger.info(f"Loaded existing: {len(output_dataset.get('instances', []))} instances")
        else:
            output_dataset = self._create_empty_dataset(model_name, prompt_type, strategy, repetitions)
            logger.info(f"Created new dataset")
        
        # Filter successful patches
        to_measure = []
        for result_file in result_files:
            try:
                exp_result = load_json(result_file)
                instance_id = exp_result.get("instance_id")
                
                if exp_result.get("status") != "success":
                    continue
                
                if skip_completed and self._is_instance_in_dataset(output_dataset, instance_id):
                    continue
                
                to_measure.append({
                    "instance_id": instance_id,
                    "llm_response": exp_result.get("llm_response", "")
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
            repetitions: int = 5, skip_completed: bool = True, limit: Optional[int] = None) -> Dict:
        """Run measurements for strategies."""
        
        start_time = time.time()
        summary = {"prompt_type": prompt_type, "results": {}}
        
        for strategy in strategies:
            stats = self.measure_strategy(strategy, prompt_type, repetitions, skip_completed, limit)
            summary["results"][strategy] = stats
        
        summary["duration_seconds"] = time.time() - start_time
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 SUMMARY - {prompt_type.upper()}")
        logger.info(f"{'='*70}")
        for strategy, stats in summary["results"].items():
            if "error" in stats:
                logger.info(f"{strategy}: ERROR - {stats['error']}")
            else:
                logger.info(f"{strategy}: {stats.get('success',0)} success, {stats.get('failed',0)} failed")
        logger.info(f"Duration: {summary['duration_seconds']/60:.1f} minutes")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Measure LLM patches")
    parser.add_argument('--strategy', '-s', choices=['oracle', 'realistic', 'both'], default='oracle')
    parser.add_argument('--prompt-type', '-p', choices=['zero_shot', 'cot'], default='zero_shot')
    parser.add_argument('--repetitions', '-k', type=int, default=5)
    parser.add_argument('--limit', '-l', type=int, default=None)
    parser.add_argument('--no-skip', action='store_true')
    
    args = parser.parse_args()
    
    strategies = ['oracle', 'realistic'] if args.strategy == 'both' else [args.strategy]
    
    measurer = LLMPatchMeasurer(
        original_dataset_path=ORIGINAL_DATASET,
        reduced_dataset_path=DEFAULT_DATASET
    )
    
    measurer.run(
        strategies=strategies,
        prompt_type=args.prompt_type,
        repetitions=args.repetitions,
        skip_completed=not args.no_skip,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
