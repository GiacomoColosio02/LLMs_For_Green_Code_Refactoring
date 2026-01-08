"""
LLM Patch Measurement Engine for Green Code Refactoring.

Measures energy consumption of LLM-generated patches and creates
datasets COMPATIBLE with swe_perf_green.json format.

Output format:
    data/processed/green/[MODEL]_[PROMPT_TYPE]_[STRATEGY]_k[REPS].json
    
Example:
    data/processed/green/Qwen2.5-Coder-7B_ZeroShot_Oracle_k5.json

Usage:
    # Measure oracle patches with 5 repetitions
    python measure_llm_patches.py --strategy oracle --repetitions 5
    
    # Measure realistic patches
    python measure_llm_patches.py --strategy realistic --repetitions 5
    
    # Measure both strategies
    python measure_llm_patches.py --strategy both --repetitions 5
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
from collections import defaultdict

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

# Metrics classification (same as measure_and_create_green_dataset.py)
GREEN_METRICS = [
    'cpu_energy_joules',
    'gpu_energy_joules', 
    'total_energy_joules',
    'power_watts',
    'carbon_grams',
    'energy_efficiency'
]

EFFICIENCY_METRICS = [
    'duration_seconds',
    'cpu_usage_mean_percent',
    'cpu_usage_peak_percent',
    'ram_usage_mean_mb',
    'ram_usage_peak_mb',
    'gpu_temperature_mean_celsius',
    'gpu_temperature_peak_celsius'
]

ALL_METRICS = GREEN_METRICS + EFFICIENCY_METRICS
AGGREGATIONS = ['mean', 'std', 'min', 'max']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sanitize_model_name(model_name: str) -> str:
    """
    Convert model name to filesystem-safe string.
    
    Example: "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ" -> "Qwen2.5-Coder-7B"
    """
    # Remove provider prefix
    name = model_name.split("/")[-1]
    
    # Remove common suffixes
    for suffix in ["-Instruct-AWQ", "-AWQ", "-Instruct", "-instruct", "-awq"]:
        name = name.replace(suffix, "")
    
    # Remove special characters
    name = re.sub(r'[^\w\-.]', '_', name)
    
    return name


def get_output_filename(
    model_name: str,
    prompt_type: str,
    strategy: str,
    repetitions: int
) -> str:
    """
    Generate descriptive output filename.
    
    Format: [MODEL]_[PROMPT_TYPE]_[STRATEGY]_k[REPS].json
    Example: Qwen2.5-Coder-7B_ZeroShot_Oracle_k5.json
    """
    model_clean = sanitize_model_name(model_name)
    strategy_clean = strategy.capitalize()
    
    return f"{model_clean}_{prompt_type}_{strategy_clean}_k{repetitions}.json"


def load_json(path: Path) -> dict:
    """Load JSON file."""
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    """Save JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_aggregated_metrics(test_result: dict) -> Optional[Dict]:
    """
    Extract aggregated metrics from test result.
    Compatible with measure_and_create_green_dataset.py format.
    """
    if not test_result:
        return None
    
    # Check if already aggregated
    if 'aggregated' in test_result:
        aggregated = test_result['aggregated']
    elif 'metrics' in test_result:
        # Direct metrics dict
        aggregated = test_result['metrics']
    else:
        return None
    
    result = {}
    
    for metric in ALL_METRICS:
        for agg in AGGREGATIONS:
            key = f"{metric}_{agg}"
            if key in aggregated:
                result[key] = aggregated[key]
    
    return result if result else None


# =============================================================================
# MEASUREMENT ENGINE
# =============================================================================

class LLMPatchMeasurer:
    """
    Measures energy consumption of LLM-generated patches.
    
    Creates datasets in the SAME FORMAT as swe_perf_green.json:
    - green_metrics per test with base/head structure
    - Compatible metadata
    - Same aggregation format
    """
    
    def __init__(
        self,
        original_dataset_path: Path,
        reduced_dataset_path: Path,
        country_code: str = "ESP"
    ):
        """
        Initialize measurer.
        
        Args:
            original_dataset_path: Path to original SWE-Perf dataset (for measurer)
            reduced_dataset_path: Path to reduced dataset (for instance list)
            country_code: ISO country code for carbon intensity
        """
        self.original_dataset_path = Path(original_dataset_path)
        self.reduced_dataset_path = Path(reduced_dataset_path)
        self.country_code = country_code
        
        # Load datasets
        self.original_data = self._load_dataset_as_dict(self.original_dataset_path)
        self.reduced_data = self._load_dataset_as_dict(self.reduced_dataset_path)
        
        # Initialize measurer
        self.measurer = SWEPerfMeasurer(str(original_dataset_path), country_code=country_code)
        
        logger.info(f"LLMPatchMeasurer initialized")
        logger.info(f"  Original dataset: {len(self.original_data)} instances")
        logger.info(f"  Reduced dataset: {len(self.reduced_data)} instances")
    
    def _load_dataset_as_dict(self, path: Path) -> Dict[str, Dict]:
        """Load dataset as dict keyed by instance_id."""
        data = load_json(path)
        
        if isinstance(data, list):
            instances = data
        else:
            instances = data.get("instances", [])
        
        return {item["instance_id"]: item for item in instances}
    
    def _create_empty_dataset(
        self,
        model_name: str,
        prompt_type: str,
        strategy: str,
        repetitions: int
    ) -> Dict:
        """Create empty dataset with proper metadata."""
        return {
            'metadata': {
                'name': f'LLM Green Dataset - {model_name} - {prompt_type} - {strategy}',
                'description': f'Green software metrics for LLM-optimized code',
                'model': model_name,
                'prompt_type': prompt_type,
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
        """Check if instance already in dataset."""
        return any(
            inst.get('instance_id') == instance_id 
            for inst in dataset.get('instances', [])
        )
    
    def _calculate_aggregates_from_measurements(self, measurements: List[Dict]) -> Dict[str, float]:
        """Calculate aggregated metrics from raw measurement list."""
        import numpy as np
        
        if not measurements:
            return {}
        
        aggregated = {}
        
        for metric in ALL_METRICS:
            values = []
            for m in measurements:
                val = m.get(metric)
                # Handle alternative names
                if val is None and metric == 'duration_seconds':
                    val = m.get('runtime_seconds')
                if val is None and metric == 'total_energy_joules':
                    val = m.get('energy_joules')
                if val is not None:
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        pass
            
            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))
                aggregated[f"{metric}_min"] = float(np.min(values))
                aggregated[f"{metric}_max"] = float(np.max(values))
        
        return aggregated
    
    # =========================================================================
    # MEASUREMENT METHODS
    # =========================================================================
    
    def measure_instance(
        self,
        instance_id: str,
        llm_response: str,
        repetitions: int = 5
    ) -> Optional[Dict]:
        """
        Measure a single instance with LLM patch.
        
        Returns data in swe_perf_green.json compatible format:
        {
            "instance_id": "...",
            "green_metrics": {
                "test_name": {
                    "head": { metrics... }  # LLM-optimized version
                }
            }
        }
        """
        logger.info(f"  📏 Measuring {instance_id}")
        
        # Get instance data
        instance = self.reduced_data.get(instance_id) or self.original_data.get(instance_id)
        if not instance:
            logger.error(f"    Instance {instance_id} not found")
            return None
        
        test_list = instance.get('efficiency_test', [])
        if not test_list:
            logger.warning(f"    No efficiency tests")
            return None
        
        temp_dir = None
        
        try:
            # Setup repository
            temp_dir = Path(tempfile.mkdtemp(prefix="llm_meas_"))
            logger.info(f"    📦 Cloning repository...")
            
            repo_path = self.measurer.setup_repository(
                instance, temp_dir, instance['base_commit']
            )
            
            # Apply LLM patch
            logger.info(f"    🔧 Applying LLM patch...")
            patch_engine = PatchEngine(repo_path)
            
            # Get candidate files
            patch_funcs = instance.get("patch_functions", {})
            if isinstance(patch_funcs, str):
                try:
                    patch_funcs = json.loads(patch_funcs)
                except:
                    patch_funcs = {}
            candidates = list(patch_funcs.keys()) if isinstance(patch_funcs, dict) else []
            
            patch_result = patch_engine.apply_patch(llm_response, candidates)
            
            if not patch_result.success:
                logger.warning(f"    ⚠️ Patch failed: {patch_result.error_message}")
                return None
            
            logger.info(f"    ✅ Patch applied to: {patch_result.modified_files}")
            
            # Install dependencies
            logger.info(f"    📦 Installing dependencies...")
            python_path, conda_env = self.measurer.install_dependencies(
                repo_path, instance['repo'], instance['version'], instance['base_commit']
            )
            
            if not python_path:
                logger.error(f"    ❌ Dependency installation failed")
                return None
            
            # Measure each test
            logger.info(f"    🧪 Measuring {len(test_list)} tests (k={repetitions})...")
            collector = MetricsCollector(instance_id=instance_id, country_code=self.country_code)
            
            green_metrics = {}
            valid_tests = 0
            
            for test_name in test_list:
                test_cmd = f"cd {repo_path} && {python_path} -m pytest '{repo_path}/{test_name}' -v"
                
                try:
                    test_result = collector.measure_test_execution(
                        test_command=test_cmd,
                        repetitions=repetitions
                    )
                    
                    if test_result and test_result.get('status') == 'success':
                        # Get aggregated metrics
                        if 'aggregated' in test_result:
                            metrics = {}
                            for metric in ALL_METRICS:
                                for agg in AGGREGATIONS:
                                    key = f"{metric}_{agg}"
                                    if key in test_result['aggregated']:
                                        metrics[key] = test_result['aggregated'][key]
                            
                            if metrics:
                                green_metrics[test_name] = {'head': metrics}
                                valid_tests += 1
                                logger.info(f"      ✅ {test_name.split('::')[-1]}")
                        
                        elif 'measurements' in test_result:
                            # Calculate from raw measurements
                            valid_measurements = [
                                m for m in test_result['measurements']
                                if m and m.get('return_code') == 0
                            ]
                            if valid_measurements:
                                metrics = self._calculate_aggregates_from_measurements(valid_measurements)
                                if metrics:
                                    green_metrics[test_name] = {'head': metrics}
                                    valid_tests += 1
                                    logger.info(f"      ✅ {test_name.split('::')[-1]}")
                    else:
                        logger.warning(f"      ❌ {test_name.split('::')[-1]} - test failed")
                        
                except Exception as e:
                    logger.warning(f"      ❌ {test_name.split('::')[-1]} - {str(e)[:50]}")
            
            # Cleanup conda env
            if conda_env:
                self.measurer.cleanup_conda_env(conda_env)
            
            if valid_tests == 0:
                logger.warning(f"    ⚠️ No valid test measurements")
                return None
            
            # Build result in swe_perf_green.json format
            result = {
                'instance_id': instance_id,
                'repo': instance.get('repo'),
                'version': instance.get('version'),
                'base_commit': instance.get('base_commit'),
                'patch': instance.get('patch'),  # Original human patch for reference
                'test_patch': instance.get('test_patch'),
                'efficiency_test': list(green_metrics.keys()),  # Only tests that worked
                'green_metrics': green_metrics,
                '_green_metadata': {
                    'valid_tests': valid_tests,
                    'total_tests': len(test_list),
                    'repetitions': repetitions,
                    'green_metrics_count': len(GREEN_METRICS),
                    'efficiency_metrics_count': len(EFFICIENCY_METRICS),
                    'aggregations': AGGREGATIONS,
                    'creation_date': datetime.now().isoformat(),
                    'patch_files': patch_result.modified_files
                }
            }
            
            logger.info(f"    ✅ Success: {valid_tests}/{len(test_list)} tests measured")
            return result
            
        except Exception as e:
            logger.error(f"    ❌ Error: {e}", exc_info=True)
            return None
            
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def measure_strategy(
        self,
        strategy: str,
        prompt_type: str = "ZeroShot",
        repetitions: int = 5,
        skip_completed: bool = True,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Measure all successful patches for a strategy.
        
        Args:
            strategy: "oracle" or "realistic"
            prompt_type: Prompt type name for output filename
            repetitions: Number of measurement repetitions
            skip_completed: Skip already measured instances
            limit: Maximum instances to process
            
        Returns:
            Summary statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📏 MEASURING: {prompt_type} - {strategy.upper()}")
        logger.info(f"{'='*70}")
        
        # Find experiment results
        results_dir = RESULTS_DIR / f"zs_{strategy}"
        if not results_dir.exists():
            logger.error(f"Results directory not found: {results_dir}")
            return {"error": "no_results_directory"}
        
        result_files = list(results_dir.glob("*.json"))
        logger.info(f"Found {len(result_files)} experiment results")
        
        # Detect model from first result
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
        
        # Generate output filename
        output_filename = get_output_filename(model_name, prompt_type, strategy, repetitions)
        output_path = GREEN_OUTPUT_DIR / output_filename
        
        logger.info(f"Output: {output_path}")
        
        # Load or create output dataset
        if output_path.exists():
            output_dataset = load_json(output_path)
            logger.info(f"Loaded existing dataset: {len(output_dataset.get('instances', []))} instances")
        else:
            output_dataset = self._create_empty_dataset(model_name, prompt_type, strategy, repetitions)
            logger.info(f"Created new dataset")
        
        # Filter to successful patches
        to_measure = []
        for result_file in result_files:
            try:
                exp_result = load_json(result_file)
                
                instance_id = exp_result.get("instance_id")
                status = exp_result.get("status")
                
                # Only measure successful patches
                if status != "success":
                    continue
                
                # Skip if already measured
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
            logger.info(f"✅ All instances already measured!")
            return {
                "total": len(result_files),
                "already_done": already_done,
                "measured": 0,
                "success": 0,
                "failed": 0
            }
        
        # Measure each instance
        stats = {
            "total": len(result_files),
            "already_done": already_done,
            "measured": 0,
            "success": 0,
            "failed": 0,
            "timing": []
        }
        
        for idx, item in enumerate(to_measure):
            instance_id = item["instance_id"]
            
            logger.info(f"\n[{idx+1}/{len(to_measure)}] {instance_id}")
            
            start_time = time.time()
            
            result = self.measure_instance(
                instance_id=instance_id,
                llm_response=item["llm_response"],
                repetitions=repetitions
            )
            
            elapsed = time.time() - start_time
            stats["timing"].append(elapsed)
            stats["measured"] += 1
            
            if result:
                stats["success"] += 1
                
                # Remove old entry if exists
                output_dataset['instances'] = [
                    i for i in output_dataset['instances']
                    if i.get('instance_id') != instance_id
                ]
                
                # Add new entry
                output_dataset['instances'].append(result)
                
                # Update metadata
                output_dataset['metadata']['last_updated'] = datetime.now().isoformat()
                output_dataset['metadata']['instance_count'] = len(output_dataset['instances'])
                
                # Save incrementally
                save_json(output_dataset, output_path)
                
                logger.info(f"  💾 Saved ({elapsed:.1f}s)")
            else:
                stats["failed"] += 1
                logger.warning(f"  ❌ Failed ({elapsed:.1f}s)")
            
            # Progress update every 5 instances
            if (idx + 1) % 5 == 0 and stats["timing"]:
                avg_time = sum(stats["timing"]) / len(stats["timing"])
                remaining = len(to_measure) - (idx + 1)
                eta_min = (remaining * avg_time) / 60
                success_rate = stats["success"] / stats["measured"] * 100
                logger.info(f"\n📊 Progress: {idx+1}/{len(to_measure)} | Success: {success_rate:.0f}% | ETA: {eta_min:.1f}min")
        
        # Final save
        save_json(output_dataset, output_path)
        
        logger.info(f"\n✅ Strategy complete!")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   Total instances: {len(output_dataset['instances'])}")
        
        return stats
    
    def run(
        self,
        strategies: List[str],
        prompt_type: str = "ZeroShot",
        repetitions: int = 5,
        skip_completed: bool = True,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Run measurements for all specified strategies.
        
        Args:
            strategies: List of strategies ["oracle", "realistic"]
            prompt_type: Prompt type name for filename
            repetitions: Number of repetitions per test
            skip_completed: Skip already measured
            limit: Max instances per strategy
            
        Returns:
            Summary dictionary
        """
        start_time = time.time()
        
        summary = {
            "start_time": datetime.now().isoformat(),
            "prompt_type": prompt_type,
            "strategies": strategies,
            "repetitions": repetitions,
            "results": {}
        }
        
        for strategy in strategies:
            stats = self.measure_strategy(
                strategy=strategy,
                prompt_type=prompt_type,
                repetitions=repetitions,
                skip_completed=skip_completed,
                limit=limit
            )
            summary["results"][strategy] = stats
        
        summary["end_time"] = datetime.now().isoformat()
        summary["total_duration_seconds"] = time.time() - start_time
        
        # Print final summary
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print final summary."""
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 MEASUREMENT SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Prompt Type: {summary['prompt_type']}")
        logger.info(f"Repetitions: {summary['repetitions']}")
        
        for strategy, stats in summary["results"].items():
            if "error" in stats:
                logger.info(f"\n{strategy.upper()}: ERROR - {stats['error']}")
                continue
                
            logger.info(f"\n{strategy.upper()}:")
            logger.info(f"  Already done: {stats.get('already_done', 0)}")
            logger.info(f"  Measured now: {stats.get('measured', 0)}")
            logger.info(f"  ✅ Success: {stats.get('success', 0)}")
            logger.info(f"  ❌ Failed: {stats.get('failed', 0)}")
            
            if stats.get('measured', 0) > 0:
                success_rate = stats['success'] / stats['measured'] * 100
                logger.info(f"  Success rate: {success_rate:.1f}%")
            
            if stats.get('timing'):
                avg_time = sum(stats['timing']) / len(stats['timing'])
                logger.info(f"  Avg time: {avg_time:.1f}s per instance")
        
        duration_min = summary['total_duration_seconds'] / 60
        logger.info(f"\nTotal duration: {duration_min:.1f} minutes")
        logger.info(f"Output directory: {GREEN_OUTPUT_DIR}")
        logger.info(f"{'='*70}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure LLM-generated patches and create green datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files are saved to:
    data/processed/green/[MODEL]_[PROMPT]_[STRATEGY]_k[REPS].json

Examples:
  # Measure oracle patches with 5 repetitions
  python measure_llm_patches.py --strategy oracle --repetitions 5
  
  # Measure realistic patches
  python measure_llm_patches.py --strategy realistic --repetitions 5
  
  # Measure both strategies
  python measure_llm_patches.py --strategy both --repetitions 5
  
  # Test with 2 instances
  python measure_llm_patches.py --strategy oracle --limit 2 --repetitions 3
  
  # Force re-measure all
  python measure_llm_patches.py --strategy oracle --no-skip
        """
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['oracle', 'realistic', 'both'],
        default='both',
        help='Strategy to measure (default: both)'
    )
    
    parser.add_argument(
        '--prompt-type', '-p',
        type=str,
        default='ZeroShot',
        help='Prompt type name for output filename (default: ZeroShot)'
    )
    
    parser.add_argument(
        '--repetitions', '-k',
        type=int,
        default=5,
        help='Number of measurement repetitions (default: 5)'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=str(DEFAULT_DATASET),
        help='Path to reduced dataset JSON'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of instances to measure'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip already measured instances'
    )
    
    args = parser.parse_args()
    
    # Determine strategies
    if args.strategy == 'both':
        strategies = ['oracle', 'realistic']
    else:
        strategies = [args.strategy]
    
    # Run measurements
    measurer = LLMPatchMeasurer(
        original_dataset_path=ORIGINAL_DATASET,
        reduced_dataset_path=Path(args.dataset)
    )
    
    summary = measurer.run(
        strategies=strategies,
        prompt_type=args.prompt_type,
        repetitions=args.repetitions,
        skip_completed=not args.no_skip,
        limit=args.limit
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())