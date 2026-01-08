"""
LLM Patch Measurement Engine for Green Code Refactoring.

Measures energy consumption of LLM-generated patches and creates
a comparative dataset: base vs head_human vs head_llm.

Usage:
    # Measure all successful patches for oracle strategy
    python measure_llm_patches.py --strategy oracle
    
    # Measure all successful patches for realistic strategy
    python measure_llm_patches.py --strategy realistic
    
    # Measure both strategies
    python measure_llm_patches.py --strategy both
    
    # Limit number of instances
    python measure_llm_patches.py --strategy oracle --limit 5
"""
import sys
import os
import json
import logging
import argparse
import tempfile
import shutil
import time
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
GREEN_DATASET = PROJECT_ROOT / "data" / "processed" / "green" / "swe_perf_green.json"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "llm_green"

# Metrics to collect
GREEN_METRICS = [
    'cpu_energy_joules', 'gpu_energy_joules', 'total_energy_joules',
    'power_watts', 'carbon_grams', 'energy_efficiency'
]
EFFICIENCY_METRICS = [
    'duration_seconds', 'cpu_usage_mean_percent', 'cpu_usage_peak_percent',
    'ram_usage_mean_mb', 'ram_usage_peak_mb', 'gpu_temperature_mean_celsius'
]
ALL_METRICS = GREEN_METRICS + EFFICIENCY_METRICS
AGGREGATIONS = ['mean', 'std', 'min', 'max']


# =============================================================================
# MEASUREMENT ENGINE
# =============================================================================

class LLMPatchMeasurer:
    """
    Measures energy consumption of LLM-generated patches.
    
    Creates a comparative dataset with:
    - base: Original code (before human optimization)
    - head_human: Human-optimized code (gold standard)
    - head_llm: LLM-optimized code (our patches)
    """
    
    def __init__(
        self,
        original_dataset_path: Path,
        green_dataset_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        country_code: str = "ESP"
    ):
        """
        Initialize measurer.
        
        Args:
            original_dataset_path: Path to original SWE-Perf dataset
            green_dataset_path: Path to green dataset with human measurements (optional)
            output_path: Path for output LLM green dataset
            country_code: ISO country code for carbon intensity
        """
        self.original_dataset_path = Path(original_dataset_path)
        self.green_dataset_path = Path(green_dataset_path) if green_dataset_path else GREEN_DATASET
        self.output_path = Path(output_path) if output_path else OUTPUT_DIR / "llm_green_dataset.json"
        self.country_code = country_code
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.original_data = self._load_dataset(self.original_dataset_path)
        self.green_data = self._load_green_dataset()
        
        # Initialize measurer
        self.measurer = SWEPerfMeasurer(str(original_dataset_path), country_code=country_code)
        
        # Load or create output dataset
        self.output_data = self._load_or_create_output()
        
        logger.info(f"LLMPatchMeasurer initialized")
        logger.info(f"  Original dataset: {len(self.original_data)} instances")
        logger.info(f"  Green dataset: {len(self.green_data)} instances")
        logger.info(f"  Output: {self.output_path}")
    
    def _load_dataset(self, path: Path) -> Dict[str, Dict]:
        """Load dataset as dict keyed by instance_id."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        instances = data if isinstance(data, list) else data.get("instances", [])
        return {item["instance_id"]: item for item in instances}
    
    def _load_green_dataset(self) -> Dict[str, Dict]:
        """Load green dataset with human measurements."""
        if not self.green_dataset_path.exists():
            logger.warning(f"Green dataset not found: {self.green_dataset_path}")
            return {}
        
        with open(self.green_dataset_path, 'r') as f:
            data = json.load(f)
        
        instances = data.get("instances", [])
        return {item["instance_id"]: item for item in instances}
    
    def _load_or_create_output(self) -> Dict:
        """Load existing output or create new."""
        if self.output_path.exists():
            try:
                with open(self.output_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "metadata": {
                "name": "LLM Green Code Refactoring Dataset",
                "description": "Comparative measurements: base vs head_human vs head_llm",
                "created": datetime.now().isoformat(),
                "metrics": {
                    "green": GREEN_METRICS,
                    "efficiency": EFFICIENCY_METRICS
                },
                "aggregations": AGGREGATIONS
            },
            "instances": []
        }
    
    def _save_output(self):
        """Save output dataset."""
        self.output_data["metadata"]["last_updated"] = datetime.now().isoformat()
        self.output_data["metadata"]["instance_count"] = len(self.output_data["instances"])
        
        with open(self.output_path, 'w') as f:
            json.dump(self.output_data, f, indent=2)
    
    def _get_existing_instance(self, instance_id: str, strategy: str) -> Optional[Dict]:
        """Check if instance+strategy already measured."""
        for inst in self.output_data["instances"]:
            if inst.get("instance_id") == instance_id and inst.get("strategy") == strategy:
                return inst
        return None
    
    def _load_experiment_result(self, instance_id: str, strategy: str) -> Optional[Dict]:
        """Load experiment result from results directory."""
        result_file = RESULTS_DIR / f"zs_{strategy}" / f"{instance_id}.json"
        
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _calculate_aggregates(self, measurements: List[Dict]) -> Dict[str, float]:
        """Calculate aggregated metrics from raw measurements."""
        if not measurements:
            return {}
        
        import numpy as np
        
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
                    values.append(float(val))
            
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
        strategy: str,
        llm_response: str,
        repetitions: int = 3
    ) -> Optional[Dict]:
        """
        Measure a single instance with LLM patch.
        
        Args:
            instance_id: Instance identifier
            strategy: "oracle" or "realistic"
            llm_response: Raw LLM response containing the patch
            repetitions: Number of measurement repetitions
            
        Returns:
            Dictionary with measurements or None if failed
        """
        logger.info(f"📏 Measuring {instance_id} ({strategy})")
        
        # Get original instance data
        instance = self.original_data.get(instance_id)
        if not instance:
            logger.error(f"Instance {instance_id} not found in original dataset")
            return None
        
        test_list = instance.get('efficiency_test', [])
        if not test_list:
            logger.warning(f"No efficiency tests for {instance_id}")
            return None
        
        temp_dir = None
        result = {
            "instance_id": instance_id,
            "strategy": strategy,
            "repo": instance.get("repo"),
            "version": instance.get("version"),
            "measurement_date": datetime.now().isoformat(),
            "tests": {},
            "status": "failed"
        }
        
        try:
            # Setup repository at base commit
            temp_dir = Path(tempfile.mkdtemp(prefix="llm_meas_"))
            logger.info(f"  📦 Setting up repository...")
            
            repo_path = self.measurer.setup_repository(
                instance, temp_dir, instance['base_commit']
            )
            
            # Apply LLM patch
            logger.info(f"  🔧 Applying LLM patch...")
            patch_engine = PatchEngine(repo_path)
            
            # Get candidate files from the patch
            candidates = list(instance.get("patch_functions", {}).keys()) if isinstance(
                instance.get("patch_functions"), dict
            ) else []
            
            patch_result = patch_engine.apply_patch(llm_response, candidates)
            
            if not patch_result.success:
                logger.warning(f"  ⚠️ Patch application failed")
                result["patch_status"] = "failed"
                result["patch_error"] = patch_result.error_message
                return result
            
            result["patch_status"] = "success"
            result["patch_files"] = patch_result.modified_files
            
            # Install dependencies
            logger.info(f"  📦 Installing dependencies...")
            python_path, conda_env = self.measurer.install_dependencies(
                repo_path, instance['repo'], instance['version'], instance['base_commit']
            )
            
            if not python_path:
                logger.error(f"  ❌ Dependency installation failed")
                result["error"] = "dependency_installation_failed"
                return result
            
            # Measure each test
            logger.info(f"  🧪 Measuring {len(test_list)} tests ({repetitions} repetitions each)...")
            collector = MetricsCollector(instance_id=instance_id, country_code=self.country_code)
            
            successful_tests = 0
            
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
                            result["tests"][test_name] = {
                                "status": "success",
                                "metrics": test_result['aggregated']
                            }
                            successful_tests += 1
                        elif 'measurements' in test_result:
                            # Calculate aggregates from raw measurements
                            valid_measurements = [
                                m for m in test_result['measurements']
                                if m and m.get('return_code') == 0
                            ]
                            if valid_measurements:
                                result["tests"][test_name] = {
                                    "status": "success",
                                    "metrics": self._calculate_aggregates(valid_measurements)
                                }
                                successful_tests += 1
                    else:
                        result["tests"][test_name] = {
                            "status": "failed",
                            "error": "test_execution_failed"
                        }
                        
                except Exception as e:
                    logger.warning(f"    ❌ Test {test_name} failed: {e}")
                    result["tests"][test_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            # Cleanup conda env
            if conda_env:
                self.measurer.cleanup_conda_env(conda_env)
            
            # Set final status
            if successful_tests > 0:
                result["status"] = "success"
                result["successful_tests"] = successful_tests
                result["total_tests"] = len(test_list)
                logger.info(f"  ✅ Measured {successful_tests}/{len(test_list)} tests successfully")
            else:
                result["status"] = "all_tests_failed"
                logger.warning(f"  ⚠️ All tests failed")
            
            return result
            
        except Exception as e:
            logger.error(f"  ❌ Measurement error: {e}", exc_info=True)
            result["error"] = str(e)
            return result
            
        finally:
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def measure_strategy(
        self,
        strategy: str,
        skip_completed: bool = True,
        limit: Optional[int] = None,
        repetitions: int = 3
    ) -> Dict:
        """
        Measure all instances for a strategy.
        
        Args:
            strategy: "oracle" or "realistic"
            skip_completed: Skip already measured instances
            limit: Maximum instances to process
            repetitions: Number of measurement repetitions
            
        Returns:
            Summary statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📏 MEASURING STRATEGY: {strategy.upper()}")
        logger.info(f"{'='*70}")
        
        # Find successful experiment results
        results_dir = RESULTS_DIR / f"zs_{strategy}"
        if not results_dir.exists():
            logger.warning(f"Results directory not found: {results_dir}")
            return {"error": "no_results_directory"}
        
        result_files = list(results_dir.glob("*.json"))
        logger.info(f"Found {len(result_files)} experiment results")
        
        # Filter to successful patches
        to_measure = []
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    exp_result = json.load(f)
                
                instance_id = exp_result.get("instance_id")
                status = exp_result.get("status")
                
                # Only measure successful patches
                if status != "success":
                    continue
                
                # Skip if already measured
                if skip_completed and self._get_existing_instance(instance_id, strategy):
                    continue
                
                to_measure.append({
                    "instance_id": instance_id,
                    "llm_response": exp_result.get("llm_response", ""),
                    "model": exp_result.get("model", "unknown")
                })
                
                if limit and len(to_measure) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Error reading {result_file}: {e}")
        
        logger.info(f"Instances to measure: {len(to_measure)}")
        
        # Measure each instance
        stats = {
            "total": len(to_measure),
            "success": 0,
            "failed": 0,
            "timing": []
        }
        
        for idx, item in enumerate(to_measure):
            instance_id = item["instance_id"]
            
            logger.info(f"\n[{idx+1}/{len(to_measure)}] {instance_id}")
            
            start_time = time.time()
            
            measurement = self.measure_instance(
                instance_id=instance_id,
                strategy=strategy,
                llm_response=item["llm_response"],
                repetitions=repetitions
            )
            
            elapsed = time.time() - start_time
            stats["timing"].append(elapsed)
            
            if measurement and measurement.get("status") == "success":
                stats["success"] += 1
                
                # Add to output dataset
                measurement["model"] = item["model"]
                
                # Add human measurements if available
                if instance_id in self.green_data:
                    measurement["human_metrics"] = self.green_data[instance_id].get("green_metrics", {})
                
                # Remove old entry if exists
                self.output_data["instances"] = [
                    i for i in self.output_data["instances"]
                    if not (i.get("instance_id") == instance_id and i.get("strategy") == strategy)
                ]
                
                self.output_data["instances"].append(measurement)
                self._save_output()
                
                logger.info(f"  ✅ Saved ({elapsed:.1f}s)")
            else:
                stats["failed"] += 1
                logger.warning(f"  ❌ Failed ({elapsed:.1f}s)")
        
        return stats
    
    def run(
        self,
        strategies: List[str],
        skip_completed: bool = True,
        limit: Optional[int] = None,
        repetitions: int = 3
    ) -> Dict:
        """
        Run measurements for all specified strategies.
        
        Args:
            strategies: List of strategies to measure
            skip_completed: Skip already measured
            limit: Max instances per strategy
            repetitions: Measurement repetitions
            
        Returns:
            Summary dictionary
        """
        start_time = time.time()
        
        summary = {
            "start_time": datetime.now().isoformat(),
            "strategies": strategies,
            "results": {}
        }
        
        for strategy in strategies:
            stats = self.measure_strategy(
                strategy=strategy,
                skip_completed=skip_completed,
                limit=limit,
                repetitions=repetitions
            )
            summary["results"][strategy] = stats
        
        summary["end_time"] = datetime.now().isoformat()
        summary["total_duration_seconds"] = time.time() - start_time
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 MEASUREMENT SUMMARY")
        logger.info(f"{'='*70}")
        
        for strategy, stats in summary["results"].items():
            logger.info(f"\n{strategy.upper()}:")
            logger.info(f"  Measured: {stats.get('total', 0)}")
            logger.info(f"  Success: {stats.get('success', 0)}")
            logger.info(f"  Failed: {stats.get('failed', 0)}")
        
        logger.info(f"\nOutput saved: {self.output_path}")
        logger.info(f"Total instances in dataset: {len(self.output_data['instances'])}")
        
        return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure LLM-generated patches for green code refactoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Measure oracle strategy patches
  python measure_llm_patches.py --strategy oracle
  
  # Measure realistic strategy patches
  python measure_llm_patches.py --strategy realistic
  
  # Measure both strategies
  python measure_llm_patches.py --strategy both
  
  # Test with 3 instances
  python measure_llm_patches.py --strategy oracle --limit 3
  
  # Use 5 repetitions for more stable results
  python measure_llm_patches.py --strategy oracle --repetitions 5
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
        '--dataset', '-d',
        type=str,
        default=str(DEFAULT_DATASET),
        help='Path to original dataset JSON'
    )
    
    parser.add_argument(
        '--green-dataset', '-g',
        type=str,
        default=str(GREEN_DATASET),
        help='Path to green dataset with human measurements'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for LLM green dataset'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of instances to measure'
    )
    
    parser.add_argument(
        '--repetitions', '-r',
        type=int,
        default=3,
        help='Number of measurement repetitions (default: 3)'
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
    
    # Create logs directory
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    
    # Run measurements
    measurer = LLMPatchMeasurer(
        original_dataset_path=Path(args.dataset),
        green_dataset_path=Path(args.green_dataset) if args.green_dataset else None,
        output_path=Path(args.output) if args.output else None
    )
    
    summary = measurer.run(
        strategies=strategies,
        skip_completed=not args.no_skip,
        limit=args.limit,
        repetitions=args.repetitions
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())