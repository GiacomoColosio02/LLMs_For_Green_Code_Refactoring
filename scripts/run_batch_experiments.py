"""
Batch Experiment Runner for Green Code Refactoring.

Runs experiments for all instances in a dataset.
Simply loops through instances and calls run_experiment.py for each.

Supports all prompt types:
- zero_shot: Zero-Shot prompting
- cot: Chain-of-Thought prompting
- self_collab: Self-Collaboration (multi-turn)
- ldb: LDB iterative refinement

Usage:
    # Run all instances with oracle strategy
    python run_batch_experiments.py --strategy oracle --prompt-type zero_shot
    
    # Run Self-Collaboration
    python run_batch_experiments.py --strategy oracle --prompt-type self_collab
    
    # Run LDB
    python run_batch_experiments.py --strategy realistic --prompt-type ldb
    
    # Run both strategies
    python run_batch_experiments.py --strategy both --prompt-type cot
    
    # Use full dataset (131 instances)
    python run_batch_experiments.py --dataset data/processed/swe_perf_reduced.json
    
    # Limit number of instances (for testing)
    python run_batch_experiments.py --strategy oracle --limit 5
"""
import sys
import os
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the single experiment runner
from scripts.run_experiment import ExperimentRunner

# --- LOGGING ---
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "batch_experiments.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BatchExperiments")

# --- CONSTANTS ---
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "swe_perf_reduced.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Prompt type to directory prefix mapping
PROMPT_TYPE_TO_DIR_PREFIX = {
    "zero_shot": "zs",
    "cot": "cot",
    "self_collab": "sc",
    "ldb": "ldb"
}


# =============================================================================
# BATCH RUNNER
# =============================================================================

class BatchExperimentRunner:
    """
    Runs experiments for multiple instances.
    
    Features:
    - Supports all prompt types: zero_shot, cot, self_collab, ldb
    - Skips already completed instances
    - Tracks progress and statistics
    - Handles errors gracefully (continues with next instance)
    - Generates summary report
    """
    
    def __init__(
        self,
        dataset_path: Path,
        strategies: List[str],
        prompt_type: str = "zero_shot",
        skip_completed: bool = True,
        limit: Optional[int] = None
    ):
        """
        Initialize batch runner.
        
        Args:
            dataset_path: Path to dataset JSON
            strategies: List of strategies to run ("oracle", "realistic", or both)
            prompt_type: "zero_shot", "cot", "self_collab", or "ldb"
            skip_completed: Skip instances that already have results
            limit: Maximum number of instances to process (None = all)
        """
        self.dataset_path = Path(dataset_path)
        self.strategies = strategies
        self.prompt_type = prompt_type.lower()
        self.skip_completed = skip_completed
        self.limit = limit
        
        # Validate prompt type
        if self.prompt_type not in PROMPT_TYPE_TO_DIR_PREFIX:
            raise ValueError(f"Invalid prompt_type: {self.prompt_type}. "
                           f"Valid options: {list(PROMPT_TYPE_TO_DIR_PREFIX.keys())}")
        
        # Load dataset
        self.instances = self._load_dataset()
        
        logger.info(f"BatchExperimentRunner initialized")
        logger.info(f"  Dataset: {self.dataset_path.name}")
        logger.info(f"  Instances: {len(self.instances)}")
        logger.info(f"  Prompt Type: {self.prompt_type.upper()}")
        logger.info(f"  Strategies: {self.strategies}")
        logger.info(f"  Skip completed: {self.skip_completed}")
        logger.info(f"  Limit: {self.limit or 'None'}")
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset from JSON file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        return data.get("instances", [])
    
    def _get_results_dir(self, strategy: str) -> Path:
        """Get results directory for current prompt type and strategy."""
        prefix = PROMPT_TYPE_TO_DIR_PREFIX.get(self.prompt_type, self.prompt_type)
        return RESULTS_DIR / f"{prefix}_{strategy}"
    
    def _is_completed(self, instance_id: str, strategy: str) -> bool:
        """Check if instance already has successful results."""
        result_file = self._get_results_dir(strategy) / f"{instance_id}.json"
        
        if not result_file.exists():
            return False
        
        # Check if result was successful
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            return result.get("status") == "success"
        except:
            return False
    
    def _get_pending_instances(self, strategy: str) -> List[Dict]:
        """Get list of instances that need processing."""
        pending = []
        
        for instance in self.instances:
            instance_id = instance.get("instance_id", instance.get("id", ""))
            if not instance_id:
                continue
            
            if self.skip_completed and self._is_completed(instance_id, strategy):
                continue
            
            pending.append(instance)
            
            # Apply limit
            if self.limit and len(pending) >= self.limit:
                break
        
        return pending
    
    def run(self) -> Dict:
        """
        Run all experiments.
        
        Returns:
            Summary dictionary with statistics
        """
        start_time = time.time()
        
        summary = {
            "start_time": datetime.now().isoformat(),
            "dataset": str(self.dataset_path),
            "prompt_type": self.prompt_type,
            "strategies": self.strategies,
            "results": {}
        }
        
        for strategy in self.strategies:
            logger.info(f"\n{'='*70}")
            logger.info(f"🚀 STARTING BATCH: {self.prompt_type.upper()} - {strategy.upper()}")
            logger.info(f"{'='*70}")
            
            strategy_results = self._run_strategy(strategy)
            summary["results"][strategy] = strategy_results
        
        # Final summary
        total_time = time.time() - start_time
        summary["end_time"] = datetime.now().isoformat()
        summary["total_duration_seconds"] = total_time
        
        self._print_summary(summary)
        self._save_summary(summary)
        
        return summary
    
    def _run_strategy(self, strategy: str) -> Dict:
        """Run all instances for a single strategy."""
        
        # Get pending instances
        pending = self._get_pending_instances(strategy)
        total_in_dataset = len(self.instances)
        already_done = total_in_dataset - len(pending) if self.skip_completed else 0
        
        logger.info(f"  Total instances: {total_in_dataset}")
        logger.info(f"  Already completed: {already_done}")
        logger.info(f"  To process: {len(pending)}")
        
        if not pending:
            logger.info(f"  ✅ All instances already completed!")
            return {
                "total": total_in_dataset,
                "skipped": already_done,
                "processed": 0,
                "success": 0,
                "failed": 0,
                "errors": []
            }
        
        # Initialize runner for this strategy
        runner = ExperimentRunner(
            dataset_path=self.dataset_path,
            strategy=strategy,
            prompt_type=self.prompt_type
        )
        
        # Process instances
        results = {
            "total": total_in_dataset,
            "skipped": already_done,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "patch_failed": 0,
            "errors": [],
            "timing": []
        }
        
        for idx, instance in enumerate(pending):
            instance_id = instance.get("instance_id", instance.get("id", ""))
            
            logger.info(f"\n[{idx+1}/{len(pending)}] Processing: {instance_id}")
            
            instance_start = time.time()
            
            try:
                result = runner.run(instance_id)
                
                instance_time = time.time() - instance_start
                results["timing"].append(instance_time)
                results["processed"] += 1
                
                if result["status"] == "success":
                    results["success"] += 1
                    logger.info(f"  ✅ Success ({instance_time:.1f}s)")
                elif result["status"] == "patch_failed":
                    results["patch_failed"] += 1
                    logger.warning(f"  ⚠️ Patch failed ({instance_time:.1f}s)")
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "instance_id": instance_id,
                        "error": result.get("error", "Unknown error")
                    })
                    logger.error(f"  ❌ Failed: {result.get('error', 'Unknown')}")
                
            except Exception as e:
                instance_time = time.time() - instance_start
                results["processed"] += 1
                results["failed"] += 1
                results["errors"].append({
                    "instance_id": instance_id,
                    "error": str(e)
                })
                logger.error(f"  ❌ Exception: {e}")
            
            # Progress update every 5 instances
            if (idx + 1) % 5 == 0:
                self._print_progress(results, len(pending), idx + 1)
        
        return results
    
    def _print_progress(self, results: Dict, total: int, current: int):
        """Print progress update."""
        success_rate = results["success"] / results["processed"] * 100 if results["processed"] > 0 else 0
        avg_time = sum(results["timing"]) / len(results["timing"]) if results["timing"] else 0
        remaining = total - current
        eta_minutes = (remaining * avg_time) / 60
        
        logger.info(f"\n📊 Progress: {current}/{total} ({current/total*100:.1f}%)")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        logger.info(f"   Avg time: {avg_time:.1f}s")
        logger.info(f"   ETA: {eta_minutes:.1f} minutes")
    
    def _print_summary(self, summary: Dict):
        """Print final summary."""
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 BATCH EXPERIMENT SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Prompt Type: {summary['prompt_type'].upper()}")
        
        for strategy, results in summary["results"].items():
            logger.info(f"\n{strategy.upper()}:")
            logger.info(f"  Total instances: {results['total']}")
            logger.info(f"  Skipped (already done): {results['skipped']}")
            logger.info(f"  Processed: {results['processed']}")
            logger.info(f"  ✅ Success: {results['success']}")
            logger.info(f"  ⚠️ Patch failed: {results.get('patch_failed', 0)}")
            logger.info(f"  ❌ Errors: {results['failed']}")
            
            if results['processed'] > 0:
                success_rate = results['success'] / results['processed'] * 100
                logger.info(f"  Success rate: {success_rate:.1f}%")
            
            if results.get('timing'):
                avg_time = sum(results['timing']) / len(results['timing'])
                logger.info(f"  Avg time per instance: {avg_time:.1f}s")
        
        logger.info(f"\nTotal duration: {summary['total_duration_seconds']/60:.1f} minutes")
        logger.info(f"{'='*70}")
    
    def _save_summary(self, summary: Dict):
        """Save summary to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = PROJECT_ROOT / "logs" / f"batch_{self.prompt_type}_{timestamp}.json"
        
        # Remove timing arrays for cleaner output
        clean_summary = json.loads(json.dumps(summary))
        for strategy in clean_summary.get("results", {}).values():
            if "timing" in strategy:
                del strategy["timing"]
        
        with open(summary_file, 'w') as f:
            json.dump(clean_summary, f, indent=2)
        
        logger.info(f"Summary saved: {summary_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run batch experiments for all instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Zero-Shot oracle strategy for all instances
  python run_batch_experiments.py --strategy oracle --prompt-type zero_shot
  
  # Run Chain-of-Thought realistic strategy
  python run_batch_experiments.py --strategy realistic --prompt-type cot
  
  # Run Self-Collaboration oracle
  python run_batch_experiments.py --strategy oracle --prompt-type self_collab
  
  # Run LDB both strategies
  python run_batch_experiments.py --strategy both --prompt-type ldb
  
  # Use full dataset (131 instances)
  python run_batch_experiments.py --dataset data/processed/swe_perf_reduced.json -p zero_shot
  
  # Test with only 3 instances
  python run_batch_experiments.py --strategy oracle --limit 3
  
  # Force re-run (don't skip completed)
  python run_batch_experiments.py --strategy oracle --no-skip
        """
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['oracle', 'realistic', 'both'],
        default='both',
        help='Strategy to run (default: both)'
    )
    
    parser.add_argument(
        '--prompt-type', '-p',
        type=str,
        choices=['zero_shot', 'cot', 'self_collab', 'ldb'],
        default='zero_shot',
        help='Prompt type (default: zero_shot)'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=str(DEFAULT_DATASET),
        help='Path to dataset JSON (default: swe_perf_reduced.json)'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of instances to process'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip already completed instances'
    )
    
    args = parser.parse_args()
    
    # Determine strategies
    if args.strategy == 'both':
        strategies = ['oracle', 'realistic']
    else:
        strategies = [args.strategy]
    
    # Run batch
    try:
        runner = BatchExperimentRunner(
            dataset_path=Path(args.dataset),
            strategies=strategies,
            prompt_type=args.prompt_type,
            skip_completed=not args.no_skip,
            limit=args.limit
        )
        
        summary = runner.run()
        
        # Return exit code based on results
        total_success = sum(r.get('success', 0) for r in summary['results'].values())
        total_processed = sum(r.get('processed', 0) for r in summary['results'].values())
        
        if total_processed == 0:
            return 0  # Nothing to do is OK
        
        return 0 if total_success > 0 else 1
        
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())