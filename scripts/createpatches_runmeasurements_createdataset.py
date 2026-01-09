"""
Complete Pipeline: Create Patches → Run Measurements → Create Dataset

This script runs the entire pipeline for ALL prompt strategies:
- Zero-Shot (Oracle & Realistic)
- Chain-of-Thought (Oracle & Realistic)
- Self-Collaboration (Oracle & Realistic)
- LDB (Oracle & Realistic)

Output: data/processed/green/ALL_STRATEGIES_green_dataset.json

Usage:
    # Run everything
    python scripts/createpatches_runmeasurements_createdataset.py
    
    # Skip patch generation (use existing patches)
    python scripts/createpatches_runmeasurements_createdataset.py --skip-patches
    
    # Skip measurements (use existing measurements)
    python scripts/createpatches_runmeasurements_createdataset.py --skip-measurements
    
    # Only create dataset from existing data
    python scripts/createpatches_runmeasurements_createdataset.py --only-dataset
    
    # Run only specific strategies
    python scripts/createpatches_runmeasurements_createdataset.py --strategies zero_shot cot
"""
import sys
import os
import json
import csv
import subprocess
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Directories
RESULTS_DIR = PROJECT_ROOT / "results"
GREEN_DIR = PROJECT_ROOT / "data" / "processed" / "green"
RAW_MEASUREMENTS_DIR = PROJECT_ROOT / "data" / "raw" / "measurements"
REDUCED_DATASET = PROJECT_ROOT / "data" / "processed" / "swe_perf_reduced_test.json"

# GREEN DATASET WITH BASE/HEAD - This is the source of truth for Base/Head measurements
GREEN_K3_DATASET = GREEN_DIR / "swe_perf_green_k3.json"

# Output files
OUTPUT_JSON = GREEN_DIR / "ALL_STRATEGIES_green_dataset.json"
OUTPUT_CSV = GREEN_DIR / "ALL_STRATEGIES_green_dataset.csv"

# =============================================================================
# ALL 8 CONFIGURATIONS (4 prompt types × 2 strategies)
# =============================================================================
PATCH_CONFIGS = [
    # Zero-Shot
    {"strategy": "oracle", "prompt_type": "zero_shot", "name": "ZS_Oracle", "results_dir": "zs_oracle"},
    {"strategy": "realistic", "prompt_type": "zero_shot", "name": "ZS_Realistic", "results_dir": "zs_realistic"},
    # Chain-of-Thought
    {"strategy": "oracle", "prompt_type": "cot", "name": "CoT_Oracle", "results_dir": "cot_oracle"},
    {"strategy": "realistic", "prompt_type": "cot", "name": "CoT_Realistic", "results_dir": "cot_realistic"},
    # Self-Collaboration (NEW)
    {"strategy": "oracle", "prompt_type": "self_collab", "name": "SC_Oracle", "results_dir": "sc_oracle"},
    {"strategy": "realistic", "prompt_type": "self_collab", "name": "SC_Realistic", "results_dir": "sc_realistic"},
    # LDB (NEW)
    {"strategy": "oracle", "prompt_type": "ldb", "name": "LDB_Oracle", "results_dir": "ldb_oracle"},
    {"strategy": "realistic", "prompt_type": "ldb", "name": "LDB_Realistic", "results_dir": "ldb_realistic"},
]

# Prompt type to clean name mapping (for measurement files)
PROMPT_TYPE_CLEAN = {
    "zero_shot": "ZeroShot",
    "cot": "CoT",
    "self_collab": "SelfCollab",
    "ldb": "LDB"
}

# Metrics
GREEN_METRICS = [
    'total_energy_joules', 'cpu_energy_joules', 'gpu_energy_joules',
    'power_watts', 'carbon_grams', 'energy_efficiency'
]
EFFICIENCY_METRICS = [
    'duration_seconds', 'cpu_usage_mean_percent', 'cpu_usage_peak_percent',
    'ram_usage_mean_mb', 'ram_usage_peak_mb',
    'gpu_temperature_mean_celsius', 'gpu_temperature_peak_celsius'
]
ALL_METRICS = GREEN_METRICS + EFFICIENCY_METRICS

# Key metrics we want to extract
KEY_METRICS_RAW = ['total_energy_joules', 'duration_seconds', 'power_watts', 'carbon_grams', 'cpu_usage_mean_percent']
KEY_METRICS_WITH_SUFFIX = [f"{m}_mean" for m in KEY_METRICS_RAW]


class PipelineRunner:
    """Complete pipeline for patch generation, measurement, and dataset creation."""
    
    def __init__(self, repetitions: int = 5, strategies: Optional[List[str]] = None):
        self.repetitions = repetitions
        self.start_time = time.time()
        
        # Filter configurations if specific strategies requested
        if strategies:
            self.configs = [c for c in PATCH_CONFIGS if c['prompt_type'] in strategies]
        else:
            self.configs = PATCH_CONFIGS
        
    def log(self, msg: str, level: str = "INFO"):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.1f}s] {level}: {msg}")
    
    def log_header(self, title: str):
        print(f"\n{'='*70}")
        print(f"🚀 {title}")
        print(f"{'='*70}\n")
    
    # =========================================================================
    # PHASE 1: PATCH GENERATION
    # =========================================================================
    
    def generate_patches(self) -> Dict[str, int]:
        """Generate patches for all configurations."""
        self.log_header("PHASE 1: GENERATING PATCHES")
        
        results = {}
        
        for config in self.configs:
            self.log(f"Generating {config['name']} patches...")
            
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_batch_experiments.py"),
                "--strategy", config['strategy'],
                "--prompt-type", config['prompt_type']
            ]
            
            try:
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, 
                             capture_output=False)
                
                # Count successful patches
                results_dir = RESULTS_DIR / config['results_dir']
                success_count = self._count_successful_patches(results_dir)
                results[config['name']] = success_count
                self.log(f"  ✅ {config['name']}: {success_count} patches generated")
                
            except subprocess.CalledProcessError as e:
                self.log(f"  ❌ {config['name']} failed: {e}", "ERROR")
                results[config['name']] = 0
        
        return results
    
    def _count_successful_patches(self, results_dir: Path) -> int:
        """Count successful patches in a results directory."""
        if not results_dir.exists():
            return 0
        
        count = 0
        for f in results_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                if data.get('status') == 'success':
                    count += 1
            except:
                pass
        return count
    
    # =========================================================================
    # PHASE 2: MEASUREMENTS
    # =========================================================================
    
    def run_measurements(self) -> Dict[str, int]:
        """Run measurements for all configurations."""
        self.log_header("PHASE 2: RUNNING MEASUREMENTS")
        
        results = {}
        
        for config in self.configs:
            self.log(f"Measuring {config['name']} patches...")
            
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "measure_llm_patches.py"),
                "--strategy", config['strategy'],
                "--prompt-type", config['prompt_type'],
                "--repetitions", str(self.repetitions)
            ]
            
            try:
                subprocess.run(cmd, cwd=PROJECT_ROOT, check=True,
                             capture_output=False)
                
                # Find output file and count instances
                output_file = self._find_measurement_file(config['prompt_type'], config['strategy'])
                if output_file:
                    with open(output_file) as f:
                        data = json.load(f)
                    count = len(data.get('instances', []))
                    results[config['name']] = count
                    self.log(f"  ✅ {config['name']}: {count} instances measured")
                else:
                    results[config['name']] = 0
                    self.log(f"  ⚠️ {config['name']}: output file not found")
                    
            except subprocess.CalledProcessError as e:
                self.log(f"  ❌ {config['name']} measurement failed: {e}", "ERROR")
                results[config['name']] = 0
        
        return results
    
    def _find_measurement_file(self, prompt_type: str, strategy: str) -> Optional[Path]:
        """Find measurement output file."""
        prompt_clean = PROMPT_TYPE_CLEAN.get(prompt_type, prompt_type)
        strategy_clean = strategy.capitalize()
        
        # Pattern: *_SelfCollab_Oracle_k5.json
        pattern = f"*_{prompt_clean}_{strategy_clean}_k*.json"
        matches = list(GREEN_DIR.glob(pattern))
        
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
        return None
    
    # =========================================================================
    # PHASE 3: DATASET CREATION
    # =========================================================================
    
    def create_dataset(self) -> Dict:
        """Create consolidated dataset from all measurements."""
        self.log_header("PHASE 3: CREATING CONSOLIDATED DATASET")
        
        # Load base/head measurements from swe_perf_green_k3.json
        self.log("Loading base/head measurements from swe_perf_green_k3.json...")
        base_head_data = self._load_base_head_from_green_k3()
        self.log(f"  Found {len(base_head_data)} instances with base/head data")
        
        # Load LLM measurements for ALL configurations
        llm_data = {}
        for config in self.configs:
            self.log(f"Loading {config['name']} measurements...")
            measurement_file = self._find_measurement_file(config['prompt_type'], config['strategy'])
            if measurement_file:
                llm_data[config['name']] = self._load_llm_measurements(measurement_file)
                self.log(f"  Found {len(llm_data[config['name']])} instances")
            else:
                llm_data[config['name']] = {}
                self.log(f"  ⚠️ No data found")
        
        # Get all instance IDs that have LLM data
        llm_instances = set()
        for name, data in llm_data.items():
            llm_instances.update(data.keys())
        
        self.log(f"Total instances with LLM data: {len(llm_instances)}")
        
        # Build consolidated dataset
        dataset_instances = []
        
        for instance_id in sorted(llm_instances):
            instance_entry = {
                'instance_id': instance_id,
                'repo': '',
                'metrics': {}
            }
            
            # Add base/head metrics from green_k3
            if instance_id in base_head_data:
                bh = base_head_data[instance_id]
                instance_entry['repo'] = bh.get('repo', '')
                instance_entry['metrics']['Base'] = bh.get('base_metrics', {})
                instance_entry['metrics']['Head'] = bh.get('head_metrics', {})
            
            # Add LLM metrics for all configurations
            for config in self.configs:
                if instance_id in llm_data.get(config['name'], {}):
                    instance_entry['metrics'][f"Head_{config['name']}"] = \
                        llm_data[config['name']][instance_id]
            
            # Only include if we have some metrics
            if instance_entry['metrics']:
                dataset_instances.append(instance_entry)
        
        # Build variants list dynamically
        variants = ['Base', 'Head']
        for config in self.configs:
            variants.append(f"Head_{config['name']}")
        
        # Create final dataset
        dataset = {
            'metadata': {
                'name': 'Green Code Refactoring - All Strategies Comparison Dataset',
                'description': 'Energy measurements comparing Base, Human (Head), and all LLM strategies',
                'variants': variants,
                'strategies': [c['name'] for c in self.configs],
                'metrics': KEY_METRICS_RAW,
                'all_metrics': ALL_METRICS,
                'repetitions': self.repetitions,
                'creation_date': datetime.now().isoformat(),
                'instance_count': len(dataset_instances),
                'base_head_source': 'swe_perf_green_k3.json'
            },
            'instances': dataset_instances
        }
        
        return dataset
    
    def _load_base_head_from_green_k3(self) -> Dict[str, Dict]:
        """Load base and head measurements from swe_perf_green_k3.json."""
        data = {}
        
        if not GREEN_K3_DATASET.exists():
            self.log(f"  ⚠️ Green k3 dataset not found: {GREEN_K3_DATASET}", "WARN")
            return data
        
        try:
            with open(GREEN_K3_DATASET) as f:
                green_k3 = json.load(f)
            
            for instance in green_k3.get('instances', []):
                instance_id = instance.get('instance_id')
                if not instance_id:
                    continue
                
                green_metrics = instance.get('green_metrics', {})
                
                # Aggregate metrics across all tests
                base_metrics = self._aggregate_metrics_from_green(green_metrics, 'base')
                head_metrics = self._aggregate_metrics_from_green(green_metrics, 'head')
                
                data[instance_id] = {
                    'repo': instance.get('repo', ''),
                    'base_metrics': base_metrics,
                    'head_metrics': head_metrics
                }
                
        except Exception as e:
            self.log(f"  ⚠️ Error loading {GREEN_K3_DATASET}: {e}", "WARN")
        
        return data
    
    def _aggregate_metrics_from_green(self, green_metrics: Dict, variant: str) -> Dict[str, float]:
        """Aggregate metrics from green_metrics structure for a specific variant."""
        if not green_metrics:
            return {}
        
        metric_values = defaultdict(list)
        
        for test_name, test_data in green_metrics.items():
            if not isinstance(test_data, dict):
                continue
            
            variant_data = test_data.get(variant, {})
            if not isinstance(variant_data, dict):
                continue
            
            for metric_raw in KEY_METRICS_RAW:
                metric_with_suffix = f"{metric_raw}_mean"
                
                if metric_with_suffix in variant_data:
                    val = variant_data[metric_with_suffix]
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):
                        metric_values[metric_raw].append(val)
                elif metric_raw in variant_data:
                    val = variant_data[metric_raw]
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):
                        metric_values[metric_raw].append(val)
        
        result = {}
        for metric, values in metric_values.items():
            if values:
                result[metric] = sum(values) / len(values)
        
        return result
    
    def _load_llm_measurements(self, measurement_file: Path) -> Dict[str, Dict]:
        """Load LLM measurements from a measurement file."""
        data = {}
        
        try:
            with open(measurement_file) as f:
                raw = json.load(f)
            
            for instance in raw.get('instances', []):
                instance_id = instance.get('instance_id')
                if not instance_id:
                    continue
                
                green_metrics = instance.get('green_metrics', {})
                aggregated = self._aggregate_llm_metrics(green_metrics)
                
                if aggregated:
                    data[instance_id] = aggregated
                
        except Exception as e:
            self.log(f"  ⚠️ Error loading {measurement_file}: {e}", "WARN")
        
        return data
    
    def _aggregate_llm_metrics(self, green_metrics: Dict) -> Dict[str, float]:
        """Aggregate metrics from LLM measurement green_metrics."""
        if not green_metrics:
            return {}
        
        metric_values = defaultdict(list)
        
        for test_name, test_data in green_metrics.items():
            if not isinstance(test_data, dict):
                continue
            
            metrics = test_data.get('head', test_data)
            if not isinstance(metrics, dict):
                continue
            
            for metric_raw in KEY_METRICS_RAW:
                metric_with_suffix = f"{metric_raw}_mean"
                
                if metric_with_suffix in metrics:
                    val = metrics[metric_with_suffix]
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):
                        metric_values[metric_raw].append(val)
                elif metric_raw in metrics:
                    val = metrics[metric_raw]
                    if isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):
                        metric_values[metric_raw].append(val)
        
        result = {}
        for metric, values in metric_values.items():
            if values:
                result[metric] = sum(values) / len(values)
        
        return result
    
    def save_dataset(self, dataset: Dict):
        """Save dataset as JSON and CSV."""
        self.log_header("SAVING DATASET")
        
        GREEN_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(dataset, f, indent=2)
        self.log(f"✅ JSON saved: {OUTPUT_JSON}")
        
        # Save CSV
        self._save_csv(dataset)
        self.log(f"✅ CSV saved: {OUTPUT_CSV}")
    
    def _save_csv(self, dataset: Dict):
        """Save dataset as CSV."""
        instances = dataset.get('instances', [])
        variants = dataset['metadata']['variants']
        metrics = KEY_METRICS_RAW
        
        # Build headers
        headers = ['instance_id', 'repo']
        for variant in variants:
            for metric in metrics:
                headers.append(f"{variant}_{metric}")
        
        # Add comparison columns
        headers.extend([
            'human_energy_reduction_%',
            'best_llm_energy_reduction_%',
            'llm_vs_human_%',
            'best_llm_variant'
        ])
        
        # Build rows
        rows = []
        for inst in instances:
            row = {
                'instance_id': inst['instance_id'],
                'repo': inst.get('repo', '')
            }
            
            # Add metrics for each variant
            for variant in variants:
                variant_metrics = inst.get('metrics', {}).get(variant, {})
                for metric in metrics:
                    col_name = f"{variant}_{metric}"
                    row[col_name] = variant_metrics.get(metric, '')
            
            # Calculate comparisons
            base_energy = inst.get('metrics', {}).get('Base', {}).get('total_energy_joules')
            head_energy = inst.get('metrics', {}).get('Head', {}).get('total_energy_joules')
            
            # Human energy reduction vs base
            if base_energy and head_energy and base_energy > 0:
                row['human_energy_reduction_%'] = round((base_energy - head_energy) / base_energy * 100, 2)
            else:
                row['human_energy_reduction_%'] = ''
            
            # Find best LLM variant
            llm_energies = {}
            for variant in variants:
                if variant.startswith('Head_'):
                    energy = inst.get('metrics', {}).get(variant, {}).get('total_energy_joules')
                    if energy:
                        llm_energies[variant] = energy
            
            if llm_energies:
                best_llm = min(llm_energies, key=llm_energies.get)
                best_llm_energy = llm_energies[best_llm]
                
                if base_energy and base_energy > 0:
                    row['best_llm_energy_reduction_%'] = round((base_energy - best_llm_energy) / base_energy * 100, 2)
                else:
                    row['best_llm_energy_reduction_%'] = ''
                
                if head_energy and head_energy > 0:
                    row['llm_vs_human_%'] = round(best_llm_energy / head_energy * 100, 2)
                else:
                    row['llm_vs_human_%'] = ''
                
                row['best_llm_variant'] = best_llm.replace('Head_', '')
            else:
                row['best_llm_energy_reduction_%'] = ''
                row['llm_vs_human_%'] = ''
                row['best_llm_variant'] = ''
            
            rows.append(row)
        
        # Write CSV
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    
    def print_summary(self, dataset: Dict):
        """Print dataset summary."""
        self.log_header("DATASET SUMMARY")
        
        instances = dataset.get('instances', [])
        variants = dataset['metadata']['variants']
        
        print(f"Total instances: {len(instances)}")
        print(f"Strategies tested: {len(self.configs)}")
        print(f"Variants: {', '.join(variants)}")
        print()
        
        # Coverage per variant
        print("Coverage per variant:")
        for variant in variants:
            count = sum(1 for i in instances if i.get('metrics', {}).get(variant))
            pct = count / len(instances) * 100 if instances else 0
            print(f"  {variant}: {count}/{len(instances)} ({pct:.1f}%)")
        
        # Strategy performance comparison
        print("\n" + "="*50)
        print("STRATEGY PERFORMANCE COMPARISON")
        print("="*50)
        
        strategy_stats = defaultdict(lambda: {'energy_reductions': [], 'vs_human': []})
        
        for inst in instances:
            base_energy = inst.get('metrics', {}).get('Base', {}).get('total_energy_joules')
            head_energy = inst.get('metrics', {}).get('Head', {}).get('total_energy_joules')
            
            if not base_energy or base_energy <= 0:
                continue
            
            for config in self.configs:
                variant_name = f"Head_{config['name']}"
                llm_energy = inst.get('metrics', {}).get(variant_name, {}).get('total_energy_joules')
                
                if llm_energy:
                    # Energy reduction vs base
                    reduction = (base_energy - llm_energy) / base_energy * 100
                    strategy_stats[config['name']]['energy_reductions'].append(reduction)
                    
                    # Vs human (if available)
                    if head_energy and head_energy > 0:
                        vs_human = llm_energy / head_energy * 100
                        strategy_stats[config['name']]['vs_human'].append(vs_human)
        
        # Print stats per strategy
        print("\n📊 Average Energy Reduction vs Base (higher = better):")
        for config in self.configs:
            stats = strategy_stats[config['name']]
            if stats['energy_reductions']:
                avg = sum(stats['energy_reductions']) / len(stats['energy_reductions'])
                count = len(stats['energy_reductions'])
                print(f"  {config['name']}: {avg:+.1f}% ({count} instances)")
            else:
                print(f"  {config['name']}: No data")
        
        print("\n📊 LLM vs Human (100% = same, <100% = LLM better):")
        for config in self.configs:
            stats = strategy_stats[config['name']]
            if stats['vs_human']:
                avg = sum(stats['vs_human']) / len(stats['vs_human'])
                print(f"  {config['name']}: {avg:.1f}%")
        
        # Find best strategy overall
        print("\n" + "-"*50)
        best_strategy = None
        best_reduction = float('-inf')
        for config in self.configs:
            stats = strategy_stats[config['name']]
            if stats['energy_reductions']:
                avg = sum(stats['energy_reductions']) / len(stats['energy_reductions'])
                if avg > best_reduction:
                    best_reduction = avg
                    best_strategy = config['name']
        
        if best_strategy:
            print(f"🏆 Best Strategy: {best_strategy} ({best_reduction:+.1f}% avg energy reduction)")
    
    def run(self, skip_patches: bool = False, skip_measurements: bool = False, only_dataset: bool = False):
        """Run the complete pipeline."""
        
        print("\n" + "="*70)
        print("🔬 COMPLETE PIPELINE: ALL STRATEGIES")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Strategies: {[c['name'] for c in self.configs]}")
        print(f"Repetitions: {self.repetitions}")
        print()
        
        # Phase 1: Generate patches
        if not skip_patches and not only_dataset:
            patch_results = self.generate_patches()
            print(f"\nPatch generation complete: {sum(patch_results.values())} total patches")
        else:
            self.log("Skipping patch generation")
        
        # Phase 2: Run measurements
        if not skip_measurements and not only_dataset:
            measurement_results = self.run_measurements()
            print(f"\nMeasurement complete: {sum(measurement_results.values())} total instances")
        else:
            self.log("Skipping measurements")
        
        # Phase 3: Create dataset
        dataset = self.create_dataset()
        
        # Save dataset
        self.save_dataset(dataset)
        
        # Print summary
        self.print_summary(dataset)
        
        # Final timing
        elapsed = time.time() - self.start_time
        print(f"\n{'='*70}")
        print(f"✅ PIPELINE COMPLETE!")
        print(f"   Total time: {elapsed/60:.1f} minutes")
        print(f"   Output JSON: {OUTPUT_JSON}")
        print(f"   Output CSV: {OUTPUT_CSV}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: Create Patches → Run Measurements → Create Dataset (All Strategies)"
    )
    parser.add_argument('--skip-patches', action='store_true',
                       help='Skip patch generation (use existing patches)')
    parser.add_argument('--skip-measurements', action='store_true',
                       help='Skip measurements (use existing measurements)')
    parser.add_argument('--only-dataset', action='store_true',
                       help='Only create dataset from existing data')
    parser.add_argument('--repetitions', '-k', type=int, default=5,
                       help='Number of measurement repetitions (default: 5)')
    parser.add_argument('--strategies', nargs='+', 
                       choices=['zero_shot', 'cot', 'self_collab', 'ldb'],
                       help='Only run specific strategies (default: all)')
    
    args = parser.parse_args()
    
    runner = PipelineRunner(
        repetitions=args.repetitions,
        strategies=args.strategies
    )
    runner.run(
        skip_patches=args.skip_patches,
        skip_measurements=args.skip_measurements,
        only_dataset=args.only_dataset
    )


if __name__ == "__main__":
    main()