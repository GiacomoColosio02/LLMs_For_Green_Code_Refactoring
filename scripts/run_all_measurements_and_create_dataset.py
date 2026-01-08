"""
Complete LLM Patch Measurement Pipeline.

1. Runs all measurements (ZS Oracle, ZS Realistic, CoT Oracle, CoT Realistic)
2. Creates consolidated dataset with Base, Head, and all LLM variants

Output: data/processed/green/consolidated_green_dataset.json
"""
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Directories
GREEN_DIR = PROJECT_ROOT / "data" / "processed" / "green"
ORIGINAL_DATASET = PROJECT_ROOT / "data" / "original" / "swe_perf_original_20251124.json"
REDUCED_DATASET = PROJECT_ROOT / "data" / "processed" / "swe_perf_reduced_test.json"

# Measurement configurations
MEASUREMENTS = [
    {"strategy": "oracle", "prompt_type": "zero_shot", "name": "ZS_Oracle"},
    {"strategy": "realistic", "prompt_type": "zero_shot", "name": "ZS_Realistic"},
    {"strategy": "oracle", "prompt_type": "cot", "name": "CoT_Oracle"},
    {"strategy": "realistic", "prompt_type": "cot", "name": "CoT_Realistic"},
]

# Metrics to extract
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
AGGREGATIONS = ['mean', 'std', 'min', 'max']


def run_measurement(strategy: str, prompt_type: str, repetitions: int = 5) -> bool:
    """Run a single measurement configuration."""
    print(f"\n{'='*70}")
    print(f"🚀 RUNNING: {prompt_type.upper()} - {strategy.upper()}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, 
        str(PROJECT_ROOT / "scripts" / "measure_llm_patches.py"),
        "--strategy", strategy,
        "--prompt-type", prompt_type,
        "--repetitions", str(repetitions)
    ]
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        print(f"✅ {prompt_type} {strategy} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {prompt_type} {strategy} failed: {e}")
        return False


def find_llm_dataset(prompt_type: str, strategy: str) -> Optional[Path]:
    """Find the LLM measurement dataset file."""
    prompt_clean = "ZeroShot" if prompt_type == "zero_shot" else "CoT"
    strategy_clean = strategy.capitalize()
    
    # Look for matching files
    pattern = f"*_{prompt_clean}_{strategy_clean}_k*.json"
    matches = list(GREEN_DIR.glob(pattern))
    
    if matches:
        # Return most recent
        return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def load_base_measurements() -> Dict[str, Dict]:
    """Load base/head measurements from the original green dataset."""
    base_data = {}
    
    # Look for the original measurement files
    measurements_dir = PROJECT_ROOT / "data" / "raw" / "measurements"
    
    if measurements_dir.exists():
        for instance_dir in measurements_dir.iterdir():
            if instance_dir.is_dir():
                measurement_file = instance_dir / "measurements.json"
                if measurement_file.exists():
                    try:
                        with open(measurement_file) as f:
                            data = json.load(f)
                        instance_id = data.get('instance_id', instance_dir.name)
                        base_data[instance_id] = data
                    except Exception as e:
                        print(f"  ⚠️ Error loading {measurement_file}: {e}")
    
    # Also check processed green datasets
    for green_file in GREEN_DIR.glob("*.json"):
        if "consolidated" not in green_file.name and "LLM" not in str(green_file):
            try:
                with open(green_file) as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'instances' in data:
                    for inst in data['instances']:
                        inst_id = inst.get('instance_id')
                        if inst_id and inst_id not in base_data:
                            base_data[inst_id] = inst
            except:
                pass
    
    return base_data


def load_llm_dataset(path: Path) -> Dict[str, Dict]:
    """Load LLM measurement dataset."""
    if not path or not path.exists():
        return {}
    
    try:
        with open(path) as f:
            data = json.load(f)
        
        instances = data.get('instances', [])
        return {inst['instance_id']: inst for inst in instances}
    except Exception as e:
        print(f"  ⚠️ Error loading {path}: {e}")
        return {}


def extract_metrics_from_tests(tests_data: Dict) -> Dict[str, Dict]:
    """Extract aggregated metrics from test results."""
    if not tests_data:
        return {}
    
    # Aggregate across all tests
    aggregated = {}
    
    for test_name, test_result in tests_data.items():
        # Handle different structures
        if isinstance(test_result, dict):
            # Could be {'head': {...}} or direct metrics
            metrics = test_result.get('head', test_result)
            
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if metric in ALL_METRICS or any(agg in metric for agg in AGGREGATIONS):
                        if metric not in aggregated:
                            aggregated[metric] = []
                        if isinstance(value, (int, float)):
                            aggregated[metric].append(value)
    
    # Compute mean across tests
    result = {}
    for metric, values in aggregated.items():
        if values:
            result[metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
    
    return result


def create_consolidated_dataset(
    base_data: Dict[str, Dict],
    llm_datasets: Dict[str, Dict[str, Dict]]
) -> Dict:
    """Create consolidated dataset with all measurements."""
    
    # Get all instance IDs
    all_instances = set(base_data.keys())
    for llm_name, llm_data in llm_datasets.items():
        all_instances.update(llm_data.keys())
    
    print(f"\n📊 Creating consolidated dataset...")
    print(f"   Total instances: {len(all_instances)}")
    
    consolidated_instances = []
    
    for instance_id in sorted(all_instances):
        instance_entry = {
            'instance_id': instance_id,
            'measurements': {}
        }
        
        # Add base measurements if available
        if instance_id in base_data:
            base_inst = base_data[instance_id]
            
            # Extract base commit metrics
            base_measurements = base_inst.get('base_measurements', {})
            if base_measurements and base_measurements.get('tests'):
                base_metrics = extract_metrics_from_tests(
                    {t['test_name']: t for t in base_measurements['tests'] if isinstance(t, dict)}
                )
                instance_entry['measurements']['Base'] = base_metrics
            
            # Extract head commit metrics (original optimization)
            head_measurements = base_inst.get('head_measurements', {})
            if head_measurements and head_measurements.get('tests'):
                head_metrics = extract_metrics_from_tests(
                    {t['test_name']: t for t in head_measurements['tests'] if isinstance(t, dict)}
                )
                instance_entry['measurements']['Head'] = head_metrics
            
            # Copy metadata
            instance_entry['repo'] = base_inst.get('repo', '')
            instance_entry['efficiency_test'] = base_inst.get('efficiency_test', [])
        
        # Add LLM measurements
        for llm_name, llm_data in llm_datasets.items():
            if instance_id in llm_data:
                llm_inst = llm_data[instance_id]
                green_metrics = llm_inst.get('green_metrics', {})
                
                if green_metrics:
                    llm_metrics = extract_metrics_from_tests(green_metrics)
                    instance_entry['measurements'][f'Head_{llm_name}'] = llm_metrics
                    
                    # Also copy repo info if not present
                    if 'repo' not in instance_entry:
                        instance_entry['repo'] = llm_inst.get('repo', '')
                    if 'efficiency_test' not in instance_entry:
                        instance_entry['efficiency_test'] = llm_inst.get('efficiency_test', [])
        
        # Only add if we have some measurements
        if instance_entry['measurements']:
            consolidated_instances.append(instance_entry)
    
    # Create final dataset
    consolidated = {
        'metadata': {
            'name': 'Consolidated Green Dataset - LLM Patches',
            'description': 'Energy measurements for Base, Head (original), and LLM-generated patches',
            'variants': ['Base', 'Head'] + [f'Head_{m["name"]}' for m in MEASUREMENTS],
            'green_metrics': GREEN_METRICS,
            'efficiency_metrics': EFFICIENCY_METRICS,
            'aggregations': AGGREGATIONS,
            'creation_date': datetime.now().isoformat(),
            'instance_count': len(consolidated_instances)
        },
        'instances': consolidated_instances
    }
    
    return consolidated


def print_dataset_summary(dataset: Dict):
    """Print summary of the consolidated dataset."""
    print(f"\n{'='*70}")
    print("📊 CONSOLIDATED DATASET SUMMARY")
    print(f"{'='*70}")
    
    instances = dataset.get('instances', [])
    variants = dataset['metadata']['variants']
    
    print(f"\nTotal instances: {len(instances)}")
    print(f"Variants: {', '.join(variants)}")
    
    # Count coverage per variant
    coverage = {v: 0 for v in variants}
    for inst in instances:
        for variant in variants:
            if variant in inst.get('measurements', {}):
                coverage[variant] += 1
    
    print(f"\nCoverage per variant:")
    for variant, count in coverage.items():
        pct = count / len(instances) * 100 if instances else 0
        print(f"  {variant}: {count}/{len(instances)} ({pct:.1f}%)")
    
    # Sample metrics
    if instances:
        print(f"\nSample instance: {instances[0]['instance_id']}")
        for variant, metrics in instances[0].get('measurements', {}).items():
            if metrics:
                sample_metric = list(metrics.keys())[0] if metrics else "N/A"
                print(f"  {variant}: {len(metrics)} metrics (e.g., {sample_metric})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all measurements and create consolidated dataset")
    parser.add_argument('--skip-measurements', action='store_true', help='Skip measurements, only create dataset')
    parser.add_argument('--repetitions', '-k', type=int, default=5, help='Repetitions per test')
    parser.add_argument('--only', type=str, choices=['zs_oracle', 'zs_realistic', 'cot_oracle', 'cot_realistic'],
                        help='Run only specific measurement')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Step 1: Run measurements
    if not args.skip_measurements:
        print("\n" + "="*70)
        print("🔬 PHASE 1: RUNNING MEASUREMENTS")
        print("="*70)
        
        measurements_to_run = MEASUREMENTS
        if args.only:
            measurements_to_run = [m for m in MEASUREMENTS 
                                   if f"{m['prompt_type']}_{m['strategy']}" == args.only.replace('-', '_')]
        
        for config in measurements_to_run:
            success = run_measurement(
                strategy=config['strategy'],
                prompt_type=config['prompt_type'],
                repetitions=args.repetitions
            )
            if not success:
                print(f"⚠️ {config['name']} failed, continuing...")
    
    # Step 2: Load all datasets
    print("\n" + "="*70)
    print("📂 PHASE 2: LOADING DATASETS")
    print("="*70)
    
    # Load base/head measurements
    print("\nLoading base measurements...")
    base_data = load_base_measurements()
    print(f"  Found {len(base_data)} base instances")
    
    # Load LLM datasets
    llm_datasets = {}
    for config in MEASUREMENTS:
        print(f"\nLooking for {config['name']}...")
        dataset_path = find_llm_dataset(config['prompt_type'], config['strategy'])
        if dataset_path:
            print(f"  Found: {dataset_path.name}")
            llm_datasets[config['name']] = load_llm_dataset(dataset_path)
            print(f"  Loaded {len(llm_datasets[config['name']])} instances")
        else:
            print(f"  ⚠️ Not found")
            llm_datasets[config['name']] = {}
    
    # Step 3: Create consolidated dataset
    print("\n" + "="*70)
    print("🔧 PHASE 3: CREATING CONSOLIDATED DATASET")
    print("="*70)
    
    consolidated = create_consolidated_dataset(base_data, llm_datasets)
    
    # Save consolidated dataset
    output_path = GREEN_DIR / "consolidated_green_dataset.json"
    GREEN_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    print(f"\n💾 Saved: {output_path}")
    
    # Print summary
    print_dataset_summary(consolidated)
    
    # Also create a CSV version for easy analysis
    create_csv_export(consolidated, GREEN_DIR / "consolidated_green_dataset.csv")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Complete! Total time: {elapsed/60:.1f} minutes")


def create_csv_export(dataset: Dict, output_path: Path):
    """Create CSV export for easy analysis."""
    import csv
    
    instances = dataset.get('instances', [])
    if not instances:
        return
    
    # Get all variants and metrics
    variants = dataset['metadata']['variants']
    
    # Key metrics to export
    key_metrics = [
        'total_energy_joules', 'duration_seconds', 'power_watts',
        'cpu_usage_mean_percent', 'carbon_grams'
    ]
    
    # Build CSV rows
    rows = []
    headers = ['instance_id', 'repo']
    
    for variant in variants:
        for metric in key_metrics:
            headers.append(f"{variant}_{metric}")
    
    for inst in instances:
        row = {
            'instance_id': inst['instance_id'],
            'repo': inst.get('repo', '')
        }
        
        for variant in variants:
            measurements = inst.get('measurements', {}).get(variant, {})
            for metric in key_metrics:
                metric_data = measurements.get(metric, {})
                value = metric_data.get('mean', '') if isinstance(metric_data, dict) else ''
                row[f"{variant}_{metric}"] = value
        
        rows.append(row)
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"📄 CSV exported: {output_path}")


if __name__ == "__main__":
    main()
