"""
Create the SWE-Perf Green Extended dataset.
Adds green metrics (energy, carbon, efficiency) to the reduced dataset.
Generates separate files based on the number of repetitions (k).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
from collections import defaultdict

# Metrics classification
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

# Aggregation suffixes
AGGREGATIONS = ['mean', 'std', 'min', 'max']

# Fields to remove from original dataset
FIELDS_TO_REMOVE = [
    'problem_statement_oracle',
    'problem_statement_realistic', 
    'duration_changes'
]


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    """Save JSON file with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_aggregated_metrics(test_data: dict) -> Optional[Dict]:
    """
    Extract aggregated metrics (mean, std, min, max) for a test.
    
    Args:
        test_data: Single test measurement data
        
    Returns:
        Dictionary with aggregated metrics or None if invalid
    """
    if not test_data or 'aggregated' not in test_data:
        return None
    
    aggregated = test_data['aggregated']
    result = {}
    
    for metric in ALL_METRICS:
        for agg in AGGREGATIONS:
            key = f"{metric}_{agg}"
            if key in aggregated:
                result[key] = aggregated[key]
    
    return result if result else None


def get_repetition_count(test_data: dict) -> int:
    """Get number of valid repetitions for a test."""
    if not test_data or 'measurements' not in test_data:
        return 0
    
    valid_reps = sum(
        1 for m in test_data['measurements'] 
        if m and m.get('return_code') == 0
    )
    return valid_reps


def process_instance(
    instance: dict,
    measurements_dir: Path
) -> Optional[Dict]:
    """
    Process a single instance and add green metrics.
    
    Args:
        instance: Instance from reduced dataset
        measurements_dir: Path to measurements directory
        
    Returns:
        Instance with green metrics or None if no valid measurements
    """
    instance_id = instance['instance_id']
    meas_file = measurements_dir / instance_id / "measurements.json"
    
    if not meas_file.exists():
        return None
    
    measurements = load_json(meas_file)
    
    # Get efficiency tests from instance
    efficiency_tests = instance.get('efficiency_test', [])
    if not efficiency_tests:
        return None
    
    # Process base and head measurements
    green_metrics = {}
    min_repetitions = float('inf')
    valid_tests = 0
    
    for test_name in efficiency_tests:
        test_metrics = {'base': None, 'head': None}
        test_reps = {'base': 0, 'head': 0}
        
        for commit_type in ['base', 'head']:
            commit_data = measurements.get(f'{commit_type}_measurements', {})
            tests = commit_data.get('tests', [])
            
            # Find matching test
            for i, test in enumerate(tests):
                if test and test.get('test_name') == test_name:
                    # Check if test passed
                    if test.get('status') == 'success':
                        metrics = get_aggregated_metrics(test)
                        if metrics:
                            test_metrics[commit_type] = metrics
                            test_reps[commit_type] = get_repetition_count(test)
                    break
                # Fallback: match by index if test_name not set
                elif i < len(efficiency_tests) and efficiency_tests[i] == test_name:
                    if test and 'aggregated' in test:
                        metrics = get_aggregated_metrics(test)
                        if metrics:
                            test_metrics[commit_type] = metrics
                            test_reps[commit_type] = get_repetition_count(test)
                    break
        
        # Only include test if both base and head have valid metrics
        if test_metrics['base'] and test_metrics['head']:
            green_metrics[test_name] = test_metrics
            min_reps = min(test_reps['base'], test_reps['head'])
            min_repetitions = min(min_repetitions, min_reps)
            valid_tests += 1
    
    if not green_metrics or min_repetitions == float('inf'):
        return None
    
    # Create new instance with green metrics
    new_instance = {}
    
    # Copy fields (excluding ones to remove)
    for key, value in instance.items():
        if key not in FIELDS_TO_REMOVE and key != '_reduction_metadata':
            new_instance[key] = value
    
    # Add green metrics
    new_instance['green_metrics'] = green_metrics
    
    # Add metadata
    new_instance['_green_metadata'] = {
        'valid_tests': valid_tests,
        'total_tests': len(efficiency_tests),
        'repetitions': min_repetitions,
        'green_metrics_count': len(GREEN_METRICS),
        'efficiency_metrics_count': len(EFFICIENCY_METRICS),
        'aggregations': AGGREGATIONS,
        'creation_date': datetime.now().isoformat()
    }
    
    return new_instance, min_repetitions


def create_green_dataset(
    reduced_dataset_path: Path,
    measurements_dir: Path,
    output_dir: Path
):
    """
    Create green extended dataset(s) grouped by repetition count.
    
    Args:
        reduced_dataset_path: Path to reduced dataset JSON
        measurements_dir: Path to measurements directory
        output_dir: Output directory for green datasets
    """
    print("=" * 100)
    print("🌱 CREATING SWE-PERF GREEN EXTENDED DATASET")
    print("=" * 100)
    
    # Load reduced dataset
    print(f"\n📂 Loading reduced dataset: {reduced_dataset_path}")
    reduced_dataset = load_json(reduced_dataset_path)
    print(f"   Found {len(reduced_dataset)} instances")
    
    # Print metrics info
    print(f"\n📊 METRICS CLASSIFICATION:")
    print(f"   🟢 GREEN ({len(GREEN_METRICS)}): {', '.join(GREEN_METRICS)}")
    print(f"   🔵 EFFICIENCY ({len(EFFICIENCY_METRICS)}): {', '.join(EFFICIENCY_METRICS)}")
    print(f"   📈 AGGREGATIONS: {', '.join(AGGREGATIONS)}")
    print(f"   📏 Total metrics per test: {len(ALL_METRICS)} × {len(AGGREGATIONS)} = {len(ALL_METRICS) * len(AGGREGATIONS)}")
    
    # Group instances by repetition count
    instances_by_k = defaultdict(list)
    failed_instances = []
    
    print(f"\n🔄 Processing instances...")
    print("-" * 100)
    
    for i, instance in enumerate(reduced_dataset):
        instance_id = instance['instance_id']
        
        # Progress indicator
        progress = (i + 1) / len(reduced_dataset) * 100
        print(f"\r   [{i+1:3d}/{len(reduced_dataset)}] ({progress:5.1f}%) Processing: {instance_id:<50}", end="", flush=True)
        
        result = process_instance(instance, measurements_dir)
        
        if result:
            new_instance, k = result
            instances_by_k[k].append(new_instance)
        else:
            failed_instances.append(instance_id)
    
    print(f"\r   {'Processing complete!':<100}")
    print("-" * 100)
    
    # Summary
    print(f"\n📈 RESULTS BY REPETITION COUNT (k):")
    print("-" * 60)
    
    total_saved = 0
    saved_files = []
    
    for k in sorted(instances_by_k.keys()):
        instances = instances_by_k[k]
        count = len(instances)
        total_tests = sum(inst['_green_metadata']['valid_tests'] for inst in instances)
        
        # Save dataset for this k
        filename = f"swe_perf_green_k{k}.json"
        output_path = output_dir / filename
        
        dataset_info = {
            'metadata': {
                'name': f'SWE-Perf Green Extended (k={k})',
                'description': f'Green software metrics with {k} measurement repetitions',
                'repetitions': k,
                'instance_count': count,
                'total_tests': total_tests,
                'green_metrics': GREEN_METRICS,
                'efficiency_metrics': EFFICIENCY_METRICS,
                'aggregations': AGGREGATIONS,
                'creation_date': datetime.now().isoformat()
            },
            'instances': instances
        }
        
        save_json(dataset_info, output_path)
        saved_files.append((k, count, total_tests, output_path))
        total_saved += count
        
        print(f"   k={k}: {count:3d} instances, {total_tests:4d} tests → {filename}")
    
    # Failed instances
    if failed_instances:
        print(f"\n⚠️  FAILED INSTANCES ({len(failed_instances)}):")
        for inst_id in failed_instances[:10]:
            print(f"   - {inst_id}")
        if len(failed_instances) > 10:
            print(f"   ... and {len(failed_instances) - 10} more")
    
    # Final summary
    print(f"\n" + "=" * 100)
    print(f"✅ GREEN DATASET CREATION COMPLETE!")
    print("=" * 100)
    print(f"\n📊 SUMMARY:")
    print(f"   Input instances:    {len(reduced_dataset)}")
    print(f"   Output instances:   {total_saved}")
    print(f"   Failed instances:   {len(failed_instances)}")
    print(f"   Success rate:       {total_saved / len(reduced_dataset) * 100:.1f}%")
    
    print(f"\n💾 SAVED FILES:")
    for k, count, tests, path in saved_files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   {path.name}: {count} instances, {tests} tests ({size_mb:.2f} MB)")
    
    print(f"\n📁 Output directory: {output_dir}")
    print("=" * 100)
    
    return instances_by_k, failed_instances


def main():
    # Auto-detect base directory
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent
    
    reduced_dataset_path = base_dir / "data" / "processed" / "swe_perf_reduced.json"
    measurements_dir = base_dir / "data" / "raw" / "measurements"
    output_dir = base_dir / "data" / "processed" / "green"
    
    create_green_dataset(
        reduced_dataset_path=reduced_dataset_path,
        measurements_dir=measurements_dir,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()