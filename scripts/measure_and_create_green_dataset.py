"""
Measure all SWE-Perf instances and create the Green Extended dataset.
For each instance: measures with measure_instance.py, then adds to green dataset.
Automatically skips already completed instances.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from measure_instance import SWEPerfMeasurer

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
AGGREGATIONS = ['mean', 'std', 'min', 'max']

# Fields to remove from original dataset
FIELDS_TO_REMOVE = [
    'problem_statement_oracle',
    'problem_statement_realistic', 
    'duration_changes'
]


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


def get_aggregated_metrics(test_data: dict) -> Optional[Dict]:
    """Extract aggregated metrics (mean, std, min, max) for a test."""
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


def is_instance_in_green_dataset(green_dataset: dict, instance_id: str) -> bool:
    """Check if instance is already in green dataset."""
    instances = green_dataset.get('instances', [])
    return any(inst.get('instance_id') == instance_id for inst in instances)


def process_instance_to_green(
    instance: dict,
    measurements_dir: Path
) -> Optional[tuple]:
    """
    Process a single instance and create green metrics entry.
    
    Returns:
        Tuple of (green_instance, repetitions) or None if invalid
    """
    instance_id = instance['instance_id']
    meas_file = measurements_dir / instance_id / "measurements.json"
    
    if not meas_file.exists():
        return None
    
    measurements = load_json(meas_file)
    
    efficiency_tests = instance.get('efficiency_test', [])
    if not efficiency_tests:
        return None
    
    green_metrics = {}
    min_repetitions = float('inf')
    valid_tests = 0
    
    for idx, test_name in enumerate(efficiency_tests):
        test_metrics = {'base': None, 'head': None}
        test_reps = {'base': 0, 'head': 0}
        
        for commit_type in ['base', 'head']:
            commit_data = measurements.get(f'{commit_type}_measurements', {})
            tests = commit_data.get('tests', [])
            
            # Match by index (more reliable)
            if idx < len(tests):
                test = tests[idx]
                if test and test.get('status') == 'success':
                    metrics = get_aggregated_metrics(test)
                    if metrics:
                        test_metrics[commit_type] = metrics
                        test_reps[commit_type] = get_repetition_count(test)
        
        if test_metrics['base'] and test_metrics['head']:
            green_metrics[test_name] = test_metrics
            min_reps = min(test_reps['base'], test_reps['head'])
            min_repetitions = min(min_repetitions, min_reps)
            valid_tests += 1
    
    if not green_metrics or min_repetitions == float('inf'):
        return None
    
    # Create green instance
    new_instance = {}
    
    for key, value in instance.items():
        if key not in FIELDS_TO_REMOVE and key != '_reduction_metadata':
            new_instance[key] = value
    
    new_instance['green_metrics'] = green_metrics
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


def measure_and_create_green_dataset(
    reduced_dataset_path: str,
    measurements_dir: str,
    green_output_path: str,
    country_code: str = 'ESP',
    force_remeasure: bool = False
):
    """
    Measure all instances and create green dataset incrementally.
    
    Args:
        reduced_dataset_path: Path to reduced dataset JSON
        measurements_dir: Directory for raw measurements
        green_output_path: Path to green dataset JSON output
        country_code: ISO country code for carbon
        force_remeasure: If True, remeasure even completed instances
    """
    print("=" * 100)
    print("🌱 SWE-PERF GREEN EXTENDED - MEASURE & CREATE DATASET")
    print("=" * 100)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌍 Country: {country_code}")
    print(f"📂 Measurements: {measurements_dir}")
    print(f"📄 Green output: {green_output_path}")
    print(f"🔄 Force remeasure: {force_remeasure}")
    print("=" * 100)
    
    # Paths
    reduced_path = Path(reduced_dataset_path)
    meas_dir = Path(measurements_dir)
    green_path = Path(green_output_path)
    
    # Load reduced dataset
    print(f"\n📂 Loading reduced dataset...")
    reduced_dataset = load_json(reduced_path)
    if isinstance(reduced_dataset, list):
        instances_list = reduced_dataset
    else:
        instances_list = reduced_dataset.get('instances', reduced_dataset)
    print(f"   Found {len(instances_list)} instances")
    
    # Load existing green dataset (to resume)
    print(f"\n📂 Loading existing green dataset...")
    if green_path.exists():
        green_dataset = load_json(green_path)
        existing_count = len(green_dataset.get('instances', []))
        print(f"   Found {existing_count} existing instances")
    else:
        green_dataset = {
            'metadata': {
                'name': 'SWE-Perf Green Extended',
                'description': 'Green software metrics for SWE-Perf instances',
                'green_metrics': GREEN_METRICS,
                'efficiency_metrics': EFFICIENCY_METRICS,
                'aggregations': AGGREGATIONS,
                'creation_date': datetime.now().isoformat()
            },
            'instances': []
        }
        print(f"   Creating new green dataset")
    
    # Initialize measurer
    print(f"\n📂 Initializing measurer...")
    
    # Find original dataset path for measurer
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent
    original_dataset_path = base_dir / "data" / "original" / "swe_perf_original_20251124.json"
    
    measurer = SWEPerfMeasurer(
        dataset_path=str(original_dataset_path),
        country_code=country_code
    )
    
    # Print metrics info
    print(f"\n📊 METRICS CLASSIFICATION:")
    print(f"   🟢 GREEN ({len(GREEN_METRICS)}): {', '.join(GREEN_METRICS)}")
    print(f"   🔵 EFFICIENCY ({len(EFFICIENCY_METRICS)}): {', '.join(EFFICIENCY_METRICS)}")
    print(f"   📈 AGGREGATIONS: {', '.join(AGGREGATIONS)}")
    print(f"   📏 Total metrics per test: {len(ALL_METRICS)} × {len(AGGREGATIONS)} = {len(ALL_METRICS) * len(AGGREGATIONS)}")
    
    # Track progress
    successes = []
    failures = []
    skipped = []
    
    total_instances = len(instances_list)
    
    print(f"\n🔄 Processing {total_instances} instances...")
    print("=" * 100)
    
    for idx, instance in enumerate(instances_list):
        instance_id = instance['instance_id']
        
        # Check if already in green dataset
        if not force_remeasure and is_instance_in_green_dataset(green_dataset, instance_id):
            print(f"\n⭐ [{idx+1}/{total_instances}] {instance_id}")
            print(f"   Already in green dataset, skipping...")
            skipped.append(instance_id)
            continue
        
        print(f"\n{'='*100}")
        print(f"🔬 [{idx+1}/{total_instances}] {instance_id}")
        print(f"{'='*100}")
        
        start_time = time.time()
        
        try:
            # Step 1: Measure instance
            print(f"\n📏 Step 1: Measuring instance...")
            measurer.measure_instance(
                instance_id=instance_id,
                output_dir=str(meas_dir)
            )
            
            # Step 2: Convert to green format
            print(f"\n🌱 Step 2: Converting to green format...")
            result = process_instance_to_green(instance, meas_dir)
            
            if result:
                green_instance, repetitions = result
                
                # Step 3: Add to green dataset
                print(f"\n💾 Step 3: Adding to green dataset...")
                
                # Remove if exists (for force_remeasure)
                green_dataset['instances'] = [
                    inst for inst in green_dataset['instances']
                    if inst.get('instance_id') != instance_id
                ]
                
                # Add new instance
                green_dataset['instances'].append(green_instance)
                
                # Update metadata
                green_dataset['metadata']['last_updated'] = datetime.now().isoformat()
                green_dataset['metadata']['instance_count'] = len(green_dataset['instances'])
                
                # Save immediately (incremental)
                save_json(green_dataset, green_path)
                
                elapsed = time.time() - start_time
                valid_tests = green_instance['_green_metadata']['valid_tests']
                
                successes.append({
                    'instance_id': instance_id,
                    'valid_tests': valid_tests,
                    'repetitions': repetitions,
                    'elapsed_seconds': elapsed
                })
                
                print(f"\n✅ Success! {valid_tests} tests, k={repetitions}, {elapsed:.1f}s")
                print(f"   Green dataset now has {len(green_dataset['instances'])} instances")
            else:
                elapsed = time.time() - start_time
                failures.append({
                    'instance_id': instance_id,
                    'error': 'Could not extract green metrics',
                    'elapsed_seconds': elapsed
                })
                print(f"\n⚠️  Could not extract green metrics")
                
        except Exception as e:
            elapsed = time.time() - start_time
            failures.append({
                'instance_id': instance_id,
                'error': str(e),
                'elapsed_seconds': elapsed
            })
            print(f"\n❌ Error: {str(e)[:100]}")
            print(f"   Continuing with next instance...")
        
        # Progress summary
        total_processed = len(successes) + len(failures)
        total_done = total_processed + len(skipped)
        success_rate = len(successes) / total_processed * 100 if total_processed > 0 else 0
        
        print(f"\n📊 Progress: {total_done}/{total_instances}")
        print(f"   ✅ New successes: {len(successes)}")
        print(f"   ❌ Failures: {len(failures)}")
        print(f"   ⭐ Skipped: {len(skipped)}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        
        if successes:
            avg_time = sum(s['elapsed_seconds'] for s in successes) / len(successes)
            remaining = total_instances - total_done
            eta_hours = (remaining * avg_time) / 3600
            print(f"   ⏱️  Avg time: {avg_time:.1f}s | ETA: {eta_hours:.1f}h")
    
    # Final summary
    print("\n" + "=" * 100)
    print("🎉 MEASUREMENT & GREEN DATASET CREATION COMPLETE!")
    print("=" * 100)
    
    total_tests = sum(
        inst['_green_metadata']['valid_tests'] 
        for inst in green_dataset['instances']
    )
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   📁 Green dataset: {green_path}")
    print(f"   📦 Total instances: {len(green_dataset['instances'])}")
    print(f"   🧪 Total tests: {total_tests}")
    print(f"   ✅ New successes: {len(successes)}")
    print(f"   ❌ Failures: {len(failures)}")
    print(f"   ⭐ Skipped: {len(skipped)}")
    print(f"   ⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save failure log
    if failures:
        failures_path = green_path.parent / "green_failures.json"
        save_json(failures, failures_path)
        print(f"\n⚠️  Failures saved to: {failures_path}")
        print(f"   Failed instances:")
        for f in failures[:10]:
            print(f"   - {f['instance_id']}: {f['error'][:50]}")
        if len(failures) > 10:
            print(f"   ... and {len(failures) - 10} more")
    
    print("=" * 100)
    
    return green_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Measure all SWE-Perf instances and create Green Extended dataset"
    )
    
    # Auto-detect paths
    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent
    
    default_reduced = base_dir / "data" / "processed" / "swe_perf_reduced.json"
    default_measurements = base_dir / "data" / "raw" / "measurements"
    default_green = base_dir / "data" / "processed" / "green" / "swe_perf_green.json"
    
    parser.add_argument(
        '--reduced-dataset',
        type=str,
        default=str(default_reduced),
        help='Path to reduced dataset JSON'
    )
    parser.add_argument(
        '--measurements-dir',
        type=str,
        default=str(default_measurements),
        help='Directory for raw measurements'
    )
    parser.add_argument(
        '--green-output',
        type=str,
        default=str(default_green),
        help='Path to green dataset JSON output'
    )
    parser.add_argument(
        '--country',
        type=str,
        default='ESP',
        help='ISO country code for carbon intensity'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force remeasure even already completed instances'
    )
    
    args = parser.parse_args()
    
    measure_and_create_green_dataset(
        reduced_dataset_path=args.reduced_dataset,
        measurements_dir=args.measurements_dir,
        green_output_path=args.green_output,
        country_code=args.country,
        force_remeasure=args.force
    )


if __name__ == "__main__":
    main()