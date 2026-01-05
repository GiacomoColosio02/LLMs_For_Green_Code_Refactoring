"""
Measure all SWE-Perf instances and create the Green Extended dataset.
For each instance: measures with measure_instance.py, then adds to green dataset.
Automatically skips already completed instances.
AUTO-CLEANS reduced dataset: removes failed instances and failed tests.
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

# Repos to skip (known issues)
SKIP_REPOS = []


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


def should_skip_instance(instance_id: str) -> bool:
    """Check if instance should be skipped due to known issues."""
    instance_lower = instance_id.lower()
    for skip_repo in SKIP_REPOS:
        if skip_repo.lower() in instance_lower:
            return True
    return False


def is_instance_in_green_dataset(green_dataset: dict, instance_id: str) -> bool:
    """Check if instance is already in green dataset."""
    instances = green_dataset.get('instances', [])
    return any(inst.get('instance_id') == instance_id for inst in instances)


def is_test_valid(test_data: dict) -> bool:
    """Check if a test has valid measurements (return_code == 0)."""
    if not test_data or 'measurements' not in test_data:
        return False
    return any(m.get('return_code') == 0 for m in test_data['measurements'])


def process_instance_to_green(
    instance: dict,
    measurements_dir: Path
) -> Optional[tuple]:
    """
    Process a single instance and create green metrics entry.
    Uses test_name matching for accuracy.
    
    Returns:
        Tuple of (green_instance, repetitions, valid_test_names) or None if invalid
    """
    instance_id = instance['instance_id']
    meas_file = measurements_dir / instance_id / "measurements.json"
    
    if not meas_file.exists():
        return None
    
    measurements = load_json(meas_file)
    
    efficiency_tests = instance.get('efficiency_test', [])
    if not efficiency_tests:
        return None
    
    # Build lookup dictionaries by test_name
    def build_test_lookup(commit_data):
        lookup = {}
        for test in commit_data.get('tests', []):
            if test and 'test_name' in test:
                lookup[test['test_name']] = test
        return lookup
    
    base_lookup = build_test_lookup(measurements.get('base_measurements', {}))
    head_lookup = build_test_lookup(measurements.get('head_measurements', {}))
    
    green_metrics = {}
    min_repetitions = float('inf')
    valid_tests = 0
    valid_test_names = []  # Track which tests are valid
    
    for test_name in efficiency_tests:
        test_metrics = {'base': None, 'head': None}
        test_reps = {'base': 0, 'head': 0}
        
        for commit_type, lookup in [('base', base_lookup), ('head', head_lookup)]:
            test = lookup.get(test_name)
            
            if test and is_test_valid(test):
                metrics = get_aggregated_metrics(test)
                if metrics:
                    test_metrics[commit_type] = metrics
                    test_reps[commit_type] = get_repetition_count(test)
        
        # Only include test if both base and head have valid metrics
        if test_metrics['base'] and test_metrics['head']:
            green_metrics[test_name] = test_metrics
            min_reps = min(test_reps['base'], test_reps['head'])
            min_repetitions = min(min_repetitions, min_reps)
            valid_tests += 1
            valid_test_names.append(test_name)
    
    if not green_metrics or min_repetitions == float('inf'):
        return None
    
    # Create green instance
    new_instance = {}
    
    for key, value in instance.items():
        if key not in FIELDS_TO_REMOVE and key != '_reduction_metadata':
            new_instance[key] = value
    
    # Update efficiency_test to only include valid tests
    new_instance['efficiency_test'] = valid_test_names
    
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
    
    return new_instance, min_repetitions, valid_test_names


def update_reduced_dataset(
    reduced_path: Path,
    instances_list: List[Dict],
    failed_instances: List[str],
    test_updates: Dict[str, List[str]]
) -> int:
    """
    Update reduced dataset by removing failed instances and invalid tests.
    
    Args:
        reduced_path: Path to reduced dataset
        instances_list: Current instances list
        failed_instances: List of instance_ids that completely failed
        test_updates: Dict mapping instance_id -> list of valid test names
        
    Returns:
        Number of changes made
    """
    changes = 0
    
    # Remove failed instances
    original_count = len(instances_list)
    instances_list[:] = [
        inst for inst in instances_list 
        if inst['instance_id'] not in failed_instances
    ]
    removed_instances = original_count - len(instances_list)
    changes += removed_instances
    
    # Update test lists for instances with partial failures
    for instance in instances_list:
        inst_id = instance['instance_id']
        if inst_id in test_updates:
            valid_tests = test_updates[inst_id]
            original_tests = instance.get('efficiency_test', [])
            if set(valid_tests) != set(original_tests):
                instance['efficiency_test'] = valid_tests
                changes += 1
    
    # Save updated reduced dataset
    if changes > 0:
        save_json(instances_list, reduced_path)
    
    return changes


def measure_and_create_green_dataset(
    reduced_dataset_path: str,
    measurements_dir: str,
    green_output_path: str,
    country_code: str = 'ESP',
    force_remeasure: bool = False
):
    """
    Measure all instances and create green dataset incrementally.
    Auto-cleans reduced dataset by removing failed instances/tests.
    
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
    print(f"⏭️  Skipping repos: {', '.join(SKIP_REPOS)}")
    print(f"🧹 Auto-clean reduced dataset: ENABLED")
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
    skipped_repos = []
    
    # Track for reduced dataset cleanup
    failed_instances = []  # Instances to remove completely
    test_updates = {}  # instance_id -> valid_test_names
    
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
        
        # Check if should skip due to known issues (e.g., astropy)
        if should_skip_instance(instance_id):
            print(f"\n⏭️  [{idx+1}/{total_instances}] {instance_id}")
            print(f"   Skipping (known issue - will fix later)...")
            skipped_repos.append(instance_id)
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
                green_instance, repetitions, valid_test_names = result
                
                # Track valid tests for reduced dataset update
                original_tests = instance.get('efficiency_test', [])
                if len(valid_test_names) < len(original_tests):
                    test_updates[instance_id] = valid_test_names
                    removed_count = len(original_tests) - len(valid_test_names)
                    print(f"   🧹 {removed_count} invalid tests will be removed from reduced dataset")
                
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
                failed_instances.append(instance_id)
                print(f"\n⚠️  Could not extract green metrics")
                print(f"   🧹 Instance will be removed from reduced dataset")
                
        except Exception as e:
            elapsed = time.time() - start_time
            failures.append({
                'instance_id': instance_id,
                'error': str(e),
                'elapsed_seconds': elapsed
            })
            failed_instances.append(instance_id)
            print(f"\n❌ Error: {str(e)[:100]}")
            print(f"   🧹 Instance will be removed from reduced dataset")
            print(f"   Continuing with next instance...")
        
        # Progress summary
        total_processed = len(successes) + len(failures)
        total_done = total_processed + len(skipped) + len(skipped_repos)
        success_rate = len(successes) / total_processed * 100 if total_processed > 0 else 0
        
        print(f"\n📊 Progress: {total_done}/{total_instances}")
        print(f"   ✅ New successes: {len(successes)}")
        print(f"   ❌ Failures: {len(failures)}")
        print(f"   ⭐ Skipped (done): {len(skipped)}")
        print(f"   ⏭️  Skipped (repos): {len(skipped_repos)}")
        print(f"   📈 Success rate: {success_rate:.1f}%")
        
        if successes:
            avg_time = sum(s['elapsed_seconds'] for s in successes) / len(successes)
            remaining = total_instances - total_done
            eta_hours = (remaining * avg_time) / 3600
            print(f"   ⏱️  Avg time: {avg_time:.1f}s | ETA: {eta_hours:.1f}h")
    
    # Step 4: Update reduced dataset (remove failed instances/tests)
    print(f"\n{'='*100}")
    print("🧹 UPDATING REDUCED DATASET")
    print("=" * 100)
    
    if failed_instances or test_updates:
        changes = update_reduced_dataset(
            reduced_path,
            instances_list,
            failed_instances,
            test_updates
        )
        print(f"   Removed {len(failed_instances)} failed instances")
        print(f"   Updated tests for {len(test_updates)} instances")
        print(f"   Total changes: {changes}")
        print(f"   💾 Reduced dataset saved: {reduced_path}")
        print(f"   📦 New instance count: {len(instances_list)}")
    else:
        print("   No changes needed - all instances/tests valid")
    
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
    print(f"   ⭐ Skipped (done): {len(skipped)}")
    print(f"   ⏭️  Skipped (repos): {len(skipped_repos)}")
    print(f"   ⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Reduced dataset cleanup summary
    if failed_instances or test_updates:
        print(f"\n🧹 REDUCED DATASET CLEANUP:")
        print(f"   Instances removed: {len(failed_instances)}")
        print(f"   Instances with tests removed: {len(test_updates)}")
        print(f"   Final instance count: {len(instances_list)}")
    
    # List skipped repos
    if skipped_repos:
        print(f"\n⏭️  SKIPPED REPOS ({len(skipped_repos)}):")
        for inst_id in skipped_repos[:10]:
            print(f"   - {inst_id}")
        if len(skipped_repos) > 10:
            print(f"   ... and {len(skipped_repos) - 10} more")
    
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