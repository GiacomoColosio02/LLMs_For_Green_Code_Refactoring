"""
Script to create the reduced SWE-Perf dataset.
Removes instances without valid tests and removes invalid tests from partial instances.
"""
import json
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime


def load_original_dataset(dataset_path: Path) -> List[Dict]:
    """Load the original SWE-Perf dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def get_valid_tests_for_instance(measurements_dir: Path, instance_id: str, expected_tests: List[str]) -> Set[str]:
    """
    Get the set of valid tests for an instance.
    A test is valid if it passes (return_code == 0) in both base and head commits.
    
    Args:
        measurements_dir: Path to measurements directory
        instance_id: Instance identifier
        expected_tests: List of expected test names
        
    Returns:
        Set of valid test names
    """
    json_file = measurements_dir / instance_id / "measurements.json"
    
    if not json_file.exists():
        return set()
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    base_tests = data.get('base_measurements', {}).get('tests', [])
    head_tests = data.get('head_measurements', {}).get('tests', [])
    
    def get_passed_tests(tests: List, expected: List[str]) -> Set[str]:
        """Get set of test names that passed."""
        passed = set()
        for i, t in enumerate(tests):
            if t and 'measurements' in t:
                # Check if at least one repetition passed
                if any(m.get('return_code') == 0 for m in t['measurements']):
                    if i < len(expected):
                        passed.add(expected[i])
        return passed
    
    base_passed = get_passed_tests(base_tests, expected_tests)
    head_passed = get_passed_tests(head_tests, expected_tests)
    
    # Valid tests = passed in BOTH base and head
    return base_passed & head_passed


def create_reduced_dataset(
    original_dataset_path: Path,
    measurements_dir: Path,
    output_path: Path
) -> Dict:
    """
    Create the reduced dataset by:
    1. Removing instances without any valid tests
    2. Removing invalid tests from instances with partial valid tests
    
    Args:
        original_dataset_path: Path to original SWE-Perf JSON
        measurements_dir: Path to measurements directory
        output_path: Path to save reduced dataset
        
    Returns:
        Statistics dictionary
    """
    # Load original dataset
    original_dataset = load_original_dataset(original_dataset_path)
    
    print("=" * 100)
    print("📊 CREATING SWE-PERF REDUCED DATASET")
    print("=" * 100)
    print(f"\nOriginal dataset: {len(original_dataset)} instances")
    
    # Statistics
    stats = {
        'original_instances': len(original_dataset),
        'original_tests': 0,
        'reduced_instances': 0,
        'reduced_tests': 0,
        'removed_instances': [],
        'instances_with_removed_tests': [],
        'fully_valid_instances': 0,
        'partially_valid_instances': 0,
    }
    
    reduced_dataset = []
    
    for instance in original_dataset:
        instance_id = instance['instance_id']
        expected_tests = instance.get('efficiency_test', [])
        stats['original_tests'] += len(expected_tests)
        
        # Get valid tests for this instance
        valid_tests = get_valid_tests_for_instance(
            measurements_dir, instance_id, expected_tests
        )
        
        if len(valid_tests) == 0:
            # No valid tests - remove instance entirely
            stats['removed_instances'].append({
                'instance_id': instance_id,
                'reason': 'no_valid_tests',
                'original_tests': len(expected_tests)
            })
            continue
        
        # Create reduced instance
        reduced_instance = instance.copy()
        
        # Filter to only valid tests (maintaining original order)
        valid_tests_list = [t for t in expected_tests if t in valid_tests]
        reduced_instance['efficiency_test'] = valid_tests_list
        
        # Track removed tests
        removed_tests = [t for t in expected_tests if t not in valid_tests]
        
        if len(removed_tests) > 0:
            stats['instances_with_removed_tests'].append({
                'instance_id': instance_id,
                'original_tests': len(expected_tests),
                'valid_tests': len(valid_tests_list),
                'removed_tests': removed_tests
            })
            stats['partially_valid_instances'] += 1
        else:
            stats['fully_valid_instances'] += 1
        
        # Add metadata about reduction
        reduced_instance['_reduction_metadata'] = {
            'original_test_count': len(expected_tests),
            'valid_test_count': len(valid_tests_list),
            'removed_test_count': len(removed_tests),
            'reduction_date': datetime.now().isoformat(),
            'reduction_reason': 'full' if len(removed_tests) == 0 else 'partial'
        }
        
        reduced_dataset.append(reduced_instance)
        stats['reduced_instances'] += 1
        stats['reduced_tests'] += len(valid_tests_list)
    
    # Save reduced dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(reduced_dataset, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 100}")
    print("📈 REDUCTION SUMMARY")
    print("=" * 100)
    
    print(f"\n📦 INSTANCES:")
    print(f"   Original:           {stats['original_instances']}")
    print(f"   Reduced:            {stats['reduced_instances']}")
    print(f"   Removed:            {len(stats['removed_instances'])}")
    print(f"   ├─ Fully valid:     {stats['fully_valid_instances']}")
    print(f"   └─ Partially valid: {stats['partially_valid_instances']}")
    
    print(f"\n🧪 TESTS:")
    print(f"   Original:           {stats['original_tests']}")
    print(f"   Valid:              {stats['reduced_tests']}")
    print(f"   Removed:            {stats['original_tests'] - stats['reduced_tests']}")
    
    retention_instances = stats['reduced_instances'] / stats['original_instances'] * 100
    retention_tests = stats['reduced_tests'] / stats['original_tests'] * 100
    print(f"\n📊 RETENTION RATE:")
    print(f"   Instances:          {retention_instances:.1f}%")
    print(f"   Tests:              {retention_tests:.1f}%")
    
    # Print removed instances
    if stats['removed_instances']:
        print(f"\n❌ REMOVED INSTANCES ({len(stats['removed_instances'])}):")
        for removed in stats['removed_instances']:
            print(f"   - {removed['instance_id']} ({removed['original_tests']} tests)")
    
    # Print instances with removed tests (top 10)
    if stats['instances_with_removed_tests']:
        print(f"\n⚠️  INSTANCES WITH REMOVED TESTS ({len(stats['instances_with_removed_tests'])}):")
        sorted_partial = sorted(
            stats['instances_with_removed_tests'],
            key=lambda x: len(x['removed_tests']),
            reverse=True
        )
        for inst in sorted_partial[:10]:
            removed_count = len(inst['removed_tests'])
            print(f"   - {inst['instance_id']}: {inst['valid_tests']}/{inst['original_tests']} valid ({removed_count} removed)")
        if len(sorted_partial) > 10:
            print(f"   ... and {len(sorted_partial) - 10} more")
    
    print(f"\n💾 Reduced dataset saved to: {output_path}")
    print("=" * 100)
    
    # Save statistics as well
    stats_path = output_path.parent / "reduction_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Statistics saved to: {stats_path}")
    
    return stats


def main():
    # Paths
    base_dir = Path("/home/giacomo/LLMs_For_Green_Code_Refactoring")
    original_dataset_path = base_dir / "data" / "original" / "swe_perf_original_20251124.json"
    measurements_dir = base_dir / "data" / "raw" / "measurements"
    output_path = base_dir / "data" / "processed" / "swe_perf_reduced.json"
    
    # Create reduced dataset
    stats = create_reduced_dataset(
        original_dataset_path=original_dataset_path,
        measurements_dir=measurements_dir,
        output_path=output_path
    )
    
    return stats


if __name__ == "__main__":
    main()