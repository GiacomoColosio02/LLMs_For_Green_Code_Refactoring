#!/usr/bin/env python3
"""
Verify GSMM measurements for SWE-Perf instances.

This script validates that all measurement JSON files contain the required
13 GSMM metrics (6 GREEN + 7 EFFICIENCY) in the correct structure.

Expected JSON structure:
{
    "instance_id": "...",
    "base_measurements": {
        "tests": [
            {
                "measurements": [
                    {
                        "gpu_energy_joules": ...,
                        "cpu_energy_joules": ...,
                        // ... 13 total metrics
                    }
                ]
            }
        ]
    },
    "head_measurements": {
        "tests": [...]
    }
}

Usage:
    python scripts/verify_measurements.py
    python scripts/verify_measurements.py --output-dir data/raw/measurements
    python scripts/verify_measurements.py --verbose
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


# Required GSMM metrics (13 total)
REQUIRED_GREEN_METRICS = [
    'gpu_energy_joules',
    'cpu_energy_joules', 
    'total_energy_joules',
    'power_watts',
    'carbon_grams',
    'energy_efficiency'
]

REQUIRED_EFFICIENCY_METRICS = [
    'duration_seconds',
    'cpu_usage_mean_percent',
    'cpu_usage_peak_percent',
    'ram_usage_peak_mb',
    'gpu_usage_mean_percent',
    'gpu_usage_peak_percent',
    'gpu_memory_peak_mb'
]

ALL_REQUIRED_METRICS = REQUIRED_GREEN_METRICS + REQUIRED_EFFICIENCY_METRICS


class MeasurementVerifier:
    """Verify GSMM measurement JSON files."""
    
    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Results
        self.valid_instances = []
        self.incomplete_instances = []
        self.invalid_structure = []
        self.missing_files = []
        
    def verify_metrics_in_measurement(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """
        Verify a single measurement dict has all 13 metrics.
        
        Returns:
            (is_complete, missing_metrics)
        """
        missing = [m for m in ALL_REQUIRED_METRICS if m not in metrics]
        return len(missing) == 0, missing
    
    def verify_commit_measurements(self, commit_data: Dict, commit_type: str) -> Tuple[bool, str]:
        """
        Verify measurements for a commit (base or head).
        
        Returns:
            (is_valid, error_message)
        """
        # Check structure
        if 'tests' not in commit_data:
            return False, f"{commit_type}: missing 'tests' key"
        
        if not isinstance(commit_data['tests'], list):
            return False, f"{commit_type}: 'tests' is not a list"
        
        if len(commit_data['tests']) == 0:
            return False, f"{commit_type}: 'tests' is empty"
        
        # Check first test has measurements
        first_test = commit_data['tests'][0]
        if 'measurements' not in first_test:
            return False, f"{commit_type}: test missing 'measurements' key"
        
        if not isinstance(first_test['measurements'], list):
            return False, f"{commit_type}: 'measurements' is not a list"
        
        if len(first_test['measurements']) == 0:
            return False, f"{commit_type}: 'measurements' is empty"
        
        # Verify metrics in first measurement
        first_measurement = first_test['measurements'][0]
        is_complete, missing = self.verify_metrics_in_measurement(first_measurement)
        
        if not is_complete:
            return False, f"{commit_type}: missing metrics: {missing}"
        
        return True, ""
    
    def verify_instance(self, json_file: Path) -> Dict:
        """
        Verify a single instance JSON file.
        
        Returns:
            Dict with verification results
        """
        instance_id = json_file.parent.name
        
        result = {
            'instance_id': instance_id,
            'valid': False,
            'error': None,
            'base_valid': False,
            'head_valid': False
        }
        
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            # Check top-level structure
            if 'base_measurements' not in data:
                result['error'] = "Missing 'base_measurements' key"
                return result
            
            if 'head_measurements' not in data:
                result['error'] = "Missing 'head_measurements' key"
                return result
            
            # Verify base measurements
            base_valid, base_error = self.verify_commit_measurements(
                data['base_measurements'], 'BASE'
            )
            result['base_valid'] = base_valid
            if not base_valid:
                result['error'] = base_error
                return result
            
            # Verify head measurements
            head_valid, head_error = self.verify_commit_measurements(
                data['head_measurements'], 'HEAD'
            )
            result['head_valid'] = head_valid
            if not head_valid:
                result['error'] = head_error
                return result
            
            # All checks passed
            result['valid'] = True
            return result
            
        except json.JSONDecodeError as e:
            result['error'] = f"Invalid JSON: {e}"
            return result
        except Exception as e:
            result['error'] = f"Error reading file: {e}"
            return result
    
    def verify_all(self) -> Dict:
        """
        Verify all instances in output directory.
        
        Returns:
            Summary dict
        """
        print("=" * 80)
        print("🔍 GSMM MEASUREMENT VERIFICATION")
        print("=" * 80)
        print(f"Directory: {self.output_dir}")
        print()
        
        # Find all measurement JSON files
        json_files = list(self.output_dir.glob("*/measurements.json"))
        
        if len(json_files) == 0:
            print("❌ No measurement files found!")
            return {
                'total': 0,
                'valid': 0,
                'invalid': 0
            }
        
        print(f"Found {len(json_files)} measurement files\n")
        
        # Verify each instance
        results = []
        for json_file in sorted(json_files):
            result = self.verify_instance(json_file)
            results.append(result)
            
            if self.verbose:
                if result['valid']:
                    print(f"✅ {result['instance_id']}")
                else:
                    print(f"❌ {result['instance_id']}: {result['error']}")
        
        # Categorize results
        self.valid_instances = [r['instance_id'] for r in results if r['valid']]
        self.incomplete_instances = [r for r in results if not r['valid']]
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 80)
        
        total = len(json_files)
        valid = len(self.valid_instances)
        invalid = len(self.incomplete_instances)
        
        print(f"Total instances: {total}")
        print(f"✅ Valid (13 metrics): {valid} ({valid/total*100:.1f}%)")
        print(f"❌ Invalid/Incomplete: {invalid} ({invalid/total*100:.1f}%)")
        
        # Show invalid instances
        if invalid > 0:
            print(f"\n❌ INVALID INSTANCES ({invalid}):")
            for result in self.incomplete_instances[:20]:  # Show first 20
                print(f"   {result['instance_id']}: {result['error']}")
            if invalid > 20:
                print(f"   ... and {invalid - 20} more")
        
        # Show valid instances (if not too many)
        if valid > 0 and valid <= 10:
            print(f"\n✅ VALID INSTANCES ({valid}):")
            for instance_id in self.valid_instances:
                print(f"   {instance_id}")
        elif valid > 10:
            print(f"\n✅ Valid instances: {valid} (use --verbose to see all)")
        
        print("=" * 80)
        
        return {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'valid_instances': self.valid_instances,
            'invalid_instances': [r['instance_id'] for r in self.incomplete_instances]
        }


def main():
    parser = argparse.ArgumentParser(
        description='Verify GSMM measurement JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify default directory
  python scripts/verify_measurements.py
  
  # Verify specific directory
  python scripts/verify_measurements.py --output-dir data/raw/measurements
  
  # Verbose output (show all instances)
  python scripts/verify_measurements.py --verbose
  
  # Save invalid instances to file
  python scripts/verify_measurements.py --save-invalid invalid_instances.txt
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/measurements',
        help='Directory containing measurement JSON files (default: data/raw/measurements)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output for each instance'
    )
    
    parser.add_argument(
        '--save-invalid',
        type=str,
        help='Save list of invalid instances to file'
    )
    
    args = parser.parse_args()
    
    # Verify measurements
    verifier = MeasurementVerifier(
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    summary = verifier.verify_all()
    
    # Save invalid instances if requested
    if args.save_invalid and summary['invalid'] > 0:
        with open(args.save_invalid, 'w') as f:
            for instance_id in summary['invalid_instances']:
                f.write(f"{instance_id}\n")
        print(f"\n💾 Invalid instances saved to: {args.save_invalid}")
    
    # Exit with appropriate code
    if summary['invalid'] > 0:
        print(f"\n⚠️  Found {summary['invalid']} invalid instances")
        sys.exit(1)
    else:
        print("\n✅ All instances are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
