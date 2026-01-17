#!/usr/bin/env python3
"""
QWEN_DATASET_BENCHMARK.py

Creates a unified benchmark dataset combining:
- BASE measurements (original code before fix)
- HUMAN HEAD measurements (human-written fix)
- QWEN HEAD measurements (LLM-generated fix) for all 8 prompt/strategy combinations

Output: A flat CSV/JSON where each row is one instance with all measurement columns.

Usage:
    python scripts/QWEN_DATASET_BENCHMARK.py
    python scripts/QWEN_DATASET_BENCHMARK.py --output-format csv
    python scripts/QWEN_DATASET_BENCHMARK.py --output-format both

Author: Giacomo Colosio
Project: LLMs For Green Code Refactoring - UPC Thesis
"""

import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

# Metrics to extract from aggregated measurements
METRICS_TO_EXTRACT = [
    'duration_seconds_mean',
    'duration_seconds_std',
    'total_energy_joules_mean',
    'total_energy_joules_std',
    'cpu_energy_joules_mean',
    'cpu_energy_joules_std',
    'gpu_energy_joules_mean',
    'gpu_energy_joules_std',
    'carbon_grams_mean',
    'carbon_grams_std',
    'power_watts_mean',
    'cpu_usage_mean_percent_mean',
    'cpu_usage_peak_percent_mean',
    'ram_usage_mean_mb_mean',
    'ram_usage_peak_mb_mean',
    'gpu_temperature_mean_celsius_mean',
    'gpu_temperature_peak_celsius_mean',
]

# Qwen configurations: (prompt_type, strategy, short_key)
QWEN_CONFIGS = [
    ('ZeroShot', 'Oracle', 'zs_oracle'),
    ('ZeroShot', 'Realistic', 'zs_realistic'),
    ('CoT', 'Oracle', 'cot_oracle'),
    ('CoT', 'Realistic', 'cot_realistic'),
    ('LDB', 'Oracle', 'ldb_oracle'),
    ('LDB', 'Realistic', 'ldb_realistic'),
    ('SelfCollab', 'Oracle', 'sc_oracle'),
    ('SelfCollab', 'Realistic', 'sc_realistic'),
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_aggregated_metrics(
    instance: Dict, 
    prefix: str
) -> Dict[str, Optional[float]]:
    """
    Extract aggregated metrics from an instance's green_metrics.
    
    Args:
        instance: Instance dict containing green_metrics
        prefix: Prefix for column names (e.g., 'base', 'human_head', 'qwen_zs_oracle')
    
    Returns:
        Dict with prefixed metric names and values (or None if not available)
    """
    result = {}
    
    # Initialize all metrics as None
    for metric in METRICS_TO_EXTRACT:
        result[f'{prefix}_{metric}'] = None
    
    # Try to extract from green_metrics
    green_metrics = instance.get('green_metrics', {})
    
    if not green_metrics:
        return result
    
    # Get the first test's metrics (they should be consistent across tests)
    first_test_key = next(iter(green_metrics.keys()), None)
    if not first_test_key:
        return result
    
    test_data = green_metrics[first_test_key]
    
    # Determine which key to use based on prefix
    if prefix == 'base':
        measurement_key = 'base'
    else:
        measurement_key = 'head'
    
    if measurement_key not in test_data:
        return result
    
    aggregated = test_data[measurement_key].get('aggregated', {})
    
    for metric in METRICS_TO_EXTRACT:
        if metric in aggregated:
            result[f'{prefix}_{metric}'] = aggregated[metric]
    
    return result


def load_human_dataset(green_dir: Path) -> Dict[str, Dict]:
    """Load the human baseline dataset (base + head measurements)."""
    human_file = green_dir / 'swe_perf_green_k3.json'
    
    if not human_file.exists():
        raise FileNotFoundError(f"Human dataset not found: {human_file}")
    
    with open(human_file) as f:
        data = json.load(f)
    
    # Index by instance_id
    return {inst['instance_id']: inst for inst in data['instances']}


def load_qwen_datasets(green_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """Load all Qwen measurement datasets."""
    qwen_data = {}
    
    for prompt, strategy, key in QWEN_CONFIGS:
        fname = f'Qwen2.5-Coder-7B_{prompt}_{strategy}_k3.json'
        fpath = green_dir / fname
        
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            qwen_data[key] = {inst['instance_id']: inst for inst in data['instances']}
            print(f"  ✓ Loaded {key}: {len(qwen_data[key])} instances")
        else:
            print(f"  ✗ Not found: {fname}")
            qwen_data[key] = {}
    
    return qwen_data


def create_unified_dataset(
    human_data: Dict[str, Dict],
    qwen_data: Dict[str, Dict[str, Dict]]
) -> List[Dict]:
    """Create the unified dataset combining all measurements."""
    unified = []
    
    # Get all unique instance IDs
    all_ids = set(human_data.keys())
    for key in qwen_data:
        all_ids.update(qwen_data[key].keys())
    
    print(f"\nProcessing {len(all_ids)} unique instances...")
    
    for instance_id in sorted(all_ids):
        row = {
            'instance_id': instance_id,
            'repo': None,
            'has_human_baseline': instance_id in human_data,
        }
        
        # Count how many Qwen configs have measurements for this instance
        qwen_count = sum(1 for key in qwen_data if instance_id in qwen_data[key])
        row['qwen_configs_available'] = qwen_count
        
        # Extract human baseline metrics (base + head)
        if instance_id in human_data:
            human_inst = human_data[instance_id]
            row['repo'] = human_inst.get('repo')
            
            # Extract BASE metrics
            base_metrics = extract_aggregated_metrics(human_inst, 'base')
            row.update(base_metrics)
            
            # Extract HUMAN HEAD metrics
            # Need to temporarily modify the extraction to look for 'head'
            human_head_metrics = {}
            green_metrics = human_inst.get('green_metrics', {})
            if green_metrics:
                first_test_key = next(iter(green_metrics.keys()), None)
                if first_test_key and 'head' in green_metrics[first_test_key]:
                    aggregated = green_metrics[first_test_key]['head'].get('aggregated', {})
                    for metric in METRICS_TO_EXTRACT:
                        human_head_metrics[f'human_head_{metric}'] = aggregated.get(metric)
            
            # Fill missing metrics with None
            for metric in METRICS_TO_EXTRACT:
                if f'human_head_{metric}' not in human_head_metrics:
                    human_head_metrics[f'human_head_{metric}'] = None
            
            row.update(human_head_metrics)
        else:
            # No human baseline - fill with None
            for metric in METRICS_TO_EXTRACT:
                row[f'base_{metric}'] = None
                row[f'human_head_{metric}'] = None
        
        # Extract QWEN metrics for each config
        for _, _, key in QWEN_CONFIGS:
            prefix = f'qwen_{key}'
            
            if instance_id in qwen_data[key]:
                qwen_inst = qwen_data[key][instance_id]
                
                # Extract metrics (Qwen files only have 'head')
                qwen_metrics = {}
                green_metrics = qwen_inst.get('green_metrics', {})
                if green_metrics:
                    first_test_key = next(iter(green_metrics.keys()), None)
                    if first_test_key and 'head' in green_metrics[first_test_key]:
                        aggregated = green_metrics[first_test_key]['head'].get('aggregated', {})
                        for metric in METRICS_TO_EXTRACT:
                            qwen_metrics[f'{prefix}_{metric}'] = aggregated.get(metric)
                
                # Fill missing metrics with None
                for metric in METRICS_TO_EXTRACT:
                    if f'{prefix}_{metric}' not in qwen_metrics:
                        qwen_metrics[f'{prefix}_{metric}'] = None
                
                row.update(qwen_metrics)
            else:
                # No Qwen measurement for this config - fill with None
                for metric in METRICS_TO_EXTRACT:
                    row[f'{prefix}_{metric}'] = None
        
        unified.append(row)
    
    return unified


def compute_improvement_columns(unified: List[Dict]) -> List[Dict]:
    """Add computed improvement columns (vs base and vs human)."""
    
    key_metrics = ['duration_seconds_mean', 'total_energy_joules_mean', 'cpu_energy_joules_mean']
    
    for row in unified:
        # Compute improvements for each Qwen config
        for _, _, key in QWEN_CONFIGS:
            prefix = f'qwen_{key}'
            
            for metric in key_metrics:
                qwen_val = row.get(f'{prefix}_{metric}')
                base_val = row.get(f'base_{metric}')
                human_val = row.get(f'human_head_{metric}')
                
                # Improvement vs BASE (positive = better/faster/less energy)
                if qwen_val is not None and base_val is not None and base_val > 0:
                    improvement = ((base_val - qwen_val) / base_val) * 100
                    row[f'{prefix}_{metric}_improvement_vs_base_pct'] = round(improvement, 2)
                else:
                    row[f'{prefix}_{metric}_improvement_vs_base_pct'] = None
                
                # Improvement vs HUMAN (positive = Qwen is better than human)
                if qwen_val is not None and human_val is not None and human_val > 0:
                    improvement = ((human_val - qwen_val) / human_val) * 100
                    row[f'{prefix}_{metric}_improvement_vs_human_pct'] = round(improvement, 2)
                else:
                    row[f'{prefix}_{metric}_improvement_vs_human_pct'] = None
    
    return unified


def save_as_json(unified: List[Dict], output_path: Path, metadata: Dict):
    """Save unified dataset as JSON."""
    output = {
        'metadata': metadata,
        'instances': unified
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved JSON: {output_path}")


def save_as_csv(unified: List[Dict], output_path: Path):
    """Save unified dataset as CSV."""
    if not unified:
        print("✗ No data to save")
        return
    
    # Get all column names
    columns = list(unified[0].keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(unified)
    
    print(f"✓ Saved CSV: {output_path}")


def print_statistics(unified: List[Dict]):
    """Print summary statistics of the unified dataset."""
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("="*70)
    
    total = len(unified)
    with_human = sum(1 for r in unified if r['has_human_baseline'])
    
    print(f"\nTotal instances: {total}")
    print(f"With human baseline: {with_human}")
    print(f"Without human baseline: {total - with_human}")
    
    print("\n📈 Qwen coverage by config:")
    for _, _, key in QWEN_CONFIGS:
        count = sum(1 for r in unified if r.get(f'qwen_{key}_duration_seconds_mean') is not None)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {key}: {count}/{total} ({pct:.1f}%)")
    
    # Compute average improvements where available
    print("\n📉 Average improvements vs BASE (where available):")
    for _, _, key in QWEN_CONFIGS:
        col = f'qwen_{key}_total_energy_joules_mean_improvement_vs_base_pct'
        values = [r[col] for r in unified if r.get(col) is not None]
        if values:
            avg = sum(values) / len(values)
            print(f"  {key}: {avg:+.2f}% energy (n={len(values)})")
    
    print("\n📉 Average improvements vs HUMAN (where available):")
    for _, _, key in QWEN_CONFIGS:
        col = f'qwen_{key}_total_energy_joules_mean_improvement_vs_human_pct'
        values = [r[col] for r in unified if r.get(col) is not None]
        if values:
            avg = sum(values) / len(values)
            print(f"  {key}: {avg:+.2f}% energy (n={len(values)})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create unified Qwen benchmark dataset'
    )
    parser.add_argument(
        '--green-dir',
        type=str,
        default='data/processed/green',
        help='Directory containing green measurement files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/green',
        help='Output directory'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='both',
        help='Output format'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='QWEN_UNIFIED_BENCHMARK',
        help='Output filename (without extension)'
    )
    
    args = parser.parse_args()
    
    green_dir = Path(args.green_dir)
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("🔧 QWEN DATASET BENCHMARK - Unified Dataset Creation")
    print("="*70)
    print(f"\nInput directory: {green_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {args.output_format}")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    
    print("\n  Loading human baseline (base + head)...")
    human_data = load_human_dataset(green_dir)
    print(f"  ✓ Loaded human baseline: {len(human_data)} instances")
    
    print("\n  Loading Qwen measurements...")
    qwen_data = load_qwen_datasets(green_dir)
    
    # Create unified dataset
    print("\n🔄 Creating unified dataset...")
    unified = create_unified_dataset(human_data, qwen_data)
    
    # Add improvement columns
    print("📊 Computing improvement metrics...")
    unified = compute_improvement_columns(unified)
    
    # Print statistics
    print_statistics(unified)
    
    # Prepare metadata
    metadata = {
        'description': 'Unified green benchmark dataset for Qwen LLM code refactoring',
        'created_at': datetime.now().isoformat(),
        'human_instances': len(human_data),
        'total_instances': len(unified),
        'qwen_configs': [key for _, _, key in QWEN_CONFIGS],
        'metrics_extracted': METRICS_TO_EXTRACT,
        'repetitions': 3,
        'model': 'Qwen2.5-Coder-7B-Instruct-AWQ'
    }
    
    # Save outputs
    print("\n💾 Saving outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output_format in ['json', 'both']:
        json_path = output_dir / f'{args.output_name}.json'
        save_as_json(unified, json_path, metadata)
    
    if args.output_format in ['csv', 'both']:
        csv_path = output_dir / f'{args.output_name}.csv'
        save_as_csv(unified, csv_path)
    
    print("\n" + "="*70)
    print("✅ DONE!")
    print("="*70)


if __name__ == '__main__':
    main()