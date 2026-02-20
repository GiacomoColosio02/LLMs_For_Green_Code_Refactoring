"""
Measure BASE and HUMAN_HEAD commits for all SWE-Perf instances with k=5 repetitions.
Creates a CSV dataset with raw measurements (no aggregation).
Supports resume: skips tests already measured.

Output CSV schema:
  instance_id, test_name, repo, path,
  {metric}_base_Attempt{1..5},
  {metric}_human_head_Attempt{1..5}

Total columns: 4 + (13 metrics × 2 configs × 5 attempts) = 134
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import csv
import argparse
import time
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from measure_instance import SWEPerfMeasurer
from src.measurement.collector import MetricsCollector
from src.utils.config import load_config


# ============================================================================
# CONFIGURATION
# ============================================================================

# 13 metrics to extract per attempt
METRICS = [
    # GREEN (7)
    'cpu_energy_joules',
    'gpu_energy_joules',
    'total_energy_joules',
    'system_energy_joules',
    'power_watts',
    'carbon_grams',
    'carbon_grams_system',
    # EFFICIENCY (6)
    'duration_seconds',
    'cpu_usage_mean_percent',
    'cpu_usage_peak_percent',
    'ram_usage_mean_mb',
    'ram_usage_peak_mb',
    'gpu_temperature_mean_celsius',
]

CONFIGS = ['base', 'human_head']
K = 5  # Number of repetitions per test


# ============================================================================
# CSV UTILITIES
# ============================================================================

def build_csv_header() -> List[str]:
    """Build the full CSV header: 4 id cols + 13 metrics × 2 configs × 5 attempts."""
    header = ['instance_id', 'test_name', 'repo', 'path']
    for config in CONFIGS:
        for metric in METRICS:
            for attempt in range(1, K + 1):
                header.append(f"{metric}_{config}_Attempt{attempt}")
    return header


def load_existing_csv(csv_path: Path) -> Dict[str, Dict]:
    """
    Load existing CSV into a dict keyed by "instance_id||test_name".
    Used for resume functionality.
    """
    existing = {}
    if not csv_path.exists():
        return existing

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['instance_id']}||{row['test_name']}"
            existing[key] = row

    print(f"  📂 Loaded {len(existing)} existing rows from CSV")
    return existing


def save_csv(csv_path: Path, header: List[str], rows: Dict[str, Dict]):
    """Overwrite CSV with all current rows (used for progressive save)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)


def is_config_complete(row: Dict, config: str) -> bool:
    """Check if a row has all K measurements for a given config."""
    for metric in METRICS:
        for attempt in range(1, K + 1):
            col = f"{metric}_{config}_Attempt{attempt}"
            if col not in row or row[col] == '' or row[col] is None:
                return False
    return True


# ============================================================================
# ROW HELPERS
# ============================================================================

def init_empty_row(instance_id: str, test_name: str, repo: str) -> Dict:
    """Create an empty row with all columns initialized to ''."""
    row = {
        'instance_id': instance_id,
        'test_name': test_name,
        'repo': repo,
        'path': test_name.rsplit('::', 1)[0] if '::' in test_name else test_name,
    }
    for config in CONFIGS:
        for metric in METRICS:
            for attempt in range(1, K + 1):
                row[f"{metric}_{config}_Attempt{attempt}"] = ''
    return row


def fill_row_from_measurements(row: Dict, config: str, measurements: List[Dict]):
    """
    Fill a row's columns for a given config from raw measurement dicts.
    Each measurement dict comes directly from the collector (one per attempt).
    """
    for attempt_idx, measurement in enumerate(measurements):
        if attempt_idx >= K:
            break
        attempt_num = attempt_idx + 1
        for metric in METRICS:
            col = f"{metric}_{config}_Attempt{attempt_num}"
            row[col] = measurement.get(metric, '')


# ============================================================================
# INSTANCE MEASUREMENT
# ============================================================================

def measure_instance_configs(
    measurer: SWEPerfMeasurer,
    instance: Dict,
    existing_rows: Dict[str, Dict],
    csv_path: Path,
    header: List[str]
) -> Tuple[int, int, int]:
    """
    Measure base and/or human_head for one instance.
    Follows the same clone → install → measure flow as measure_instance.py.
    Skips configs where all tests are already complete.

    Returns:
        Tuple of (tests_measured, tests_skipped, tests_failed)
    """
    instance_id = instance['instance_id']
    repo = instance['repo']
    efficiency_tests = instance.get('efficiency_test', [])

    if not efficiency_tests:
        print(f"  ⚠️  No efficiency tests, skipping")
        return (0, 0, 0)

    # Determine which configs still need measuring
    configs_needed = []
    for config in CONFIGS:
        all_complete = True
        for test_name in efficiency_tests:
            key = f"{instance_id}||{test_name}"
            if key not in existing_rows or not is_config_complete(existing_rows[key], config):
                all_complete = False
                break
        if not all_complete:
            configs_needed.append(config)

    if not configs_needed:
        print(f"  ⭐ All configs complete, skipping")
        return (0, len(efficiency_tests) * len(CONFIGS), 0)

    print(f"  📋 Tests: {len(efficiency_tests)} | Configs needed: {configs_needed}")

    # Create temp dir for cloning repos
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)

    tests_measured = 0
    tests_skipped = 0
    tests_failed = 0

    try:
        for config in configs_needed:
            commit = instance['base_commit'] if config == 'base' else instance['head_commit']
            print(f"\n  🔬 [{config.upper()}] commit {commit[:8]}...")

            # ---- Clone & install (same flow as measure_instance.py) ----
            try:
                repo_path = measurer.setup_repository(instance, temp_path, commit)
            except Exception as e:
                print(f"  ❌ Clone failed: {str(e)[:100]}")
                tests_failed += len(efficiency_tests)
                continue

            python_path, conda_env = measurer.install_dependencies(
                repo_path, repo, instance['version'], commit
            )

            if python_path is None:
                print(f"  ❌ Dependencies failed for {config}")
                tests_failed += len(efficiency_tests)
                if conda_env:
                    measurer.cleanup_conda_env(conda_env)
                shutil.rmtree(repo_path, ignore_errors=True)
                continue

            # ---- Initialize collector (same as measure_instance.py) ----
            collector = MetricsCollector(
                instance_id=instance_id,
                country_code=measurer.country_code
            )

            # Measure baseline (system idle reference)
            collector.measure_baseline(
                duration=measurer.config['measurement']['baseline_duration_sec']
            )

            # ---- Measure each test with K repetitions ----
            for test_idx, test_name in enumerate(efficiency_tests):
                key = f"{instance_id}||{test_name}"

                # Skip if already complete for this config
                if key in existing_rows and is_config_complete(existing_rows[key], config):
                    print(f"    ⭐ [{test_idx+1}/{len(efficiency_tests)}] Already complete for {config}")
                    tests_skipped += 1
                    continue

                print(f"    📝 [{test_idx+1}/{len(efficiency_tests)}] {test_name}")

                # Run K repetitions via measure_single_test
                # This uses the same method from measure_instance.py
                result = measurer.measure_single_test(
                    collector=collector,
                    test_name=test_name,
                    repo_path=repo_path,
                    python_path=python_path,
                    repetitions=K,
                    timeout=600  # 10 min per test
                )

                if result is None or result.get('status') != 'success':
                    err = result.get('error', 'unknown') if result else 'None returned'
                    print(f"    ❌ Failed: {err[:100]}")
                    tests_failed += 1
                    continue

                # Extract raw measurements list
                raw_measurements = result.get('measurements', [])

                if len(raw_measurements) < K:
                    print(f"    ⚠️  Only {len(raw_measurements)}/{K} attempts succeeded")

                # Init row if first time seeing this test
                if key not in existing_rows:
                    existing_rows[key] = init_empty_row(instance_id, test_name, repo)

                # Fill measurements into row
                fill_row_from_measurements(existing_rows[key], config, raw_measurements)

                tests_measured += 1

                # Progressive save after each test
                save_csv(csv_path, header, existing_rows)
                print(f"    ✅ Saved ({len(raw_measurements)} attempts)")

            # ---- Cleanup this commit's repo ----
            shutil.rmtree(repo_path, ignore_errors=True)
            if conda_env:
                measurer.cleanup_conda_env(conda_env)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return (tests_measured, tests_skipped, tests_failed)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure BASE and HUMAN_HEAD with k=5 for all SWE-Perf instances"
    )

    script_path = Path(__file__).resolve()
    base_dir = script_path.parent.parent

    parser.add_argument(
        '--dataset',
        type=str,
        default=str(base_dir / "data" / "processed(k=5)" / "swe_perf_reduced.json"),
        help='Path to reduced dataset JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(base_dir / "data" / "processed(k=5)" / "base_human_head_k5.csv"),
        help='Output CSV path'
    )
    parser.add_argument(
        '--country',
        type=str,
        default='ESP',
        help='ISO country code for carbon intensity'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Start from instance index (for manual resume)'
    )

    args = parser.parse_args()

    # ---- Banner ----
    print("=" * 80)
    print("🌱 SWE-PERF GREEN — BASE & HUMAN_HEAD MEASUREMENTS (k=5)")
    print("=" * 80)
    print(f"⏰ Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Dataset:      {args.dataset}")
    print(f"📄 Output CSV:   {args.output}")
    print(f"🌍 Country:      {args.country}")
    print(f"🔁 Repetitions:  {K}")
    print(f"📊 Metrics:      {len(METRICS)}")
    print(f"📋 Configs:      {CONFIGS}")
    total_cols = 4 + len(METRICS) * len(CONFIGS) * K
    print(f"📐 CSV columns:  {total_cols}")
    print("=" * 80)

    # ---- Build header ----
    header = build_csv_header()
    assert len(header) == total_cols, f"Header mismatch: {len(header)} vs {total_cols}"

    # ---- Load dataset ----
    print(f"\n📂 Loading dataset...")
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    instances = dataset if isinstance(dataset, list) else dataset.get('instances', dataset)
    print(f"  Found {len(instances)} instances")

    total_tests = sum(len(inst.get('efficiency_test', [])) for inst in instances)
    print(f"  Total tests across all instances: {total_tests}")

    # ---- Load existing CSV for resume ----
    csv_path = Path(args.output)
    existing_rows = load_existing_csv(csv_path)

    # ---- Initialize measurer ----
    # The measurer needs the full original dataset (with all repo info).
    # Fall back to reduced dataset if original not found.
    original_dataset = base_dir / "data" / "original" / "swe_perf_original_20251124.json"
    dataset_for_measurer = str(original_dataset) if original_dataset.exists() else args.dataset

    print(f"\n📂 Initializing measurer with: {dataset_for_measurer}")
    measurer = SWEPerfMeasurer(
        dataset_path=dataset_for_measurer,
        country_code=args.country
    )

    # ---- Process all instances ----
    grand_measured = 0
    grand_skipped = 0
    grand_failed = 0
    instance_times = []

    for idx, instance in enumerate(instances):
        if idx < args.start_from:
            continue

        instance_id = instance['instance_id']
        n_tests = len(instance.get('efficiency_test', []))

        print(f"\n{'='*80}")
        print(f"🔬 [{idx+1}/{len(instances)}] {instance_id}  ({n_tests} tests)")
        print(f"{'='*80}")

        t0 = time.time()

        try:
            measured, skipped, failed = measure_instance_configs(
                measurer=measurer,
                instance=instance,
                existing_rows=existing_rows,
                csv_path=csv_path,
                header=header,
            )
            grand_measured += measured
            grand_skipped += skipped
            grand_failed += failed

        except Exception as e:
            grand_failed += n_tests
            print(f"\n  ❌ Instance error: {str(e)[:200]}")
            print(f"  Continuing...")

        elapsed = time.time() - t0
        instance_times.append(elapsed)

        # ---- Progress ----
        done = idx + 1 - args.start_from
        remaining = len(instances) - (idx + 1)
        avg_time = sum(instance_times) / len(instance_times)
        eta_h = (remaining * avg_time) / 3600

        print(f"\n  ⏱️  Instance time: {elapsed:.1f}s")
        print(f"  📊 Progress: {idx+1}/{len(instances)} instances")
        print(f"     Tests measured: {grand_measured} | Skipped: {grand_skipped} | Failed: {grand_failed}")
        print(f"     CSV rows: {len(existing_rows)}")
        print(f"     ETA: ~{eta_h:.1f}h ({avg_time:.0f}s/instance avg)")

    # ---- Final save ----
    save_csv(csv_path, header, existing_rows)

    print(f"\n{'='*80}")
    print("🎉 COMPLETE!")
    print(f"{'='*80}")
    print(f"  📄 CSV:            {csv_path}")
    print(f"  📊 Total rows:     {len(existing_rows)}")
    print(f"  📐 Total columns:  {len(header)}")
    print(f"  🧪 Measured:       {grand_measured}")
    print(f"  ⭐ Skipped:        {grand_skipped}")
    print(f"  ❌ Failed:          {grand_failed}")
    print(f"  ⏰ Finished:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()