"""
Measure all 140 SWE-Perf instances with green metrics.
Automatically skips already completed instances.
"""
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

from measure_instance import SWEPerfMeasurer


def is_instance_complete(measurements_dir: Path, instance_id: str) -> bool:
    """
    Verifica se un'istanza ha misurazioni complete (base E head).
    
    Args:
        measurements_dir: Directory delle misurazioni
        instance_id: ID dell'istanza
        
    Returns:
        True se completa, False altrimenti
    """
    json_file = measurements_dir / instance_id / "measurements.json"
    
    if not json_file.exists():
        return False
    
    try:
        with open(json_file) as f:
            data = json.load(f)
        
        has_base = bool(data.get('base_measurements', {}).get('tests'))
        has_head = bool(data.get('head_measurements', {}).get('tests'))
        
        return has_base and has_head
    except (json.JSONDecodeError, Exception):
        return False


def measure_all_instances(
    dataset_path: str,
    output_dir: str,
    country_code: str = 'ESP',
    start_from: int = 0,
    limit: int = None,
    force_remeasure: bool = False
):
    """
    Measure all instances in the dataset.
    Automatically skips already completed instances.
    
    Args:
        dataset_path: Path to SWE-Perf JSON
        output_dir: Output directory for measurements
        country_code: ISO country code for carbon
        start_from: Index to start from (for resuming)
        limit: Maximum number of instances to measure
        force_remeasure: If True, remeasure even completed instances
    """
    print("=" * 80)
    print("🔬 SWE-PERF GREEN METRICS MEASUREMENT - ALL INSTANCES")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌍 Country: {country_code}")
    print(f"📂 Output: {output_dir}")
    print(f"🔄 Force remeasure: {force_remeasure}")
    print("=" * 80)
    
    # Load dataset
    measurer = SWEPerfMeasurer(
        dataset_path=dataset_path,
        country_code=country_code
    )
    
    total_instances = len(measurer.dataset)
    print(f"\n📋 Total instances in dataset: {total_instances}")
    
    # Determine which instances to measure
    if limit:
        end_idx = min(start_from + limit, total_instances)
    else:
        end_idx = total_instances
    
    instances_to_measure = measurer.dataset[start_from:end_idx]
    print(f"📋 Instances in range: {start_from} to {end_idx-1} ({len(instances_to_measure)} total)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Count already completed
    already_complete = 0
    if not force_remeasure:
        for instance in instances_to_measure:
            if is_instance_complete(output_path, instance['instance_id']):
                already_complete += 1
        print(f"✅ Already completed: {already_complete}")
        print(f"📋 To measure: {len(instances_to_measure) - already_complete}")
    
    # Log file for tracking progress
    log_file = output_path / "measurement_log.txt"
    
    # Track successes, failures, and skipped
    successes = []
    failures = []
    skipped = []
    
    # Measure each instance
    for idx, instance in enumerate(instances_to_measure, start=start_from):
        instance_id = instance['instance_id']
        
        # Check if already complete
        if not force_remeasure and is_instance_complete(output_path, instance_id):
            print(f"\n⏭️  [{idx+1}/{total_instances}] {instance_id} - Already complete, skipping")
            skipped.append(instance_id)
            continue
        
        print("\n" + "=" * 80)
        print(f"🔬 INSTANCE {idx+1}/{total_instances}: {instance_id}")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Measure instance
            measurer.measure_instance(
                instance_id=instance_id,
                output_dir=output_dir
            )
            
            elapsed = time.time() - start_time
            
            # Verify it's actually complete
            if is_instance_complete(output_path, instance_id):
                successes.append({
                    'index': idx,
                    'instance_id': instance_id,
                    'elapsed_seconds': elapsed
                })
                
                # Log success
                with open(log_file, 'a') as f:
                    f.write(f"✅ {idx+1}/{total_instances} | {instance_id} | {elapsed:.1f}s\n")
                
                print(f"\n✅ Success! Elapsed: {elapsed:.1f}s")
            else:
                # Measurement ran but didn't produce complete results
                failures.append({
                    'index': idx,
                    'instance_id': instance_id,
                    'error': 'Incomplete measurements (missing base or head)',
                    'elapsed_seconds': elapsed
                })
                
                with open(log_file, 'a') as f:
                    f.write(f"⚠️  {idx+1}/{total_instances} | {instance_id} | INCOMPLETE\n")
                
                print(f"\n⚠️  Incomplete measurements")
            
        except Exception as e:
            elapsed = time.time() - start_time
            failures.append({
                'index': idx,
                'instance_id': instance_id,
                'error': str(e),
                'elapsed_seconds': elapsed
            })
            
            # Log failure
            with open(log_file, 'a') as f:
                f.write(f"❌ {idx+1}/{total_instances} | {instance_id} | ERROR: {str(e)}\n")
            
            print(f"\n❌ Failed! Error: {str(e)}")
            print("   Continuing with next instance...")
        
        # Print progress summary
        total_processed = len(successes) + len(failures)
        total_with_skipped = total_processed + len(skipped)
        success_rate = len(successes) / total_processed * 100 if total_processed > 0 else 0
        
        print(f"\n📊 Progress: {total_with_skipped}/{len(instances_to_measure)}")
        print(f"   ✅ Successes: {len(successes)}")
        print(f"   ❌ Failures: {len(failures)}")
        print(f"   ⏭️  Skipped: {len(skipped)}")
        print(f"   📈 Success rate (new): {success_rate:.1f}%")
        
        if successes:
            avg_time = sum(s['elapsed_seconds'] for s in successes) / len(successes)
            remaining = len(instances_to_measure) - total_with_skipped
            eta_seconds = remaining * avg_time
            eta_hours = eta_seconds / 3600
            print(f"   ⏱️  Avg time per instance: {avg_time:.1f}s")
            print(f"   ⏳ ETA: {eta_hours:.1f} hours")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏁 MEASUREMENT COMPLETE!")
    print("=" * 80)
    print(f"✅ New successes: {len(successes)}")
    print(f"❌ Failures: {len(failures)}")
    print(f"⏭️  Skipped (already complete): {len(skipped)}")
    print(f"📊 Total complete: {len(successes) + len(skipped)}/{len(instances_to_measure)}")
    
    if len(successes) + len(failures) > 0:
        print(f"📈 Success rate (new measurements): {len(successes)/(len(successes)+len(failures))*100:.1f}%")
    
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_in_range': len(instances_to_measure),
        'new_successes': len(successes),
        'failures': len(failures),
        'skipped_already_complete': len(skipped),
        'total_complete': len(successes) + len(skipped),
        'success_details': successes,
        'failure_details': failures,
        'skipped_list': skipped
    }
    
    summary_file = output_path / "measurement_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    
    # Save list of still-failing instances for easy retry
    if failures:
        failing_file = output_path / "instances_still_failing.txt"
        with open(failing_file, 'w') as f:
            for item in failures:
                f.write(f"{item['instance_id']}\n")
        print(f"💾 Failing instances saved to: {failing_file}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Measure all SWE-Perf instances with green metrics"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/original/swe_perf_original_20251124.json',
        help='Path to SWE-Perf dataset JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/measurements',
        help='Output directory for measurements'
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
        help='Index to start from (for resuming)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of instances to measure'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force remeasure even already completed instances'
    )
    
    args = parser.parse_args()
    
    measure_all_instances(
        dataset_path=args.dataset,
        output_dir=args.output,
        country_code=args.country,
        start_from=args.start_from,
        limit=args.limit,
        force_remeasure=args.force
    )


if __name__ == "__main__":
    main()