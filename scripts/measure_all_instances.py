"""
Measure all 140 SWE-Perf instances with green metrics.
"""
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

from measure_instance import SWEPerfMeasurer


def measure_all_instances(
    dataset_path: str,
    output_dir: str,
    country_code: str = 'ESP',
    start_from: int = 0,
    limit: int = None
):
    """
    Measure all instances in the dataset.
    
    Args:
        dataset_path: Path to SWE-Perf JSON
        output_dir: Output directory for measurements
        country_code: ISO country code for carbon
        start_from: Index to start from (for resuming)
        limit: Maximum number of instances to measure
    """
    print("=" * 80)
    print(" SWE-PERF GREEN METRICS MEASUREMENT - ALL INSTANCES")
    print("=" * 80)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Country: {country_code}")
    print(f" Output: {output_dir}")
    print("=" * 80)
    
    # Load dataset
    measurer = SWEPerfMeasurer(
        dataset_path=dataset_path,
        country_code=country_code
    )
    
    total_instances = len(measurer.dataset)
    print(f"\n Total instances in dataset: {total_instances}")
    
    # Determine which instances to measure
    if limit:
        end_idx = min(start_from + limit, total_instances)
    else:
        end_idx = total_instances
    
    instances_to_measure = measurer.dataset[start_from:end_idx]
    print(f" Measuring instances {start_from} to {end_idx-1} ({len(instances_to_measure)} total)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Log file for tracking progress
    log_file = output_path / "measurement_log.txt"
    
    # Track successes and failures
    successes = []
    failures = []
    
    # Measure each instance
    for idx, instance in enumerate(instances_to_measure, start=start_from):
        instance_id = instance['instance_id']
        
        print("\n" + "=" * 80)
        print(f" INSTANCE {idx+1}/{total_instances}: {instance_id}")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Measure instance
            measurer.measure_instance(
                instance_id=instance_id,
                output_dir=output_dir
            )
            
            elapsed = time.time() - start_time
            successes.append({
                'index': idx,
                'instance_id': instance_id,
                'elapsed_seconds': elapsed
            })
            
            # Log success
            with open(log_file, 'a') as f:
                f.write(f"✅ {idx+1}/{total_instances} | {instance_id} | {elapsed:.1f}s\n")
            
            print(f"\n✅ Success! Elapsed: {elapsed:.1f}s")
            
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
                f.write(f" {idx+1}/{total_instances} | {instance_id} | ERROR: {str(e)}\n")
            
            print(f"\n Failed! Error: {str(e)}")
            print("  Continuing with next instance...")
        
        # Print progress summary
        total_measured = len(successes) + len(failures)
        success_rate = len(successes) / total_measured * 100 if total_measured > 0 else 0
        
        print(f"\n📊 Progress: {total_measured}/{len(instances_to_measure)} measured "
              f"({success_rate:.1f}% success rate)")
        
        if successes:
            avg_time = sum(s['elapsed_seconds'] for s in successes) / len(successes)
            remaining = len(instances_to_measure) - total_measured
            eta_seconds = remaining * avg_time
            eta_hours = eta_seconds / 3600
            print(f"⏱️  Average time per instance: {avg_time:.1f}s")
            print(f"⏳ ETA: {eta_hours:.1f} hours")
    
    # Final summary
    print("\n" + "=" * 80)
    print(" MEASUREMENT COMPLETE!")
    print("=" * 80)
    print(f" Successes: {len(successes)}")
    print(f" Failures: {len(failures)}")
    print(f" Success rate: {len(successes)/(len(successes)+len(failures))*100:.1f}%")
    print(f" Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_measured': len(successes) + len(failures),
        'successes': successes,
        'failures': failures,
        'success_rate': len(successes)/(len(successes)+len(failures))*100 if (len(successes)+len(failures)) > 0 else 0
    }
    
    summary_file = output_path / "measurement_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
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
    
    args = parser.parse_args()
    
    measure_all_instances(
        dataset_path=args.dataset,
        output_dir=args.output,
        country_code=args.country,
        start_from=args.start_from,
        limit=args.limit
    )


if __name__ == "__main__":
    main()