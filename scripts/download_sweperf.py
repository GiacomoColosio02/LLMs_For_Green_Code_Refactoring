"""
Download SWE-Perf original dataset from HuggingFace.
"""
import json
from pathlib import Path
from datasets import load_dataset
from datetime import datetime


def download_sweperf(output_dir: str = "data/original"):
    """
    Download SWE-Perf dataset and save to JSON.
    
    Args:
        output_dir: Directory to save the dataset
    """
    print("📥 Downloading SWE-Perf dataset from HuggingFace...")
    print("=" * 60)
    
    # Download dataset
    try:
        ds = load_dataset("SWE-Perf/SWE-Perf")
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total instances: {len(ds['test'])}")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return
    
    # Convert to list of dicts
    print("\n📦 Converting to JSON format...")
    instances = []
    
    for i, sample in enumerate(ds['test']):
        instances.append(dict(sample))
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1} instances...")
    
    print(f"✅ Converted {len(instances)} instances")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = output_path / f"swe_perf_original_{timestamp}.json"
    
    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(instances, f, indent=2)
    
    file_size_mb = output_file.stat().st_size / (1024 ** 2)
    print(f"✅ Saved! File size: {file_size_mb:.2f} MB")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY:")
    print(f"   Total instances: {len(instances)}")
    print(f"   Output file: {output_file}")
    
    # Show example fields from first instance
    if instances:
        print(f"\n   Example instance fields:")
        for key in list(instances[0].keys())[:10]:
            print(f"     - {key}")
        if len(instances[0].keys()) > 10:
            print(f"     ... and {len(instances[0].keys()) - 10} more fields")
    
    print("\n✨ Download complete!")
    
    return output_file


if __name__ == "__main__":
    download_sweperf()