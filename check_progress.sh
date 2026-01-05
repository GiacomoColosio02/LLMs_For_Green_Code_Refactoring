#!/bin/bash

python3 << 'EOF'
from pathlib import Path
import json
from datetime import datetime

measurements_dir = Path("data/raw/measurements")
dataset_path = Path("data/original/swe_perf_original_20251124.json")

with open(dataset_path) as f:
    dataset = json.load(f)

total = len(dataset)
complete = 0
base_only = 0
head_only = 0
empty = 0
missing = 0

for instance in dataset:
    instance_id = instance['instance_id']
    json_file = measurements_dir / instance_id / "measurements.json"
    
    if not json_file.exists():
        missing += 1
        continue
    
    try:
        with open(json_file) as f:
            data = json.load(f)
        
        base = data.get('base_measurements', {})
        head = data.get('head_measurements', {})
        
        has_base = bool(base.get('tests'))
        has_head = bool(head.get('tests'))
        
        if has_base and has_head:
            complete += 1
        elif has_base and not has_head:
            base_only += 1
        elif has_head and not has_base:
            head_only += 1
        else:
            empty += 1
    except:
        empty += 1

measured = total - missing
progress_pct = (complete / total) * 100

print(f"{'=' * 70}")
print(f"📊 PROGRESS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 70}")
print(f"✅ Complete (BASE+HEAD): {complete:>3d}/{total} ({complete/total*100:>5.1f}%)")
print(f"📊 Total measured:        {measured:>3d}/{total} ({measured/total*100:>5.1f}%)")
print(f"⚠️  BASE only:            {base_only:>3d}")
print(f"⚠️  HEAD only:            {head_only:>3d}")
print(f"⚠️  Empty/Invalid:        {empty:>3d}")
print(f"❌ Missing:              {missing:>3d}")
print(f"{'=' * 70}")
print(f"🎯 Target: {140 - complete} more needed for 100%")
print(f"{'=' * 70}")

EOF
