"""
Test script to verify energy monitoring tools on server
Run this on gaissa.essi.upc.edu to check available tools
"""
import subprocess
import os
import sys
from pathlib import Path
import time

print("=" * 70)
print("🔍 ENERGY MONITORING TOOLS - SERVER TEST")
print("=" * 70)

results = {}

# Test 1: NVIDIA-SMI (GPU)
print("\n📊 TEST 1: NVIDIA-SMI (GPU Monitoring)")
print("-" * 70)
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw,temperature.gpu', '--format=csv'],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print("✅ nvidia-smi available")
        print("\nOutput preview:")
        print(result.stdout[:300])
        results['nvidia-smi'] = True
    else:
        print("❌ nvidia-smi failed")
        print(result.stderr)
        results['nvidia-smi'] = False
except Exception as e:
    print(f"❌ nvidia-smi not found: {e}")
    results['nvidia-smi'] = False

# Test 2: EnergiBridge (CPU Energy)
print("\n📊 TEST 2: ENERGIBRIDGE (CPU Energy Monitoring)")
print("-" * 70)

# Check multiple possible locations
energibridge_paths = [
    './energibridge',
    '/usr/local/bin/energibridge',
    '/usr/bin/energibridge',
    str(Path.home() / 'energibridge'),
    str(Path.home() / 'bin' / 'energibridge')
]

energibridge_found = None
for path in energibridge_paths:
    if os.path.exists(path) and os.path.isfile(path):
        energibridge_found = path
        print(f"✅ EnergiBridge found at: {path}")
        break

if energibridge_found:
    try:
        # Try to run it for 2 seconds
        print("\nTesting EnergiBridge (2 seconds)...")
        proc = subprocess.Popen(
            [energibridge_found, '200'],  # 200ms sampling
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(2)
        proc.terminate()
        stdout, stderr = proc.communicate()
        
        print("\nOutput preview:")
        print(stdout[:500])
        results['energibridge'] = True
        results['energibridge_path'] = energibridge_found
        
    except Exception as e:
        print(f"❌ EnergiBridge execution failed: {e}")
        results['energibridge'] = False
else:
    print("❌ EnergiBridge not found in common locations")
    print("\n💡 Please provide the path to energibridge binary")
    results['energibridge'] = False

# Test 3: Check for Wattmeter CSV
print("\n📊 TEST 3: WATTMETER (System Power)")
print("-" * 70)

# Common locations for wattmeter data
wattmeter_paths = [
    '/var/log/wattmeter/',
    str(Path.home() / 'wattmeter'),
    './wattmeter'
]

wattmeter_found = False
for path in wattmeter_paths:
    if os.path.exists(path):
        print(f"✅ Wattmeter directory found: {path}")
        # List CSV files
        csv_files = list(Path(path).glob('*.csv'))
        if csv_files:
            print(f"   Found {len(csv_files)} CSV file(s)")
            print(f"   Latest: {csv_files[-1].name}")
            wattmeter_found = True
            results['wattmeter'] = True
            results['wattmeter_path'] = path
        break

if not wattmeter_found:
    print("⚠️  Wattmeter data directory not found")
    print("💡 Ask professor for wattmeter CSV location")
    results['wattmeter'] = False

# Test 4: pynvml (Python GPU library)
print("\n📊 TEST 4: PYNVML (Python GPU Library)")
print("-" * 70)
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(handle)
    print(f"✅ pynvml available")
    print(f"   GPU: {name}")
    print(f"   Device count: {device_count}")
    pynvml.nvmlShutdown()
    results['pynvml'] = True
except Exception as e:
    print(f"❌ pynvml not available: {e}")
    results['pynvml'] = False

# Summary
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)

tools_status = [
    ("nvidia-smi", "GPU metrics via command line"),
    ("pynvml", "GPU metrics via Python library"),
    ("energibridge", "CPU energy consumption"),
    ("wattmeter", "Total system power")
]

for tool, description in tools_status:
    status = "✅ AVAILABLE" if results.get(tool, False) else "❌ NOT FOUND"
    print(f"{status:20s} {tool:15s} - {description}")

# Recommendations
print("\n" + "=" * 70)
print("💡 RECOMMENDATIONS")
print("=" * 70)

if results.get('pynvml'):
    print("✅ Use pynvml for real-time GPU monitoring (already implemented)")
else:
    print("⚠️  Install pynvml: pip install pynvml")

if results.get('energibridge'):
    print(f"✅ Use EnergiBridge for CPU energy: {results.get('energibridge_path')}")
else:
    print("⚠️  Need to locate or install EnergiBridge binary")
    print("   Check with professor or in ~/bin/ directory")

if results.get('wattmeter'):
    print(f"✅ Wattmeter data available at: {results.get('wattmeter_path')}")
else:
    print("⚠️  Ask professor for Wattmeter CSV file location")
    print("   This is optional if we have GPU + CPU energy")

print("\n" + "=" * 70)
print("🎯 NEXT STEPS")
print("=" * 70)
print("1. Run this script on server: python test_energy_tools_server.py")
print("2. Send output to me")
print("3. I'll create the integration plan based on available tools")
print("=" * 70)
