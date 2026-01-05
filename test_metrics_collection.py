"""
Quick test to verify all metrics are collected on server
"""
import sys
sys.path.insert(0, '.')

from src.measurement.collector import MetricsCollector

print("="*70)
print("🧪 TEST: Metrics Collection on Server")
print("="*70)

# Create collector
collector = MetricsCollector(
    instance_id="test_server",
    country_code="ESP"
)

print(f"\n📊 GPU Status: {'✅ ENABLED' if collector.gpu_enabled else '❌ DISABLED'}")

# Measure baseline
print("\n📏 Measuring baseline (3 seconds)...")
baseline = collector.measure_baseline(duration=3.0)

print("\n✅ Baseline metrics collected:")
for key, value in baseline.items():
    print(f"  {key:35s}: {value:8.2f}")

# Simulate test execution with simple command
print("\n🔬 Measuring test execution (2 repetitions)...")
results = collector.measure_test_execution(
    test_command="python3 -c 'import time; sum(range(10000000)); time.sleep(1)'",
    repetitions=2
)

print("\n✅ Test metrics collected:")
agg = results['aggregated']
for key in sorted(agg.keys()):
    if '_mean' in key:
        print(f"  {key:45s}: {agg[key]:8.2f}")

print("\n"+"="*70)
print("✨ Test completed successfully!")
print("="*70)
