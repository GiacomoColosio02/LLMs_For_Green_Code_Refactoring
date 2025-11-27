"""
Test script for GPU monitoring integration
Tests GPU metrics collection with and without GPU hardware
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.measurement.gpu_monitor import is_gpu_available, GPUMonitor
from src.measurement.collector import MetricsCollector


def test_gpu_availability():
    """Test GPU detection"""
    print("="*70)
    print("🧪 TEST 1: GPU Availability Check")
    print("="*70)
    
    available = is_gpu_available()
    
    if available:
        print("✅ GPU detected and available")
        
        # Try to get GPU info
        try:
            monitor = GPUMonitor(device_index=0)
            print(f"   GPU Name: {monitor.gpu_name}")
            monitor.shutdown()
        except Exception as e:
            print(f"⚠️  Could not initialize GPU: {e}")
    else:
        print("ℹ️  No GPU available (expected on systems without NVIDIA GPU)")
    
    return available


def test_gpu_monitor():
    """Test GPU monitor if available"""
    print("\n" + "="*70)
    print("🧪 TEST 2: GPU Monitor (if available)")
    print("="*70)
    
    if not is_gpu_available():
        print("⏭️  Skipping - No GPU available")
        return
    
    try:
        monitor = GPUMonitor(
            device_index=0,
            track_temperature=True,
            track_power=True
        )
        
        print(f"✅ GPU Monitor initialized: {monitor.gpu_name}")
        
        # Collect some samples
        monitor.start_monitoring()
        import time
        for _ in range(10):
            monitor.add_sample()
            time.sleep(0.1)
        
        stats = monitor.get_statistics()
        
        print(f"   Samples collected: {stats['num_samples']}")
        print(f"   GPU Utilization: {stats['gpu_utilization_mean_percent']:.1f}% (mean)")
        print(f"   GPU Memory: {stats['gpu_memory_mean_mb']:.1f} MB (mean)")
        
        if 'gpu_temperature_mean_celsius' in stats:
            print(f"   Temperature: {stats['gpu_temperature_mean_celsius']:.1f}°C")
        
        if 'gpu_power_mean_watts' in stats:
            print(f"   Power: {stats['gpu_power_mean_watts']:.1f} W")
        
        monitor.shutdown()
        print("✅ GPU Monitor test passed")
        
    except Exception as e:
        print(f"❌ GPU Monitor test failed: {e}")
        return False
    
    return True


def test_metrics_collector_integration():
    """Test MetricsCollector with GPU support"""
    print("\n" + "="*70)
    print("🧪 TEST 3: MetricsCollector Integration")
    print("="*70)
    
    try:
        collector = MetricsCollector(
            instance_id="test_gpu_integration",
            country_code="ESP"
        )
        
        gpu_status = "✅ ENABLED" if collector.gpu_enabled else "ℹ️  DISABLED"
        print(f"   GPU monitoring: {gpu_status}")
        print(f"   Config loaded: {'gpu' in collector.config}")
        
        # Test baseline measurement (short duration)
        print("\n   Testing baseline measurement (2s)...")
        baseline = collector.measure_baseline(duration=2.0)
        
        print(f"   ✅ Baseline collected:")
        print(f"      CPU: {baseline['cpu_usage_mean_percent']:.1f}%")
        print(f"      RAM: {baseline['ram_usage_mean_mb']:.1f} MB")
        
        if collector.gpu_enabled:
            print(f"      GPU: {baseline['gpu_utilization_mean_percent']:.1f}%")
            print(f"      GPU Memory: {baseline['gpu_memory_mean_mb']:.1f} MB")
        
        print("\n✅ MetricsCollector integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ MetricsCollector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 GPU MONITORING INTEGRATION TESTS")
    print("="*70)
    print()
    
    results = {}
    
    # Test 1: GPU availability
    results['gpu_available'] = test_gpu_availability()
    
    # Test 2: GPU monitor (only if GPU available)
    if results['gpu_available']:
        results['gpu_monitor'] = test_gpu_monitor()
    else:
        results['gpu_monitor'] = None  # Skipped
    
    # Test 3: MetricsCollector integration (always run)
    results['collector_integration'] = test_metrics_collector_integration()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        print(f"  {status}  {test_name}")
    
    # Overall result
    failed = [k for k, v in results.items() if v is False]
    
    if failed:
        print(f"\n❌ {len(failed)} test(s) failed")
        sys.exit(1)
    else:
        print("\n✨ All tests passed!")
        print("\n💡 Note: GPU tests will run fully on server with RTX 4090")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
