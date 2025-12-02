"""
Test rapido per vedere quali metriche vengono misurate in locale
"""
import sys
import json
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("🧪 TEST METRICHE IN LOCALE")
print("=" * 60)
print()

# Test 1: Resource Monitor
print("📊 TEST 1: Resource Monitor")
print("-" * 60)
try:
    from src.measurement.resource_monitor import ResourceMonitor
    import time
    
    monitor = ResourceMonitor(interval=0.1)
    monitor.start_monitoring()
    
    # Simula lavoro
    for i in range(10):
        monitor.add_sample()
        time.sleep(0.1)
    
    stats = monitor.get_statistics()
    
    print("✅ Resource Monitor funziona!")
    print(f"  CPU mean: {stats.get('cpu_usage_mean_percent', 'N/A'):.2f}%")
    print(f"  CPU peak: {stats.get('cpu_usage_peak_percent', 'N/A'):.2f}%")
    print(f"  RAM peak: {stats.get('ram_usage_peak_mb', 'N/A'):.2f} MB")
    print()
    
    print("📋 Tutte le metriche disponibili:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()

print()
print("-" * 60)

# Test 2: Energy Monitor
print()
print("⚡ TEST 2: Energy Monitor")
print("-" * 60)
try:
    from src.measurement.energy_monitor import EnergyMonitor
    
    monitor = EnergyMonitor(country_code='ESP')
    monitor.start()
    
    # Simula lavoro
    import time
    time.sleep(2)
    
    metrics = monitor.stop()
    
    print("✅ Energy Monitor funziona!")
    print(f"  Energy: {metrics.get('energy_consumption_joules', 'N/A'):.2f} J")
    print(f"  Power: {metrics.get('mean_power_draw_watts', 'N/A'):.2f} W")
    print(f"  Carbon: {metrics.get('carbon_emissions_grams', 'N/A'):.6f} g")
    print(f"  Efficiency: {metrics.get('energy_efficiency_tests_per_joule', 'N/A'):.6f} test/J")
    print()
    
    print("📋 Tutte le metriche disponibili:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")
    
except Exception as e:
    print(f"❌ Errore: {e}")
    import traceback
    traceback.print_exc()

print()
print("-" * 60)

# Test 3: GPU Monitor (se esiste)
print()
print("🎮 TEST 3: GPU Monitor (opzionale)")
print("-" * 60)
try:
    from src.measurement.gpu_monitor import GPUMonitor
    
    monitor = GPUMonitor()
    sample = monitor.sample_once()
    
    print("✅ GPU Monitor funziona!")
    print(f"  GPU: {sample.get('gpu_utilization_percent', 'N/A')}%")
    print(f"  GPU Memory: {sample.get('gpu_memory_used_mb', 'N/A')} MB")
    
except Exception as e:
    print(f"⚠️ GPU Monitor non disponibile (normale se non hai GPU): {e}")

print()
print("=" * 60)
print("✅ TEST COMPLETATO")
print("=" * 60)
print()
print("📊 SUMMARY:")
print("  Metriche Resource: CPU mean, CPU peak, RAM peak")
print("  Metriche Energy: Energy (J), Power (W), Carbon (g), Efficiency")
print("  Metriche GPU: Opzionali se disponibili")
print()

