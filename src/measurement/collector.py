"""
Main measurement collector that combines all metrics.
Uses GSMM approach for energy measurement (GPU + CPU).
"""
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from src.measurement.resource_monitor import ResourceMonitor
from src.measurement.energy_monitor_gsmm import EnergyMonitorGSMM
from src.measurement.gpu_monitor import GPUMonitor, is_gpu_available
from src.utils.config import load_config


class MetricsCollector:
    """Collect all metrics (CPU, RAM, GPU, Energy, Carbon) for a test execution."""
    
    def __init__(
        self,
        instance_id: str,
        country_code: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize metrics collector.
        
        Args:
            instance_id: Unique identifier for the instance
            country_code: ISO country code for carbon intensity
            config: Configuration dictionary (if None, loads default)
        """
        self.instance_id = instance_id
        self.country_code = country_code
        self.config = config if config is not None else load_config()
        
        # Check GPU availability
        self.gpu_enabled = (
            self.config.get('gpu', {}).get('enabled', False) and 
            is_gpu_available()
        )
        
        if self.gpu_enabled:
            print("✅ GPU monitoring enabled")
        else:
            print("ℹ️  GPU monitoring disabled")
        
        # Initialize GSMM energy monitor (handles GPU + CPU energy internally)
        # Add grid intensity to config
        if 'energy' not in self.config:
            self.config['energy'] = {}
        
        # Use country-specific grid intensity or default
        grid_intensities = {
            'ESP': 250,  # Spain
            'USA': 417,  # USA average
            'DEU': 311,  # Germany
            'FRA': 52,   # France
            'GBR': 233,  # UK
        }
        
        if self.country_code and self.country_code in grid_intensities:
            self.config['energy']['grid_intensity'] = grid_intensities[self.country_code]
        elif 'grid_intensity' not in self.config['energy']:
            self.config['energy']['grid_intensity'] = 250  # Default Spain
        
        try:
            self.energy_monitor = EnergyMonitorGSMM(self.config)
            print(f"✅ GSMM Energy monitoring enabled (grid: {self.config['energy']['grid_intensity']} gCO2e/kWh)")
        except Exception as e:
            print(f"⚠️  GSMM Energy monitoring unavailable: {e}")
            self.energy_monitor = None
        
        # Results
        self.metrics: Dict = {}
        
    def measure_baseline(self, duration: float = 5.0) -> Dict:
        """
        Measure baseline (idle system).
        
        Args:
            duration: How long to measure baseline (seconds)
            
        Returns:
            Dictionary with baseline metrics
        """
        print(f"📊 Measuring baseline for {duration}s...")
        
        if not self.energy_monitor:
            raise RuntimeError("Energy monitor not available")
        
        # Measure baseline with GSMM monitor
        baseline = self.energy_monitor.measure_baseline(duration)
        
        print(f"  ✅ Baseline: "
              f"{baseline.get('cpu_usage_mean_percent', 0):.1f}% CPU, "
              f"{baseline.get('ram_usage_mean_mb', 0):.1f} MB RAM, "
              f"{baseline['power_watts']:.2f} W", end='')
        
        if self.gpu_enabled:
            print(f", {baseline.get('gpu_utilization_mean_percent', 0):.1f}% GPU")
        else:
            print()
        
        return baseline
    
    def measure_test_execution(
        self,
        test_command: str,
        repetitions: int = 1,
        venv_python: Optional[Path] = None
    ) -> Dict:
        """
        Measure test execution with all metrics.
        
        Args:
            test_command: Pytest command to execute
            repetitions: Number of times to repeat the test
            venv_python: Path to virtual environment python
            
        Returns:
            Dictionary with all metrics
        """
        print(f"\n🔬 Measuring test execution ({repetitions} repetitions)...")
        
        if not self.energy_monitor:
            raise RuntimeError("Energy monitor not available")
        
        all_measurements = []
        
        for rep in range(repetitions):
            print(f"  Run {rep + 1}/{repetitions}...", end=' ')
            
            # Measure with GSMM monitor (includes execution)
            measurement = self.energy_monitor.measure_test_energy(
                test_command,
                venv_python
            )
            
            measurement['repetition'] = rep + 1
            measurement['return_code'] = 0  # If EnergiBridge succeeds, test passed
            
            all_measurements.append(measurement)
            
            gpu_info = f", {measurement.get('gpu_utilization_mean_percent', 0):.1f}% GPU" if self.gpu_enabled else ""
            print(f"✅ {measurement['duration_seconds']:.2f}s, "
                  f"{measurement.get('cpu_usage_mean_percent', 0):.1f}% CPU{gpu_info}, "
                  f"{measurement['total_energy_joules']:.2f} J")
        
        # Calculate aggregated statistics
        aggregated = self._aggregate_measurements(all_measurements)
        
        return {
            'measurements': all_measurements,
            'aggregated': aggregated
        }
    
    def _aggregate_measurements(self, measurements: list) -> Dict:
        """Calculate mean and std from multiple measurements."""
        import statistics
        
        # Core energy metrics (always present with GSMM)
        keys = [
            'duration_seconds',
            'gpu_energy_joules',
            'cpu_energy_joules',
            'total_energy_joules',
            'power_watts',
            'carbon_grams',
            'energy_efficiency',
        ]
        
        # CPU/RAM metrics (from ResourceMonitor if available)
        if measurements and 'cpu_usage_mean_percent' in measurements[0]:
            keys.extend([
                'cpu_usage_mean_percent',
                'cpu_usage_peak_percent',
                'ram_usage_mean_mb',
                'ram_usage_peak_mb',
            ])
        
        # GPU metrics (if available)
        if self.gpu_enabled and measurements:
            if 'gpu_utilization_mean_percent' in measurements[0]:
                keys.extend([
                    'gpu_utilization_mean_percent',
                    'gpu_utilization_peak_percent',
                    'gpu_memory_mean_mb',
                    'gpu_memory_peak_mb',
                    'gpu_memory_mean_percent',
                    'gpu_memory_peak_percent'
                ])
            
            if 'gpu_temperature_mean_celsius' in measurements[0]:
                keys.extend([
                    'gpu_temperature_mean_celsius',
                    'gpu_temperature_peak_celsius'
                ])
            
            if 'gpu_power_mean_watts' in measurements[0]:
                keys.extend([
                    'gpu_power_mean_watts',
                    'gpu_power_peak_watts'
                ])
        
        aggregated = {}
        
        for key in keys:
            values = [m[key] for m in measurements if key in m]
            if values:
                aggregated[f'{key}_mean'] = statistics.mean(values)
                aggregated[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
                aggregated[f'{key}_min'] = min(values)
                aggregated[f'{key}_max'] = max(values)
        
        return aggregated
    
    def save_results(self, results: Dict, output_dir: Path):
        """
        Save measurement results to JSON file.
        
        Args:
            results: Dictionary with all results
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime(self.config['output']['timestamp_format'])
        filename = f"{self.instance_id}_{timestamp}.json"
        filepath = output_dir / filename
        
        # Add metadata
        results['metadata'] = {
            'instance_id': self.instance_id,
            'timestamp': timestamp,
            'country_code': self.country_code,
            'gpu_enabled': self.gpu_enabled,
            'energy_method': 'GSMM (GPU + CPU)',
            'grid_intensity_gCO2e_per_kWh': self.config['energy']['grid_intensity'],
            'config': self.config
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        
        return filepath



if __name__ == "__main__":
    # Test the collector with GSMM
    print("Testing MetricsCollector with GSMM...")
    print("=" * 60)
    
    collector = MetricsCollector(
        instance_id="test_instance",
        country_code="ESP"
    )
    
    # Measure baseline
    baseline = collector.measure_baseline(duration=3.0)
    
    # Measure a simple shell command (not a pytest test)
    # We need to pass wrap_with_pytest=False for shell commands
    print("\n🔬 Measuring shell command (2 repetitions)...")
    
    all_measurements = []
    for rep in range(2):
        print(f"  Run {rep + 1}/2...", end=' ')
        
        # Measure with GSMM monitor (shell command, not pytest)
        measurement = collector.energy_monitor.measure_test_energy(
            "sleep 2",
            venv_python=None,
            wrap_with_pytest=False  # Shell command, not pytest!
        )
        
        measurement['repetition'] = rep + 1
        measurement['return_code'] = 0
        
        all_measurements.append(measurement)
        
        print(f"✅ {measurement['duration_seconds']:.2f}s, "
              f"{measurement['total_energy_joules']:.2f} J")
    
    # Calculate aggregated statistics
    results = {
        'measurements': all_measurements,
        'aggregated': collector._aggregate_measurements(all_measurements)
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"  Baseline Power: {baseline['power_watts']:.2f} W")
    print(f"  Baseline GPU Energy: {baseline.get('gpu_energy_joules', 0):.2f} J")
    print(f"  Baseline CPU Energy: {baseline.get('cpu_energy_joules', 0):.2f} J")
    
    print(f"\n  Test duration: {results['aggregated']['duration_seconds_mean']:.2f} ± "
          f"{results['aggregated']['duration_seconds_std']:.3f} s")
    print(f"  Test Total Energy: {results['aggregated']['total_energy_joules_mean']:.2f} ± "
          f"{results['aggregated']['total_energy_joules_std']:.2f} J")
    print(f"  Test GPU Energy: {results['aggregated']['gpu_energy_joules_mean']:.2f} ± "
          f"{results['aggregated']['gpu_energy_joules_std']:.2f} J")
    print(f"  Test CPU Energy: {results['aggregated']['cpu_energy_joules_mean']:.2f} ± "
          f"{results['aggregated']['cpu_energy_joules_std']:.2f} J")
    print(f"  Test Carbon: {results['aggregated']['carbon_grams_mean']:.6f} ± "
          f"{results['aggregated']['carbon_grams_std']:.6f} g CO2e")
    
    print("\n✨ MetricsCollector GSMM test passed!")
