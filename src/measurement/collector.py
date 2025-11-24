"""
Main measurement collector that combines all metrics.
"""
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from src.measurement.resource_monitor import ResourceMonitor
from src.measurement.energy_monitor import EnergyMonitor
from src.utils.config import load_config


class MetricsCollector:
    """Collect all metrics (CPU, RAM, Energy, Carbon) for a test execution."""
    
    def __init__(
        self,
        instance_id: str,
        country_code: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize metrics collector.
        
        Args:
            instance_id: Unique identifier for the instance (e.g., 'astropy__astropy-16065')
            country_code: ISO country code for carbon intensity
            config: Configuration dictionary (if None, loads default)
        """
        self.instance_id = instance_id
        self.country_code = country_code
        self.config = config if config is not None else load_config()
        
        # Monitoring components
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.energy_monitor: Optional[EnergyMonitor] = None
        
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
        
        # Initialize monitors
        resource_monitor = ResourceMonitor(
            interval=self.config['resources']['cpu_interval']
        )
        
        energy_monitor = EnergyMonitor(
            country_code=self.country_code,
            measure_power_secs=self.config['energy']['measure_power_secs'],
            tracking_mode=self.config['energy']['tracking_mode']
        )
        
        # Start monitoring
        resource_monitor.start_monitoring()
        energy_monitor.start()
        
        # Wait for baseline duration
        start_time = time.time()
        while time.time() - start_time < duration:
            resource_monitor.add_sample()
            time.sleep(self.config['resources']['cpu_interval'])
        
        # Stop and collect
        resource_stats = resource_monitor.get_statistics()
        energy_stats = energy_monitor.stop()
        
        baseline = {
            'duration_seconds': duration,
            'cpu_usage_mean_percent': resource_stats['cpu_usage_mean_percent'],
            'ram_usage_mean_mb': resource_stats['ram_usage_mean_mb'],
            'energy_joules': energy_stats['energy_consumption_joules'],
            'power_watts': energy_stats['mean_power_draw_watts']
        }
        
        print(f"  ✅ Baseline: {baseline['cpu_usage_mean_percent']:.1f}% CPU, "
              f"{baseline['ram_usage_mean_mb']:.1f} MB RAM, "
              f"{baseline['power_watts']:.2f} W")
        
        return baseline
    
    def measure_test_execution(
        self,
        test_command: str,
        repetitions: int = 1
    ) -> Dict:
        """
        Measure test execution with all metrics.
        
        Args:
            test_command: Command to execute (e.g., 'pytest test.py::test_func')
            repetitions: Number of times to repeat the test
            
        Returns:
            Dictionary with all metrics
        """
        print(f"\n🔬 Measuring test execution ({repetitions} repetitions)...")
        
        all_measurements = []
        
        for rep in range(repetitions):
            print(f"  Run {rep + 1}/{repetitions}...", end=' ')
            
            # Initialize monitors
            resource_monitor = ResourceMonitor(
                interval=self.config['resources']['cpu_interval']
            )
            
            energy_monitor = EnergyMonitor(
                country_code=self.country_code,
                measure_power_secs=self.config['energy']['measure_power_secs'],
                tracking_mode=self.config['energy']['tracking_mode']
            )
            
            # Start monitoring
            resource_monitor.start_monitoring()
            energy_monitor.start()
            
            # Execute test
            start_time = time.time()
            
            proc = subprocess.Popen(
                test_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Monitor while test runs
            while proc.poll() is None:
                resource_monitor.add_sample()
                time.sleep(self.config['resources']['cpu_interval'])
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Get results
            stdout, stderr = proc.communicate()
            return_code = proc.returncode
            
            # Stop monitoring
            resource_stats = resource_monitor.get_statistics()
            energy_stats = energy_monitor.stop()
            
            # Combine metrics
            measurement = {
                'repetition': rep + 1,
                'duration_seconds': duration,
                'return_code': return_code,
                'cpu_usage_mean_percent': resource_stats['cpu_usage_mean_percent'],
                'cpu_usage_peak_percent': resource_stats['cpu_usage_peak_percent'],
                'ram_usage_mean_mb': resource_stats['ram_usage_mean_mb'],
                'ram_usage_peak_mb': resource_stats['ram_usage_peak_mb'],
                'energy_joules': energy_stats['energy_consumption_joules'],
                'power_watts': energy_stats['mean_power_draw_watts'],
                'carbon_grams': energy_stats['carbon_emissions_grams'],
                'energy_efficiency': energy_stats['energy_efficiency_tests_per_joule']
            }
            
            all_measurements.append(measurement)
            
            print(f"✅ {duration:.2f}s, {measurement['cpu_usage_mean_percent']:.1f}% CPU, "
                  f"{measurement['energy_joules']:.2f} J")
        
        # Calculate aggregated statistics
        aggregated = self._aggregate_measurements(all_measurements)
        
        return {
            'measurements': all_measurements,
            'aggregated': aggregated
        }
    
    def _aggregate_measurements(self, measurements: list) -> Dict:
        """Calculate mean and std from multiple measurements."""
        import statistics
        
        keys = [
            'duration_seconds',
            'cpu_usage_mean_percent',
            'cpu_usage_peak_percent',
            'ram_usage_mean_mb',
            'ram_usage_peak_mb',
            'energy_joules',
            'power_watts',
            'carbon_grams',
            'energy_efficiency'
        ]
        
        aggregated = {}
        
        for key in keys:
            values = [m[key] for m in measurements]
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
            'config': self.config
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        
        return filepath


if __name__ == "__main__":
    # Test the collector with a simple command
    print("Testing MetricsCollector...")
    print("=" * 60)
    
    collector = MetricsCollector(
        instance_id="test_instance",
        country_code="ESP"
    )
    
    # Measure baseline
    baseline = collector.measure_baseline(duration=3.0)
    
    # Measure a simple test (just sleep command)
    test_command = "sleep 2"
    results = collector.measure_test_execution(
        test_command=test_command,
        repetitions=3
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"  Baseline CPU: {baseline['cpu_usage_mean_percent']:.1f}%")
    print(f"  Baseline Power: {baseline['power_watts']:.2f} W")
    print(f"\n  Test duration: {results['aggregated']['duration_seconds_mean']:.2f} ± "
          f"{results['aggregated']['duration_seconds_std']:.3f} s")
    print(f"  Test CPU: {results['aggregated']['cpu_usage_mean_percent_mean']:.1f} ± "
          f"{results['aggregated']['cpu_usage_mean_percent_std']:.1f} %")
    print(f"  Test Energy: {results['aggregated']['energy_joules_mean']:.2f} ± "
          f"{results['aggregated']['energy_joules_std']:.2f} J")
    print(f"  Test Carbon: {results['aggregated']['carbon_grams_mean']:.6f} ± "
          f"{results['aggregated']['carbon_grams_std']:.6f} g CO2e")
    
    print("\n✨ MetricsCollector test passed!")