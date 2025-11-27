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
        
        # Monitoring components
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.energy_monitor: Optional[EnergyMonitor] = None
        self.gpu_monitor: Optional[GPUMonitor] = None
        
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
        
        gpu_monitor = None
        if self.gpu_enabled:
            gpu_config = self.config.get('gpu', {})
            gpu_monitor = GPUMonitor(
                device_index=gpu_config.get('device_index', 0),
                track_temperature=gpu_config.get('track_temperature', True),
                track_power=gpu_config.get('track_power', True)
            )
            gpu_monitor.start_monitoring()
        
        # Start monitoring
        resource_monitor.start_monitoring()
        energy_monitor.start()
        
        # Wait for baseline duration
        start_time = time.time()
        while time.time() - start_time < duration:
            resource_monitor.add_sample()
            if gpu_monitor:
                gpu_monitor.add_sample()
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
        
        # Add GPU metrics if available
        if gpu_monitor:
            gpu_stats = gpu_monitor.get_statistics()
            baseline.update({
                'gpu_utilization_mean_percent': gpu_stats['gpu_utilization_mean_percent'],
                'gpu_memory_mean_mb': gpu_stats['gpu_memory_mean_mb']
            })
            gpu_monitor.shutdown()
        
        print(f"  ✅ Baseline: {baseline['cpu_usage_mean_percent']:.1f}% CPU, "
              f"{baseline['ram_usage_mean_mb']:.1f} MB RAM, "
              f"{baseline['power_watts']:.2f} W", end='')
        
        if self.gpu_enabled:
            print(f", {baseline['gpu_utilization_mean_percent']:.1f}% GPU")
        else:
            print()
        
        return baseline
    
    def measure_test_execution(
        self,
        test_command: str,
        repetitions: int = 1
    ) -> Dict:
        """
        Measure test execution with all metrics.
        
        Args:
            test_command: Command to execute
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
            
            gpu_monitor = None
            if self.gpu_enabled:
                gpu_config = self.config.get('gpu', {})
                gpu_monitor = GPUMonitor(
                    device_index=gpu_config.get('device_index', 0),
                    track_temperature=gpu_config.get('track_temperature', True),
                    track_power=gpu_config.get('track_power', True)
                )
                gpu_monitor.start_monitoring()
            
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
                if gpu_monitor:
                    gpu_monitor.add_sample()
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
            
            # Add GPU metrics if available
            if gpu_monitor:
                gpu_stats = gpu_monitor.get_statistics()
                measurement.update({
                    'gpu_utilization_mean_percent': gpu_stats['gpu_utilization_mean_percent'],
                    'gpu_utilization_peak_percent': gpu_stats['gpu_utilization_peak_percent'],
                    'gpu_memory_mean_mb': gpu_stats['gpu_memory_mean_mb'],
                    'gpu_memory_peak_mb': gpu_stats['gpu_memory_peak_mb'],
                    'gpu_memory_mean_percent': gpu_stats['gpu_memory_mean_percent'],
                    'gpu_memory_peak_percent': gpu_stats['gpu_memory_peak_percent']
                })
                
                if 'gpu_temperature_mean_celsius' in gpu_stats:
                    measurement['gpu_temperature_mean_celsius'] = gpu_stats['gpu_temperature_mean_celsius']
                    measurement['gpu_temperature_peak_celsius'] = gpu_stats['gpu_temperature_peak_celsius']
                
                if 'gpu_power_mean_watts' in gpu_stats:
                    measurement['gpu_power_mean_watts'] = gpu_stats['gpu_power_mean_watts']
                    measurement['gpu_power_peak_watts'] = gpu_stats['gpu_power_peak_watts']
                
                gpu_monitor.shutdown()
            
            all_measurements.append(measurement)
            
            gpu_info = f", {measurement.get('gpu_utilization_mean_percent', 0):.1f}% GPU" if self.gpu_enabled else ""
            print(f"✅ {duration:.2f}s, {measurement['cpu_usage_mean_percent']:.1f}% CPU{gpu_info}, "
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
        
        # Base metrics (always present)
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
        
        # Add GPU metrics if available
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
            'config': self.config
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        
        return filepath


if __name__ == "__main__":
    # Test the collector with a simple command
    print("Testing MetricsCollector with GPU support...")
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
    
    if collector.gpu_enabled:
        print(f"  Baseline GPU: {baseline.get('gpu_utilization_mean_percent', 0):.1f}%")
    
    print(f"\n  Test duration: {results['aggregated']['duration_seconds_mean']:.2f} ± "
          f"{results['aggregated']['duration_seconds_std']:.3f} s")
    print(f"  Test CPU: {results['aggregated']['cpu_usage_mean_percent_mean']:.1f} ± "
          f"{results['aggregated']['cpu_usage_mean_percent_std']:.1f} %")
    
    if collector.gpu_enabled:
        print(f"  Test GPU: {results['aggregated'].get('gpu_utilization_mean_percent_mean', 0):.1f} ± "
              f"{results['aggregated'].get('gpu_utilization_mean_percent_std', 0):.1f} %")
    
    print(f"  Test Energy: {results['aggregated']['energy_joules_mean']:.2f} ± "
          f"{results['aggregated']['energy_joules_std']:.2f} J")
    print(f"  Test Carbon: {results['aggregated']['carbon_grams_mean']:.6f} ± "
          f"{results['aggregated']['carbon_grams_std']:.6f} g CO2e")
    
    print("\n✨ MetricsCollector test passed!")
