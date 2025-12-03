"""
GSMM Energy Monitor - GPU + CPU energy monitoring.
Implements the Green Software Maturity Model approach.
"""
from pathlib import Path
from typing import Dict, Optional
import time

from .gpu_monitor import GPUMonitor
from .cpu_energy_monitor import CPUEnergyMonitor


class EnergyMonitorGSMM:
    """
    Energy monitor following GSMM methodology.
    Currently measures GPU + CPU energy (~75-90% coverage).
    Can be extended with Wattmeter for 100% coverage.
    """
    
    def __init__(self, config: dict):
        """
        Initialize GSMM energy monitor.
        
        Args:
            config: Configuration dictionary with GPU and energy settings
        """
        self.config = config
        
        # Initialize GPU monitor if available and enabled
        self.gpu_monitor = None
        if config.get('gpu', {}).get('enabled', False):
            try:
                self.gpu_monitor = GPUMonitor(config)
            except Exception as e:
                print(f"Warning: GPU monitoring unavailable: {e}")
        
        # Initialize CPU energy monitor
        try:
            self.cpu_monitor = CPUEnergyMonitor()
        except Exception as e:
            raise RuntimeError(f"CPU energy monitoring required but unavailable: {e}")
        
        # Grid intensity for carbon calculation (gCO2e/kWh)
        self.grid_intensity = config.get('energy', {}).get('grid_intensity', 250)
    
    def measure_test_energy(self, test_command: str, venv_python: Optional[Path] = None) -> Dict:
        """
        Measure energy consumption for a test execution.
        
        Args:
            test_command: pytest command to execute
            venv_python: Path to virtual environment python (if any)
            
        Returns:
            Dictionary with energy metrics
        """
        start_time = time.time()
        
        # Start GPU monitoring if available
        if self.gpu_monitor:
            self.gpu_monitor.start_monitoring()
        
        # Measure CPU energy (includes test execution)
        # Wrap command with venv python if provided
        if venv_python:
            full_command = f"{venv_python} -m pytest {test_command}"
        else:
            full_command = f"pytest {test_command}"
        
        cpu_metrics = self.cpu_monitor.measure_energy(full_command)
        
        # Stop GPU monitoring if available
        gpu_metrics = {}
        if self.gpu_monitor:
            gpu_metrics = self.gpu_monitor.stop_monitoring()
        
        duration = time.time() - start_time
        
        # Extract energy values
        gpu_energy_joules = gpu_metrics.get('total_energy', 0.0)
        cpu_energy_joules = cpu_metrics.get('cpu_energy_joules', 0.0)
        
        # Calculate total energy (GPU + CPU)
        # Note: This is ~75-90% coverage (missing RAM, Storage, PSU overhead)
        # With Wattmeter, total_energy would be from system measurement (100% coverage)
        total_energy_joules = gpu_energy_joules + cpu_energy_joules
        
        # Calculate derived metrics
        power_watts = total_energy_joules / duration if duration > 0 else 0
        
        # Carbon emissions: energy (kWh) * grid_intensity (gCO2e/kWh)
        energy_kwh = total_energy_joules / 3600 / 1000  # J -> kWh
        carbon_grams = energy_kwh * self.grid_intensity
        
        # Energy efficiency: tests per Joule (inverse of energy per test)
        energy_efficiency = 1.0 / total_energy_joules if total_energy_joules > 0 else 0
        
        # Compile all metrics
        metrics = {
            # Core energy metrics (GSMM)
            'gpu_energy_joules': gpu_energy_joules,
            'cpu_energy_joules': cpu_energy_joules,
            'total_energy_joules': total_energy_joules,
            'power_watts': power_watts,
            'carbon_grams': carbon_grams,
            'energy_efficiency': energy_efficiency,
            
            # Duration
            'duration_seconds': duration,
            
            # GPU details (if available)
            **{f'gpu_{k}': v for k, v in gpu_metrics.items() if k != 'total_energy'},
            
            # CPU details
            'cpu_power_watts': cpu_metrics.get('cpu_power_watts', 0),
            'cpu_samples': cpu_metrics.get('samples', 0),
        }
        
        return metrics
    
    def measure_baseline(self, duration_seconds: float = 5.0) -> Dict:
        """
        Measure baseline energy consumption (system idle).
        
        Args:
            duration_seconds: How long to measure baseline
            
        Returns:
            Dictionary with baseline energy metrics
        """
        # Use sleep command as baseline
        baseline_command = f"sleep {duration_seconds}"
        return self.measure_test_energy(baseline_command)


if __name__ == "__main__":
    # Test
    print("Testing EnergyMonitorGSMM...")
    
    config = {
        'gpu': {
            'enabled': True,
            'sampling_interval': 0.1,
            'track_temperature': True,
            'track_power': True
        },
        'energy': {
            'grid_intensity': 250  # Spain
        }
    }
    
    monitor = EnergyMonitorGSMM(config)
    
    print("\n1. Testing baseline measurement (5s)...")
    baseline = monitor.measure_baseline(5.0)
    print(f"   Baseline Energy: {baseline['total_energy_joules']:.2f} J")
    print(f"   GPU: {baseline['gpu_energy_joules']:.2f} J")
    print(f"   CPU: {baseline['cpu_energy_joules']:.2f} J")
    print(f"   Power: {baseline['power_watts']:.2f} W")
    print(f"   Carbon: {baseline['carbon_grams']:.4f} g")
    
    print("\n✅ EnergyMonitorGSMM working!")
