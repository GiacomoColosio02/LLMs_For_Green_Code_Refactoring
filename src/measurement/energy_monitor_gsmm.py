"""
GSMM Energy Monitor - GPU + CPU energy monitoring with resource tracking.
Implements the Green Software Maturity Model approach.
"""
from pathlib import Path
from typing import Dict, Optional
import time
import threading
import psutil

from .gpu_monitor import GPUMonitor
from .cpu_energy_monitor import CPUEnergyMonitor


class SystemResourceTracker:
    """Track system-wide CPU and RAM usage during test execution."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.cpu_samples = []
        self.ram_samples = []
        self.running = False
        self.thread = None
    
    def _sample_loop(self):
        """Sampling loop running in separate thread."""
        while self.running:
            # System CPU percentage
            cpu_pct = psutil.cpu_percent(interval=self.interval)
            self.cpu_samples.append(cpu_pct)
            
            # System RAM usage
            mem = psutil.virtual_memory()
            ram_mb = mem.used / (1024 ** 2)
            self.ram_samples.append(ram_mb)
    
    def start(self):
        """Start sampling."""
        self.cpu_samples = []
        self.ram_samples = []
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> Dict:
        """Stop sampling and return statistics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if not self.cpu_samples:
            return {
                'cpu_usage_mean_percent': 0.0,
                'cpu_usage_peak_percent': 0.0,
                'ram_usage_mean_mb': 0.0,
                'ram_usage_peak_mb': 0.0
            }
        
        import statistics
        return {
            'cpu_usage_mean_percent': statistics.mean(self.cpu_samples),
            'cpu_usage_peak_percent': max(self.cpu_samples),
            'ram_usage_mean_mb': statistics.mean(self.ram_samples),
            'ram_usage_peak_mb': max(self.ram_samples)
        }


class GPUMonitorThread:
    """Wrapper for GPU monitor with threading."""
    
    def __init__(self, gpu_monitor: GPUMonitor, interval: float = 0.1):
        self.gpu_monitor = gpu_monitor
        self.interval = interval
        self.running = False
        self.thread = None
    
    def _sample_loop(self):
        """Sampling loop for GPU."""
        while self.running:
            try:
                self.gpu_monitor.add_sample()
                time.sleep(self.interval)
            except Exception as e:
                print(f"Warning: GPU sampling error: {e}")
                break
    
    def start(self):
        """Start GPU sampling thread."""
        self.gpu_monitor.start_monitoring()
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> Dict:
        """Stop sampling and return statistics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        stats = self.gpu_monitor.get_statistics()
        self.gpu_monitor.shutdown()
        
        # Rename utilization -> usage for consistency
        renamed = {}
        for key, value in stats.items():
            new_key = key.replace('utilization', 'usage')
            renamed[new_key] = value
        
        return renamed


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
        self.gpu_monitor_thread = None
        if config.get('gpu', {}).get('enabled', False):
            try:
                gpu_config = config.get('gpu', {})
                gpu_monitor = GPUMonitor(
                    device_index=gpu_config.get('device_index', 0),
                    track_temperature=gpu_config.get('track_temperature', True),
                    track_power=gpu_config.get('track_power', True)
                )
                self.gpu_monitor_thread = GPUMonitorThread(
                    gpu_monitor, 
                    interval=gpu_config.get('sampling_interval', 0.1)
                )
            except Exception as e:
                print(f"Warning: GPU monitoring unavailable: {e}")
        
        # Initialize CPU energy monitor
        try:
            self.cpu_monitor = CPUEnergyMonitor()
        except Exception as e:
            raise RuntimeError(f"CPU energy monitoring required but unavailable: {e}")
        
        # Grid intensity for carbon calculation (gCO2e/kWh)
        self.grid_intensity = config.get('energy', {}).get('grid_intensity', 250)
    
    def measure_test_energy(self, test_command: str, venv_python: Optional[Path] = None, wrap_with_pytest: Optional[bool] = None) -> Dict:
        """
        Measure energy consumption for a test execution.
        
        Args:
            test_command: Command to execute (pytest test or shell command)
            venv_python: Path to virtual environment python (if any) - IGNORED if command already contains python/pytest
            wrap_with_pytest: If True, wraps command with pytest. If None, auto-detects. If False, uses command as-is.
            
        Returns:
            Dictionary with energy metrics
        """
        start_time = time.time()
        
        # Start GPU monitoring thread if available
        if self.gpu_monitor_thread:
            self.gpu_monitor_thread.start()
        
        # Start system resource tracking
        resource_tracker = SystemResourceTracker(interval=0.1)
        resource_tracker.start()
        
        # Auto-detect if command needs pytest wrapping
        if wrap_with_pytest is None:
            # If command already contains pytest or python, don't wrap
            wrap_with_pytest = not ('pytest' in test_command or 'python' in test_command)
        
        # Build full command
        if wrap_with_pytest:
            # This is a test name that needs to be wrapped with pytest
            if venv_python:
                full_command = f"{venv_python} -m pytest {test_command}"
            else:
                full_command = f"pytest {test_command}"
        else:
            # This is already a complete command (e.g., "cd /path && python -m pytest test")
            # Use it as-is
            full_command = test_command
        
        # Measure CPU energy (includes test execution)
        # cpu_energy_monitor will wrap complex commands (with cd, &&) in bash script
        cpu_metrics = self.cpu_monitor.measure_energy(full_command)
        
        # Stop resource tracking
        resource_stats = resource_tracker.stop()
        
        # Stop GPU monitoring thread if available
        gpu_metrics = {}
        if self.gpu_monitor_thread:
            gpu_metrics = self.gpu_monitor_thread.stop()
        
        duration = time.time() - start_time
        
        # Extract energy values
        gpu_energy_joules = 0.0
        if gpu_metrics and 'gpu_power_mean_watts' in gpu_metrics:
            # Calculate GPU energy from power samples
            gpu_power_mean = gpu_metrics['gpu_power_mean_watts']
            gpu_energy_joules = gpu_power_mean * duration
        
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
            
            # CPU/RAM usage (from resource tracker)
            'cpu_usage_mean_percent': resource_stats['cpu_usage_mean_percent'],
            'cpu_usage_peak_percent': resource_stats['cpu_usage_peak_percent'],
            'ram_usage_mean_mb': resource_stats['ram_usage_mean_mb'],
            'ram_usage_peak_mb': resource_stats['ram_usage_peak_mb'],
            
            # GPU details (if available) - already renamed to usage
            **{k: v for k, v in gpu_metrics.items() if k not in ['gpu_power_mean_watts', 'gpu_power_peak_watts']},
            
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
        # Use sleep command as baseline (explicitly don't wrap with pytest)
        baseline_command = f"sleep {duration_seconds}"
        return self.measure_test_energy(baseline_command, wrap_with_pytest=False)


if __name__ == "__main__":
    # Test
    print("Testing EnergyMonitorGSMM with Threading...")
    
    config = {
        'gpu': {
            'enabled': True,
            'device_index': 0,
            'sampling_interval': 0.1,
            'track_temperature': True,
            'track_power': True
        },
        'energy': {
            'grid_intensity': 250  # Spain
        }
    }
    
    monitor = EnergyMonitorGSMM(config)
    
    print("\n1. Testing baseline measurement (3s)...")
    baseline = monitor.measure_baseline(3.0)
    print(f"   Baseline Energy: {baseline['total_energy_joules']:.2f} J")
    print(f"   CPU Usage: {baseline['cpu_usage_mean_percent']:.1f}%")
    print(f"   RAM Usage: {baseline['ram_usage_mean_mb']:.1f} MB")
    print(f"   GPU Usage: {baseline.get('gpu_usage_mean_percent', 0):.1f}%")
    
    print("\n✅ EnergyMonitorGSMM with threading working!")
