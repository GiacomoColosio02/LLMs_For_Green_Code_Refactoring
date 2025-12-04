"""
GSMM Energy Monitor - GPU + CPU + Wattmeter energy monitoring with resource tracking.
Implements the Green Software Maturity Model approach.
"""
from pathlib import Path
from typing import Dict, Optional
import time
import threading
import psutil

from .gpu_monitor import GPUMonitor
from .cpu_energy_monitor import CPUEnergyMonitor
from .wattmeter_monitor import WattmeterMonitor, WattmeterMonitorThread


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
        
        # DON'T shutdown NVML here - it breaks subsequent measurements!
        # NVML will be shutdown when the whole EnergyMonitorGSMM is destroyed
        
        # Rename utilization -> usage for consistency
        renamed = {}
        for key, value in stats.items():
            new_key = key.replace('utilization', 'usage')
            renamed[new_key] = value
        
        return renamed


class EnergyMonitorGSMM:
    """
    Energy monitor following GSMM methodology.
    Supports GPU + CPU energy (~75-90% coverage) and optional Wattmeter (100% coverage).
    """
    
    def __init__(self, config: dict):
        """
        Initialize GSMM energy monitor.
        
        Args:
            config: Configuration dictionary with GPU, CPU, and wattmeter settings
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
                print("✅ GPU monitoring enabled")
            except Exception as e:
                print(f"⚠️  GPU monitoring unavailable: {e}")
        
        # Initialize CPU energy monitor
        try:
            self.cpu_monitor = CPUEnergyMonitor()
        except Exception as e:
            raise RuntimeError(f"CPU energy monitoring required but unavailable: {e}")
        
        # Initialize Wattmeter (optional, provides 100% system coverage)
        self.wattmeter_enabled = config.get('wattmeter', {}).get('enabled', False)
        self.wattmeter_thread = None
        if self.wattmeter_enabled:
            try:
                wattmeter_config = config.get('wattmeter', {})
                wattmeter = WattmeterMonitor(
                    ip=wattmeter_config.get('ip', '10.4.60.25'),
                    output_id=wattmeter_config.get('output_id', 1),
                    timeout=wattmeter_config.get('timeout', 5),
                    polling_interval=wattmeter_config.get('polling_interval', 0.1)
                )
                self.wattmeter_thread = WattmeterMonitorThread(wattmeter)
                print("✅ Wattmeter monitoring enabled (SYSTEM-LEVEL, 100% coverage)")
            except Exception as e:
                print(f"⚠️  Wattmeter unavailable: {e}")
                print("   Continuing with GPU+CPU measurements only (~75-90% coverage)")
                self.wattmeter_enabled = False
        
        # Grid intensity for carbon calculation (gCO2e/kWh)
        self.grid_intensity = config.get('energy', {}).get('grid_intensity', 250)
        
        # Print coverage status
        if self.wattmeter_enabled:
            print(f"✅ GSMM Energy monitoring enabled (grid: {self.grid_intensity} gCO2e/kWh)")
            print("   Coverage: 100% (wattmeter system-level)")
        else:
            print(f"✅ GSMM Energy monitoring enabled (grid: {self.grid_intensity} gCO2e/kWh)")
            print("   Coverage: ~75-90% (GPU + CPU only)")
    
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
        
        # Start wattmeter thread FIRST (for complete system coverage)
        if self.wattmeter_thread:
            self.wattmeter_thread.start()
        
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
        
        # Stop wattmeter thread if available
        wattmeter_metrics = {}
        if self.wattmeter_thread:
            wattmeter_metrics = self.wattmeter_thread.stop()
        
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
        # With Wattmeter, system_energy provides 100% coverage
        total_energy_joules = gpu_energy_joules + cpu_energy_joules
        
        # Calculate derived metrics
        power_watts = total_energy_joules / duration if duration > 0 else 0
        
        # Carbon emissions: energy (kWh) * grid_intensity (gCO2e/kWh)
        energy_kwh = total_energy_joules / 3600 / 1000  # J -> kWh
        carbon_grams = energy_kwh * self.grid_intensity
        
        # Energy efficiency: tests per Joule (inverse of energy per test)
        energy_efficiency = 1.0 / total_energy_joules if total_energy_joules > 0 else 0
        
        # Compile GREEN metrics (GSMM)
        green_metrics = {
            'gpu_energy_joules': gpu_energy_joules,
            'cpu_energy_joules': cpu_energy_joules,
            'total_energy_joules': total_energy_joules,
            'power_watts': power_watts,
            'carbon_grams': carbon_grams,
            'energy_efficiency': energy_efficiency,
        }
        
        # Add wattmeter metrics if available (100% system coverage)
        if wattmeter_metrics:
            green_metrics['system_energy_joules'] = wattmeter_metrics.get('system_energy_joules', 0)
            green_metrics['system_power_mean_watts'] = wattmeter_metrics.get('system_power_mean_watts', 0)
            green_metrics['system_power_peak_watts'] = wattmeter_metrics.get('system_power_peak_watts', 0)
            
            # Recalculate carbon with system energy (more accurate, 100% coverage)
            if green_metrics['system_energy_joules'] > 0:
                system_energy_kwh = green_metrics['system_energy_joules'] / 3600000.0  # J -> kWh
                green_metrics['carbon_grams_system'] = system_energy_kwh * self.grid_intensity
        
        # Compile all metrics
        metrics = {
            # GREEN metrics (6 core + 3 wattmeter if available)
            **green_metrics,
            
            # EFFICIENCY metrics (7 total)
            'duration_seconds': duration,
            'cpu_usage_mean_percent': resource_stats['cpu_usage_mean_percent'],
            'cpu_usage_peak_percent': resource_stats['cpu_usage_peak_percent'],
            'ram_usage_peak_mb': resource_stats['ram_usage_peak_mb'],
            
            # GPU usage metrics (if available)
            'gpu_usage_mean_percent': gpu_metrics.get('gpu_usage_mean_percent', 0),
            'gpu_usage_peak_percent': gpu_metrics.get('gpu_usage_peak_percent', 0),
            'gpu_memory_peak_mb': gpu_metrics.get('gpu_memory_peak_mb', 0),
            
            # Additional details
            'ram_usage_mean_mb': resource_stats['ram_usage_mean_mb'],
            **{k: v for k, v in gpu_metrics.items() if k not in ['gpu_power_mean_watts', 'gpu_power_peak_watts', 'gpu_usage_mean_percent', 'gpu_usage_peak_percent', 'gpu_memory_peak_mb']},
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
    
    def __del__(self):
        """Cleanup NVML when EnergyMonitorGSMM is destroyed."""
        if self.gpu_monitor_thread and self.gpu_monitor_thread.gpu_monitor:
            try:
                self.gpu_monitor_thread.gpu_monitor.shutdown()
            except:
                pass


if __name__ == "__main__":
    # Test
    print("Testing EnergyMonitorGSMM with Wattmeter...")
    
    config = {
        'gpu': {
            'enabled': True,
            'device_index': 0,
            'sampling_interval': 0.1,
            'track_temperature': True,
            'track_power': True
        },
        'wattmeter': {
            'enabled': True,
            'ip': '10.4.60.25',
            'output_id': 1,
            'timeout': 5,
            'polling_interval': 0.1
        },
        'energy': {
            'grid_intensity': 250  # Spain
        }
    }
    
    monitor = EnergyMonitorGSMM(config)
    
    print("\n1. Testing baseline measurement (3s)...")
    baseline = monitor.measure_baseline(3.0)
    print(f"   Total Energy: {baseline['total_energy_joules']:.2f} J")
    if 'system_energy_joules' in baseline:
        print(f"   System Energy (Wattmeter): {baseline['system_energy_joules']:.2f} J")
    print(f"   CPU Usage: {baseline['cpu_usage_mean_percent']:.1f}%")
    print(f"   RAM Usage: {baseline['ram_usage_mean_mb']:.1f} MB")
    
    print("\n✅ EnergyMonitorGSMM with Wattmeter working!")
