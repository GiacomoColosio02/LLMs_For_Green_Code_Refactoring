"""
GPU monitoring using NVIDIA Management Library (NVML).
"""
import time
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("⚠️ pynvml not available. GPU monitoring disabled.")


@dataclass
class GPUSample:
    """Single sample of GPU usage."""
    timestamp: float
    gpu_utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_celsius: Optional[float] = None
    power_draw_watts: Optional[float] = None


class GPUMonitor:
    """Monitor GPU usage using NVML."""
    
    def __init__(
        self,
        device_index: int = 0,
        track_temperature: bool = True,
        track_power: bool = True
    ):
        """
        Initialize GPU monitor.
        
        Args:
            device_index: GPU device index (0 = first GPU)
            track_temperature: Track GPU temperature
            track_power: Track GPU power draw
        """
        if not NVML_AVAILABLE:
            raise RuntimeError("pynvml not available. Install with: pip install pynvml")
        
        self.device_index = device_index
        self.track_temperature = track_temperature
        self.track_power = track_power
        self.samples: List[GPUSample] = []
        
        # Initialize NVML
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            
            # Get GPU name
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            print(f"📊 GPU Monitor initialized: {self.gpu_name}")
            
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize GPU {device_index}: {e}")
    
    def sample_once(self) -> GPUSample:
        """
        Take a single GPU measurement.
        
        Returns:
            GPUSample with current usage
        """
        try:
            # GPU utilization (compute)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_percent = utilization.gpu
            
            # Memory usage
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used_mb = mem_info.used / (1024 ** 2)
            memory_total_mb = mem_info.total / (1024 ** 2)
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            # Optional: Temperature
            temperature = None
            if self.track_temperature:
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        self.handle,
                        pynvml.NVML_TEMPERATURE_GPU
                    )
                except pynvml.NVMLError:
                    pass
            
            # Optional: Power draw
            power_watts = None
            if self.track_power:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                    power_watts = power_mw / 1000.0
                except pynvml.NVMLError:
                    pass
            
            sample = GPUSample(
                timestamp=time.time(),
                gpu_utilization_percent=gpu_percent,
                memory_used_mb=memory_used_mb,
                memory_total_mb=memory_total_mb,
                memory_percent=memory_percent,
                temperature_celsius=temperature,
                power_draw_watts=power_watts
            )
            
            return sample
            
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to sample GPU: {e}")
    
    def start_monitoring(self):
        """Start collecting samples."""
        self.samples = []
    
    def add_sample(self):
        """Add a sample to the collection."""
        sample = self.sample_once()
        self.samples.append(sample)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics from collected samples.
        
        Returns:
            Dictionary with mean, peak, min for GPU metrics
        """
        if not self.samples:
            return {}
        
        gpu_util = [s.gpu_utilization_percent for s in self.samples]
        mem_percent = [s.memory_percent for s in self.samples]
        mem_used = [s.memory_used_mb for s in self.samples]
        
        stats = {
            'gpu_utilization_mean_percent': statistics.mean(gpu_util),
            'gpu_utilization_peak_percent': max(gpu_util),
            'gpu_utilization_min_percent': min(gpu_util),
            'gpu_utilization_std_percent': statistics.stdev(gpu_util) if len(gpu_util) > 1 else 0.0,
            
            'gpu_memory_mean_percent': statistics.mean(mem_percent),
            'gpu_memory_peak_percent': max(mem_percent),
            'gpu_memory_mean_mb': statistics.mean(mem_used),
            'gpu_memory_peak_mb': max(mem_used),
            
            'num_samples': len(self.samples)
        }
        
        # Add optional metrics if tracked
        if self.track_temperature:
            temps = [s.temperature_celsius for s in self.samples if s.temperature_celsius is not None]
            if temps:
                stats['gpu_temperature_mean_celsius'] = statistics.mean(temps)
                stats['gpu_temperature_peak_celsius'] = max(temps)
        
        if self.track_power:
            powers = [s.power_draw_watts for s in self.samples if s.power_draw_watts is not None]
            if powers:
                stats['gpu_power_mean_watts'] = statistics.mean(powers)
                stats['gpu_power_peak_watts'] = max(powers)
        
        return stats
    
    def get_raw_samples(self) -> List[Dict]:
        """Get all raw samples as list of dicts."""
        return [asdict(s) for s in self.samples]
    
    def shutdown(self):
        """Cleanup NVML."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    
    def __del__(self):
        """Ensure NVML is shutdown on deletion."""
        self.shutdown()


def is_gpu_available() -> bool:
    """Check if GPU monitoring is available."""
    if not NVML_AVAILABLE:
        return False
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return device_count > 0
    except:
        return False


if __name__ == "__main__":
    # Test the GPU monitor
    print("Testing GPU Monitor...")
    print("=" * 60)
    
    if not is_gpu_available():
        print("❌ No GPU available for testing")
        exit(1)
    
    monitor = GPUMonitor(device_index=0)
    monitor.start_monitoring()
    
    # Simulate some monitoring
    print("\n📊 Monitoring GPU for 3 seconds...\n")
    for i in range(30):
        monitor.add_sample()
        time.sleep(0.1)
    
    stats = monitor.get_statistics()
    
    print("GPU Statistics:")
    print(f"  GPU Name: {monitor.gpu_name}")
    print(f"  Utilization mean: {stats['gpu_utilization_mean_percent']:.1f}%")
    print(f"  Utilization peak: {stats['gpu_utilization_peak_percent']:.1f}%")
    print(f"  Memory mean: {stats['gpu_memory_mean_mb']:.1f} MB ({stats['gpu_memory_mean_percent']:.1f}%)")
    print(f"  Memory peak: {stats['gpu_memory_peak_mb']:.1f} MB ({stats['gpu_memory_peak_percent']:.1f}%)")
    
    if 'gpu_temperature_mean_celsius' in stats:
        print(f"  Temperature mean: {stats['gpu_temperature_mean_celsius']:.1f}°C")
        print(f"  Temperature peak: {stats['gpu_temperature_peak_celsius']:.1f}°C")
    
    if 'gpu_power_mean_watts' in stats:
        print(f"  Power mean: {stats['gpu_power_mean_watts']:.1f} W")
        print(f"  Power peak: {stats['gpu_power_peak_watts']:.1f} W")
    
    print(f"  Samples: {stats['num_samples']}")
    
    monitor.shutdown()
    print("\n✨ GPU monitor test passed!")
