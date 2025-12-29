"""
GPU monitoring using NVIDIA Management Library (NVML).
Uses singleton pattern for NVML to prevent segfaults from multiple init/shutdown.
"""
import time
import statistics
import atexit
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("⚠️ pynvml not available. GPU monitoring disabled.")


# =============================================================================
# NVML SINGLETON MANAGER
# =============================================================================
class NVMLManager:
    """
    Singleton manager for NVML initialization.
    Ensures NVML is initialized only ONCE per process and never shutdown
    until process exit.
    """
    _instance = None
    _initialized = False
    _device_count = 0
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def ensure_initialized(self) -> bool:
        """
        Ensure NVML is initialized. Safe to call multiple times.
        
        Returns:
            True if NVML is available and initialized, False otherwise
        """
        if not NVML_AVAILABLE:
            return False
        
        if self._initialized:
            return True
        
        try:
            pynvml.nvmlInit()
            self._device_count = pynvml.nvmlDeviceGetCount()
            self._initialized = True
            
            # Register shutdown at process exit (not before!)
            atexit.register(self._shutdown_at_exit)
            
            print(f"✅ NVML initialized (found {self._device_count} GPU(s))")
            return True
            
        except pynvml.NVMLError as e:
            print(f"⚠️ NVML initialization failed: {e}")
            return False
    
    def _shutdown_at_exit(self):
        """Called only at process exit by atexit."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
                print("✅ NVML shutdown complete")
            except:
                pass
            self._initialized = False
    
    def is_available(self) -> bool:
        """Check if GPU is available (initializes if needed)."""
        return self.ensure_initialized() and self._device_count > 0
    
    def get_device_count(self) -> int:
        """Get number of GPUs."""
        if self.ensure_initialized():
            return self._device_count
        return 0
    
    def get_handle(self, device_index: int = 0):
        """Get device handle."""
        if not self.ensure_initialized():
            raise RuntimeError("NVML not available")
        if device_index >= self._device_count:
            raise ValueError(f"Device {device_index} not found (have {self._device_count})")
        return pynvml.nvmlDeviceGetHandleByIndex(device_index)


# Global singleton instance
_nvml_manager = NVMLManager()


def is_gpu_available() -> bool:
    """
    Check if GPU monitoring is available.
    Safe to call multiple times - uses singleton.
    """
    return _nvml_manager.is_available()


# =============================================================================
# GPU SAMPLE DATA CLASS
# =============================================================================
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


# =============================================================================
# GPU MONITOR CLASS
# =============================================================================
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
        if not _nvml_manager.is_available():
            raise RuntimeError("GPU not available. Install pynvml and ensure NVIDIA driver is loaded.")
        
        self.device_index = device_index
        self.track_temperature = track_temperature
        self.track_power = track_power
        self.samples: List[GPUSample] = []
        
        # Get handle from singleton manager
        self.handle = _nvml_manager.get_handle(device_index)
        
        # Get GPU name
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
        print(f"📊 GPU Monitor initialized: {self.gpu_name}")
    
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
        """
        Cleanup method - DOES NOTHING now.
        NVML shutdown is handled by NVMLManager at process exit.
        Kept for API compatibility.
        """
        # DO NOT call nvmlShutdown() here!
        # The singleton manager handles this at process exit.
        pass
    
    def __del__(self):
        """Destructor - does nothing, shutdown handled by singleton."""
        pass


# =============================================================================
# MAIN TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing GPU Monitor with Singleton NVML Manager...")
    print("=" * 60)
    
    # Test 1: Check availability multiple times (should not crash)
    print("\n1. Testing is_gpu_available() multiple times...")
    for i in range(5):
        result = is_gpu_available()
        print(f"   Call {i+1}: {result}")
    
    if not is_gpu_available():
        print("❌ No GPU available for testing")
        exit(1)
    
    # Test 2: Create multiple GPUMonitor instances
    print("\n2. Creating multiple GPUMonitor instances...")
    monitors = []
    for i in range(3):
        m = GPUMonitor(device_index=0)
        monitors.append(m)
        print(f"   Created monitor {i+1}: {m.gpu_name}")
    
    # Test 3: Sample from first monitor
    print("\n3. Sampling from first monitor (3 seconds)...")
    monitor = monitors[0]
    monitor.start_monitoring()
    
    for i in range(30):
        monitor.add_sample()
        time.sleep(0.1)
    
    stats = monitor.get_statistics()
    
    print("   GPU Statistics:")
    print(f"     Utilization mean: {stats['gpu_utilization_mean_percent']:.1f}%")
    print(f"     Utilization peak: {stats['gpu_utilization_peak_percent']:.1f}%")
    print(f"     Memory mean: {stats['gpu_memory_mean_mb']:.1f} MB")
    print(f"     Memory peak: {stats['gpu_memory_peak_mb']:.1f} MB")
    
    if 'gpu_temperature_mean_celsius' in stats:
        print(f"     Temperature mean: {stats['gpu_temperature_mean_celsius']:.1f}°C")
    
    if 'gpu_power_mean_watts' in stats:
        print(f"     Power mean: {stats['gpu_power_mean_watts']:.1f} W")
    
    # Test 4: Delete monitors (should NOT cause segfault)
    print("\n4. Deleting monitors (testing no segfault)...")
    del monitors
    del monitor
    
    # Test 5: Create new monitor after deletion (should work!)
    print("\n5. Creating new monitor after deletion...")
    new_monitor = GPUMonitor(device_index=0)
    sample = new_monitor.sample_once()
    print(f"   ✅ New monitor works! GPU at {sample.gpu_utilization_percent}%")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! GPU monitoring is stable.")
    print("   NVML will shutdown automatically at process exit.")
    print("=" * 60)