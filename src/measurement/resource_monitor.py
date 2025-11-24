"""
CPU and RAM monitoring utilities using psutil.
"""
import psutil
import time
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ResourceSample:
    """Single sample of resource usage."""
    timestamp: float
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float


class ResourceMonitor:
    """Monitor CPU and memory usage of a process."""
    
    def __init__(self, pid: Optional[int] = None, interval: float = 0.1):
        """
        Initialize resource monitor.
        
        Args:
            pid: Process ID to monitor. If None, monitors current process.
            interval: Sampling interval in seconds
        """
        self.pid = pid if pid is not None else psutil.Process().pid
        self.interval = interval
        self.samples: List[ResourceSample] = []
        
    def sample_once(self) -> ResourceSample:
        """
        Take a single resource measurement.
        
        Returns:
            ResourceSample with current usage
        """
        try:
            proc = psutil.Process(self.pid)
            
            # CPU percent (can be > 100% on multi-core)
            cpu_percent = proc.cpu_percent(interval=self.interval)
            
            # Memory info
            mem_info = proc.memory_info()
            memory_rss_mb = mem_info.rss / (1024 ** 2)  # Resident Set Size
            memory_vms_mb = mem_info.vms / (1024 ** 2)  # Virtual Memory Size
            
            sample = ResourceSample(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_rss_mb=memory_rss_mb,
                memory_vms_mb=memory_vms_mb
            )
            
            return sample
            
        except psutil.NoSuchProcess:
            raise RuntimeError(f"Process {self.pid} no longer exists")
    
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
            Dictionary with mean, peak, min for CPU and RAM
        """
        if not self.samples:
            return {}
        
        cpu_values = [s.cpu_percent for s in self.samples]
        ram_values = [s.memory_rss_mb for s in self.samples]
        
        stats = {
            'cpu_usage_mean_percent': statistics.mean(cpu_values),
            'cpu_usage_peak_percent': max(cpu_values),
            'cpu_usage_min_percent': min(cpu_values),
            'cpu_usage_std_percent': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0.0,
            
            'ram_usage_mean_mb': statistics.mean(ram_values),
            'ram_usage_peak_mb': max(ram_values),
            'ram_usage_min_mb': min(ram_values),
            'ram_usage_std_mb': statistics.stdev(ram_values) if len(ram_values) > 1 else 0.0,
            
            'num_samples': len(self.samples)
        }
        
        return stats
    
    def get_raw_samples(self) -> List[Dict]:
        """Get all raw samples as list of dicts."""
        return [asdict(s) for s in self.samples]


def monitor_process_resources(pid: int, duration: float, interval: float = 0.1) -> Dict:
    """
    Monitor a process for a given duration.
    
    Args:
        pid: Process ID to monitor
        duration: How long to monitor (seconds)
        interval: Sampling interval (seconds)
        
    Returns:
        Dictionary with statistics
    """
    monitor = ResourceMonitor(pid=pid, interval=interval)
    monitor.start_monitoring()
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        monitor.add_sample()
        time.sleep(interval)
    
    return monitor.get_statistics()


if __name__ == "__main__":
    # Test the resource monitor
    print("Testing Resource Monitor...")
    print("Monitoring current process for 3 seconds...\n")
    
    import os
    
    monitor = ResourceMonitor(pid=os.getpid(), interval=0.1)
    monitor.start_monitoring()
    
    # Simulate some work
    for i in range(30):
        monitor.add_sample()
        # Do some CPU work
        _ = sum(range(10000))
        time.sleep(0.1)
    
    stats = monitor.get_statistics()
    
    print("Statistics:")
    print(f"  CPU mean: {stats['cpu_usage_mean_percent']:.2f}%")
    print(f"  CPU peak: {stats['cpu_usage_peak_percent']:.2f}%")
    print(f"  RAM mean: {stats['ram_usage_mean_mb']:.2f} MB")
    print(f"  RAM peak: {stats['ram_usage_peak_mb']:.2f} MB")
    print(f"  Samples: {stats['num_samples']}")
    
    print("\n Resource monitor test passed!")