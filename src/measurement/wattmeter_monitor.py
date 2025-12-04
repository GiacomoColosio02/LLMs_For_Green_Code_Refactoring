"""
Wattmeter Monitor - NETIO PowerBOX 4KF measurements
Provides system-level power measurements (100% energy coverage)
"""
import time
import requests
import statistics
import threading
from typing import Dict, Optional, List
from .base_monitor import BaseMonitor


class WattmeterMonitor(BaseMonitor):
    """
    Monitor power consumption using NETIO PowerBOX 4KF wattmeter.
    
    Endpoints:
        - JSON API: http://{ip}/netio.json
        - Output ID: 1 (server connected to Output 1)
    
    Features:
        - System-level power measurement (wall power)
        - 100% energy coverage (GPU + CPU + RAM + PSU + all components)
        - Replaces GPU+CPU partial measurements when available
    """
    
    def __init__(self, ip: str = "10.4.60.25", output_id: int = 1, 
                 timeout: int = 5, polling_interval: float = 0.1):
        """
        Initialize wattmeter monitor.
        
        Args:
            ip: Wattmeter IP address (default: 10.4.60.25)
            output_id: Output ID where server is connected (default: 1)
            timeout: Request timeout in seconds
            polling_interval: Sampling interval in seconds
        """
        super().__init__()
        self.ip = ip
        self.endpoint = f"http://{ip}/netio.json"
        self.output_id = output_id
        self.timeout = timeout
        self.polling_interval = polling_interval
        
        self.power_samples: List[float] = []
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify wattmeter is accessible."""
        try:
            response = requests.get(self.endpoint, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Verify output exists
            outputs = data.get('Outputs', [])
            if not outputs or len(outputs) < self.output_id:
                raise ValueError(f"Output {self.output_id} not found in wattmeter response")
            
            print(f"✅ Wattmeter connected: {self.ip}")
            print(f"   Output {self.output_id}: {outputs[self.output_id-1].get('Name', 'Unknown')}")
            
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to connect to wattmeter at {self.ip}: {e}")
    
    def get_current_power(self) -> Optional[float]:
        """
        Get current power reading from wattmeter.
        
        Returns:
            Current power in Watts, or None if read fails
        """
        try:
            response = requests.get(self.endpoint, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            outputs = data.get('Outputs', [])
            if len(outputs) < self.output_id:
                return None
            
            output = outputs[self.output_id - 1]
            power_watts = output.get('Load', 0)  # Load is in Watts
            
            return power_watts
            
        except Exception as e:
            print(f"Warning: Wattmeter read error: {e}")
            return None
    
    def start_monitoring(self):
        """Start power monitoring."""
        self.power_samples = []
        print(f"📊 Wattmeter monitoring started (output {self.output_id})")
    
    def add_sample(self):
        """Add a power sample (called by monitoring thread)."""
        power = self.get_current_power()
        if power is not None:
            self.power_samples.append(power)
    
    def stop_monitoring(self) -> Dict[str, float]:
        """
        Stop monitoring and calculate statistics.
        
        Returns:
            Dict with power and energy metrics
        """
        if not self.power_samples:
            return {}
        
        duration = len(self.power_samples) * self.polling_interval
        
        metrics = {
            'system_power_mean_watts': statistics.mean(self.power_samples),
            'system_power_peak_watts': max(self.power_samples),
            'system_power_min_watts': min(self.power_samples),
            'system_energy_joules': statistics.mean(self.power_samples) * duration,
            'samples_count': len(self.power_samples),
            'duration_seconds': duration
        }
        
        print(f"✅ Wattmeter: {metrics['system_energy_joules']:.2f} J "
              f"({metrics['system_power_mean_watts']:.2f} W avg)")
        
        return metrics
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics without stopping."""
        return self.stop_monitoring()
    
    def shutdown(self):
        """Cleanup (wattmeter needs no shutdown)."""
        pass


class WattmeterMonitorThread:
    """Thread wrapper for wattmeter continuous sampling."""
    
    def __init__(self, wattmeter: WattmeterMonitor):
        self.wattmeter = wattmeter
        self.running = False
        self.thread = None
    
    def _sample_loop(self):
        """Continuous sampling loop."""
        while self.running:
            try:
                self.wattmeter.add_sample()
                time.sleep(self.wattmeter.polling_interval)
            except Exception as e:
                print(f"Warning: Wattmeter sampling error: {e}")
                break
    
    def start(self):
        """Start monitoring thread."""
        self.wattmeter.start_monitoring()
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and get results."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.wattmeter.stop_monitoring()
