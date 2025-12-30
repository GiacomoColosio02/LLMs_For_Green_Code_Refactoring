"""
Wattmeter Monitor - NETIO PowerBOX 4KF measurements
Provides system-level power measurements (100% energy coverage)

Uses persistent HTTP session to avoid connection timeout issues
when polling continuously.
"""
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import statistics
import threading
from typing import Dict, Optional, List


class WattmeterMonitor:
    """
    Monitor power consumption using NETIO PowerBOX 4KF wattmeter.
    
    Endpoints:
        - JSON API: http://{ip}/netio.json
        - Output ID: 1 (server connected to Output 1)
    
    Features:
        - System-level power measurement (wall power)
        - 100% energy coverage (GPU + CPU + RAM + PSU + all components)
        - Persistent HTTP session for reliable polling
    """
    
    def __init__(self, ip: str = "10.4.60.25", output_id: int = 1, 
                 timeout: int = 10, polling_interval: float = 1.0):
        """
        Initialize wattmeter monitor.
        
        Args:
            ip: Wattmeter IP address
            output_id: Output ID where server is connected
            timeout: Request timeout in seconds
            polling_interval: Sampling interval in seconds (minimum 1.0 recommended)
        """
        self.ip = ip
        self.endpoint = f"http://{ip}/netio.json"
        self.output_id = output_id
        self.timeout = timeout
        self.polling_interval = max(polling_interval, 1.0)  # Minimum 1 second
        
        self.power_samples: List[float] = []
        self._consecutive_failures = 0
        self._max_failures = 5
        
        # Create persistent session with retry logic
        self.session = self._create_session()
        
        self._verify_connection()
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy and connection pooling."""
        session = requests.Session()
        
        # Retry strategy: 3 retries with exponential backoff
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,  # 0.5s, 1s, 2s
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        # Single connection pool (reuse connection)
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=1,
            pool_maxsize=1
        )
        
        session.mount("http://", adapter)
        
        # Keep-alive header
        session.headers.update({
            'Connection': 'keep-alive'
        })
        
        return session
    
    def _verify_connection(self):
        """Verify wattmeter is accessible."""
        try:
            response = self.session.get(self.endpoint, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            # Verify output exists
            outputs = data.get('Outputs', [])
            if not outputs or len(outputs) < self.output_id:
                raise ValueError(f"Output {self.output_id} not found")
            
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
            response = self.session.get(self.endpoint, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            outputs = data.get('Outputs', [])
            if len(outputs) < self.output_id:
                return None
            
            output = outputs[self.output_id - 1]
            power_watts = output.get('Load', 0)
            
            # Reset failure counter on success
            self._consecutive_failures = 0
            
            return power_watts
            
        except Exception as e:
            self._consecutive_failures += 1
            
            # Only print warning every few failures to reduce noise
            if self._consecutive_failures <= 3 or self._consecutive_failures % 10 == 0:
                print(f"Warning: Wattmeter read error ({self._consecutive_failures}): {e}")
            
            # Try to recreate session if too many failures
            if self._consecutive_failures == self._max_failures:
                print("⚠️  Recreating wattmeter session...")
                self.session.close()
                self.session = self._create_session()
            
            return None
    
    def start_monitoring(self):
        """Start power monitoring."""
        self.power_samples = []
        self._consecutive_failures = 0
    
    def add_sample(self):
        """Add a power sample."""
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
        
        return metrics
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics."""
        return self.stop_monitoring()
    
    def shutdown(self):
        """Cleanup - close session."""
        if self.session:
            self.session.close()


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
            except Exception as e:
                print(f"Warning: Wattmeter sampling error: {e}")
            
            # Always sleep, even on error
            time.sleep(self.wattmeter.polling_interval)
    
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
            self.thread.join(timeout=2.0)
        return self.wattmeter.stop_monitoring()


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("Testing WattmeterMonitor with persistent session...")
    print("=" * 60)
    
    try:
        wattmeter = WattmeterMonitor(
            ip="10.4.60.25",
            output_id=1,
            timeout=10,
            polling_interval=1.0
        )
        
        print("\n📊 Sampling for 10 seconds (1 sample/sec)...")
        wattmeter.start_monitoring()
        
        for i in range(10):
            wattmeter.add_sample()
            if wattmeter.power_samples:
                print(f"   Sample {i+1}: {wattmeter.power_samples[-1]} W")
            else:
                print(f"   Sample {i+1}: (failed)")
            time.sleep(1)
        
        stats = wattmeter.stop_monitoring()
        
        print(f"\n📈 Results:")
        print(f"   Samples collected: {stats.get('samples_count', 0)}")
        print(f"   Mean power: {stats.get('system_power_mean_watts', 0):.1f} W")
        print(f"   Peak power: {stats.get('system_power_peak_watts', 0):.1f} W")
        print(f"   Energy: {stats.get('system_energy_joules', 0):.1f} J")
        
        wattmeter.shutdown()
        print("\n✅ WattmeterMonitor test passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")