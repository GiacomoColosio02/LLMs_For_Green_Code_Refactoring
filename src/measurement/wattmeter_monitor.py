"""
System power monitoring using Wattmeter (NETIO PowerBOX).
Accesses wattmeter via HTTP API at 147.83.72.195.
"""
import requests
import time
from typing import Dict


class WattmeterMonitor:
    """Monitor system power using NETIO PowerBOX wattmeter."""
    
    def __init__(self, url: str = "http://147.83.72.195/netio.json", output_id: int = 1):
        """
        Initialize wattmeter monitor.
        
        Args:
            url: Wattmeter API endpoint
            output_id: Output ID (server connected to output 1)
        """
        self.url = url
        self.output_id = output_id
        self.initial_energy_wh = None
        self.initial_timestamp = None
    
    def _get_current_state(self) -> Dict:
        """
        Get current wattmeter state.
        
        Returns:
            Dictionary with Load (W) and Energy (Wh cumulative)
        """
        try:
            response = requests.get(self.url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Find output by ID
            for output in data['Outputs']:
                if output['ID'] == self.output_id:
                    return {
                        'load_watts': output['Load'],
                        'energy_wh': output['Energy'],
                        'timestamp': time.time()
                    }
            
            raise ValueError(f"Output {self.output_id} not found in wattmeter response")
            
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to contact wattmeter: {e}")
    
    def start_monitoring(self):
        """Start monitoring - record initial state."""
        state = self._get_current_state()
        self.initial_energy_wh = state['energy_wh']
        self.initial_timestamp = state['timestamp']
    
    def stop_monitoring(self) -> Dict:
        """
        Stop monitoring and return metrics.
        
        Returns:
            Dictionary with system energy metrics
        """
        if self.initial_energy_wh is None:
            raise RuntimeError("Monitoring not started. Call start_monitoring() first.")
        
        final_state = self._get_current_state()
        
        # Calculate energy consumed
        energy_wh = final_state['energy_wh'] - self.initial_energy_wh
        energy_joules = energy_wh * 3600  # Wh to Joules
        
        duration_seconds = final_state['timestamp'] - self.initial_timestamp
        
        metrics = {
            'system_energy_joules': energy_joules,
            'system_power_watts': energy_joules / duration_seconds if duration_seconds > 0 else 0,
            'duration_seconds': duration_seconds,
            'initial_load_watts': None,  # Not tracked
            'final_load_watts': final_state['load_watts']
        }
        
        return metrics


if __name__ == "__main__":
    # Test
    print("Testing WattmeterMonitor...")
    print("Make sure you're on gaissa server with access to wattmeter!")
    
    monitor = WattmeterMonitor()
    
    print("Starting monitoring...")
    monitor.start_monitoring()
    
    print("Waiting 5 seconds...")
    time.sleep(5)
    
    print("Stopping monitoring...")
    metrics = monitor.stop_monitoring()
    
    print(f"System Energy: {metrics['system_energy_joules']:.2f} J")
    print(f"System Power: {metrics['system_power_watts']:.2f} W")
    print(f"Duration: {metrics['duration_seconds']:.2f} s")
