"""
Energy and carbon monitoring using CodeCarbon.
"""
import time
from typing import Dict, Optional
from codecarbon import EmissionsTracker
from src.utils.config import load_config, get_grid_intensity


class EnergyMonitor:
    """Monitor energy consumption and carbon emissions."""
    
    def __init__(
        self,
        country_code: Optional[str] = None,
        measure_power_secs: float = 0.1,
        tracking_mode: str = "process"
    ):
        """
        Initialize energy monitor.
        
        Args:
            country_code: ISO country code for grid intensity (e.g., 'ESP', 'ITA')
            measure_power_secs: Power sampling interval
            tracking_mode: 'process' or 'machine'
        """
        self.country_code = country_code
        self.measure_power_secs = measure_power_secs
        self.tracking_mode = tracking_mode
        
        # Get grid intensity
        self.grid_intensity = get_grid_intensity(country_code)
        
        # Initialize tracker (will be created on start)
        self.tracker: Optional[EmissionsTracker] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def start(self):
        """Start energy tracking."""
        self.tracker = EmissionsTracker(
            measure_power_secs=self.measure_power_secs,
            save_to_file=False,  # We handle file saving
            logging_logger=None,  # Disable logging
            tracking_mode=self.tracking_mode,
            log_level="error"
        )
        
        self.start_time = time.time()
        self.tracker.start()
        
    def stop(self) -> Dict[str, float]:
        """
        Stop tracking and return metrics.
        
        Returns:
            Dictionary with energy and carbon metrics
        """
        if self.tracker is None:
            raise RuntimeError("Tracker not started. Call start() first.")
        
        self.end_time = time.time()
        
        # Stop tracker and get emissions (in kg CO2e)
        emissions_kg = self.tracker.stop()
        
        # Get energy consumed (in kWh)
        energy_kwh = self.tracker.final_emissions_data.energy_consumed
        
        # Calculate duration
        duration_sec = self.end_time - self.start_time
        
        # Convert to our units
        energy_joules = energy_kwh * 3.6e6  # kWh to Joules
        carbon_grams = emissions_kg * 1000   # kg to grams
        
        # Calculate mean power draw
        mean_power_watts = energy_joules / duration_sec if duration_sec > 0 else 0.0
        
        # Calculate energy efficiency (placeholder, will be set by caller)
        # Useful work = 1 test execution
        energy_efficiency = 1.0 / energy_joules if energy_joules > 0 else 0.0
        
        metrics = {
            'energy_consumption_joules': energy_joules,
            'energy_consumption_kwh': energy_kwh,
            'mean_power_draw_watts': mean_power_watts,
            'carbon_emissions_grams': carbon_grams,
            'carbon_emissions_kg': emissions_kg,
            'energy_efficiency_tests_per_joule': energy_efficiency,
            'duration_seconds': duration_sec,
            'grid_intensity_g_per_kwh': self.grid_intensity,
            'country_code': self.country_code if self.country_code else 'default'
        }
        
        return metrics


def measure_energy(func, country_code: Optional[str] = None) -> Dict[str, float]:
    """
    Measure energy consumption of a function.
    
    Args:
        func: Function to measure
        country_code: ISO country code for carbon intensity
        
    Returns:
        Dictionary with energy metrics
    """
    config = load_config()
    
    monitor = EnergyMonitor(
        country_code=country_code,
        measure_power_secs=config['energy']['measure_power_secs'],
        tracking_mode=config['energy']['tracking_mode']
    )
    
    monitor.start()
    
    # Execute function
    result = func()
    
    metrics = monitor.stop()
    
    return metrics, result


if __name__ == "__main__":
    # Test the energy monitor
    print("Testing Energy Monitor...")
    print("Monitoring energy for 3 seconds of CPU work...\n")
    
    def dummy_work():
        """Simulate some CPU work."""
        total = 0
        for i in range(3):
            # Do some computation
            for j in range(1000000):
                total += j
            time.sleep(1)
        return total
    
    # Measure energy
    monitor = EnergyMonitor(country_code='ESP')
    monitor.start()
    
    result = dummy_work()
    
    metrics = monitor.stop()
    
    print("⚡ Energy Metrics:")
    print(f"  Energy: {metrics['energy_consumption_joules']:.4f} J")
    print(f"  Energy: {metrics['energy_consumption_kwh']:.8f} kWh")
    print(f"  Power: {metrics['mean_power_draw_watts']:.2f} W")
    print(f"  Carbon: {metrics['carbon_emissions_grams']:.6f} g CO2e")
    print(f"  Efficiency: {metrics['energy_efficiency_tests_per_joule']:.2f} tests/J")
    print(f"  Duration: {metrics['duration_seconds']:.2f} s")
    print(f"  Grid (Spain): {metrics['grid_intensity_g_per_kwh']} g/kWh")
    
    print(f"\n  Dummy work result: {result}")
    print("\n✨ Energy monitor test passed!")