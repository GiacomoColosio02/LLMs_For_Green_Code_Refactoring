"""
CPU Energy monitoring using EnergiBridge.
Requires sudo privileges and energibridge binary.
"""
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
from typing import Optional


class CPUEnergyMonitor:
    """Monitor CPU energy using EnergiBridge."""
    
    def __init__(self, energibridge_path: str = "~/LLMs_For_Green_Code_Refactoring/energibridge"):
        """
        Initialize CPU energy monitor.
        
        Args:
            energibridge_path: Path to energibridge binary
        """
        self.energibridge_path = Path(energibridge_path).expanduser()
        
        if not self.energibridge_path.exists():
            raise FileNotFoundError(f"EnergiBridge not found at: {self.energibridge_path}")
    
    def measure_energy(self, command: str, output_csv: Optional[Path] = None) -> dict:
        """
        Measure CPU energy for a command execution.
        
        Args:
            command: Shell command to execute and measure
            output_csv: Optional path for CSV output (temp file if None)
            
        Returns:
            Dictionary with CPU energy metrics
        """
        # Create temp CSV if not provided
        if output_csv is None:
            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            output_csv = Path(temp_csv.name)
            temp_csv.close()
        
        # Build energibridge command
        energibridge_cmd = f"sudo {self.energibridge_path} -o {output_csv} -- {command}"
        
        # Execute
        try:
            result = subprocess.run(
                energibridge_cmd,
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"EnergiBridge failed: {e.stderr}")
        
        # Parse CSV output
        try:
            df = pd.read_csv(output_csv)
            
            # EnergiBridge provides cumulative energy
            # Columns: timestamp, energy_joules
            initial_energy = df['energy_joules'].iloc[0]
            final_energy = df['energy_joules'].iloc[-1]
            
            cpu_energy_joules = final_energy - initial_energy
            duration_seconds = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
            
            metrics = {
                'cpu_energy_joules': cpu_energy_joules,
                'cpu_power_watts': cpu_energy_joules / duration_seconds if duration_seconds > 0 else 0,
                'duration_seconds': duration_seconds,
                'samples': len(df)
            }
            
            # Cleanup temp file
            if output_csv.name.startswith('/tmp/'):
                output_csv.unlink()
            
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to parse EnergiBridge output: {e}")


if __name__ == "__main__":
    # Test
    print("Testing CPUEnergyMonitor...")
    
    monitor = CPUEnergyMonitor()
    
    # Test with simple command
    metrics = monitor.measure_energy("sleep 2")
    
    print(f"CPU Energy: {metrics['cpu_energy_joules']:.2f} J")
    print(f"CPU Power: {metrics['cpu_power_watts']:.2f} W")
    print(f"Duration: {metrics['duration_seconds']:.2f} s")
