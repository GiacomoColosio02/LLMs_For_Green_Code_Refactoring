"""
CPU Energy monitoring using EnergiBridge.
Requires sudo privileges and energibridge binary.
"""
import subprocess
import tempfile
import pandas as pd
import os
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
        # Create temp CSV in current directory (not /tmp) to avoid permission issues
        cleanup_csv = False
        if output_csv is None:
            output_csv = Path(f"energibridge_temp_{os.getpid()}.csv")
            cleanup_csv = True
        
        # For complex commands with cd/&&, wrap in a bash script
        # EnergiBridge cannot handle shell operators like cd, &&, ||, etc.
        if 'cd ' in command or '&&' in command or '||' in command or ';' in command:
            # Create temporary bash script
            script_path = Path(f"energibridge_script_{os.getpid()}.sh")
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("set -e\n")  # Exit on error
                f.write(f"{command}\n")
            script_path.chmod(0o755)
            
            # Use bash script instead of direct command
            energibridge_cmd = f"sudo {self.energibridge_path} -o {output_csv} -- bash {script_path}"
        else:
            # Simple command, pass directly
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
            
            # Change ownership of CSV to current user so we can read it
            subprocess.run(f"sudo chown {os.getuid()}:{os.getgid()} {output_csv}", 
                          shell=True, check=True)
            
        except subprocess.CalledProcessError as e:
            # Cleanup on error
            if cleanup_csv and output_csv.exists():
                output_csv.unlink()
            if 'script_path' in locals() and script_path.exists():
                script_path.unlink()
            raise RuntimeError(f"EnergiBridge failed: {e.stderr}")
        finally:
            # Cleanup script if created
            if 'script_path' in locals() and script_path.exists():
                script_path.unlink()
        
        # Parse CSV output
        try:
            df = pd.read_csv(output_csv)
            
            # EnergiBridge provides cumulative CPU_ENERGY (J)
            if 'CPU_ENERGY (J)' not in df.columns:
                raise ValueError(f"Column 'CPU_ENERGY (J)' not found. Available: {df.columns.tolist()}")
            
            # Time is in milliseconds, convert to seconds
            initial_time_ms = df['Time'].iloc[0]
            final_time_ms = df['Time'].iloc[-1]
            duration_seconds = (final_time_ms - initial_time_ms) / 1e3
            
            # Energy is cumulative in Joules
            initial_energy = df['CPU_ENERGY (J)'].iloc[0]
            final_energy = df['CPU_ENERGY (J)'].iloc[-1]
            cpu_energy_joules = final_energy - initial_energy
            
            metrics = {
                'cpu_energy_joules': cpu_energy_joules,
                'cpu_power_watts': cpu_energy_joules / duration_seconds if duration_seconds > 0 else 0,
                'duration_seconds': duration_seconds,
                'samples': len(df)
            }
            
            # Cleanup temp file
            if cleanup_csv and output_csv.exists():
                output_csv.unlink()
            
            return metrics
            
        except Exception as e:
            # Cleanup on error
            if cleanup_csv and output_csv.exists():
                output_csv.unlink()
            raise RuntimeError(f"Failed to parse EnergiBridge output: {e}")


if __name__ == "__main__":
    # Test
    print("Testing CPUEnergyMonitor...")
    
    monitor = CPUEnergyMonitor()
    
    # Test with simple command
    print("Running 'sleep 2' test...")
    metrics = monitor.measure_energy("sleep 2")
    
    print(f"\n✅ Results:")
    print(f"CPU Energy: {metrics['cpu_energy_joules']:.2f} J")
    print(f"CPU Power: {metrics['cpu_power_watts']:.2f} W")
    print(f"Duration: {metrics['duration_seconds']:.2f} s")
    print(f"Samples: {metrics['samples']}")
