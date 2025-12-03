"""
Measure a single SWE-Perf instance with all metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import subprocess
import tempfile
import shutil
from typing import Dict, Optional
from src.utils.config import load_config
from src.measurement.collector import MetricsCollector


class SWEPerfMeasurer:
    """Measure SWE-Perf instance with green metrics."""
    
    def __init__(self, dataset_path: str, country_code: Optional[str] = None):
        """
        Initialize measurer.
        
        Args:
            dataset_path: Path to SWE-Perf JSON dataset
            country_code: ISO country code for carbon intensity
        """
        self.dataset_path = Path(dataset_path)
        self.country_code = country_code
        self.config = load_config()
        
        # Load dataset
        print(f"📂 Loading dataset from {self.dataset_path}...")
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)
        print(f"✅ Loaded {len(self.dataset)} instances")
    
    def get_instance(self, instance_id: str) -> Optional[Dict]:
        """
        Get instance by ID.
        
        Args:
            instance_id: Instance identifier (e.g., 'astropy__astropy-16065')
            
        Returns:
            Instance dictionary or None if not found
        """
        for instance in self.dataset:
            if instance['instance_id'] == instance_id:
                return instance
        return None
    
    def setup_repository(self, instance: Dict, temp_dir: Path, commit: str) -> Path:
        """
        Clone repository and checkout specific commit.
        
        Args:
            instance: SWE-Perf instance
            temp_dir: Temporary directory for cloning
            commit: Commit hash to checkout
            
        Returns:
            Path to repository
        """
        repo_name = instance['repo']
        repo_url = f"https://github.com/{repo_name}.git"
        
        # Use commit hash as subdirectory to avoid conflicts between base/head
        repo_path = temp_dir / f"{repo_name.split('/')[-1]}_{commit[:8]}"
        
        print(f"  📦 Cloning {repo_name}...")
        
        # Clone repository (FULL clone to reach old commits)
        subprocess.run(
            ['git', 'clone', repo_url, str(repo_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        # Fetch specific commit (might not be in default branches)
        print(f"  🔀 Fetching commit {commit[:8]}...")
        subprocess.run(
            ['git', 'fetch', 'origin', commit],
            cwd=repo_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False  # Don't fail if already present
        )
        
        # Checkout specific commit
        print(f"  🔀 Checking out commit {commit[:8]}...")
        subprocess.run(
            ['git', 'checkout', commit],
            cwd=repo_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        
        return repo_path
    
    def install_dependencies(self, repo_path: Path, version: str) -> Optional[Path]:
        """
        Create virtual environment and install package dependencies.
        
        Args:
            repo_path: Path to repository
            version: Version string from SWE-Perf
            
        Returns:
            Path to venv directory, or None if failed
        """
        print(f"  📦 Creating virtual environment...")
        
        venv_path = repo_path / "venv_sweperf"
        
        try:
            # Create virtual environment
            subprocess.run(
                ['python3', '-m', 'venv', str(venv_path)],
                check=True,
                timeout=60
            )
            print(f"  ✅ Virtual environment created")
            
            # Upgrade pip in venv
            subprocess.run(
                [str(venv_path / 'bin' / 'pip'), 'install', '--upgrade', 'pip'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60
            )
            
            # Install package with dependencies
            print(f"  📦 Installing dependencies (version: {version})...")
            subprocess.run(
                [str(venv_path / 'bin' / 'pip'), 'install', '-e', '.'],
                cwd=repo_path,
                check=True,
                timeout=600  # 10 minutes timeout
            )
            
            # Install test dependencies (pytest, hypothesis, scipy)
            print(f"  📦 Installing test dependencies...")
            subprocess.run(
                [str(venv_path / 'bin' / 'pip'), 'install', 
                 'pytest>=8.0', 'hypothesis', 'scipy', 'pytest-astropy'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120
            )
            
            # Fallback: downgrade numpy for old repos if needed
            try:
                subprocess.run(
                    [str(venv_path / 'bin' / 'pip'), 'install', 'numpy<2.0'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=60
                )
            except:
                pass  # If this fails, not critical
            
            print(f"  ✅ Dependencies installed")
            return venv_path
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  ⚠️  Warning: Could not install dependencies: {e}")
            return None
    
    def measure_commit(
        self,
        instance: Dict,
        commit: str,
        commit_type: str,
        temp_dir: Path
    ) -> Dict:
        """
        Measure metrics for a specific commit.
        
        Args:
            instance: SWE-Perf instance
            commit: Commit hash
            commit_type: 'base' or 'head'
            temp_dir: Temporary directory
            
        Returns:
            Dictionary with all measurements
        """
        print(f"\n🔬 Measuring {commit_type} commit...")
        print("=" * 60)
        
        # Setup repository
        repo_path = self.setup_repository(instance, temp_dir, commit)
        
        # Install dependencies and get venv path
        venv_path = self.install_dependencies(repo_path, instance['version'])
        
        if venv_path is None:
            print(f"  ⚠️  Skipping measurements - dependencies failed")
            return {}
        
        # Get test commands
        efficiency_tests = instance['efficiency_test']
        
        if not efficiency_tests:
            print(f"  ⚠️  No efficiency tests found!")
            return {}
        
        print(f"  🧪 Found {len(efficiency_tests)} efficiency tests")
        
        # Initialize collector
        collector = MetricsCollector(
            instance_id=instance['instance_id'],
            country_code=self.country_code
        )
        
        # Measure baseline
        baseline = collector.measure_baseline(
            duration=self.config['measurement']['baseline_duration_sec']
        )
        
        # Measure each test
        all_test_results = []
        
        for i, test_name in enumerate(efficiency_tests):
            print(f"\n  📝 Test {i+1}/{len(efficiency_tests)}: {test_name}")
            
            # Build pytest command using venv pytest
            pytest_bin = venv_path / 'bin' / 'python'
            test_command = f"cd {repo_path} && {pytest_bin} -m pytest {test_name} -v"
            
            # Measure test execution
            test_results = collector.measure_test_execution(
                test_command=test_command,
                repetitions=self.config['measurement']['repetitions']
            )
            
            test_results['test_name'] = test_name
            all_test_results.append(test_results)
        
        # Combine results
        results = {
            'commit': commit,
            'commit_type': commit_type,
            'baseline': baseline,
            'tests': all_test_results
        }
        
        # Cleanup
        shutil.rmtree(repo_path, ignore_errors=True)
        
        return results
    
    def measure_instance(self, instance_id: str, output_dir: str = "data/raw/measurements"):
        """
        Measure a single SWE-Perf instance.
        
        Args:
            instance_id: Instance identifier
            output_dir: Directory to save measurements
        """
        print("=" * 60)
        print(f"🎯 MEASURING INSTANCE: {instance_id}")
        print("=" * 60)
        
        # Get instance
        instance = self.get_instance(instance_id)
        if instance is None:
            print(f"❌ Instance '{instance_id}' not found in dataset!")
            return
        
        print(f"\n📊 Instance info:")
        print(f"  Repository: {instance['repo']}")
        print(f"  Base commit: {instance['base_commit'][:8]}")
        print(f"  Head commit: {instance['head_commit'][:8]}")
        print(f"  Efficiency tests: {len(instance['efficiency_test'])}")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            temp_path = Path(temp_dir)
            
            # Measure base commit
            base_results = self.measure_commit(
                instance=instance,
                commit=instance['base_commit'],
                commit_type='base',
                temp_dir=temp_path
            )
            
            # Measure head commit
            head_results = self.measure_commit(
                instance=instance,
                commit=instance['head_commit'],
                commit_type='head',
                temp_dir=temp_path
            )
        
        # Combine all results
        final_results = {
            'instance_id': instance_id,
            'repo': instance['repo'],
            'base_commit': instance['base_commit'],
            'head_commit': instance['head_commit'],
            'base_measurements': base_results,
            'head_measurements': head_results,
            'original_duration_changes': instance['duration_changes']
        }
        
        # Save results
        output_path = Path(output_dir) / instance_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"measurements.json"
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"✅ MEASUREMENT COMPLETE!")
        print(f"💾 Results saved to: {output_file}")
        print("=" * 60)
        
        return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Measure a single SWE-Perf instance with green metrics"
    )
    parser.add_argument(
        '--instance',
        type=str,
        required=True,
        help='Instance ID to measure (e.g., astropy__astropy-16065)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/original/swe_perf_original_20251124.json',
        help='Path to SWE-Perf dataset JSON'
    )
    parser.add_argument(
        '--country',
        type=str,
        default=None,
        help='ISO country code for carbon intensity (e.g., ESP, ITA)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/measurements',
        help='Output directory for measurements'
    )
    
    args = parser.parse_args()
    
    # Create measurer
    measurer = SWEPerfMeasurer(
        dataset_path=args.dataset,
        country_code=args.country
    )
    
    # Measure instance
    measurer.measure_instance(
        instance_id=args.instance,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
