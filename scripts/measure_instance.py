"""
Measure a single SWE-Perf instance with all metrics.
Optimized for compatibility with legacy scientific Python projects.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import subprocess
import tempfile
import shutil
from typing import Dict, Optional, List
from src.utils.config import load_config
from src.measurement.collector import MetricsCollector

# Python version to use for virtual environments (matching SWE-Perf paper)
PYTHON_EXECUTABLE = "python3.9"

# Version constraints for compatibility with legacy scientific Python
COMPATIBLE_VERSIONS = {
    'setuptools': '<70',      # Keep dep_util module (removed in 70+)
    'numpy': '<2.0',          # Keep np.product, np.cumproduct (removed in 2.0)
    'matplotlib': '<3.9',     # Keep register_cmap (removed in 3.9)
}


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
        
        # Verify Python 3.9 is available
        try:
            result = subprocess.run(
                [PYTHON_EXECUTABLE, '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ Using {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                f"❌ {PYTHON_EXECUTABLE} not found! "
                "SWE-Perf requires Python 3.9 for compatibility."
            )
        
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
        print(f"  📦 Creating virtual environment with {PYTHON_EXECUTABLE}...")
        
        venv_path = repo_path / "venv_sweperf"
        venv_pip = str(venv_path / 'bin' / 'pip')
        
        try:
            # Create virtual environment with Python 3.9
            subprocess.run(
                [PYTHON_EXECUTABLE, '-m', 'venv', str(venv_path)],
                check=True,
                timeout=60
            )
            print(f"  ✅ Virtual environment created")
            
            # Upgrade pip and install base packages with version constraints
            # CRITICAL: setuptools<70 keeps dep_util module
            print(f"  📦 Installing base packages (setuptools{COMPATIBLE_VERSIONS['setuptools']})...")
            subprocess.run(
                [venv_pip, 'install', '--upgrade', 
                 'pip', f"setuptools{COMPATIBLE_VERSIONS['setuptools']}", 'wheel'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120
            )
            
            # Install build dependencies for scientific Python projects
            # These are needed when using --no-build-isolation
            print(f"  📦 Installing build dependencies (numpy{COMPATIBLE_VERSIONS['numpy']})...")
            subprocess.run(
                [venv_pip, 'install',
                 'extension_helpers', 'setuptools_scm', 'cython', 
                 f"numpy{COMPATIBLE_VERSIONS['numpy']}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180
            )
            
            # Install package with dependencies
            # Use --no-build-isolation to use our setuptools<70 instead of isolated env
            print(f"  📦 Installing project dependencies (version: {version})...")
            subprocess.run(
                [venv_pip, 'install', '-e', '.', '--no-build-isolation'],
                cwd=repo_path,
                check=True,
                timeout=600  # 10 minutes timeout
            )
            
            # Install test dependencies with version constraints
            # CRITICAL: matplotlib<3.9 keeps register_cmap
            print(f"  📦 Installing test dependencies (matplotlib{COMPATIBLE_VERSIONS['matplotlib']})...")
            subprocess.run(
                [venv_pip, 'install', 
                 'pytest', 'hypothesis', 'scipy', 'pytest-astropy', 'urllib3',
                 f"matplotlib{COMPATIBLE_VERSIONS['matplotlib']}",
                 f"numpy{COMPATIBLE_VERSIONS['numpy']}"],  # Re-enforce numpy constraint
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180
            )
            
            print(f"  ✅ Dependencies installed")
            return venv_path
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  ⚠️  Warning: Could not install dependencies: {e}")
            return None
    
    def measure_single_test(
        self,
        collector: MetricsCollector,
        test_name: str,
        repo_path: Path,
        venv_path: Path,
        repetitions: int
    ) -> Optional[Dict]:
        """
        Measure a single test with error handling.
        
        Args:
            collector: MetricsCollector instance
            test_name: Name of the test to run
            repo_path: Path to repository
            venv_path: Path to virtual environment
            repetitions: Number of repetitions
            
        Returns:
            Test results dict or None if failed
        """
        try:
            # Build pytest command using venv python
            pytest_bin = venv_path / 'bin' / 'python'
            test_command = f"cd {repo_path} && {pytest_bin} -m pytest {test_name} -v"
            
            # Measure test execution
            test_results = collector.measure_test_execution(
                test_command=test_command,
                repetitions=repetitions
            )
            
            test_results['test_name'] = test_name
            test_results['status'] = 'success'
            return test_results
            
        except Exception as e:
            print(f"    ❌ Test failed: {str(e)[:100]}")
            return {
                'test_name': test_name,
                'status': 'failed',
                'error': str(e)[:500]
            }
    
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
            return {'status': 'dependency_failed'}
        
        # Get test commands
        efficiency_tests = instance['efficiency_test']
        
        if not efficiency_tests:
            print(f"  ⚠️  No efficiency tests found!")
            return {'status': 'no_tests'}
        
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
        
        # Measure each test (with individual error handling)
        all_test_results = []
        successful_tests = 0
        failed_tests = 0
        
        for i, test_name in enumerate(efficiency_tests):
            print(f"\n  📝 Test {i+1}/{len(efficiency_tests)}: {test_name}")
            
            result = self.measure_single_test(
                collector=collector,
                test_name=test_name,
                repo_path=repo_path,
                venv_path=venv_path,
                repetitions=self.config['measurement']['repetitions']
            )
            
            if result and result.get('status') == 'success':
                successful_tests += 1
            else:
                failed_tests += 1
            
            all_test_results.append(result)
        
        print(f"\n  📊 Test summary: {successful_tests} passed, {failed_tests} failed")
        
        # Combine results
        results = {
            'commit': commit,
            'commit_type': commit_type,
            'baseline': baseline,
            'tests': all_test_results,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'status': 'success' if successful_tests > 0 else 'all_tests_failed'
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
            return None
        
        print(f"\n📊 Instance info:")
        print(f"  Repository: {instance['repo']}")
        print(f"  Base commit: {instance['base_commit'][:8]}")
        print(f"  Head commit: {instance['head_commit'][:8]}")
        print(f"  Efficiency tests: {len(instance['efficiency_test'])}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
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
        finally:
            # Cleanup with ignore_errors (survives permission errors)
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Check if we have any valid results
        base_ok = base_results.get('status') == 'success'
        head_ok = head_results.get('status') == 'success'
        
        if not base_ok and not head_ok:
            print(f"\n❌ Both commits failed - not saving results")
            return None
        
        # Combine all results
        final_results = {
            'instance_id': instance_id,
            'repo': instance['repo'],
            'base_commit': instance['base_commit'],
            'head_commit': instance['head_commit'],
            'base_measurements': base_results,
            'head_measurements': head_results,
            'original_duration_changes': instance['duration_changes'],
            'measurement_status': {
                'base_ok': base_ok,
                'head_ok': head_ok,
                'total_tests': len(instance['efficiency_test']),
                'base_successful': base_results.get('successful_tests', 0),
                'head_successful': head_results.get('successful_tests', 0)
            }
        }
        
        # Save results
        output_path = Path(output_dir) / instance_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / "measurements.json"
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"✅ MEASUREMENT COMPLETE!")
        print(f"  Base: {base_results.get('successful_tests', 0)} tests passed")
        print(f"  Head: {head_results.get('successful_tests', 0)} tests passed")
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